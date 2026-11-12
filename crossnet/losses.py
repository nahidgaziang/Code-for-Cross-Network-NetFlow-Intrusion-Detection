"""
Training objectives and test-time adaptation steps.

  coral_loss       Deep CORAL: match second-order statistics of source and target embeddings
  dann_loss        DANN: gradient-reversal domain discriminator
  irm_loss         IRMv1 penalty over source testbeds
  vrex_loss        V-REx: variance of per-testbed risks
  groupdro_loss    GroupDRO: online worst-group reweighting over source testbeds
  iw_dann_loss     importance-weighted DANN, simplified: class weights for the domain loss come
                   from the batch-level target prior estimate (not the confusion-matrix estimator
                   of the original IW-DANN)
  dann_recon_loss  DANN-R: DANN plus a feature-reconstruction decoder (a simplified variant written
                   for this study, not a reproduction of a published system)
  adabn, tent      re-estimate BatchNorm statistics / entropy-minimise BatchNorm affine
                   parameters on the unlabelled target buffer
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import DEVICE, to_dev


def _ce(logits, yb, cw):
    return F.cross_entropy(logits.float(), yb, weight=cw)


def _domain_risks(logits, yb, idx, D, doms, cw):
    """Class-weighted risk of each source domain in the batch (no host sync)."""
    d = D[idx]
    loss = F.cross_entropy(logits.float(), yb, weight=cw, reduction="none")
    w = cw[yb] if cw is not None else torch.ones_like(loss)
    risks = []
    for g in doms:
        m = (d == g).float()
        risks.append((loss * m).sum() / (w * m).sum().clamp_min(1e-8))
    return torch.stack(risks)


def coral_loss(lam):
    def f(model, xb, yb, idx, xt, step, ctx):
        logits, z = model(torch.cat([xb, xt]))
        n = len(xb)
        zs, zt = z[:n].float(), z[n:].float()
        mean_diff = (zs.mean(0) - zt.mean(0)).pow(2).mean()
        cov_diff = (torch.cov(zs.T) - torch.cov(zt.T)).pow(2).mean()
        return _ce(logits[:n], yb, ctx["cw"]) + lam * (mean_diff + cov_diff)
    return f


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef * g, None


class WithDiscriminator(nn.Module):
    """Backbone plus a domain discriminator; forward/embed delegate to the backbone,
    so scoring and the optimizer (which sees both) need no special handling."""

    def __init__(self, net, h):
        super().__init__()
        self.net = net
        self.disc = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)

    def embed(self, x):
        return self.net.embed(x)


class WithDiscriminatorAndDecoder(nn.Module):
    """Backbone plus domain discriminator and autoencoder reconstruction decoder (DANN-R)."""

    def __init__(self, net, h, in_dim):
        super().__init__()
        self.net = net
        self.disc = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.decoder = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, in_dim))

    def forward(self, x):
        return self.net(x)

    def embed(self, x):
        return self.net.embed(x)


def dann_loss(lam):
    def f(model, xb, yb, idx, xt, step, ctx):
        logits, z = model(torch.cat([xb, xt]))
        n = len(xb)
        p = min(step / max(ctx["total_steps"], 1), 1.0)
        coef = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        dlog = model.disc(_GradReverse.apply(z, coef)).squeeze(1).float()
        dy = torch.cat([torch.zeros(n, device=z.device), torch.ones(len(xt), device=z.device)])
        return _ce(logits[:n], yb, ctx["cw"]) + lam * F.binary_cross_entropy_with_logits(dlog, dy)
    return f


def iw_dann_loss(lam, p_src_att=0.5):
    """Importance-Weighted DANN (Tachet des Combes et al., NeurIPS 2020) under label shift."""
    ps1 = max(float(p_src_att), 1e-4)
    ps0 = max(1.0 - ps1, 1e-4)

    def f(model, xb, yb, idx, xt, step, ctx):
        x_all = torch.cat([xb, xt])
        logits, z = model(x_all)
        n = len(xb)
        p = min(step / max(ctx["total_steps"], 1), 1.0)
        coef = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        dlog = model.disc(_GradReverse.apply(z, coef)).squeeze(1).float()
        dy = torch.cat([torch.zeros(n, device=z.device), torch.ones(len(xt), device=z.device)])

        # Target prior estimate from target batch predictions
        with torch.no_grad():
            pt_pred = torch.sigmoid(logits[n:, 1] - logits[n:, 0]) if logits.shape[1] > 1 else torch.sigmoid(logits[n:].squeeze(1))
            pt1 = float(pt_pred.mean().clamp(0.01, 0.99))
            pt0 = 1.0 - pt1
            beta0 = min(max(pt0 / ps0, 0.1), 10.0)
            beta1 = min(max(pt1 / ps1, 0.1), 10.0)
            w_src = torch.where(yb == 1, torch.tensor(beta1, device=z.device), torch.tensor(beta0, device=z.device))
            weights = torch.cat([w_src, torch.ones(len(xt), device=z.device)])

        adv_loss = F.binary_cross_entropy_with_logits(dlog, dy, weight=weights)
        return _ce(logits[:n], yb, ctx["cw"]) + lam * adv_loss
    return f


def dann_recon_loss(lam_adv, lam_recon):
    """Domain-Invariant NIDS (DANN-R, KBS 2023): adversarial alignment + feature reconstruction."""
    def f(model, xb, yb, idx, xt, step, ctx):
        x_all = torch.cat([xb, xt])
        logits, z = model(x_all)
        n = len(xb)
        p = min(step / max(ctx["total_steps"], 1), 1.0)
        coef = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        dlog = model.disc(_GradReverse.apply(z, coef)).squeeze(1).float()
        dy = torch.cat([torch.zeros(n, device=z.device), torch.ones(len(xt), device=z.device)])
        adv_loss = F.binary_cross_entropy_with_logits(dlog, dy)

        # Feature reconstruction loss
        x_recon = model.decoder(z)
        recon_loss = F.mse_loss(x_recon, x_all)

        return _ce(logits[:n], yb, ctx["cw"]) + lam_adv * adv_loss + lam_recon * recon_loss
    return f


def _ramped(lam, step, ctx):
    return lam * min(1.0, step / max(ctx["steps_per_epoch"], 1))


def irm_loss(lam, d_tr):
    D, doms = to_dev(d_tr, torch.long), [int(g) for g in np.unique(d_tr)]

    def f(model, xb, yb, idx, xt, step, ctx):
        logits, _ = model(xb)
        scale = torch.ones((), device=DEVICE, requires_grad=True)
        risks = _domain_risks(logits * scale, yb, idx, D, doms, ctx["cw"])
        pen = sum(torch.autograd.grad(r, scale, create_graph=True)[0].pow(2) for r in risks)
        w = _ramped(lam, step, ctx)
        return (risks.mean() + w * pen) / max(1.0, w)
    return f


def vrex_loss(lam, d_tr):
    D, doms = to_dev(d_tr, torch.long), [int(g) for g in np.unique(d_tr)]

    def f(model, xb, yb, idx, xt, step, ctx):
        logits, _ = model(xb)
        risks = _domain_risks(logits, yb, idx, D, doms, ctx["cw"])
        w = _ramped(lam, step, ctx)
        return (risks.mean() + w * ((risks - risks.mean()) ** 2).mean()) / max(1.0, w)
    return f


def groupdro_loss(eta, d_tr):
    D, doms = to_dev(d_tr, torch.long), [int(g) for g in np.unique(d_tr)]
    q = [torch.full((len(doms),), 1.0 / len(doms), device=DEVICE)]

    def f(model, xb, yb, idx, xt, step, ctx):
        logits, _ = model(xb)
        risks = _domain_risks(logits, yb, idx, D, doms, ctx["cw"])
        qn = q[0] * torch.exp(eta * risks.detach())
        q[0] = qn / qn.sum()
        return (q[0] * risks).sum()
    return f


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


@torch.no_grad()
def adabn(model, X_ref):
    """Replace BatchNorm running statistics by the exact statistics of X_ref (one full-batch pass)."""
    for m in _bn_layers(model):
        m.reset_running_stats()
        m.momentum = None  # cumulative average: one pass = exact statistics of X_ref
    model.eval()
    for m in _bn_layers(model):
        m.train()
    model(to_dev(X_ref))
    model.eval()


def tent(model, X_buf, lr, steps):
    """Entropy minimization on the buffer, updating only BatchNorm affine parameters (batch statistics),
    followed by AdaBN so that inference uses the buffer's statistics."""
    params = [p for m in _bn_layers(model) for p in (m.weight, m.bias)]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    xb = to_dev(X_buf)
    model.eval()
    for m in _bn_layers(model):
        m.train()
    for _ in range(steps):
        logits, _ = model(xb)
        p = F.softmax(logits.float(), 1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        ent.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    adabn(model, X_buf)
