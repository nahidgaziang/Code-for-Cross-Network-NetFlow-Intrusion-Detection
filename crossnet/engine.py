"""
Training engine for the neural methods.

The whole fold is kept on the GPU and mini-batches are index slices. Training uses AdamW with
early stopping on source-validation loss. Runs are bitwise reproducible: all random number
generators are seeded before model construction, cuBLAS uses a fixed workspace, and target
mini-batches for adaptation losses come from a separate generator (seed + 1).
Methods change the training objective through `loss_fn` (see crossnet/losses.py).
"""
import math
import os
import random


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(os.environ.get("NF_STRICT_DETERMINISM") == "1")


class TabularResNet(nn.Module):
    """Residual MLP backbone: width h, n_blocks residual blocks, BN + SiLU."""

    def __init__(self, d, h=96, n_blocks=1, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.SiLU()) for _ in range(n_blocks))
        self.head = nn.Linear(h, 2)

    def embed(self, x):
        z = self.inp(x)
        for b in self.blocks:
            z = z + b(z)
        return z

    def forward(self, x):
        z = self.embed(x)
        return self.head(z), z


def to_dev(a, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


def fit(model, X, y, X_val, y_val, *, sample_weight=None, loss_fn=None, target_X=None, class_balanced=True,
        lr=1e-3, weight_decay=1e-4, batch_size=1024, max_epochs=30, patience=4, amp=False, seed=0):
    """loss_fn(model, xb, yb, idx, xt, step, ctx) -> scalar replaces the default (class-weighted) cross-entropy;
    idx indexes the training arrays (e.g. domain ids), xt is a batch drawn with replacement from target_X
    (unlabeled target flows) or None, ctx holds {"cw", "steps_per_epoch", "total_steps"}. With loss_fn=None and"""
    seed_all(seed)
    model.to(DEVICE)
    Xt, yt = to_dev(X), to_dev(y, torch.long)
    Xv, yv = to_dev(X_val), to_dev(y_val, torch.long)
    wt = to_dev(sample_weight) if sample_weight is not None else None
    n, n1 = len(y), int(np.sum(y))
    cw = to_dev([n / (2.0 * max(n - n1, 1)), n / (2.0 * max(n1, 1))]) if class_balanced else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    Xtg = to_dev(target_X) if target_X is not None else None
    tgen = torch.Generator(device=DEVICE).manual_seed(seed + 1)  # separate stream: default path RNG unchanged
    ctx = {"cw": cw, "steps_per_epoch": n // batch_size, "total_steps": max_epochs * (n // batch_size)}
    best, best_state, bad, step = math.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE, generator=gen)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and DEVICE.type == "cuda"):
                if loss_fn is None:
                    logits, _ = model(Xt[idx])
                    loss = F.cross_entropy(logits.float(), yt[idx], weight=cw, reduction="none")
                    loss = (loss * wt[idx]).mean() if wt is not None else loss.mean()
                else:
                    xt = None
                    if Xtg is not None:
                        xt = Xtg[torch.randint(len(Xtg), (min(batch_size, len(Xtg)),), device=DEVICE, generator=tgen)]
                    loss = loss_fn(model, Xt[idx], yt[idx], idx, xt, step, ctx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        val = evaluate_loss(model, Xv, yv, cw)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs": epoch + 1, "best_val_loss": best, "steps": step}


@torch.no_grad()
def evaluate_loss(model, Xv, yv, cw=None, chunk=65536):
    model.eval()
    tot = 0.0
    for i in range(0, len(yv), chunk):
        logits, _ = model(Xv[i:i + chunk])
        tot += F.cross_entropy(logits.float(), yv[i:i + chunk], weight=cw, reduction="sum").item()
    return tot / len(yv)


@torch.no_grad()
def predict_proba(model, X, chunk=65536):
    model.eval()
    Xt = X if torch.is_tensor(X) else to_dev(X)
    out = [F.softmax(model(Xt[i:i + chunk])[0].float(), 1)[:, 1] for i in range(0, len(Xt), chunk)]
    return torch.cat(out).cpu().numpy()
