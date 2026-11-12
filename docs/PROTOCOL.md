# Evaluation protocol

## Data

| Release | Testbeds | Features used |
|---|---|---|
| NF-v2 | NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2 | 41 = released schema minus the two IP addresses |
| NF-v3 | NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3, NF-CSE-CIC-IDS2018-v3 | 49 = released schema minus the two IP addresses and the two absolute timestamps |

For each testbed, `scripts/build_pools.py` draws:

- a uniform pool of 500,000 flows from the full release;
- a benign pool of up to 200,000 benign flows (fewer for BoT-IoT, which has few benign flows).

Every fold samples from these pools. The NF-v3 flow start time is stored next to the features (not as a
feature) for the temporal experiments.

## Folds (crossnet/data.py)

A fold is (release, target testbed, seed); the other three testbeds are the sources. All draws come from
`numpy.random.default_rng([seed, target index])`.

| Part | Size | Content | Visible to methods |
|---|---|---|---|
| Source training / validation | 100,000 flows per source testbed, split 80/20 per testbed | labelled source flows | yes, with labels |
| Target evaluation stream | 100,000 flows from the uniform pool | natural class prior | features only, at scoring time |
| Target buffer (S1) | 1,500 flows from the uniform pool, disjoint from evaluation | natural class prior | yes, unlabelled |
| Target window (S2) | 500 flows: 475 benign-pool flows outside evaluation and buffer, plus 25 attack flows (5%) | mostly benign | yes, unlabelled |

Target labels are kept in a separate object that only the metric code receives. `tests/test_folds.py`
checks disjointness and that no target flow reaches training or validation.

## Information settings

| Setting | What the method may use about the target |
|---|---|
| S0 | nothing (labelled source testbeds only) |
| S1 | the unlabelled buffer, whose class prior is the target's natural prior |
| S2 | the window: a short reference sample of mostly benign target traffic |
| S3 | k labelled target flows (label-budget study only) |

The settings differ in the information they grant, so comparisons across settings measure the value
of that information, not only the quality of an algorithm.

## Features and models

- **Transform.** ln(1 + max(0, x)) for every feature, then a z-score clipped to [-5, 5]. Statistics come
  from the pooled source training flows, except for the reference-based methods:
  - BRN: per-testbed benign statistics for sources, window statistics for the target;
  - DomNorm: per-testbed statistics for sources, buffer statistics for the target;
  - KNN-win: window statistics.

  Standard deviations are floored at 0.1 for the per-reference transforms.
- **Neural backbone.** A residual MLP: input layer, then `n_blocks` blocks of Linear–BatchNorm–SiLU, and a
  2-class head. Trained with AdamW, class-weighted cross-entropy (except plain ERM), up to 30 epochs, and
  early stopping on source validation loss (patience 4). The learning rate is scaled by
  sqrt(batch size / 256).
- **Search spaces.** Fixed, in `crossnet/methods.py`:

  | Model | Hyperparameters searched |
  |---|---|
  | MLP | learning rate 10^U(-3.5, -2.5), weight decay 10^U(-5, -3), dropout {0, 0.1, 0.2}, width {64, 96, 128, 256}, blocks {1, 2}, batch size {256, 1024} |
  | XGBoost | trees {200, 400}, depth {4, 6, 8}, learning rate 10^U(-1.5, -0.5), subsample {0.7, 1}, column sample {0.7, 1}, min child weight {1, 5} |
  | Random forest | trees {100, 300}, depth {12, 20, unlimited}, min leaf {1, 5}, max features {sqrt, 0.5} |
  | Isolation forest | trees {100, 300}, max samples {128, 256, 512}, max features {0.5, 1} |
  | kNN | k ∈ {1, 3, 5, 10, 20, 50} |
  | Method-specific | CORAL λ 10^U(-1, 1); DANN and IW-DANN λ 10^U(-2, 0); DANN-R λ_adv 10^U(-2, 0) and λ_rec 10^U(-3, -1); IRM and V-REx λ 10^U(-1, 3); GroupDRO η 10^U(-3, -1); TENT learning rate 10^U(-4, -2) and steps {1, 5, 10} |

- **Common random numbers.** Methods that share a learner share the same backbone draws.
- **IW-DANN and DANN-R are simplified variants, not reproductions of published systems.**
  - IW-DANN weights the domain loss by class ratios derived from a batch-level estimate of the target
    prior.
  - DANN-R adds a feature-reconstruction decoder to DANN.

## Tuning and evaluation (crossnet/runner.py)

**Tuning** uses the source testbeds only. For each of the 12 candidate configurations, the method is
trained on two sources and scored on the third, which acts as a pseudo-target with its own buffer,
window and evaluation split. The three rotations are averaged. The configuration with the highest mean
AUROC is selected; ties go to the higher PR-AUC. Tuning seed: 0.

**Evaluation.** The selected configuration is run with seeds 42, 1337, 2025, 2026, 2027, 7, 11, 101, 2024
and 31337, so each release has 4 targets × 10 seeds = 40 runs per method.

## Decision rules (crossnet/decision.py)

| Rule | Uses | Definition |
|---|---|---|
| fixed | nothing | threshold 0.5 |
| em | buffer (S1) | temperature scaling on source validation, Saerens EM estimate of the target prior on the buffer (clipped to [0.04, 0.96]), threshold maximising expected Macro-F1 |
| win_mad | window (S2) | median + 3 × 1.4826 × MAD of the window scores (log-odds scale for probabilities) |
| win_fpr | window (S2) | 99th percentile of the window scores |

Diagnostic upper bounds, which use target labels and are never used for selection:

- the true target prior plugged into em;
- the best possible threshold for target Macro-F1.

## Metrics and statistics

- **Primary metric:** AUROC on the target evaluation stream. It is threshold-free and does not depend
  on the class prior.
- **Secondary metrics:**
  - Macro-F1 and realised FPR/TPR under each rule;
  - TPR at 1% and 0.1% FPR;
  - PR-AUC;
  - recall per attack family.
- **Means** weight the four targets equally.
- **Intervals** are seed-bootstrap intervals: they resample the ten per-seed means, so they describe seed
  variation for these fixed testbeds, not uncertainty over unseen networks.
- **Paired comparisons** use Wilcoxon signed-rank tests over the 40 (target × seed) pairs of a release,
  with Holm correction within a family of hypotheses. A claim is reported as holding only if it holds in
  both releases. NF-v2 and NF-v3 come from the same four testbeds, so they are not independent
  replications.
