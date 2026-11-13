# Experiments

Each experiment below lists its question, design, command, outputs and the summed fitting time
measured on an RTX 3050 Laptop GPU. All follow-up experiments reuse the benchmark's selected
hyperparameters and, except the pairwise study, its evaluation streams. They must therefore run after
the benchmark has finished.

| ID | Question | Script | Output | Fit time |
|---|---|---|---|---|
| E1 | How do 18 methods transfer to an unseen testbed? | `python -m crossnet.runner --jobs ...` | `results/benchmark/` | ≈ 46 h |
| E2 | Is the failure network-driven or caused by novel attacks? | `analysis/failure_decomposition.py` | `results/benchmark/failure_decomposition*` | minutes |
| D | How much of the loss could any threshold recover? | `analysis/decision_layer.py` | `results/benchmark/analysis/decision_layer*` | minutes |
| E3 | How many labelled target flows match the window? | `experiments/feature_label_studies.py e3` | `results/e3/` | 0.2 h |
| E4 | Do results depend on identifier-like features? | `experiments/feature_label_studies.py e4` | `results/e4/` | 2.2 h |
| E5 | Within-network temporal training vs cross-network transfer | `experiments/within_vs_cross_network.py run` | `results/e5/` | 0.2 h |
| E5-T | Does a window collected before the monitored period still work? | `experiments/window_studies.py e5t` | `results/e5t/` | 0.1 h |
| E6 | Pairwise transfer; can a divergence audit predict the loss? | `experiments/pairwise_transfer.py run` | `results/e6/` | 0.4 h |
| E7 | Which feature groups are associated with target-benign false alarms? | `experiments/feature_groups.py` | `results/e7/` | not recorded |
| E9 | How small and how contaminated can the window be? | `experiments/window_studies.py e9` | `results/e9/` | 0.8 h |
| P | Can an attacker poison the window? | `experiments/window_poisoning.py run` | `results/poisoning/` | ≈ 2 h |
| T | Threshold rules for the window detectors | `analysis/window_thresholds.py` | `results/benchmark/analysis/s2_decisions*` | minutes |
| U | Does the window work when collected without label selection? | `experiments/unselected_window.py run` | `results/unselected/` | ≈ 0.5 h |
| D-P | How much of the irrecoverable loss is a prior effect? | `analysis/prior_controlled.py` | `results/benchmark/analysis/prior_controlled*` | minutes |
| C | Controls: one-class detectors on source benign flows, XGBoost with the window labelled benign, near-duplicates of window flows, poisoning vs. number of neighbours | `experiments/controls.py all` (or `oneclass`, `xgbwin`, `neardup`, `poisonk`) | `results/controls/` | not timed in full |
| L | Low-false-alarm range of all methods; rank-average fusion of KNN-win with source models | `analysis/low_fpr_and_fusion.py` | `results/benchmark/analysis/low_fpr.csv`, `fusion.csv` | minutes |

## E1 Benchmark

- **Design.** Leave-one-testbed-out over both releases: 18 methods, 12 tuning configurations each,
  10 evaluation seeds. See [PROTOCOL.md](PROTOCOL.md).
- **Running.** The runner is resumable and can run as several processes that claim jobs through lock
  files.
- **After it finishes:**
  - `tests/verify_benchmark.py` checks completeness and recomputes every metric from the saved scores;
  - `analysis/benchmark_tests.py` writes the main tables (`results/benchmark/analysis/main_*.csv`) and the
    hypothesis tests;
  - `analysis/duplicate_check.py` checks that the window detectors' results are not explained by
    evaluation flows that duplicate window flows.

## E2 Failure decomposition

- Attack families are harmonised across testbeds. A target family counts as *seen* if some source
  testbed contains it at ≥ 0.1% of its flows.
- **Reported:**
  - AUROC of target benign flows versus seen-family attacks, and versus novel-family attacks;
  - target benign FPR and attack TPR at a threshold set to 1% FPR on source-validation benign flows.

## D Decision layer

Applies the fixed and em rules, BBSE, the true-prior bound and the best-threshold bound to every saved
probabilistic run. It reports how much of the drop from source-validation Macro-F1 to target Macro-F1 a
perfect threshold could recover.

## E3 Label budget

- **Draws.** k ∈ {10, 30, 100, 300, 1000} labelled target flows, drawn uniformly (natural prior), nested
  across k, and disjoint from evaluation, buffer and window.
- **XGB-T** is trained on the target labels only. When all k flows belong to one class, it outputs a
  constant score (AUROC 0.5) and the run is flagged.
- **XGB-ST** adds the k flows to the source training set. Their total weight equals the source total
  weight, and class weights are balanced.

## E4 Identifier-free features

- Removes L4_SRC_PORT, L4_DST_PORT, DNS_QUERY_ID, ICMP_TYPE, ICMP_IPV4_TYPE and FTP_COMMAND_RET_CODE.
- Reruns KNN-win, IF-win, ERM-CB, XGB, RF and BRN on the benchmark folds with the benchmark
  hyperparameters.

## E5 Within- vs cross-network (NF-v3)

- **Training.** XGB, RF and ERM-CB are trained on labelled flows from the early part of the target's own
  timeline: start time ≤ the 20% or 50% quantile, up to 50,000 flows, outside the benchmark's evaluation,
  buffer and window sets.
- **Evaluation.** On the benchmark evaluation flows that start after the cut.
- **Comparison.** The same late flows are scored by the benchmark's cross-network models.

## E5-T Temporal window (NF-v3)

- **Window.** Drawn only from flows that start before the cut: 475 benign plus 25 attack flows.
- **Evaluation.** On the benchmark evaluation flows that start after the cut.
- **Comparators.** The benchmark's random-time window and all S0/S1 methods, scored on the same late
  flows.
- **BoT-IoT-v3 caveat.** It has fewer than 100 late benign flows at the 20% cut, so its AUROC is reported
  but excluded from the main comparison.

## E6 Pairwise transfer and audit

- **Runs.** 12 directed source → target pairs per release, plus the four in-distribution cases; XGB, RF
  and ERM-CB; 5 seeds.
- **Audit.** Spearman correlation of transfer loss with three benign-traffic divergence measures: mean
  PSI, mean normalised Wasserstein distance, and mean absolute standardised difference.

## E7 Feature groups

- XGBoost is retrained without one feature group at a time:
  - TTL;
  - TCP window;
  - timing;
  - all three plus ports.
- **Reported.** Target AUROC and target-benign FPR at a threshold calibrated to 1% FPR on source
  validation.

## E9 Window size and contamination

- **Grid.** Window size {50, 100, 200, 500, 1000, 2000} at 5% contamination, and contamination
  {0, 5, 10, 20, 30}% at size 500.
- **Detectors.** KNN-win and IF-win, both releases, 10 seeds.
- **Regression check.** Size 500 at 5% reproduces the benchmark exactly.

## P Window poisoning

- **Attack.** A fraction p ∈ {0, 1, 2, 5, 10, 20}% of the 500-flow window is replaced by flows of the
  target's most frequent attack family. The evaluation streams are the benchmark's.
- **Defence.** Trimming: drop the 10% most outlying window flows by leave-one-out score, then rerun the
  detector.
- **Primary metric.** Targeted-family AUROC, which does not depend on a threshold.
- **Secondary metrics:**
  - recall and benign FPR at the declared operating point (win_mad, Z = 3);
  - targeted recall at a matched benign FPR of 1% and 5%.
- All scores are saved.

## T Window threshold rules

- Evaluates win_mad over Z ∈ {1, 1.5, 2, 2.5, 3} and win_fpr over α ∈ {0.01, 0.05, 0.10, 0.20} from the
  saved scores, together with the best possible threshold, which uses labels.
- Z = 3 and α = 0.01 are the declared operating points; the other values form a sensitivity analysis
  computed on evaluation labels.
