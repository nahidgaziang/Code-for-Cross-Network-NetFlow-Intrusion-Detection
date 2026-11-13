# Analysis plan: declared hypotheses

This file reproduces, verbatim, the section of the authors' working plan in which the hypotheses of the
benchmark and of the follow-up studies were declared (section 0.5 of the plan). Each block states when it
was written relative to the runs it concerns. The plan was kept as a dated working document and was not
registered externally (see the paper, Section 4.7).

Notes for readers:
- Internal labels are kept as written. E1 is the benchmark, E2 the failure decomposition, E3 the label
  budget, E4 the identifier-free features, E5 and E5-T the within-network and early-window studies, E9 the
  window size and contamination study, P and P2 the poisoning study, U the unselected-window study, and
  A1-C2 the controls added after the second review. "Families" means the NF-v2 and NF-v3 releases.
- DI-NIDS was the working name of the method called DANN-R in the paper and the code.
- References to JOURNAL_TODO, LITERATURE_REVIEW and REVISION_PLAN are internal working files that are not
  part of this repository.
- Outcomes are reported in the paper, including the hypotheses that failed.

---

### 0.5 Pre-registered hypotheses for E1/E2/E8 (declared before any E1 result)
Primary metric: AUROC on the target eval stream. Secondary: Macro-F1 under `em` (S0/S1) and `win_mad` (S2). Tests: paired Wilcoxon over the 40 (target × seed) pairs per family, Holm-corrected within each hypothesis family; effects with seed-bootstrap 95% CIs; every claim must hold in **both** families. Inference is over sampling and training randomness for these 4 networks per family, not over networks in general (stated as a limitation).
- **H-B1** No S0/S1 method improves mean AUROC over ERM-CB by ≥ 5 pp in both families.
- **H-B2** Marginal alignment (CORAL, DANN) does worse than ERM-CB on the extreme-prior target BoT-IoT (label-shift theory).
- **H-B3** The best S2 method beats the best S0/S1 method on mean AUROC in both families.
- **H-B4** BRN > DomNorm on AUROC (benign reference vs mixture reference; tests prior-invariance).
- **H-B5** (added 2026-11-03 00:05, after the literature search found L1 in LITERATURE_REVIEW.md; no CORAL result existed yet, since its E1 jobs had not started). L1 reports that CORAL restores cross-dataset AUROC on NF-UNSW-NB15-v2 "under a benign-majority condition". Under our protocol: (a) CORAL improves AUROC over ERM-CB by ≥ 5 pp on the UNSW-NB15 target in both families; (b) descriptive: CORAL's per-target AUROC gain over ERM-CB increases with the target's benign share across the 8 targets (Spearman ρ reported, n = 8, no significance claim). If (a) holds, the thesis must be qualified: label-free alignment does help on benign-majority targets.
- **H-F1** (E2) For every method, AUROC on seen attack families is < 0.75 on at least 2 of 4 targets per family (network-driven failure).
- **H-F2** (E2) Target-benign FPR at the source-calibrated 1%-FPR threshold exceeds 5% for every S0/S1 method on average.
- **H-E8** HYB ≥ max(BRN, KNN-win) on mean AUROC in both families.

**Added 2026-11-03, saved by 19:49:48 (file modification time), before E5-T and E9 were launched at 19:59:43** (smoke tests on one target and seed ran in between, in the scratchpad) (JOURNAL_TODO T1.1, T1.2). Same test conventions as above. "Best S0/S1" means the S0/S1 method with the highest mean AUROC on the same evaluation flows, chosen post hoc, which is conservative for the window detectors.
- **H-T1 (E5-T, primary cut = first 20% of the target timeline, NF-v3).** KNN-win with an *early* window beats the best S0/S1 method on the *late* eval flows by ≥ 10 pp mean AUROC (paired Wilcoxon over the target × seed pairs, p < 0.05). Targets whose late eval stream has fewer than 100 benign flows are reported but excluded from this test. Added before running, from pool counts only: BoT-IoT-v3 has 94% of its benign traffic in the first 20% of its timeline.
- **H-T2 (E5-T, descriptive).** The AUROC cost of an early window vs a random-time window for KNN-win and IF-win on the same late flows, as a mean with a seed-bootstrap CI. The cut at 50% is a sensitivity analysis.
- **H-W1 (E9).** KNN-win with only 100 window flows (5% contamination) beats the best E1 S0/S1 method by ≥ 10 pp mean AUROC in both families.
- **H-W2 (E9).** KNN-win with 500 window flows at 20% contamination beats the best E1 S0/S1 method by ≥ 10 pp mean AUROC in both families.
- Curves for size and contamination, and IF-win, are descriptive.

**Added 2026-11-04, before E4 and E3 were built or run** (JOURNAL_TODO T1.4, T1.5). Same test conventions.
- **E4 design.**
  - Identifier-free feature set: drop L4_SRC_PORT, L4_DST_PORT, DNS_QUERY_ID, ICMP_TYPE, ICMP_IPV4_TYPE and FTP_COMMAND_RET_CODE (NF-v2: 41 → 35 features; NF-v3: 49 → 43).
  - Methods: ERM-CB, XGB, RF, BRN, KNN-win, IF-win.
  - Hyperparameters: the E1-selected ones, not re-tuned (a stated deviation).
  - 10 seeds, both families, identical folds to E1.
- **H-I1.** With identifier-free features, KNN-win beats the best identifier-free source-trained method (the best of ERM-CB, XGB, RF) by ≥ 10 pp mean AUROC in both families.
- **H-I2 (descriptive).** Per-method AUROC change, identifier-free minus full features: paired mean with a seed-bootstrap CI.
- **E3 design.** k ∈ {10, 30, 100, 300, 1000} labelled target flows.
  - Draw: uniformly at random (natural prior), nested across k, from target-pool rows disjoint from eval, buffer and window.
  - **XGB-T:** target labels only; E1-selected XGB hyperparameters, no early stopping. If the k flows contain only one class, the score is constant (AUROC 0.5) and the run is flagged.
  - **XGB-ST:** source training flows plus the k target flows. Each target flow is weighted so the target's total weight equals the source's. Class-balanced weights over the combined set. Early stopping on source validation, as in E1.
  - 10 seeds, both families. Reference: KNN-win with the E1 window (500 unlabelled, mostly benign flows).
- **H-L1 (descriptive, primary).** The "exchange rate": the smallest k at which the better of XGB-T and XGB-ST reaches KNN-win's mean AUROC, per family (or "> 1000").
- **H-L2.** At k = 100, the better S3 variant has lower mean AUROC than KNN-win in both families (paired Wilcoxon, p < 0.05).

**Added 2026-11-05, before T1.3 / WP5.1 was built or run** (JOURNAL_TODO T1.3). Same test conventions.
- **T1.3 Window Poisoning Design.**
  - Setting: S2 (reference window size N = 500 flows).
  - Poisoning fractions: p ∈ {0.00, 0.01, 0.02, 0.05, 0.10, 0.20} injected into the reference window.
  - Threat Model: Targeted injection of the target's primary attack class A* (Exploits on UNSW, DoS on CIC18, scanning on ToN, DDoS on BoT-IoT) to deliberately blind the local window detector to that attack.
  - Detectors: KNN-win and IF-win (E1-selected hyperparameters).
  - Defenses:
    1. `naive`: detector fitted directly on raw poisoned window.
    2. `trimmed`: outlier-filtered reference window. Discards top γ = 10% candidate flows based on internal leave-one-out anomaly/distance scores before fitting.
  - 10 seeds, both NetFlow families (NF-v2, NF-v3), 4 target testbeds.
- **H-P1 (Targeted Evasion).** At 10% targeted poisoning (p = 0.10), naive KNN-win experiences a drop of ≥ 20 pp in recall specifically on the targeted attack class A*, while non-targeted attack classes experience < 10 pp drop (targeted evasion effect).
- **H-P2 (Trimmed Defense Efficacy).** At 10% targeted poisoning (p = 0.10), the Trimmed Window Defense recovers ≥ 15 pp of the lost recall on A* compared to naive KNN-win across both families (paired Wilcoxon over target × seed pairs, p < 0.05).
- **H-P3 (Headline Robustness Under Minor Poisoning).** Under 5% targeted poisoning (p = 0.05), naive KNN-win maintains overall AUROC ≥ 0.75, remaining superior to the best source-trained supervised baseline in both families.

**Added 2026-11-07, before the poisoning re-run (P2) was built or run.** Author-approved re-run of the poisoning experiment.
- **Why.** The first run (above, H-P1–H-P3) reported its results at win_mad Z = 1.5, a threshold picked post hoc on evaluation labels. It saved no scores, so no matched false-alarm comparison was possible. It also used its own folds rather than the benchmark folds.
- **These hypotheses are not blind.** The first run had already shown large drops in targeted recall, and that knowledge shaped these hypotheses.
- **P2 design.**
  - **Folds.** Benchmark folds (`build_fold`), so the evaluation stream is identical to E1.
  - **Window.** 500 flows: (500 − n) benign flows from the benign pool, plus n flows of the target's most frequent attack family. Both parts are drawn from rows outside the evaluation and buffer sets, and both are nested across n. The poisoning fraction p = n/500 ∈ {0, 1, 2, 5, 10, 20}%.
  - **Detectors.** KNN-win and IF-win, run through `run_method` exactly as in E1, with the E1-selected hyperparameters.
  - **Defence.** "trimmed": remove the 10% of window flows with the highest leave-one-out window score from the naive run, then rerun the detector on the remaining 450.
  - **Metrics.**
    - Primary, threshold-free: targeted-family AUROC (benign versus targeted-family evaluation flows) and other-family AUROC.
    - Secondary: realised recall and benign FPR at the declared operating point (win_mad, Z = 3); targeted-family recall at a matched benign FPR of 1% and 5% (retrospective ROC points).
  - **Saved.** All evaluation and window scores.
- **H-P4 (targeted harm).** At p = 10%, naive KNN-win's targeted-family AUROC is ≥ 10 pp below its p = 0 value in both releases (paired Wilcoxon over target × seed, p < 0.05). Other-family AUROC changes by < 5 pp.
- **H-P5 (trimming).** At p = 10%, trimming recovers ≥ 50% of KNN-win's targeted-family AUROC drop, in both releases.
- IF-win and the full p-curves are descriptive.

**Experiment U: unselected reference windows** (declared 2026-11-13, before any run; added in response to
an external pre-submission review).

*Motivation.* Every window in E1, E5-T and E9 has a composition fixed with labels (benign-pool flows plus a
set number of attacks). Experiment U measures what happens when the window is collected without any label
selection.

*Design.* Benchmark folds; KNN-win and IF-win with their benchmark-selected hyperparameters; 10 seeds.
- **U-rand.** 500 flows drawn uniformly from the target's uniform pool, outside the evaluation and buffer
  rows, so the window has the target's natural class mix. Both releases. Scored on the full benchmark
  evaluation stream.
- **U-early.** NF-v3 only. 500 flows drawn uniformly from target flows that start before the 20% time cut,
  with no label selection, again outside the evaluation and buffer rows. Scored on the late evaluation flows,
  as in E5-T.
- **Trimmed.** Each window is also run trimmed: drop the 20% of window flows with the highest
  leave-one-out window score, then refit. Fully label-free.
- The realised window attack share is recorded.

*Hypotheses.*
- **H-U1.** With U-rand, KNN-win's per-target mean AUROC exceeds that of the best S0/S1 method of the release
  (IRM for NF-v2, RF for NF-v3) on both benign-majority targets (UNSW-NB15, CSE-CIC-IDS2018), in both releases.
- **H-U2 (descriptive).** On the attack-majority targets (BoT-IoT; ToN-IoT-v2), KNN-win with U-rand is at or
  below the S0/S1 level, consistent with the E9 contamination curve.
- **H-U3 (descriptive).** The AUROC change from trimming, per target.
- **H-U4.** On the three interpretable NF-v3 targets, KNN-win with U-early beats the best S0/S1 method on the
  late flows (GroupDRO, as in H-T1) by ≥ 10 pp on average. Paired Wilcoxon, n = 30.

**Controls after the second review** (declared 2026-11-13, before any run; plan: REVISION_PLAN_2026-11-13.md)

All runs use benchmark folds, 10 seeds and both releases, with the benchmark-selected hyperparameters of the matching
method.

- **A1, one-class control.** KNN and isolation forest are fitted on *source* benign training flows instead of the target
  window: 500 flows (matching the window size) and 20,000 flows, pooled over the three sources. KNN uses standardisation
  by the reference sample, as KNN-win does by its window.
  - **H-C1.** KNN-win beats KNN fitted on 500 source benign flows by ≥ 10 pp mean AUROC in both releases, and on ≥ 3 of
    the 4 captures in both releases.
- **A4, near-duplicate check.** For each evaluation flow, the distance to its nearest window flow is computed in
  window-standardised space. Evaluation flows within ε of a window flow are removed and KNN-win's AUROC is recomputed
  from its saved scores. ε is the 1st and the 5th percentile of the window's leave-one-out nearest-neighbour distances;
  exact duplicates (distance 0) are also reported.
  - **H-C2.** Removing near-duplicates at the 5th-percentile ε lowers KNN-win's mean AUROC by < 5 pp in both releases.
- **B4, fusion (saved scores only; rules fixed here).** (i) Equal-weight average of the within-run rank percentiles of
  KNN-win and RF; (ii) the same with the release-best S0/S1 method (IRM for NF-v2, RF for NF-v3).
  - **H-C3.** Fusion raises TPR at 1% FPR over KNN-win in both releases while lowering mean AUROC by ≤ 2 pp.
- **C1, window as labelled benign data.** XGBoost is trained on the source data plus the 500 window flows labelled benign,
  with the window weighted to carry the same total weight as the source data (the XGB-ST scheme). Benchmark-selected XGB
  hyperparameters; early stopping on source validation.
  - **H-C4.** Its mean AUROC is below KNN-win's in both releases.
- **C2, poisoning vs k (descriptive).** KNN-win on the 10%-poisoned windows of the poisoning study (same draws) with
  k ∈ {5, 10, 20, 50, 100, 200}; targeted-family AUROC. (k = 10 was added before the main run, after the smoke test
  showed that one tuned k is 10; no results were looked at.)
- **Post-hoc, descriptive (reported as such):** A2, low-FPR metrics of all methods; A3, label budget by target with attack
  counts; B2, results without BoT-IoT and precision at 1% and 0.1% prevalence; B3, source vs target attack share.

**Added 2026-11-05, before Task T3.3 / E5 was built or run** (JOURNAL_TODO T3.3).
- **Task T3.3 / E5 Design (Time Shift vs. Network Shift on NF-v3).**
  - Question: Is cross-network transfer failure driven by temporal drift or by cross-network environment shift?
  - Dataset: NF-v3 (all 4 testbeds: UNSW, BoT-IoT, ToN-IoT, CSE-CIC-IDS2018), evaluating 10 random seeds.
  - Protocol:
    - Train within testbed on EARLY flows ($t_{start} \le cut$, where $cut \in \{0.2, 0.5\}$).
    - Evaluate on the exact same LATE flows ($t_{start} > cut$) used in E5-T and E1.
    - Supervised models: `XGB` (GPU histogram), `RF` (Random Forest), `ERM-CB` (class-balanced MLP), using E1-selected hyperparameters.
    - Comparators on the same late flows:
      (a) `Within-Temporal` (train early, test late within the same network);
      (b) `Cross-Network` (best S0/S1 model trained on other networks, from saved `e5t_s01_late.csv`);
      (c) `S2-Early-Window` (`KNN-win` and `IF-win` with the early window, from saved `e5t_runs.csv`).
- **H-Time1 (Temporal Stability within Network).** Within a single network, temporal training (early → late) maintains mean AUROC ≥ 0.85 across testbeds with ≥ 100 late benign flows, demonstrating that within-network time drift alone does not cause ranking collapse.
- **H-Time2 (Network Shift Dominance).** Across target testbeds, the cross-network transfer penalty ($\Delta_{net} = \text{AUROC}_{\text{within-temporal}} - \text{AUROC}_{\text{cross-network}}$) exceeds +20 pp (paired Wilcoxon over target × seed pairs, p < 0.05), proving that network environment mismatch (infrastructure shortcuts) dwarfs temporal drift.

