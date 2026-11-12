# crossnet: cross-network evaluation of NetFlow intrusion detectors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24172401.svg)](https://doi.org/10.5281/zenodo.24172401)

Code to reproduce the experiments of the paper *Cross-Network NetFlow Intrusion Detection: A Controlled
Benchmark of Transfer, Adaptation and Target Reference Windows* (venue and DOI to be added on
publication) by
Nahid Gazi, Tanzim Hasan Prappo and A.K.M. Fakhrul Hossain (Shahjalal University of Science and
Technology).

The benchmark asks what happens when a flow-based intrusion detector trained on some networks is
deployed on a network it has never seen, and which information about the new network restores
detection. It covers:

- **Testbeds and releases.** Four public testbeds (UNSW-NB15, BoT-IoT, ToN-IoT, CSE-CIC-IDS2018), in two
  standardised NetFlow feature releases (NF-v2 and NF-v3).
- **Protocol.** Leave-one-testbed-out: train on three testbeds, test on the fourth.
- **Methods.** 18 detection and adaptation methods, grouped by the information they may use about the
  target network:

  | Setting | Information about the target | Methods |
  |---|---|---|
  | S0 | none | ERM, ERM-CB, XGB, RF, IRM, V-REx, GroupDRO |
  | S1 | unlabelled target traffic | CORAL, DANN, IW-DANN, DANN-R, AdaBN, TENT, DomNorm |
  | S2 | a short, mostly benign target window | BRN, BRN-XGB, IF-win, KNN-win |
  | S3 | a few labelled target flows | XGB-T, XGB-ST (label-budget study) |

- **Fair comparison.** Every method is tuned with the same budget by leave-one-source-out validation on
  the source testbeds only, then evaluated on ten seeds. All scores are saved, so every reported number
  can be recomputed offline.

## Repository layout

| Path | Contents |
|---|---|
| `crossnet/` | Fold construction (`data.py`), feature transforms, training engine, method registry and losses, decision rules, metrics, and the resumable benchmark runner |
| `scripts/` | Dataset download and pool construction |
| `experiments/` | Follow-up studies: window size, contamination and timing; identifier-free features; label budget; within- vs cross-network training; pairwise transfer; feature-group ablation; window poisoning |
| `analysis/` | Main tables and hypothesis tests, failure decomposition, decision-layer analysis, window threshold rules, duplicate check |
| `paper/` | Scripts that recompute every number, table and figure of the paper from the run-level results |
| `tests/` | Fold leakage tests, determinism test, and integrity check of a completed benchmark |
| `docs/` | Protocol details ([PROTOCOL.md](docs/PROTOCOL.md)) and per-experiment instructions ([EXPERIMENTS.md](docs/EXPERIMENTS.md)) |

## Requirements

- Python 3.12 and the packages in `requirements.txt`.
- An NVIDIA GPU with CUDA. XGBoost and the neural methods run on the GPU, and bitwise reproduction of
  the reported numbers depends on the GPU code paths. The reported results were produced on an RTX 3050
  Laptop GPU (6 GB) with PyTorch 2.14.1 (CUDA 12.6 build).
- About 3.3 GB of disk space for the raw releases and 1 GB for the sampling pools. A full benchmark run
  writes about 0.8 GB of saved scores.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data

The NetFlow releases are public and are not redistributed here. The download script fetches them
from the University of Queensland research data repository and records SHA-256 checksums. The pool
script then builds, for each testbed:

- a uniform pool of 500,000 flows;
- a benign pool of up to 200,000 flows;
- a statistics file.

IP addresses and absolute timestamps are not used as features.

```bash
python scripts/download_datasets.py --out data/raw
python scripts/build_pools.py --raw data/raw --out data/pools
```

The code reads pools from `data/pools` by default; set `NF_POOL_DIR` to use another location.

## Reproducing the results

All commands run from the repository root. Every script is resumable: re-running it skips finished
rows. Outputs go to `results/` (set `NF_RESULTS` to change the benchmark output directory).

**1. Main benchmark.** 18 methods × 2 releases × 4 target testbeds × 10 seeds.

```bash
python -m crossnet.runner --jobs nfv2:ERM-CB nfv3:ERM-CB nfv2:KNN-win nfv3:KNN-win ...
python tests/verify_benchmark.py          # integrity check of the finished benchmark
python analysis/benchmark_tests.py        # main tables and hypothesis tests
```

`--jobs` takes `release:method` pairs, for example `nfv2:CORAL`; the method names are the keys of
`METHODS` in `crossnet/methods.py`. Several runner processes can share one job list: each claims jobs
through lock files in `results/benchmark/locks/`, which must be deleted before a fresh launch. The summed
fitting time of the full benchmark on the hardware above was about 46 GPU-hours (32 h tuning, 14 h
evaluation).

**2. Analyses of the saved benchmark scores** (minutes each):

```bash
python analysis/failure_decomposition.py results/benchmark
python analysis/decision_layer.py results/benchmark
python analysis/window_thresholds.py
python analysis/duplicate_check.py 3
```

**3. Follow-up experiments.** These need the finished benchmark: they reuse its selected
hyperparameters and evaluation streams. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the design,
outputs and run time of each.

```bash
python experiments/window_studies.py e5t && python experiments/window_studies.py e9 && python experiments/window_studies.py report
python experiments/feature_label_studies.py e4 && python experiments/feature_label_studies.py e3 && python experiments/feature_label_studies.py report
python experiments/within_vs_cross_network.py run && python experiments/within_vs_cross_network.py report
python experiments/pairwise_transfer.py run && python experiments/pairwise_transfer.py report
python experiments/feature_groups.py
python experiments/window_poisoning.py run && python experiments/window_poisoning.py report
```

**4. Numbers, tables and figures of the paper.** These read only the result files above (no training).

```bash
python paper/build_ledger.py     # every reported number -> paper/results_ledger.csv, paper/generated/
python paper/make_figures.py     # Figures 2-7 -> paper/figures/
```

`build_ledger.py` writes one LaTeX macro per value to `paper/generated/results.tex` and the paper's
tables to `paper/generated/tables/`; set `NF_RESULTS_DIR` to read results from another location.

## Determinism and checks

- `python tests/test_folds.py`: evaluation, buffer and window flows are disjoint, and no target flow
  enters training.
- `python tests/test_determinism.py`: every method gives bitwise-identical scores when re-run with the
  same configuration and seed.
- `python tests/verify_benchmark.py`: after a full run, checks completeness and recomputes every
  reported metric from the saved scores.

## Citation

If you use this code, please cite the archived release:

> Gazi, N., Prappo, T. H., & Hossain, A. K. M. F. (2026). *Code for "Cross-Network NetFlow Intrusion
> Detection: A Controlled Benchmark of Transfer, Adaptation and Target Reference Windows"* (Version
> v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24172401

```bibtex
@software{gazi2026crossnet,
  author    = {Gazi, Nahid and Prappo, Tanzim Hasan and Hossain, A. K. M. Fakhrul},
  title     = {Code for "Cross-Network NetFlow Intrusion Detection: A Controlled Benchmark of
               Transfer, Adaptation and Target Reference Windows"},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24172401},
  url       = {https://doi.org/10.5281/zenodo.24172401}
}
```

The DOI above identifies version v1.0.0. The concept DOI
[10.5281/zenodo.24172400](https://doi.org/10.5281/zenodo.24172400) always resolves to the latest
version. The paper's citation will be added on publication.

## Results archive

The saved scores and result tables of the reported runs will be deposited in a public archive. The
link will be added here.

## License

MIT; see [LICENSE](LICENSE). The NetFlow datasets are subject to their providers' terms.
