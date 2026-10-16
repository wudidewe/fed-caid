# FedCAID

**Federated Class Adapter Intrusion Detection**, version 1.0.0. Author: liulu.

FedCAID runs a shared intrusion detection procedure on four independent datasets. Each client fits a local CrossPLE TabM detector and exchanges compact class adapter updates through two additive share collectors. Detector weights stay local. Each dataset retains its own features, labels, preprocessing, trained detectors and thresholds.

This repository contains the fresh MPC experiment described in the manuscript. Historical differential privacy accounting, subtype continuation, alternative aggregation experiments, manuscript sources, datasets and trained models are excluded.

## Install

Use Python 3.11 or newer. The numerical package versions match the validated Windows research environment. Install the source checkout:

```text
git clone https://gitee.com/liuluz/fed-caid.git
cd fed-caid
python -m pip install ".[reporting]"
```

The recorded environment uses PyTorch 2.5.1 with CUDA 12.1 and an NVIDIA GeForce RTX 4060 Laptop GPU. The synthetic functional check also works on CPU. Installation does not change an existing compatible CUDA build of PyTorch.

## Run

```text
python main.py
```

The entry checks `dataset/`, obtains missing original CSV files through the official `kagglehub` library, verifies their pinned versions and SHA256 values, and reconstructs the original TRAIN, CAL and VAL roles before training. Existing complete prepared datasets are reused. A partial or different file is rejected rather than overwritten or replaced with a new split.

For CIC-IoV-2024, raw CSV files are placed directly in `dataset/archive/`, for example `dataset/archive/decimal_benign.csv`. There is no required `CICIoV2024/decimal/` directory. Other raw datasets are kept separately under `dataset/raw/<dataset>/`. Prepared arrays use `dataset/<dataset>/development.npz` for all four datasets.

```text
python main.py --dataset CIC-IDS-2017
python main.py --dataset CIC-IoV-2024 --offline
python main.py --dataset auto --data PATH_TO_DEVELOPMENT_NPZ
```

`--offline` prohibits downloading and can prepare already available raw files. `--data` reads an explicitly supplied prepared archive. `--dataset auto --data` identifies that archive from the complete ordered feature and class maps. Folder names such as `archive` and feature counts alone do not determine dataset identity.

The default experiment uses the previously validated seed 42, three clients, three aggregation rounds, five detector epochs and all eligible TRAIN rows. Per-dataset settings are centralized in `src/main.py`. Explicit command options override these settings. A changed setting defines a new experiment.

| Dataset | Features | Classes | Kaggle source, pinned version 1 |
| --- | ---: | ---: | --- |
| CIC-IoV-2024 | 9 | 6 | `pushpakattarde/ciciov2024decimalcsv` |
| Car-Hacking | 10 | 5 | `pranavjha24/car-hacking-dataset` |
| CIC-IDS-2017 | 68 | 15 | `chethuhn/network-intrusion-dataset` |
| CIC-UNSW-NB15 | 68 | 10 | `yasserhessein/cic-unsw-nb15-augmented-dataset` |

`BENIGN` is class zero. The two 68-feature datasets have different feature names and labels. The network source is the augmented CIC UNSW representation, not the original UNSW-NB15 feature format. Its remote `CICFlowMeter.csv` is mapped to the verified local filename `CICFlowMeter_out.csv` only after content verification.

See [dataset preparation](dataset/README.md) for source layout, fixed role identities, checksum verification and Kaggle access. The first run needs internet access, disk space for several GB of original CSVs, and time for preparation and detector training. Public Kaggle access can depend on provider consent or credentials; use the official Kaggle configuration outside the repository if requested. No token is embedded in this code.

## Functional checks

```text
python main.py --check
python examples/synthetic_smoke.py
```

The synthetic example requires no real data and demonstrates software operation. It is not a reported research benchmark. The installed `iov-instruction` command remains an alias of the same entry point.

## Results

Each real invocation creates a new private experiment directory under `outputs/`. Completed runs also produce a local HTML report, a compact metric CSV, and five independent PDF/600 dpi PNG figures: the CIC-IDS-2017 confusion matrix, the CIC-UNSW-NB15 confusion matrix, summed detector training time, amortized base prediction cost, and communication volume. Both confusion matrices retain all classes and have the same physical dimensions. F1 means support weighted F1 and is reported in the table rather than a repeated plot.

```text
python examples/report_experiment.py --runs ONE_COMPLETED_RUN --out NEW_REPORT_DIRECTORY
```

Prediction cost is the summed CAL and VAL base prediction time divided by the corresponding processed records. It excludes preprocessing, model loading, calibrated decisions and network latency. Numeric share payloads and serialized round communication have different scopes and are not additive. See [measurement definitions](docs/EXPERIMENT_PRESENTATION.md).

## Scope and reuse

TRAIN fits preprocessing and the detector. CAL is divided into adapter fitting and a separate threshold holdout. VAL provides development evaluation, not independent TEST evidence. The single seed does not estimate variability across training runs.

The local collectors assume noncollusion, a trusted host/coordinator and honest clipping. This release provides no differential privacy or malicious client guarantee and does not implement blockchain or remote authenticated transport.

```python
from iov_instruction import predict_private_decisions
result = predict_private_decisions(checkpoint, X, device="cpu")
```

Only load trusted checkpoints. `X` must preserve the saved feature order and width. The interface restores TRAIN preprocessing and returns class predictions, attack alerts, scores and thresholds.

## License and citation

The repository's existing [Apache-2.0 license](LICENSE) is retained. Third party model notices remain in `src/vendor/`. Dataset access and redistribution remain subject to their providers' terms; raw and prepared data are not included in Git.

Use [CITATION.cff](CITATION.cff) and record the exact Git commit and experiment configuration. No accepted publication or software DOI is asserted.
