# FedCAID

**Federated Class Adapter Intrusion Detection**, version 1.0.0. Author: liulu.

FedCAID runs independent intrusion detection experiments on four datasets. Each client fits a local CrossPLE TabM detector and exchanges class adapter updates through two additive share collectors. Detector weights stay local. Each dataset has its own features, labels, preprocessing and thresholds.

## Installation

Use Python 3.11 and clone this repository:

```text
git clone https://gitee.com/liuluz/fed-caid.git
cd fed-caid
```

The validated environment uses Windows, Python 3.11.13, PyTorch 2.5.1 with CUDA 12.1 and an NVIDIA GeForce RTX 4060 Laptop GPU. To match the validated CUDA build, install PyTorch from its [official previous version index](https://pytorch.org/get-started/previous-versions/) before installing the remaining requirements:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

An installed `torch==2.5.1+cu121` satisfies the declared PyTorch version. For other supported devices, install the dependencies with `python -m pip install -r requirements.txt`. The default device is CUDA when available, otherwise CPU.

## Dataset setup

Download the complete four-dataset bundle from the [dataset release](https://gitee.com/liuluz/fed-caid/releases/tag/datasets-v1.0.0), or obtain the datasets from their official providers. Dataset archives are separate Release attachments and are not downloaded by `git clone`. Follow the part list, integrity checks and extraction commands in [dataset/README.md](dataset/README.md). Place their CSV files under the following project directories:

| Dataset | Raw source under `./dataset/` | Features | Classes |
| --- | --- | ---: | ---: |
| CIC-IoV-2024 | `CICIoV2024/decimal/` | 9 | 6 |
| Car-Hacking | `Car-Hacking Dataset/` | 10 | 5 |
| CIC-IDS-2017 | `CIC-IDS-2017 Dataset/` | 68 | 15 |
| CIC-UNSW-NB15 | `CIC-UNSW-NB15/CICFlowMeter_out.csv` | 68 | 10 |

The reader uses only these four locations. It verifies the byte counts and SHA256 hashes of all 19 expected source files against the bundled recipes. A different dataset release or edited CSV stops execution. The same file with a different archive folder must be moved into the documented location without changing its contents. For NB15, `CICFlowMeter.csv` is an accepted filename only when its contents match the recorded `CICFlowMeter_out.csv` hash. `Data.csv` and `Label.csv` are not experiment inputs.

The 12 recipe and row identity files under `src/fedcaid_resources/preparation/` are required program resources. They fix the original TRAIN, CAL and VAL membership, exclusions and TRAIN order. They do not contain the raw feature matrices. Keep them unchanged.

## Run the experiment

Run from the repository root:

```text
python main.py
```

The first run reconstructs TRAIN, CAL and VAL from the original CSVs and writes verified caches to `./data_cache/<dataset>/`. Later runs verify the raw sources and reuse a complete cache with matching hashes. The program does not download data or substitute synthetic data for missing files. Caches, trained models and generated reports are excluded from Git.

To run one dataset, choose its explicit identifier:

```text
python main.py --dataset CIC-IDS-2017
```

The complete default experiment uses seed 42, three clients per dataset, five detector epochs, three aggregation rounds and ten adapter epochs per round. All eligible TRAIN rows are used. Settings are in `src/main.py`; shared numerical methods are in `src/utils.py`. Changing these settings defines a different experiment. Use complete option names such as `--base-epochs`; abbreviated options are rejected.

If automatic report generation fails after training, the command returns exit code 2 and preserves the experiment outputs. Regenerate the report from the completed run without retraining, using a new report directory:

```text
python src/experiment_report.py --run outputs/your_run --out outputs/your_run_report_retry/private_DO_NOT_UPLOAD
```

`BENIGN` is class zero. The two 68-feature datasets have different ordered feature and label maps. NB15 uses the augmented CICFlowMeter representation, not the original UNSW-NB15 table format.

## Results and reproducibility

A completed run creates a fresh experiment directory and a companion report directory under `outputs/`. The report contains `report.html`, `metrics_summary.csv` and five separate figures, each as a vector PDF and a 600 dpi PNG:

- CIC-IDS-2017 confusion matrix.
- CIC-UNSW-NB15 confusion matrix.
- Detector training time.
- Base prediction cost.
- Communication volume.

Reference validation run: 2026-10-17 (`local_csv_full_20261017`), using seed 42 and the complete default configuration above. All values are percentages rounded to three decimal places, matching the generated HTML report. F1 is support weighted F1, calculated from the concatenated client VAL predictions.

| Dataset | Accuracy (%) | F1 (%) | False alarm rate (%) | False negative rate (%) |
| --- | ---: | ---: | ---: | ---: |
| CIC-IoV-2024 | 100.000 | 100.000 | 0.000 | 0.000 |
| Car-Hacking | 99.980 | 99.980 | 0.000 | 0.137 |
| CIC-IDS-2017 | 99.460 | 99.376 | 0.214 | 0.477 |
| CIC-UNSW-NB15 | 95.920 | 95.757 | 0.901 | 3.548 |

The report CSV keeps the unrounded values. For example, F1 values `99.979969`, `99.375997` and `95.756787` appear as `99.980`, `99.376` and `95.757` in this table and the HTML report. This formatting does not change the measured results. Use each new run's `metrics_summary.csv` and report when updating this dated reference.

The data hashes, bundled role metadata, default parameters and random seeds preserve the input and experiment definition. Reproducing the recorded numerical results also requires a matching software environment and device. A fixed seed does not guarantee bitwise identical training on every operating system, GPU or library build. Training time, prediction time and resource measurements vary with hardware and system load. Record the Git commit, package versions and device with reported results.

TRAIN fits preprocessing, numerical bins, class priors and the detector. CAL is divided into adapter fitting and a separate threshold holdout. VAL is development evaluation, not independent TEST evidence. A single seed does not estimate training variability. NB15 `Data.csv` and `Label.csv` overlap the development source and lack two model features; they are not used as an independent TEST set.

Prediction cost uses the summed CAL and VAL base prediction time divided by their record count. It excludes preprocessing, model loading, calibrated decisions and network latency. Numeric share payloads and serialized round communication are separate quantities. `nonzero_update_rounds` counts rounds that changed the global adapter. `secure_processing_seconds` totals the recorded startup, audit, client submission and aggregation intervals; it excludes round-begin messages, worker shutdown, local adapter fitting and result-file serialization.

The collectors assume noncollusion, a trusted host/coordinator and honest clipping. This release provides no differential privacy, malicious client protection or remote authenticated transport.

## License

The existing [Apache-2.0 license](LICENSE) is retained. Third-party model licenses and notices are in `src/vendor/`. Dataset use remains subject to each provider's terms.
