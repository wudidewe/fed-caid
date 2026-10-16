# Dataset acquisition and exact preparation

Run `python main.py` from the source checkout. The default data root is this `dataset/` directory.

1. Reuse `dataset/<dataset>/development.npz` when its metadata, feature/class signature and integrity records are valid.
2. Otherwise obtain the pinned Kaggle source through `kagglehub.dataset_download`.
3. Verify every original CSV against the recipe's byte count and SHA256.
4. Reconstruct CAL and VAL in the original source row order. Construct full TRAIN from all source rows outside the recorded exclusions, then apply the original seed 42 permutation.
5. Install the three prepared files only after validation and begin training.

No new random split or feature selection is performed. The published metadata consists of feature recipes and source row numbers, not traffic features or predictions. Numerical preprocessing remains fitted from TRAIN only.

## Raw layout

```text
dataset/
  archive/                       # CIC-IoV-2024, flat CSV files
    decimal_benign.csv
    decimal_DoS.csv
    decimal_spoofing-GAS.csv
    decimal_spoofing-RPM.csv
    decimal_spoofing-SPEED.csv
    decimal_spoofing-STEERING_WHEEL.csv
  raw/
    Car-Hacking/                 # four original CSV files
    CIC-IDS-2017/                # eight original CSV files
    CIC-UNSW-NB15/               # CICFlowMeter_out.csv
  preparation/
    <dataset>/
      recipe.json
      heldout_identities.csv
      excluded_identities.csv
  <dataset>/
    development.npz
    preparation.json
    selected_train_identities.npz
```

The IoV layout uses `dataset/archive/<filename>` directly. Neither dataset recognition nor CSV preparation depends on a folder named `decimal`. Kaggle files may arrive inside an upstream subdirectory; the verified required files are installed in the flat layout above.

All Kaggle sources use version 1, with file content hashes providing the final identity check:

| Dataset | Handle |
| --- | --- |
| CIC-IoV-2024 | `pushpakattarde/ciciov2024decimalcsv` |
| Car-Hacking | `pranavjha24/car-hacking-dataset` |
| CIC-IDS-2017 | `chethuhn/network-intrusion-dataset` |
| CIC-UNSW-NB15 | `yasserhessein/cic-unsw-nb15-augmented-dataset` |

For NB15 the remote `CICFlowMeter.csv` is mapped to local `CICFlowMeter_out.csv`, with the exact byte and SHA256 checks retained. An original UNSW-NB15 file or a different CIC UNSW version is not accepted.

## Identity and labels

The selected dataset profile defines the raw parser and expected feature/label map. Original file hashes establish source version, while columns and labels are validated during streaming. For a prepared NPZ, identification compares the complete ordered feature and class maps. The two network datasets cannot be identified from their matching 68-feature width alone. `BENIGN` is index zero for every dataset.

Prepared NPZ files contain exactly `class_names`, `feature_names`, and `X/y_train`, `X/y_cal`, `X/y_val`. TEST or confirmation arrays are rejected. `allow_pickle=False` remains active. The adjacent preparation record pins the array and TRAIN identity files. Existing data is never automatically overwritten.

## Offline use and another data root

Authorized original CSV files can be placed in the raw layout above and prepared with `python main.py --offline`. To use another `--data-root`, copy this directory's `preparation/` tree into that root before the first raw preparation. Existing prepared data can be read from any selected root without copying recipes.

Kagglehub uses its official authentication mechanism when a provider requires consent or credentials. Configure access outside this repository and never commit a token or `kaggle.json`. Download, authentication, hash or label errors stop the experiment; synthetic data is only used by explicit `--smoke`.

## Optional prepared packages

A separately published exact prepared package can be selected through `download_sources.json` or `--download-manifest`. See `download_sources.example.json`. Each entry pins a public HTTPS ZIP and its three flat members. This optional path is independent of the default pinned Kaggle raw reconstruction. No prepared data URL is fabricated or required for the default Kaggle path.

Data, caches, models and private outputs are excluded by `.gitignore`.
