# Local validation

Validated on 16 October 2026 in the recorded Windows/Python 3.11.13/PyTorch 2.5.1+cu121 environment.

- 148 automated tests passed, including the default entry, exact role reconstruction, pinned downloads, complete-roster secure aggregation, checkpoint inference and successful report export.
- All 19 original CSV files from the four pinned Kaggle version 1 sources were actually downloaded. Their SHA256 values, sizes and physical data row counts matched the original experiment.
- The complete TRAIN, CAL and VAL arrays, metadata and TRAIN identities were reconstructed. All four development NPZ files and all four TRAIN identity NPZ files were byte-identical to the original caches.
- A complete fresh experiment ran on all four real datasets with the original default settings and seed 42. All recorded detection metrics, class confusion, thresholds and data signatures matched the reference experiment. All 12 detector checkpoint files were byte-identical.
- The complete run exported five standalone figures, a metric CSV and a local HTML report. Source wheel construction also succeeded.

| Dataset | Support weighted F1 (%) |
| --- | ---: |
| CIC-IoV-2024 | 100.000 |
| Car-Hacking | 99.980 |
| CIC-IDS-2017 | 99.376 |
| CIC-UNSW-NB15 | 95.757 |

These are single seed development VAL results, not independent TEST or deployment evidence. Costs vary with hardware and runtime conditions. Raw files, prepared arrays, checkpoints, prediction caches and generated experiment reports are excluded from the repository. Public preparation metadata contains feature recipes and source row identities only.
