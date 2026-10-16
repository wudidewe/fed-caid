# Experiment Presentation

FedCAID is the publication name of IOV-instruction-v1.0. The automatic report presents one completed experiment with four independent dataset federations. It reads saved JSON only; it does not train, load checkpoints, or reopen raw datasets.

## Commands

```text
python -B main.py
python -B main.py --dataset CIC-IDS-2017
python -B examples/report_experiment.py --runs ONE_COMPLETED_RUN --out NEW_DIRECTORY
```

Main performs training and then report export. Independent postprocessing accepts exactly one completed MPC run and a new directory outside that run. `--no-report` skips automatic export. `--no-observe` disables additional private round/resource measurements. Obsolete plotting options have been removed. Each experiment uses one training seed, default 42; partitioning retains its existing separate seed.

## Five Standalone Figures

```text
metrics_summary.csv
report.html
manifest.json
figures/
  confusion_cicids2017.{png,pdf}
  confusion_cicunsw_nb15.{png,pdf}
  training_time.{png,pdf}
  prediction_time.{png,pdf}
  communication_volume.{png,pdf}
```

These five are individual figures, without subpanels. Both confusion matrices appear for the full experiment; a selected dataset only receives its corresponding confusion figure. CAN-only or synthetic-fixture reports omit both matrices. All exports use vector PDF and 600 dpi PNG. Use PDF in LaTeX for clear text and lines. A separately prepared workflow illustration can complete a six-figure manuscript. The code does not generate software screenshots or additional F1, resource, loss, recall or round plots.

F1 is support-weighted F1, named `f1` in the compact summary CSV. Accuracy, FAR and FNR remain in CSV/HTML. Raw metrics, class supports and existing selection rules remain untouched and available in the private provenance manifest. The report's first view is a concise metric table; figures and audit details are collapsed.

## Confusion Matrices

CIC-IDS-2017 and CIC-UNSW-NB15 have separate 8 by 7.4 inch canvases, each with a 5.6 inch square matrix and an adjacent height-aligned Row percent colorbar. All true/predicted classes remain in the original order. Each row is normalized from validated integer counts. Diagonal zeros are shown; positive percentages below 0.1 are explicitly marked <0.1. Other zero cells are blank. Fixed support counts are not appended to labels. Short aliases and their original class names are recorded in the manifest and explained in HTML. Unsupported rows remain zero and can be identified from original support values in the manifest; they must not be interpreted as measured recall.

## Measurement Scopes

Training time sums recorded client detector-training seconds. It excludes preprocessing, calibration and secure aggregation and is not a full workflow stopwatch.

Base prediction time in microseconds per record is one million times the sum of recorded client `inference_seconds`, divided by the sum of recorded client `inference_rows`. The timer covers CAL plus VAL batches. It is amortized batch processing, not individual request latency or the full calibrated decision pipeline. Row counts must be positive integers; if a saved milliseconds-per-sample measurement is present, it must agree with the recorded seconds/count pair. Missing seconds or counts make the normalized measurement unavailable. No denominator is guessed from VAL support alone.

Communication sums each counter over completed aggregation rounds. Numeric client shares count numerical payloads uploaded to both collectors. Round IPC includes serialized commands and replies. Their scopes differ and are not additive; startup and operating system transport overhead are excluded. Missing counters remain unavailable, not zero.

Private CPU/RSS and round observations remain collected and validated for audit, without generating extra plots. Resource and cached replay scopes are preserved in the manifest. V5 continuation does not retrain the backbone; unrecorded original training/prediction costs are not attributed to continuation.

## Interpretation and Sharing

VAL is development evidence, including any release selection; independent TEST remains unopened. A single experiment does not estimate variation across independent training. Synthetic checks are functional tests, not benchmarks. Source JSON hashes and recorded partition/feature/class identities are retained, but this reporting step is not a fresh full-data-content verification.

Report output is private by default. Keep datasets, models, caches, privacy ledgers, generated reports and machine-specific paths outside source uploads. Public release identity, application license and reproducible public dataset preparation remain author tasks.
