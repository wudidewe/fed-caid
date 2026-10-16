# Source layout

`main.py` is the unified command, dataset selection and fixed per-dataset parameter configuration. `utils.py` contains shared numerical preprocessing, CrossPLE TabM construction, losses, client partitions, calibration and metrics.

Use `import utils` in new code. Loading this module also registers the historical `Utils` name for checkpoint lookup; it does not provide a separate uppercase module for direct imports. Current v3 checkpoints store state dictionaries and retain their original fields.

| Module | Role |
| --- | --- |
| `dataset_access.py` | Complete prepared-data signatures, local reader and default repository reader. |
| `dataset_preparation.py` | Frozen source row replay and exact raw-to-TRAIN/CAL/VAL preparation. |
| `kaggle_source.py` | Pinned Kaggle file downloads and verified flat installation. |
| `data_io.py` | Shared original CSV parsing, CAN hexadecimal/DLC handling and network feature recipes. |
| `privacy_run.py` | Fresh experiment parser, local detector client and v3 checkpoint inference. |
| `privacy_core.py` | Compact class adapter and its original fitting operation. |
| `secure_run.py` | Four independent secure federations, checkpoints, timings and output records. |
| `secure_aggregation.py`, `aggregation_audit.py` | Two collectors, full-roster contracts and audit chain. |
| `round_observability.py` | Private observations used for diagnostics. |
| `experiment_report_data.py`, `experiment_report.py` | Completed-run validation and five standalone plots. |
| `report_timeseries.py` | JSON-only validation of recorded round observations used by the report. |
| `validation_core.py` | Aggregation arithmetic and release scope validation. |
| `ids_core.py`, `iov_instruction.py`, `_iov_layout.py` | Shared numerical aliases, public command/inference and snapshot paths. |
| `vendor/` | Separately licensed TabM and numerical embedding implementations. |

Only the current fresh MPC experiment is distributed. Historical DP ledgers, subtype continuation and alternative aggregation methods are excluded. The model, training, calibration, update clipping, secure summation and detection decisions preserve the original numerical behavior.
