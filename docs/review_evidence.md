# Review evidence

With the four datasets in place, [main.py](../main.py) trains a new run and writes
its report to `outputs/fedcaid_review/run_*/report/`. The 17 November 2026
reference experiment used this workflow. The main benchmark evaluates the
base detector, local and federated adapters, and full weight FedAvg on grouped
development VAL. CAN quantization, transfer and temporal studies each have
their own evaluation populations.

| File | Use in the revision |
| --- | --- |
| `SUMMARY.md` and `evidence_coverage.csv` | Execution status, evaluation role and scope of each stage. |
| `benchmark_operating_points.csv` | Grouped VAL performance, seed variation and logical communication payload for the main comparison. |
| `benchmark_ablation_effects.csv` | Paired differences for the main benchmark. |
| `benchmark_class_support.csv` | Class support, precision, recall and F1 on grouped VAL. |
| `data_role_audit.csv` | Exact input overlap and role sizes. |
| `matched_branch_metrics.csv` and `paired_evidence.csv` | Subtype results by seed and paired comparisons on matched alarms. |
| `operating_decisions.csv` and `nb15_binary_working_points.csv` | Operating criteria, false alarms and missed attacks for the relevant comparisons. |

The 95% Student t intervals describe variation across seeds on a fixed
evaluation population. An exact input unique view keeps one eligible row per
input. Class support counts the records in that evaluation population. Macro
F1 averages classes with true support in the view being reported. Independent
capture confirmation requires the separately declared acquisition described
in [external confirmation](external_confirmation.md).

## Software comparison

The feature comparison describes the released workflows. Both tools simulate
collaborative intrusion detection and serve different evaluation settings.

| Feature | CIDS-Sim 1.0 | FedCAID revision |
| --- | --- | --- |
| Packaged classification | Binary attack detection | Multiclass detection on four CAN and network-flow datasets |
| Client data | Non-IID within a dataset and a separate heterogeneous scenario across datasets | Within-dataset label skew. Each benchmark runs its own federation. |
| Object exchanged | Neural detector weights with the documented layerwise mean | Compact class-score updates. A sample-count-weighted full-model FedAvg control is measured separately. |
| Run outputs | Jupyter notebooks, round metrics and CSV output | Verified input hashes, role identity manifests, held-out comparisons, paired seeds and report outputs |

Sources for the CIDS-Sim entries are the [released notebooks](https://github.com/aulwardana/CIDS-Sim/tree/main/src), [configuration guide](https://cids-sim-doc.readthedocs.io/en/latest/tutor/configuration.html), and [aggregation documentation](https://cids-sim-doc.readthedocs.io/en/latest/cids.fl.aggregate.html).

The internal FedAvg experiment measures a common-task baseline. The published
CIDS-Sim scores use its own data, labels and training budget. A head-to-head
accuracy experiment would require a matched label map, data splits, training
budget and metric definition for both tools. Logical bytes quantify payload
size. The existing TCP timings measure transfers on the local host.
