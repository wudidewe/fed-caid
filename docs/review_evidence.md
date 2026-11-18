# Review evidence

The archived 110-case revision run included these outputs in `review_evidence/`.
With the four datasets in place, [main.py](../main.py) trains a new run and writes
its report to `outputs/fedcaid_review/run_*/report/`.

| File | Use in the revision |
| --- | --- |
| `class_support.csv` | Support in TRAIN, CAL, VAL, TEST and exact-input unique TEST. A count of zero means absent. Counts 1–29 are flagged for interpretation. |
| `class_test_recall.csv` | Full FedCAID TEST recall and variability by class, paired with the sample count for that evaluation population. |
| `main_seed_intervals.csv` | Means, sample SD and seed mean intervals for the main comparisons. |
| `main_paired_intervals.csv` | Differences within each seed before averaging, with counts of positive, negative and tied differences. |
| `split_audit_summary.csv` | Exact input overlap and coverage from the source identity audit. |
| `communication_tradeoff.csv` and `communication_payload.pdf` | Detection versus logical numeric payload by method. The figure shows payload medians and ranges across seeds on a logarithmic scale. |

These evidence tables summarize the reference objective. The run directory
also contains `balance_vs_reference.csv` for the prespecified comparison
of the two detector objectives, including all metrics and fixed test views.

The 95% Student t intervals describe variation across runs on each fixed test
population. They are approximate with five seeds. Identical results across
seeds give a zero interval width. The intervals give no estimate of sampling
uncertainty across independent captures. There is one TEST population per
dataset and protocol. Report its actual class support rather than multiplying
the count by the number of seeds. The exact-input unique view keeps one row
per eligible input. The main test remains unchanged and no minority class is
duplicated to raise support. A macro F1 on a subset averages only classes with
true support in that subset.

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
