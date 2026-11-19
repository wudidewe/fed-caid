# FedCAID

FedCAID compares local and federated intrusion detection on CAN and network flow
data. It provides matched controls for local adaptation, donor pooling and
federated aggregation.

[![FedCAID checks](https://github.com/wudidewe/fed-caid/actions/workflows/revision.yml/badge.svg)](https://github.com/wudidewe/fed-caid/actions/workflows/revision.yml)

## Run the experiment

1. Open the project root in PyCharm. Select Python 3.11 and install
   [requirements.txt](requirements.txt) in that interpreter.
2. Place the original CSVs under `dataset/` using the
   [dataset preparation guide](dataset/README.md).
3. Select the **FedCAID** run configuration or run the root [main.py](main.py).
   Leave the program arguments empty for the full local experiment.

The launcher lists missing files and returns exit code 2 if the dataset layout
is incomplete. With all required files present, it verifies the inputs and
creates a new run in `outputs/fedcaid_review/`. Runtime depends on the machine
and may take several hours. The software selects CUDA when available.

```sh
python -m pip install -r requirements.txt
python main.py
```

The primary benchmark uses four datasets and five seeds, 42 through 46, and
reports grouped development validation (VAL) for local TabM, local adaptation,
the compact federated adapter and full weight FedAvg. CAN quantization, compact
adapter transfer and the later day CIC-UNSW-NB15 benchmark use separate
retrospective test protocols. The Hulk and Exploits subtype comparisons reuse
fixed binary alarms.

The reference results in the manuscript come from run
`run_20261117_114937_guhefcl1` on 17 November 2026. The
[recorded source commit](https://github.com/wudidewe/fed-caid/tree/6f72abd255d4782302fd27ac35f8061f529ec141)
and the run receipt identify the evaluated source. The reference machine used
Windows, Python 3.11.13, PyTorch 2.5.1 with CUDA 12.1 and an NVIDIA GeForce RTX
4060 Laptop GPU.

## Optional CSE-CIC-IDS2018 replay

Place the unmodified capture at `dataset/CSE-CIC-IDS2018/02-16-2018.csv`
using the [dataset guide](dataset/README.md#optional-cse-cic-ids2018-capture).
Running `main.py` with no arguments then scores this capture after the primary
workflow finishes, using the five frozen Hulk models from that same run.
Read `hulk_2018/SUMMARY.md` and `hulk_2018/diagnostics/SUMMARY.md` inside
the run directory. Without the optional CSV, the command completes the
ordinary four dataset workflow.

To replay the capture without retraining, keep the original CIC-IDS-2017
CSVs and select a completed full review directory that still contains its
verified `hulk_attribution/` models:

```sh
python main.py --hulk-2018 <completed-full-run>
```

Use `--hulk-2018-source <path-to-original-csv>` to specify the capture in
another location for either command. Focused replay writes a new directory
under `outputs/hulk_2018_review/` and leaves the selected run intact. Use
focused replay to retry an interrupted 2018 step after its primary workflow
has completed.

This is an exploratory evaluation of one published capture. The score
diagnosis compares the frozen working point with a threshold replay fitted
on 2017 CAL. The low FAR curve calculated from the 2018 labels describes
performance on that capture. Results are kept separate from the primary
benchmark and the predeclared independent confirmation.

## Check the installation

```sh
python main.py --demo
python main.py --smoke
```

`--demo` runs a small synthetic CPU example and saves a receipt in
`outputs/fedcaid_demo/`. `--smoke` exercises the synthetic integration suite.
Both print `synthetic_only=True`. These checks require Python dependencies and
run without the source CSVs. GitHub Actions runs both checks on Windows and
Ubuntu and verifies installation from a built wheel.

## Results and follow up

Start with `outputs/fedcaid_review/run_*/report/SUMMARY.md`.

| File | Contents |
| --- | --- |
| `final_plan.json` | Source and entry point hashes, input checks and planned stages |
| `final_complete.json` | Stage status, including failures and pending evidence |
| `report/SUMMARY.md` | Experiment coverage, metrics and paired results |
| `report/tables/` | Per stage working points, class support and confusion counts |

Predeclared independent Hulk confirmation and LAN measurements require separate
declarations. Create `external/` locally when configuring these optional stages.
The local run can finish with status `completed_with_pending`. The command returns exit
code 2 for failures or pending required stages. See the
[independent capture guide](docs/external_preflight.md) and the
[LAN setup guide](docs/protocol_validation.md).

Use `python main.py --refresh-report <run-directory>` to rebuild a report from a
completed run in a separate directory. The new report records the status of its
source run. `python main.py --help` lists focused experiments and resume options.
Resume requires a compatible source snapshot.

The default NB15 low FAR study preserves the candidate random streams used for
the manuscript. To run a separate comparison with one candidate stream per
seed and client, use `python main.py --nb15-low-far --independent-client-rng`.
The run plan records which setting was used.

See [validation and reproduction](docs/review_readiness.md) for the evaluation
scope and [software comparison](docs/software_comparison.md) for related packages.

## License

The [Apache License 2.0](LICENSE) covers the FedCAID source. Third party notices
are under [src/vendor](src/vendor). Dataset providers' terms apply to their data.
