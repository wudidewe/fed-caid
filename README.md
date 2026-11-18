# FedCAID

FedCAID runs reproducible local and federated intrusion detection experiments
on CAN and network flow data. It provides matched controls for local adaptation,
donor pooling, and federated aggregation.

[![FedCAID checks](https://github.com/wudidewe/fed-caid/actions/workflows/revision.yml/badge.svg)](https://github.com/wudidewe/fed-caid/actions/workflows/revision.yml)

## Quick start

1. Download the repository as a ZIP from GitHub or run
   `git clone https://github.com/wudidewe/fed-caid.git`.
2. Open the directory containing [main.py](main.py) in PyCharm. Select a Python
   3.11 interpreter and install [requirements.txt](requirements.txt) in it.
3. Select the **FedCAID** run configuration and click **Run**. You can also open
   `main.py` and click its Run icon. Leave the program arguments empty.

Run the following commands from a terminal in the repository root.

```sh
python -m pip install -r requirements.txt
python main.py
```

The launcher selects the workflow by checking the input files under `dataset/`.

| Inputs | What the launcher does |
| --- | --- |
| No required CSV files | Runs a synthetic CPU demonstration and saves a completion receipt under `outputs/fedcaid_demo/`. |
| Some required CSV files | Lists the missing files so you can complete the dataset layout. |
| All required CSV files | Starts the full experiment in a new directory under `outputs/fedcaid_review/`. |

## Reproduce the experiment

Place the provider files under `dataset/` using the
[dataset preparation instructions](dataset/README.md). Run the same **FedCAID**
configuration or use `python main.py --full-review`. The full run takes several
hours in the recorded CUDA environment. Runtime depends on the machine and
device. The software selects CUDA when available and supports CPU execution.

The experiment uses the same four source datasets and five seeds, 42 through
46. The primary benchmark compares local TabM, local adaptation, the compact
federated adapter, and full weight FedAvg on grouped development validation
(VAL). CAN quantization, compact adapter transfer, and the later day
CIC-UNSW-NB15 benchmark use separate retrospective test protocols. The controlled
Hulk and Exploits studies compare subtype predictions with fixed binary alarms.

The optional independent Hulk and LAN stages require their respective
declarations under `external/`. The full run marks those stages as pending
when the declarations are absent and completes the local experiments. See the
[independent capture guide](docs/external_preflight.md) and
[transport guide](docs/protocol_validation.md).

The manuscript reference run is `run_20261117_114937_guhefcl1` on 17 November
2026. Its exact Python source is available at
[the recorded experiment commit](https://github.com/wudidewe/fed-caid/tree/6f72abd255d4782302fd27ac35f8061f529ec141).
The reference environment was Python 3.11, Windows, PyTorch 2.5.1 with CUDA 12.1,
and an NVIDIA GeForce RTX 4060 Laptop GPU.

## Read the results

Start with `outputs/fedcaid_review/run_*/report/SUMMARY.md` after the full run.

| File | Contents |
| --- | --- |
| `final_plan.json` | Source hashes and input preflight |
| `final_complete.json` | Completed and pending stages |
| `report/SUMMARY.md` | Experiment coverage, operating points, and paired results |
| `report/tables/` | Detailed results by stage, including per class support and confusion counts |

Use `python main.py --refresh-report <run-directory>` to rebuild a report from
an existing run. Use `python main.py --help` for the focused experiment and
resume options.

## Verify the installation

```sh
python -m unittest discover -s tests -v
python main.py --smoke
```

The explicit smoke option exercises the synthetic CPU workflow in a temporary
directory. The tests and GitHub Actions workflow provide independent checks of
the software and installation. The full experiment writes results locally. Git
ignores raw data, run outputs, caches, trained models, and build files.

See [validation and reproduction](docs/review_readiness.md) for the study
scope and [the software comparison](docs/software_comparison.md) for related
packages.

## License

The [Apache License 2.0](LICENSE) covers the FedCAID source. Third party model
licenses and notices are in [src/vendor](src/vendor). The dataset providers'
terms apply to their data.
