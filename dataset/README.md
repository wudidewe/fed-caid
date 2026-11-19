# Dataset setup

Obtain the four datasets from the official providers below and place the 19 required CSVs in the listed directories. The GitHub repository and its source archives contain the software and preparation recipes. The repository excludes raw datasets and generated caches.

## Official sources and expected layout

| Dataset | Official source | Required files |
| --- | --- | --- |
| CIC-IoV-2024 | [UNB CIC](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) | Six decimal CSVs |
| Car-Hacking | [HCRL](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) | Four attack CSVs |
| CIC-IDS-2017 | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) | Eight CSVs from `MachineLearningCSV` |
| CIC-UNSW-NB15 | [UNB CIC](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html) | CICFlowMeter representation with its `Label` column |

Extract the 19 required CSVs into these exact locations relative to the repository root:

```text
dataset/
  README.md
  CICIoV2024/
    decimal/
      decimal_benign.csv
      decimal_DoS.csv
      decimal_spoofing-GAS.csv
      decimal_spoofing-RPM.csv
      decimal_spoofing-SPEED.csv
      decimal_spoofing-STEERING_WHEEL.csv
  Car-Hacking Dataset/
    DoS_dataset.csv
    Fuzzy_dataset.csv
    gear_dataset.csv
    RPM_dataset.csv
  CIC-IDS-2017 Dataset/
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Monday-WorkingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
  CIC-UNSW-NB15/
    CICFlowMeter_out.csv
```

The program compares source byte counts and SHA256 hashes with `src/fedcaid_resources/preparation/<dataset>/recipe.json`. Do not edit CSV contents, headers, encodings or line endings. A source mismatch stops execution instead of silently changing the experiment. An updated provider release may differ from the validated release; compare its hash with the recipe before running. File matching is case insensitive, but the directory locations above are fixed.

NB15 `CICFlowMeter.csv` is accepted as an alternative filename only when its bytes match the recorded `CICFlowMeter_out.csv` source. Original UNSW-NB15 tables and separate `Data.csv`/`Label.csv` are not interchangeable with this source. IoV binary/hexadecimal files, Car-Hacking `normal_run_data/` and other accompanying files are not required by the experiment.

After installing the dependencies, open the repository root in PyCharm and run `main.py` with no arguments. From a terminal at the same location, start the full review workflow with:

```text
python main.py --full-review
```

The preparation code verifies the raw CSVs and uses the bundled row identities to reconstruct the original TRAIN, CAL and VAL roles. When needed, it creates `development.npz`, `preparation.json` and `selected_train_identities.npz` under `data_cache/<dataset>/`. The review workflow derives its grouped and retrospective evaluation roles from the verified sources and writes results under `outputs/fedcaid_review/`. The workflows and reports are described in the [project README](../README.md).

Keep the 12 recipe and row identity files under `src/fedcaid_resources/preparation/` unchanged. Generated caches belong under `data_cache/`, and results under `outputs/`. Follow each provider's terms for dataset use and redistribution.

## Optional CSE-CIC-IDS2018 capture

From the [CSE-CIC-IDS2018 registry](https://registry.opendata.aws/cse-cic-ids2018/),
download the [16 February 2018 feature CSV](https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv).
Rename it to `02-16-2018.csv` and place it in
`dataset/CSE-CIC-IDS2018/` without changing its contents. The expected SHA256 is:

```text
1a4919faa0c49c7af97230b0c2d076eba23ee6dd81103a3801d51ac316355d8b
```

The capture is optional for the primary benchmark. It contains the Hulk and
SlowHTTPTest traffic from the same public acquisition day. The [project
README](../README.md#optional-cse-cic-ids2018-replay) describes the automatic
and focused replay modes. Fetch the original 2017 CSVs above as well to
reconstruct the CAL threshold used by the focused replay.

## Historical dataset bundle

The earlier [dataset release](https://gitee.com/liuluz/fed-caid/releases/tag/datasets-v1.0.0) documented the following split archives. If you already obtained this bundle under the providers' terms, keep the parts for each dataset together in a `downloads/` directory at the repository root. Use the accompanying `README_DATASETS.md`, `SHA256SUMS.txt` and `dataset_manifest.json` to identify and verify the files. Keep the original filenames.

| Dataset | Download files | Extract to |
| --- | --- | --- |
| Car-Hacking | `car-hacking.zip.001` and `.002` | `dataset/Car-Hacking Dataset/` |
| CIC-IDS-2017 | `cic-ids-2017.zip.001` through `.003` | `dataset/` |
| CIC-IoV-2024 | `cic-iov-2024.zip` | `dataset/` |
| CIC-UNSW-NB15 | `cic-unsw-nb15.zip.001` through `.006` | `dataset/CIC-UNSW-NB15/` |

Verify the downloaded files against `SHA256SUMS.txt`. On Windows, use `Get-FileHash -Algorithm SHA256 <file>`; on systems with `sha256sum`, run `sha256sum -c SHA256SUMS.txt` from the download directory. Stop if any hash differs.

Use 7-Zip to open the first part of each split ZIP. With `7z` on PATH, run from the repository root:

```text
7z x downloads/car-hacking.zip.001 "-o./dataset/Car-Hacking Dataset"
7z x downloads/cic-ids-2017.zip.001 -o./dataset
7z x downloads/cic-iov-2024.zip -o./dataset
7z x downloads/cic-unsw-nb15.zip.001 -o./dataset/CIC-UNSW-NB15
```

The IDS and IoV archives already include their outer directory. Car-Hacking and NB15 contain CSVs at the archive root. Extract them into the matching subdirectories above and verify the result against the preparation recipes.
