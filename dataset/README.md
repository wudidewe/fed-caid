# Dataset setup

The complete four-dataset bundle is provided as separate [Release attachments](https://gitee.com/liuluz/fed-caid/releases/tag/datasets-v1.0.0). These files are not included in `git clone`. Raw datasets and generated caches remain excluded from Git; only this setup document is tracked under `dataset/`. You can also obtain the datasets manually from their official providers below.

## Release bundle

Download all files listed for each dataset into a `downloads/` directory at the repository root, together with `README_DATASETS.md`, `SHA256SUMS.txt` and `dataset_manifest.json`. Do not rename the parts. All parts of an archive must be in the same directory. The automatically generated `Source code` archives contain the software only; download the named dataset attachments for the data.

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

IDS and IoV already include their outer directory. Car-Hacking and NB15 contain CSVs at archive root, so extract them into the specific subdirectories above. The experiment inputs are byte identical to the validated local sources. Accompanying source files are preserved in the data bundle but are not used by the experiment.

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

From the repository root, run:

```text
python main.py
```

On the first run, the reader uses these raw CSVs and the bundled row identities to recreate the fixed TRAIN, CAL and VAL roles. It writes `development.npz`, `preparation.json` and `selected_train_identities.npz` to `./data_cache/<dataset>/`, outside the raw dataset tree. Subsequent runs reuse complete verified caches while still checking the raw sources. The software performs no automatic downloads.

Keep the 12 bundled recipe and row identity files unchanged. They are program resources and must remain in the source release. No generated cache or preparation metadata belongs under `dataset/`. Dataset redistribution and use remain subject to the providers' terms.
