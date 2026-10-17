# Dataset setup

Download the datasets manually from their official providers. Raw datasets and generated caches are excluded from Git. Only this setup document is included under `dataset/`.

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
