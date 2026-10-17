"""Identify datasets and load frozen TRAIN/CAL/VAL from local files."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from _iov_layout import project_root


# Ordered feature/class signatures distinguish the two 68-feature datasets.
DATASET_PROFILES = {
    "CIC-IoV-2024": {
        "class_names": ["BENIGN", "DoS", "GAS", "RPM", "SPEED", "STEERING_WHEEL"],
        "feature_names": [
            "ID",
            "DATA_0",
            "DATA_1",
            "DATA_2",
            "DATA_3",
            "DATA_4",
            "DATA_5",
            "DATA_6",
            "DATA_7",
        ],
    },
    "Car-Hacking": {
        "class_names": ["BENIGN", "DoS", "Fuzzy", "Gear spoofing", "RPM spoofing"],
        "feature_names": [
            "ID",
            "DLC",
            "DATA_0",
            "DATA_1",
            "DATA_2",
            "DATA_3",
            "DATA_4",
            "DATA_5",
            "DATA_6",
            "DATA_7",
        ],
    },
    "CIC-IDS-2017": {
        "class_names": [
            "BENIGN",
            "Bot",
            "DDoS",
            "DoS GoldenEye",
            "DoS Hulk",
            "DoS Slowhttptest",
            "DoS slowloris",
            "FTP-Patator",
            "Heartbleed",
            "Infiltration",
            "PortScan",
            "SSH-Patator",
            "Web Attack - Brute Force",
            "Web Attack - Sql Injection",
            "Web Attack - XSS",
        ],
        "feature_names": [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
            "FE_packet_direction_balance",
            "FE_byte_direction_balance",
            "FE_syn_ack_balance",
        ],
    },
    "CIC-UNSW-NB15": {
        "class_names": [
            "BENIGN",
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ],
        "feature_names": [
            "Dst Port",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packet",
            "Total Bwd packets",
            "Total Length of Fwd Packet",
            "Total Length of Bwd Packet",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Packet Length Min",
            "Packet Length Max",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Fwd Segment Size Avg",
            "Bwd Segment Size Avg",
            "Bwd Bytes/Bulk Avg",
            "Bwd Packet/Bulk Avg",
            "Bwd Bulk Rate Avg",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "FWD Init Win Bytes",
            "Bwd Init Win Bytes",
            "Fwd Act Data Pkts",
            "Fwd Seg Size Min",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ],
    },
}
DATASETS = tuple(DATASET_PROFILES)
_DEVELOPMENT_KEYS = {"class_names", "feature_names"} | {
    f"{xy}_{role}" for xy in ("X", "y") for role in ("train", "val", "cal")
}
_PACKAGE_FILES = {"development.npz", "preparation.json", "selected_train_identities.npz"}


def has_prepared_files(directory):
    """Detect a complete cache and reject partial or linked cache members."""
    directory = Path(directory)
    members = [directory / filename for filename in sorted(_PACKAGE_FILES)]
    if not any(path.exists() or path.is_symlink() for path in members):
        return False
    invalid = [path.name for path in members if not path.is_file() or path.is_symlink()]
    if invalid:
        raise ValueError(
            "Prepared dataset is incomplete; missing or invalid: "
            + ", ".join(invalid)
            + ". Existing files preserved."
        )
    return True


def _canonical_name(name):
    if name not in DATASET_PROFILES:
        raise ValueError("Unknown dataset identifier: " + str(name))
    return name


def identify_dataset(path, expected="auto"):
    """Match complete ordered feature and class maps to a supported dataset."""
    if expected != "auto":
        _canonical_name(expected)
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != _DEVELOPMENT_KEYS:
            raise ValueError(
                "Input must contain exactly TRAIN/VAL/CAL and metadata; TEST/confirmation forbidden"
            )
        classes = archive["class_names"].tolist()
        features = archive["feature_names"].tolist()
    matches = [
        name
        for name, profile in DATASET_PROFILES.items()
        if classes == profile["class_names"] and features == profile["feature_names"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "Dataset requires one exact ordered feature/class signature match; got "
            + str(len(matches))
        )
    if expected != "auto" and matches[0] != expected:
        raise ValueError(
            "Declared dataset conflicts with archive content: " + expected + " != " + matches[0]
        )
    return matches[0]


def read_local_dataset(path, dataset="auto"):
    """Load frozen TRAIN/CAL/VAL with cache and TRAIN identity hash checks."""
    identify_dataset(path, expected=dataset)
    # Import the shared loader lazily to avoid circular imports.
    from utils import load_development

    return load_development(Path(path))


def check_dataset_source(dataset, dataset_dir=None):
    """Check manually installed raw files without creating files or using the network."""
    name = _canonical_name(dataset)
    root = Path(dataset_dir) if dataset_dir is not None else project_root(__file__) / "dataset"
    from dataset_preparation import check_raw_sources

    return check_raw_sources(name, root)


def read_dataset_directory(dataset, dataset_dir=None, cache_dir=None):
    """Read fixed local CSV directories and reuse a separate verified cache."""
    name = _canonical_name(dataset)
    root = (
        Path(dataset_dir) if dataset_dir is not None else project_root(__file__) / "dataset"
    ).resolve()
    from dataset_preparation import DEFAULT_CACHE, prepare_dataset

    spec = check_dataset_source(name, root)
    cache = Path(cache_dir).resolve() if cache_dir is not None else DEFAULT_CACHE
    target = cache / name
    if target.exists():
        if not target.is_dir() or target.is_symlink():
            raise ValueError("Local cache location must be a regular directory")
        if has_prepared_files(target):
            audit = json.loads((target / "preparation.json").read_text(encoding="utf-8"))
            if audit.get("source_recipe_sha256") != spec["recipe_sha256"]:
                raise ValueError("Prepared cache belongs to a different frozen role recipe")
            return read_local_dataset(target / "development.npz", dataset=name)

    path = prepare_dataset(name, root, cache_root=cache)
    return read_local_dataset(path, dataset=name)
