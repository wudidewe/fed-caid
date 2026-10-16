"""Identify datasets and load frozen TRAIN/CAL/VAL from cache or pinned sources."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

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
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _validate_download_url(url, *, resolve=False):
    if not isinstance(url, str) or not url or any(c.isspace() for c in url):
        raise ValueError("A verified public HTTPS package URL must be configured")
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port not in (None, 443)
    ):
        raise ValueError(
            "Dataset downloads require a public HTTPS URL on port 443 without credentials"
        )
    hostname = hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local/private download hosts are forbidden")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Local/private download addresses are forbidden")
    if resolve:
        addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        if not addresses or any(
            not ipaddress.ip_address(item[4][0]).is_global for item in addresses
        ):
            raise ValueError("Download host must resolve only to public addresses")
    return url


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        _validate_download_url(new_url, resolve=True)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _download_archive(url, destination, maximum_bytes):
    _validate_download_url(url, resolve=True)
    # Check public endpoints and redirects without environment proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _HttpsRedirect())
    request = urllib.request.Request(
        url, headers={"User-Agent": "IOV-instruction/1.0 dataset-reader"}
    )
    total = 0
    with opener.open(request, timeout=30) as response:
        _validate_download_url(response.geturl(), resolve=True)
        with Path(destination).open("xb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("Dataset archive exceeds the configured download limit")
                handle.write(chunk)


def _read_manifest(manifest, dataset):
    if manifest is None:
        raise ValueError(
            "Repository dataset is missing. No verified prepared-data download package is configured"
        )
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Download manifest requires schema_version=1 and datasets")
    entry = manifest["datasets"].get(dataset)
    if not isinstance(entry, dict):
        raise ValueError("No verified prepared-data package configured for " + dataset)
    _validate_download_url(entry.get("url"))
    pins = entry.get("files")
    if not isinstance(pins, dict) or set(pins) != _PACKAGE_FILES:
        raise ValueError(
            "Download package must pin development.npz, preparation.json and TRAIN identities"
        )
    if not isinstance(entry.get("archive_sha256"), str) or not _HEX_SHA256.fullmatch(
        entry["archive_sha256"]
    ):
        raise ValueError("A lowercase SHA256 archive pin is required")
    if any(
        not isinstance(value, str) or not _HEX_SHA256.fullmatch(value) for value in pins.values()
    ):
        raise ValueError("A lowercase SHA256 pin is required for every package member")
    entry = dict(entry)
    for key, default in (
        ("max_download_bytes", 8 * 1024**3),
        ("max_uncompressed_bytes", 16 * 1024**3),
    ):
        value = entry.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError("A positive integer package size limit is required: " + key)
        entry[key] = value
    return entry


def _verify_package_files(directory, entry):
    for filename, expected in entry["files"].items():
        path = directory / filename
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError("Prepared package member hash differs: " + filename)


def _unpack_verified_archive(archive, directory, entry):
    if _sha256(archive) != entry["archive_sha256"]:
        raise ValueError("Downloaded archive SHA256 differs from the configured pin")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(_PACKAGE_FILES):
            raise ValueError("Prepared ZIP must contain exactly the three flat package files")
        seen, expanded = set(), 0
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (
                name not in _PACKAGE_FILES
                or path.is_absolute()
                or len(path.parts) != 1
                or "\\" in name
                or name in seen
                or member.is_dir()
                or stat.S_ISLNK(mode)
                or member.flag_bits & 1
            ):
                raise ValueError("Unsafe, duplicate or unexpected prepared ZIP member: " + name)
            expanded += member.file_size
            if expanded > entry["max_uncompressed_bytes"]:
                raise ValueError("Prepared package exceeds the uncompressed size limit")
            seen.add(name)
        directory.mkdir()
        for member in members:
            # Explicit flat destinations avoid extractall and path traversal.
            with bundle.open(member) as source, (directory / member.filename).open("xb") as target:
                copied = 0
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > member.file_size or copied > entry["max_uncompressed_bytes"]:
                        raise ValueError("Prepared ZIP member exceeds its declared size")
                    target.write(chunk)
    _verify_package_files(directory, entry)


def _manifest_for_root(root, manifest):
    if manifest is None:
        default = root / "download_sources.json"
        return default if default.is_file() else None
    return manifest


def check_repository_source(
    dataset, dataset_dir=None, download_manifest=None, *, allow_download=True
):
    """Validate missing-data configuration without network or filesystem writes."""
    name = _canonical_name(dataset)
    root = (
        Path(dataset_dir) if dataset_dir is not None else project_root(__file__) / "dataset"
    ).resolve()
    manifest = _manifest_for_root(root, download_manifest)
    from dataset_preparation import recipe_path, check_raw_sources

    if (manifest is None or not allow_download) and recipe_path(name, root).is_file():
        check_raw_sources(name, root, allow_download=allow_download)
        return {"_source_kind": "raw"}
    if not allow_download:
        raise FileNotFoundError(
            f"Dataset {name} is missing from {root}; offline mode forbids downloading"
        )
    if manifest is None:
        raise ValueError(
            f"No verified prepared-data download package is configured for {name}. Configure {root / 'download_sources.json'} "
            "with a verified prepared ZIP URL and SHA256 pins, or install the prepared files locally"
        )
    return _read_manifest(manifest, name)


def read_repository_dataset(
    dataset, dataset_dir=None, download_manifest=None, *, allow_download=True
):
    """Load or prepare dataset/<name>/ without replacing existing files."""
    name = _canonical_name(dataset)
    root = (
        Path(dataset_dir) if dataset_dir is not None else project_root(__file__) / "dataset"
    ).resolve()
    download_manifest = _manifest_for_root(root, download_manifest)
    target = root / name
    if target.exists():
        if not target.is_dir() or target.is_symlink():
            raise ValueError("Repository dataset location must be a regular directory")
        if download_manifest is not None:
            _verify_package_files(target, _read_manifest(download_manifest, name))
        return read_local_dataset(target / "development.npz", dataset=name)
    entry = check_repository_source(name, root, download_manifest, allow_download=allow_download)
    if entry.get("_source_kind") == "raw":
        from dataset_preparation import prepare_dataset

        path = prepare_dataset(name, root, allow_download=allow_download)
        return read_local_dataset(path, dataset=name)
    root.mkdir(parents=True, exist_ok=True)
    # Install verified files atomically from the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".prepared_download_", dir=root) as temporary:
        temporary = Path(temporary)
        archive, prepared = temporary / "package.zip", temporary / "prepared"
        _download_archive(entry["url"], archive, entry["max_download_bytes"])
        _unpack_verified_archive(archive, prepared, entry)
        result = read_local_dataset(prepared / "development.npz", dataset=name)
        if target.exists():
            raise FileExistsError("Dataset appeared during download; existing files were preserved")
        os.rename(prepared, target)
    return result
