"""Filesystem integration only; no data parsing or numerical configuration."""

from __future__ import annotations

from pathlib import Path


SNAPSHOT_SUPPORT_FILES = (
    "_iov_layout.py",
    "utils.py",
    "dataset_access.py",
    "dataset_preparation.py",
    "kaggle_source.py",
    "round_observability.py",
    "experiment_report.py",
    "experiment_report_data.py",
    "report_timeseries.py",
    "iov_instruction.py",
    "main.py",
    "data_io.py",
    "ids_core.py",
    "privacy_core.py",
    "privacy_run.py",
    "secure_run.py",
    "secure_aggregation.py",
    "aggregation_audit.py",
    "validation_core.py",
    "vendor/tabm.py",
    "vendor/rtdl_num_embeddings.py",
    "vendor/TABM_LICENSE.txt",
    "vendor/RTDL_LICENSE.txt",
    "vendor/NOTICE.md",
)


def project_root(module_file):
    """Use the checkout root, archive directory or installed command working directory."""
    directory = Path(module_file).resolve().parent
    if directory.name == "src" and (directory.parent / "main.py").is_file():
        return directory.parent
    if directory.name in {"site-packages", "dist-packages"}:
        return Path.cwd().resolve()
    return directory


def source_path(relative):
    """Resolve the actual implementation, never a checkout compatibility shim."""
    return Path(__file__).resolve().parent / relative
