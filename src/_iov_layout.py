"""Filesystem integration only; no data parsing or numerical configuration."""

from __future__ import annotations

from pathlib import Path

SNAPSHOT_SUPPORT_FILES = (
    '__init__.py',
    '_iov_layout.py',
    'utils.py',
    'dataset_access.py',
    'dataset_preparation.py',
    'revision_measurement.py',
    'experiment_config.py',
    'data_io.py',
    'privacy_core.py',
    'privacy_run.py',
    'secure_run.py',
    'secure_aggregation.py',
    'aggregation_audit.py',
    'validation_core.py',
    'vendor/tabm.py',
    'vendor/rtdl_num_embeddings.py',
    'vendor/__init__.py',
    'vendor/TABM_LICENSE.txt',
    'vendor/RTDL_LICENSE.txt',
    'vendor/NOTICE.md',
    'fedcaid_resources/__init__.py',
    'fedcaid_resources/frozen_format_compatibility.json',
    'fedcaid_resources/package_import_compatibility.json',
    'fedcaid_resources/preparation/CIC-IoV-2024/recipe.json',
    'fedcaid_resources/preparation/CIC-IoV-2024/heldout_identities.csv',
    'fedcaid_resources/preparation/CIC-IoV-2024/excluded_identities.csv',
    'fedcaid_resources/preparation/Car-Hacking/recipe.json',
    'fedcaid_resources/preparation/Car-Hacking/heldout_identities.csv',
    'fedcaid_resources/preparation/Car-Hacking/excluded_identities.csv',
    'fedcaid_resources/preparation/CIC-IDS-2017/recipe.json',
    'fedcaid_resources/preparation/CIC-IDS-2017/heldout_identities.csv',
    'fedcaid_resources/preparation/CIC-IDS-2017/excluded_identities.csv',
    'fedcaid_resources/preparation/CIC-UNSW-NB15/recipe.json',
    'fedcaid_resources/preparation/CIC-UNSW-NB15/heldout_identities.csv',
    'fedcaid_resources/preparation/CIC-UNSW-NB15/excluded_identities.csv',
)


def project_root(module_file):
    """Use the checkout root, archive directory or installed command working directory."""
    directory = Path(module_file).resolve().parent
    if directory.name == 'src' and (directory.parent / 'main.py').is_file():
        return directory.parent
    package_container = directory.parent if directory.name == 'src' else directory
    if package_container.name in {'site-packages', 'dist-packages'}:
        return Path.cwd().resolve()
    return directory


def source_path(relative):
    """Resolve the actual implementation, never a checkout compatibility shim."""
    return Path(__file__).resolve().parent / relative
