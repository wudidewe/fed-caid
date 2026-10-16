"""Install only the pinned raw CSV files needed by the published experiment.

KaggleHub handles authentication and transport. Downloads are staged locally,
and every required file must pass its byte count and SHA256 contract before
any missing raw file is installed. No data frame, split or model is changed.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile


KAGGLE_HANDLES = {
    'CIC-IoV-2024': 'pushpakattarde/ciciov2024decimalcsv',
    'Car-Hacking': 'pranavjha24/car-hacking-dataset',
    'CIC-IDS-2017': 'chethuhn/network-intrusion-dataset',
    'CIC-UNSW-NB15': 'yasserhessein/cic-unsw-nb15-augmented-dataset',
}


def _filename(value, *, nested=False):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value or
            '\x00' in value or value != value.strip()):
        raise ValueError('A relative Kaggle CSV file name is required')
    path = PurePosixPath(value)
    if (path.is_absolute() or any(part in {'.', '..', ''} for part in path.parts) or
            str(path) != value or (not nested and len(path.parts) != 1) or
            path.suffix.lower() != '.csv'):
        raise ValueError('Only declared relative CSV files are supported')
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _temporary_parent():
    return Path(tempfile.gettempdir()).resolve()


def _verify(path, source):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing regular raw CSV: {source["filename"]}')
    if path.stat().st_size != source['bytes']:
        raise ValueError(f'Raw CSV byte count differs from the frozen experiment: {source["filename"]}')
    if _sha256(path) != source['sha256']:
        raise ValueError(f'Raw CSV SHA256 differs from the frozen experiment: {source["filename"]}')


def _unpack_csv(path, remote_name, expected_bytes):
    """Handle a single CSV zip when its transport URL hides the zip suffix."""
    if not zipfile.is_zipfile(path):
        return path
    with zipfile.ZipFile(path) as archive:
        members = [entry for entry in archive.infolist() if not entry.is_dir()]
        if (len(members) != 1 or members[0].filename not in {remote_name, PurePosixPath(remote_name).name} or
                members[0].file_size != expected_bytes or members[0].flag_bits & 1):
            raise ValueError('Kaggle single-file archive does not match the declared CSV contract')
        extracted = path.with_name(path.name + '.unpacked')
        # Do not extract a ZIP path onto the filesystem. Only the validated
        # single CSV member is streamed into a locally selected filename.
        with archive.open(members[0]) as source, extracted.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    return extracted


def _download_file(handle, filename, output):
    try:
        kagglehub = importlib.import_module('kagglehub')
    except ImportError as error:
        raise RuntimeError('Automatic Kaggle downloads require kagglehub. Install the project dataset dependencies first.') from error
    try:
        saved = kagglehub.dataset_download(handle, path=filename, output_dir=str(output))
    except Exception as error:
        raise RuntimeError('Kaggle download failed. Check the connection and any Kaggle account or dataset consent requirements.') from error
    path = Path(saved).resolve()
    if not path.is_relative_to(output.resolve()):
        raise RuntimeError('KaggleHub returned a file outside the requested staging directory')
    return path


def ensure_kaggle_sources(dataset, dataset_root, recipe):
    """Fetch missing declared CSVs from an explicit Kaggle dataset version.

    Existing raw files are verified and never silently replaced. The download
    routine performs no source upload and never falls back to an unpinned
    latest version, alternative data, or a different train/validation split.
    ``source_files[].path`` is deliberately ignored: destinations are derived
    from ``dataset_root/<raw_directory>/<filename>`` only. The default raw
    directory is ``raw/<dataset>``; the IoV recipe uses ``archive``.
    """
    if dataset not in KAGGLE_HANDLES or recipe.get('name') != dataset:
        raise ValueError('The recipe must identify one supported dataset')
    provenance = recipe.get('kaggle', {})
    version = provenance.get('version')
    if (provenance.get('handle') != KAGGLE_HANDLES[dataset] or
            type(version) is not int or version < 1):
        raise ValueError('A declared Kaggle source and explicit positive version are required')
    sources = recipe.get('source_files')
    if not isinstance(sources, list) or not sources:
        raise ValueError('The experiment recipe must declare its raw CSV files')
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError('Each raw file requires a verification contract')
        name = _filename(source.get('filename'))
        _filename(source.get('kaggle_path', name), nested=True)
        if (name.casefold() in seen or type(source.get('bytes')) is not int or source['bytes'] < 1 or
                not isinstance(source.get('sha256'), str) or
                re.fullmatch(r'[0-9a-f]{64}', source['sha256']) is None):
            raise ValueError('Raw CSV names must be unique and have valid byte and SHA256 contracts')
        seen.add(name.casefold())
    root = Path(dataset_root).resolve()
    relative = recipe.get('raw_directory', 'raw/' + dataset)
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative or
            '\x00' in relative or PurePosixPath(relative).is_absolute() or
            any(part in {'.', '..', ''} for part in PurePosixPath(relative).parts) or
            str(PurePosixPath(relative)) != relative):
        raise ValueError('The raw directory must be a safe relative dataset path')
    raw = root.joinpath(*PurePosixPath(relative).parts)
    if not raw.resolve().is_relative_to(root):
        raise ValueError('Raw dataset directory escapes dataset_root')
    missing = []
    for source in sources:
        target = raw / source['filename']
        if target.exists() or target.is_symlink():
            _verify(target, source)
        else:
            missing.append(source)
    if not missing:
        return None
    # KaggleHub creates nested completion markers containing the remote
    # handle and CSV name. The system temporary directory keeps those paths
    # short even when the repository is checked out in a deep Windows path.
    staging = _temporary_parent()
    handle = f'{provenance["handle"]}/versions/{version}'
    print(f'DATASET DOWNLOAD {dataset}: Kaggle version {version}; {len(missing)} required CSV files', flush=True)
    with tempfile.TemporaryDirectory(prefix='fedcaid_', dir=staging) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(staging.resolve()):
            raise ValueError('Dataset download staging directory is invalid')
        verified = []
        for index, source in enumerate(missing):
            output = temporary / str(index)
            remote_name = source.get('kaggle_path', source['filename'])
            saved = _download_file(handle, remote_name, output)
            saved = _unpack_csv(saved, remote_name, source['bytes'])
            _verify(saved, source)
            verified.append((source, saved))
        # Publish only after all downloads have passed. The subsequent data
        # preparation code checks the frozen physical row identities/counts.
        raw.mkdir(parents=True, exist_ok=True)
        for source, saved in verified:
            target = raw / source['filename']
            if target.exists() or target.is_symlink():
                _verify(target, source)
            else:
                # The system temporary directory can be on another volume.
                # Copy into a temporary file beside the destination before
                # the final atomic rename; an interrupted copy is never a
                # complete raw CSV in the persistent dataset directory.
                with tempfile.NamedTemporaryFile(prefix='.install_', dir=raw, delete=False) as stream:
                    installing = Path(stream.name)
                try:
                    shutil.copyfile(saved, installing)
                    _verify(installing, source)
                    os.replace(installing, target)
                finally:
                    if installing.exists():
                        installing.unlink()
    return None
