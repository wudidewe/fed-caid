"""Validate the capture declaration before sealing and scoring the Hulk bundles."""

from pathlib import Path

from . import revision_external
from .revision_followup_hulk import FROZEN_VERSION, _attribution_plan, _frozen_hulk_heads, _source_artifacts, run_hulk
from .revision_v12 import _hash, _output_lock, _read, _write

ARCHIVE_PREFIX = 'workflow_before_manifest_override_'
PREFLIGHT_ERRORS = (KeyError, OSError, RuntimeError, TypeError, ValueError)


def _identity(source, manifest):
    return {'source': str(source), 'required_manifest': str(manifest), 'frozen_version': FROZEN_VERSION}


def _has_placeholder(value):
    if isinstance(value, str):
        return ('<' in value and '>' in value) or value.startswith('REPLACE_')
    if isinstance(value, dict):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_placeholder(item) for item in value)
    return False


def _pending_manifest_reason(manifest):
    if manifest.name.lower().endswith('.example.json'):
        return 'The example is a template, not a declared capture manifest'
    if not manifest.exists():
        return 'Independent capture manifest is missing'
    if not manifest.is_file():
        raise ValueError('Manifest path must identify a JSON file, not a directory')
    document = _read(manifest)
    if not isinstance(document, dict):
        raise ValueError('External manifest must be a JSON object')
    attestation = document.get('attestation', {})
    if not isinstance(attestation, dict):
        raise ValueError('External attestation must be a JSON object')
    if _has_placeholder(document) or not attestation.get('custodian'):
        return 'Manifest template or custodian attestation is incomplete'
    if attestation.get('unseen_in_development') in (None, False):
        return 'Manifest template or custodian attestation is incomplete'
    criteria, _ = revision_external._config(document, FROZEN_VERSION)
    if len(document['captures']) < criteria['min_captures']:
        return f"Independent capture count is below declared minimum ({criteria['min_captures']})"
    return None


def _missing_captures(plan):
    missing = set()
    for capture in plan['captures']:
        candidates = [capture['prepared_csv'], *(path for path, _ in capture['raw_files'])]
        for name in candidates:
            path = Path(name)
            if path.exists() and not path.is_file():
                raise ValueError(f'Declared capture input is not a file: {path}')
            if not path.is_file():
                missing.add(str(path))
    return sorted(missing)


def preflight_hulk_declaration(manifest):
    """Check declared capture files before any current-run models are trained."""
    manifest = Path(manifest).resolve()
    identity = {'required_manifest': str(manifest), 'capture_contents_read': False, 'scoring': False}
    try:
        reason = _pending_manifest_reason(manifest)
        if reason:
            return {**identity, 'status': 'pending', 'reason': reason}
        captures = revision_external._capture_files(manifest, _read(manifest)['captures'], set())
        missing = _missing_captures({'captures': captures})
        if missing:
            return {
                **identity,
                'status': 'pending',
                'reason': 'Declared capture files are missing',
                'missing_files': missing,
            }
        return {**identity, 'status': 'ready', 'scope': 'declaration only; current-run models checked later'}
    except PREFLIGHT_ERRORS as error:
        return {**identity, 'status': 'failed', 'reason': str(error)}


def _inspect_sources(source, manifest):
    source_artifacts = _source_artifacts(source)
    heads = _frozen_hulk_heads(source)
    plan = _attribution_plan(heads, manifest)
    missing = _missing_captures(plan)
    receipt = _read(heads[0].with_suffix('.json'))
    details = {
        'manifest_sha256': _hash(manifest),
        'seeds': plan['seeds'],
        'capture_count': len(plan['captures']),
        'source_artifacts': source_artifacts,
        'expected_prepared_columns': [*receipt['feature_order'], 'label', 'group_id'],
        'expected_labels': receipt['class_order'],
        'schema_scope': 'Frozen model and declaration validated; CSV contents are checked only after sealing',
        'provenance_scope': (
            'Custodian declaration and development-source hash exclusion; no independent custody proof'
        ),
    }
    if missing:
        return dict(details, status='pending', reason='Declared capture files are missing', missing_files=missing)
    return {**details, 'status': 'ready', 'reason': 'Metadata and file availability checks passed'}


def preflight_hulk(source: Path, manifest: Path) -> dict:
    """Check metadata without reading capture rows, scoring, or claiming independence."""
    source, manifest = Path(source).resolve(), Path(manifest).resolve()
    identity = {
        **_identity(source, manifest),
        'capture_contents_read': False,
        'scoring': False,
        'independence_verified': False,
    }
    try:
        reason = _pending_manifest_reason(manifest)
        if reason:
            return {**identity, 'status': 'pending', 'reason': reason}
        return {**identity, **_inspect_sources(source, manifest)}
    except PREFLIGHT_ERRORS as error:
        return {**identity, 'status': 'failed', 'reason': str(error)}


def _validate_workflow_paths(output, source, manifest, workflow):
    if output.is_relative_to(source) or source.is_relative_to(output) or manifest.is_relative_to(output):
        raise ValueError('External output must be disjoint from source models and the manifest')
    for path in (workflow, output / '.v12_lock'):
        if path.resolve() != path:
            raise ValueError('External workflow control path is redirected')


def _archive_workflow(workflow):
    previous_hash = _hash(workflow)
    archive = workflow.with_name(f'{ARCHIVE_PREFIX}{previous_hash}.json')
    if archive.resolve() != archive:
        raise ValueError('External workflow archive path is redirected')
    if archive.exists():
        if _hash(archive) != previous_hash:
            raise ValueError('Previous unsealed workflow archive changed')
    else:
        with archive.open('xb') as stream:
            stream.write(workflow.read_bytes())


def _prior_workflow(output, workflow, manifest):
    history = _read(workflow)
    if not isinstance(history, dict):
        raise ValueError('External workflow receipt must be a JSON object')
    if history.get('manifest') == str(manifest):
        return None
    if any((output / name).exists() for name in ('seal.json', 'result', 'receipt.json')):
        raise ValueError('A sealed external run cannot change its manifest path; preserve the original run')
    return {key: value for key, value in history.items() if key != 'manifest'}


def _manifest_override(output, source, manifest, source_files):
    """Preserve the previous workflow when correcting an unsealed manifest path."""
    workflow = output / 'workflow.json'
    _validate_workflow_paths(output, source, manifest, workflow)
    if not workflow.exists():
        return
    with _output_lock(output):
        former = _prior_workflow(output, workflow, manifest)
        if former is None:
            return
        expected = {
            'schema_version': 1,
            'source': str(source),
            'source_artifacts': source_files,
            'frozen_version': FROZEN_VERSION,
        }
        if former != expected:
            raise ValueError('Unsealed external workflow has a different source or configuration')
        _archive_workflow(workflow)
        _write(workflow, {**expected, 'manifest': str(manifest)})


def _include_archived_workflows(output, result):
    if result['status'] != 'completed':
        return
    for archive in sorted(output.glob(f'{ARCHIVE_PREFIX}*.json')):
        digest = _hash(archive)
        if archive.resolve() != archive or archive.name != f'{ARCHIVE_PREFIX}{digest}.json':
            raise ValueError('Previous unsealed workflow archive changed')
        result['artifacts'][archive.name] = digest


def run_verified_hulk(output: Path, source: Path, manifest: Path) -> dict:
    """Seal and score only a ready declaration, preserving matching run receipts."""
    output, source, manifest = (Path(value).resolve() for value in (output, source, manifest))
    checked = preflight_hulk(source, manifest)
    if checked['status'] != 'ready':
        return {**checked, 'output': str(output), 'artifacts': {}}
    try:
        _manifest_override(output, source, manifest, checked['source_artifacts'])
        result = run_hulk(output, source, manifest)
        result['preflight'] = {key: value for key, value in checked.items() if key != 'source_artifacts'}
        _include_archived_workflows(output, result)
        return result
    except PREFLIGHT_ERRORS as error:
        identity = _identity(source, manifest)
        return {**identity, 'status': 'failed', 'reason': str(error), 'output': str(output), 'artifacts': {}}
