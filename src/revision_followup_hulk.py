"""Verify, seal and score independent captures using frozen Hulk attribution models."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from . import revision_external
from .revision_v12 import _bindings, _hash, _output_lock, _read, _write

SOURCE_RECEIPTS = ('v12_complete.json', 'v12_plan.json', 'v12_state.json')
CONFIRMATION_PATHS = ('seal.json', 'result', 'receipt.json', 'workflow.json')
STAGE_RECEIPTS = {'run.json', 'complete.json', 'case_status.csv'}
NESTED_STAGE_RECEIPTS = {'CIC-IDS-2017/data_audit.json'}
HULK_SEEDS = (42, 43, 44, 45, 46)
FROZEN_VERSION = 7
WORKFLOW_ERRORS = (KeyError, OSError, RuntimeError, TypeError, ValueError)
HULK_CASE = ('CIC-IDS-2017', 'train_absence', 'DoS Hulk', 'reserved', 'evaluated')
CASE_FIELDS = ('dataset', 'scenario', 'target_class', 'cal_policy', 'status')
RESULT_FILES = frozenset(
    {
        'complete.json',
        'metrics.csv',
        'per_class.csv',
        'paired_by_seed.csv',
        'paired_intervals.csv',
        'capture_support.csv',
        'capture_metric_summaries.csv',
        'capture_intervals.csv',
        'confusion_counts.json',
        'fixed_alarm_transitions.csv',
    }
)
VALID_ASSESSMENTS = frozenset(
    {
        'criteria_met_on_declared_external_receiver',
        'criteria_not_met',
        'insufficient_support_or_gate',
    }
)


def _validate_source_receipts(marker, plan, state, files):
    if not all(isinstance(value, dict) for value in (marker, plan, state)):
        raise ValueError('V12 attribution receipts must be JSON objects')
    identity = (marker.get('mode'), marker.get('scope'), marker.get('status'))
    if (
        identity != ('attribution', 'development', 'completed')
        or marker.get('test_evaluated') is not False
        or (plan.get('mode'), plan.get('scope')) != identity[:2]
        or set(state) != {'attribution'}
        or marker.get('plan_sha256') != files['v12_plan.json']
        or marker.get('state_sha256') != files['v12_state.json']
    ):
        raise ValueError('Expected an intact completed V12 attribution development wrapper')


def _archived_attribution_files(stage, files):
    if stage.resolve() != stage or not isinstance(files, dict) or not STAGE_RECEIPTS <= files.keys():
        raise ValueError('Attribution stage is redirected or its receipt is incomplete')
    for name, expected in files.items():
        path = stage / name
        allowed = Path(name).name == name or name in NESTED_STAGE_RECEIPTS
        if not allowed or path.resolve() != path or not path.is_file() or _hash(path) != expected:
            raise ValueError(f'Completed attribution source artifact changed: {name}')
    return {f'attribution/{name}': expected for name, expected in files.items()}


def _source_artifacts(source):
    """Verify the completed development wrapper without changing its receipts."""
    if any((source / name).resolve() != source / name for name in SOURCE_RECEIPTS):
        raise ValueError('Attribution source receipt is redirected')
    marker, plan, state = (_read(source / name) for name in SOURCE_RECEIPTS)
    files = {name: _hash(source / name) for name in SOURCE_RECEIPTS}
    _validate_source_receipts(marker, plan, state, files)
    stage = source / 'attribution'
    files.update(_archived_attribution_files(stage, state['attribution']))
    indexed = _hulk_seed_index(stage / 'case_status.csv', _completed_hulk_run(stage))
    for seed, row in indexed.items():
        case_files = _source_case_files(stage, seed, row)
        if files.keys() & case_files.keys():
            raise ValueError(f'Hulk seed {seed} reuses another case artifact')
        files.update(case_files)
    return files


def _attribution_plan(heads, manifest):
    plan, _ = revision_external._plan(heads, manifest)
    identity = (plan.get('version'), plan.get('dataset'), plan.get('target_class'))
    expected = (FROZEN_VERSION, HULK_CASE[0], HULK_CASE[2])
    if identity != expected or plan.get('seeds') != list(HULK_SEEDS):
        raise ValueError('Independent attribution requires the five version-7 Hulk bundles for seeds 42-46')
    return plan


def _json(path):
    """Read an object from a saved JSON receipt."""
    try:
        with path.open(encoding='utf-8-sig') as stream:
            value = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f'Could not read the saved JSON receipt: {path}') from error
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object at {path}')
    return value


def _bundle_root(root):
    """Resolve a wrapper to its native attribution run, when present."""
    root = Path(root)
    native = root / 'attribution'
    if (root / 'case_status.csv').is_file() or not (native / 'case_status.csv').is_file():
        return root
    resolved = native.resolve()
    if resolved.parent != root:
        raise ValueError(f'The attribution stage leaves the selected run directory: {root}')
    return resolved


def _completed_hulk_run(root):
    """Require a completed real development evaluation."""
    marker = _json(root / 'complete.json')
    if (
        marker.get('task') != 'train-absence'
        or marker.get('mode') != 'evaluation'
        or marker.get('smoke') is not False
        or marker.get('head_only') is True
        or marker.get('test_evaluated') is not False
    ):
        raise ValueError(f'Choose a completed real Hulk development run: {root}')
    return marker


def _hulk_case_rows(inventory, marker):
    """Bind the full case inventory to the completion counts."""
    if not inventory.is_file():
        raise ValueError(f'Case status file was not found: {inventory}')
    with inventory.open(encoding='utf-8-sig', newline='') as stream:
        cases = list(csv.DictReader(stream))
    evaluated = sum(row.get('status') == 'evaluated' for row in cases)
    if len(cases) != marker.get('planned_case_policies') or evaluated != marker.get('evaluated'):
        raise ValueError(f'The case inventory does not match its completion receipt: {inventory.parent}')
    return [row for row in cases if tuple(row.get(field) for field in CASE_FIELDS) == HULK_CASE]


def _hulk_seed_index(inventory, marker):
    """Require exactly one selected evaluated case for every expected seed."""
    indexed = {}
    for row in _hulk_case_rows(inventory, marker):
        try:
            seed = int(row['seed'])
        except (KeyError, ValueError) as error:
            raise ValueError(f'Invalid seed in Hulk case inventory: {inventory}') from error
        if seed in indexed:
            raise ValueError(f'Hulk seed {seed} appears twice in the case inventory.')
        indexed[seed] = row
    if set(indexed) != set(HULK_SEEDS):
        raise ValueError('The run must contain the five evaluated reserved Hulk cases for seeds 42–46.')
    return indexed


def _check_head_details(head, seed, row):
    """Check the frozen metadata against the evaluated sample and receiver."""
    info = _json(head.with_suffix('.json'))
    thresholds = info.get('thresholds')
    receiver = info.get('receiver')
    if (
        type(info.get('seed')) is not int
        or info['seed'] != seed
        or info.get('dataset') != HULK_CASE[0]
        or info.get('target_class') != HULK_CASE[2]
        or type(receiver) is not int
        or str(receiver) != row.get('receiver')
        or not isinstance(thresholds, dict)
        or not {'head_local', 'head_federated'} <= thresholds.keys()
    ):
        raise ValueError(f'Frozen receipt does not match the evaluated Hulk case: {head}')
    return info


def _source_path(root, name):
    """Require an existing, nonredirected file inside its experiment run."""
    if not isinstance(name, str) or not name:
        raise ValueError('Hulk case does not name a source artifact')
    path = root / name
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Hulk source artifact is missing or leaves its run directory: {path}')
    return path


def _bound_source_path(root, name, digest):
    path = _source_path(root, name)
    if _hash(path) != revision_external._digest(digest):
        raise ValueError(f'Completed Hulk source artifact changed: {path.name}')
    return path


def _source_case_files(root, seed, row):
    """Trace the frozen head and its detector to an evaluated native case."""
    result = _bound_source_path(root, row['result'], row['result_sha256'])
    case = _json(result)
    identity, audit = case['identity'], case['audit']
    entry = case['frozen_head_artifacts'][row['cal_policy']]
    fields = {'file': 'sha256', 'receipt': 'receipt_sha256', 'tabm_checkpoint': 'tabm_checkpoint_sha256'}
    paths = {name: _bound_source_path(root, entry[name], entry[digest]) for name, digest in fields.items()}
    head = paths['file']
    info = _check_head_details(head, seed, row)
    settings = info['settings']
    policy = audit['policies'][row['cal_policy']]
    donor = settings['single_donor_client']
    if (
        result.name != 'result.json'
        or any(path.parent != result.parent for path in paths.values())
        or head != _source_path(root, row['frozen_head'])
        or head.suffix != '.npz'
        or paths['receipt'] != head.with_suffix('.json')
        or paths['tabm_checkpoint'] != _source_path(root, info['fixed_gate']['checkpoint'])
        or info['fixed_gate']['sha256'] != entry['tabm_checkpoint_sha256']
        or info['sha256'] != entry['sha256']
        or case['status'] != row['status']
        or any(identity[name] != row[name] for name in ('dataset', 'scenario', 'target_class'))
        or identity['seed'] != seed
        or entry['seed'] != seed
        or audit.get('test_evaluated') is not False
        or audit['receiver'] != info['receiver']
        or str(donor) != row['single_donor_client']
        or donor != entry['single_donor_client']
        or case['model_spec']['donor_subsets'].get(revision_external.SINGLE_DONOR_MODE) != [donor]
        or revision_external._digest(settings['route_sha256']) != audit['route_sha256']
        or revision_external._digest(settings['cal_partition_sha256']) != policy['partition_sha256']
        or policy['partition_sha256'] != row['partition_sha256']
        or (
            info['fixed_gate']['operating_selection']['cal_group_assignment_sha256']
            != policy['operating_audit']['assignment_sha256']
        )
    ):
        raise ValueError(f'Frozen artifacts do not match the evaluated Hulk case for seed {seed}')
    return {
        f'attribution/{result.relative_to(root).as_posix()}': row['result_sha256'],
        **{f'attribution/{path.relative_to(root).as_posix()}': entry[fields[name]] for name, path in paths.items()},
    }


def _frozen_head_path(root, row, seed):
    """Resolve the expected NPZ bundle without permitting a path escape."""
    name = row.get('frozen_head')
    if not name:
        raise ValueError(f'No frozen head was recorded for Hulk seed {seed}.')
    head = _source_path(root, name)
    if head.suffix.lower() != '.npz':
        raise ValueError(f'The recorded Hulk seed {seed} bundle is missing or leaves its run directory: {head}')
    _check_head_details(head, seed, row)
    return head


def _frozen_hulk_heads(root):
    """Return five distinct frozen NPZ paths in seed order."""
    root = _bundle_root(root)
    indexed = _hulk_seed_index(root / 'case_status.csv', _completed_hulk_run(root))
    heads = [_frozen_head_path(root, indexed[seed], seed) for seed in HULK_SEEDS]
    if len(set(heads)) != len(HULK_SEEDS):
        raise ValueError('Hulk frozen head paths must be distinct.')
    return heads


def _verify_capture_hashes(plan):
    """Check that every prepared and raw capture matches the sealed hash."""
    for capture in plan['captures']:
        files = [(capture['prepared_csv'], capture['sha256']), *capture['raw_files']]
        for path, expected in files:
            if _hash(Path(path)) != expected:
                raise ValueError(f'Independent capture file differs from its original seal: {path}')


def _result_inventory(result):
    """Validate inventory completeness and containment before hashing files."""
    if not result.is_dir() or not (result / 'complete.json').is_file():
        raise ValueError(f'External result is missing or incomplete at {result}; existing outputs were preserved')
    inventory = {path.name for path in result.iterdir()}
    if inventory != RESULT_FILES or any(not (result / name).is_file() for name in inventory):
        raise ValueError(f'External result at {result} is incomplete; existing outputs were preserved')
    if any((result / name).resolve() != result / name for name in inventory):
        raise ValueError(f'External result artifact leaves its managed path: {result}')
    return sorted(inventory)


def _validate_result_marker(marker, seal, plan, seal_sha256):
    """Require prospective seal identity and a valid external assessment."""
    if (
        marker.get('seal_sha256') != seal_sha256
        or Path(marker.get('seal', '')).resolve() != seal
        or marker.get('seeds') != plan['seeds']
        or marker.get('captures') != len(plan['captures'])
        or marker.get('detector_training') is not False
        or marker.get('threshold_selection_on_test') is not False
        or marker.get('status') not in VALID_ASSESSMENTS
    ):
        raise ValueError('External result does not match the prospective Hulk seal')


def _result_bindings(result, seal, plan, seal_sha256):
    """Return verified artifact hashes and the recorded assessment."""
    inventory = _result_inventory(result)
    marker = _read(result / 'complete.json')
    _validate_result_marker(marker, seal, plan, seal_sha256)
    return _bindings(result, inventory), marker


def _validate_output(output, source, manifest):
    if output.is_relative_to(source) or source.is_relative_to(output) or manifest.is_relative_to(output):
        raise ValueError('Follow-up output must be disjoint from the frozen source and external manifest')


def _check_capture_paths(output, plan):
    for capture in plan['captures']:
        paths = [capture['prepared_csv'], *(path for path, _ in capture['raw_files'])]
        if any(Path(path).resolve().is_relative_to(output) for path in paths):
            raise ValueError('Capture inputs must be outside the follow-up output directory')


def _workflow_paths(output):
    paths = {Path(name).stem: output / name for name in CONFIRMATION_PATHS}
    for path in (*paths.values(), output / '.v12_lock'):
        if path.resolve() != path:
            raise ValueError(f'Follow-up artifact is redirected: {path}')
    if output.exists() and not paths['workflow'].exists() and any(p.name != '.v12_lock' for p in output.iterdir()):
        raise ValueError('Follow-up needs a new output directory; existing outputs were preserved')
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _write_workflow(workflow, expected):
    if workflow.exists():
        if _read(workflow) != expected:
            raise ValueError('Follow-up source or configuration changed; existing outputs were preserved')
    elif any(path.name != '.v12_lock' for path in workflow.parent.iterdir()):
        raise ValueError('Follow-up artifacts have no original workflow receipt; existing outputs were preserved')
    else:
        _write(workflow, expected)


def _ensure_seal(paths, heads, manifest, plan):
    seal = paths['seal']
    if seal.exists():
        if _read(seal) != plan:
            raise ValueError('Existing attribution seal differs from frozen models, manifest or analysis code')
    elif paths['result'].exists() or paths['receipt'].exists():
        raise ValueError('Follow-up results have no original seal; existing outputs were preserved')
    elif revision_external.seal_confirmation(heads, manifest, seal) != plan:
        raise ValueError('Attribution inputs changed while creating the seal')


def _confirmed_files(paths, plan, bindings):
    result, seal, receipt = (paths[name] for name in ('result', 'seal', 'receipt'))
    started_here = not result.exists() and not receipt.exists()
    if started_here:
        revision_external.confirm_external(seal, result)
    files, marker = _result_bindings(result, seal, plan, bindings['seal_sha256'])
    if receipt.exists():
        if _read(receipt) != {'schema_version': 1, **bindings, 'files': files}:
            raise ValueError('Completed attribution confirmation receipt or artifacts changed')
    elif not started_here:
        with TemporaryDirectory(prefix='.verify_confirmation_', dir=result.parent) as temporary:
            rescored = Path(temporary) / 'result'
            revision_external.confirm_external(seal, rescored)
            expected_files, _ = _result_bindings(rescored, seal, plan, bindings['seal_sha256'])
        if files != expected_files or _bindings(result, files) != files:
            raise ValueError('Existing attribution result differs from sealed rescoring and was preserved')
    return files, marker


def _check_unchanged(source, manifest, source_files, heads, plan, paths, bindings):
    _verify_capture_hashes(plan)
    if (
        _source_artifacts(source) != source_files
        or _attribution_plan(heads, manifest) != plan
        or _hash(paths['seal']) != bindings['seal_sha256']
        or _hash(paths['workflow']) != bindings['workflow_sha256']
    ):
        raise ValueError('Attribution inputs or receipts changed during confirmation')


def _confirmation_details(marker):
    """Expose the verified result's evidence scope to the single-entry report."""
    return {
        'capture_count': marker['captures'],
        'seed_count': len(marker['seeds']),
        'criteria_met': marker['status'] == 'criteria_met_on_declared_external_receiver',
        'detector_training': marker['detector_training'],
        'threshold_selection_on_test': marker['threshold_selection_on_test'],
        'seed_interval_scope': marker.get('interval_scope'),
        'capture_interval_scope': marker.get('capture_interval_scope'),
        'capture_evidence': marker.get('capture_evidence'),
    }


def _save_confirmation(paths, bindings, files, marker):
    receipt = paths['receipt']
    if not receipt.exists():
        _write(receipt, {'schema_version': 1, **bindings, 'files': files})
    artifacts = {
        'seal.json': bindings['seal_sha256'],
        'workflow.json': bindings['workflow_sha256'],
        'receipt.json': _hash(receipt),
        **{f'result/{name}': digest for name, digest in files.items()},
    }
    return {
        'status': 'completed',
        'reason': '',
        'result_path': str(paths['result']),
        'seal': str(paths['seal']),
        'assessment': marker['status'],
        'scoring': True,
        'capture_contents_read': True,
        'confirmation': _confirmation_details(marker),
        'artifacts': artifacts,
    }


def _run_confirmation(output, source, manifest):
    _validate_output(output, source, manifest)
    source_files = _source_artifacts(source)
    heads = _frozen_hulk_heads(source)
    plan = _attribution_plan(heads, manifest)
    _check_capture_paths(output, plan)
    paths = _workflow_paths(output)
    expected = {
        'schema_version': 1,
        'source': str(source),
        'manifest': str(manifest),
        'source_artifacts': source_files,
        'frozen_version': FROZEN_VERSION,
    }
    with _output_lock(output):
        _write_workflow(paths['workflow'], expected)
        _ensure_seal(paths, heads, manifest, plan)
        bindings = {
            'seal_sha256': _hash(paths['seal']),
            'manifest_sha256': plan['manifest_sha256'],
            'workflow_sha256': _hash(paths['workflow']),
        }
        files, marker = _confirmed_files(paths, plan, bindings)
        _check_unchanged(source, manifest, source_files, heads, plan, paths, bindings)
        return _save_confirmation(paths, bindings, files, marker)


def run_hulk(output: Path, source: Path, manifest: Path) -> dict:
    """Seal and confirm once, then verify the matching artifacts on subsequent runs."""
    output, source, manifest = (Path(path).resolve() for path in (output, source, manifest))
    identity = {
        'output': str(output),
        'source': str(source),
        'required_manifest': str(manifest),
        'frozen_version': FROZEN_VERSION,
    }
    if not manifest.is_file():
        return {
            **identity,
            'status': 'pending',
            'reason': 'Independent capture manifest is missing',
            'artifacts': {},
        }
    try:
        return {**identity, **_run_confirmation(output, source, manifest)}
    except WORKFLOW_ERRORS as error:
        return {**identity, 'status': 'failed', 'reason': str(error), 'artifacts': {}}
