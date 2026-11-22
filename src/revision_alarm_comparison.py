"""Compare locked alarm thresholds on one frozen dual-head TabM per client and seed."""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from .experiment_config import DATASETS, ROOT
from .revision_report import summarize, write_csv
from .revision_run import SEEDS, _source_hashes, grouped_validation_arguments, main as run_matrix
from .revision_v12 import _bindings, _hash, _output_lock, _read, _write


MODES = {
    'fixed_070': ('base_fixed070', 'fixed070_local', 'fixed070_full'),
    'scalar_cal': ('base_threshold', 'fixed_base_local', 'fixed_base_full'),
    'matched_scalar_cal': ('matched_scalar_base', 'matched_scalar_local', 'matched_scalar_full'),
    'class_cal': ('class_cal_base', 'class_cal_local', 'class_cal_full'),
}
METRICS = ('accuracy', 'macro_f1', 'weighted_f1', 'far', 'fnr')
ALARM_LIMIT = 0.01
PRIMARY_METHOD = {'primary_detector': 'binary_swa', 'primary_alarm_policy': 'local_scalar_cal'}


def _case(folder, dataset, seed):
    case = _read(folder / 'result.json')
    if (case['dataset'], case['seed'], case['protocol'], case['scenario']) != (dataset, seed, 'grouped', 'main'):
        raise ValueError(f'Case identity differs at {folder}')
    rows = {row['mode']: row for row in case['evaluations']}
    if ({r['role'] for r in rows.values()} != {'val'} or len(rows) != len(case['evaluations'])
            or case['settings']['test_evaluated']):
        raise ValueError(f'Unexpected role or duplicate mode at {folder}')
    return case, rows


def _support(row):
    return [(r['class'], r['support']) for r in row['per_class']]


def _matched_case(output, original, candidate, baseline, row, old_row, dataset, seed):
    old, new = baseline['settings'], candidate['settings']
    old_cal, new_cal = old['cal_split_audit'], new['cal_split_audit']
    matching = (
        'clients', 'alpha', 'base_epochs', 'batch_size', 'detector_objective', 'rows_per_client', 'train_rows',
        'target_far', 'threshold_floor', 'threshold_min_benign_rows', 'threshold_insufficient_policy',
        'ple_boundary_policy', 'cal_split_protocol', 'cal_fit_fraction',
    )
    if (any(old[k] != new[k] for k in matching)
            or old.get('tabm_attention') or old.get('tabm_binary_head') or old.get('portable_detector') not in (None, 'reference')
            or any(old.get(k) for k in ('tabm_train_retrieval', 'tabm_tree_score_fusion', 'tabm_can_id_robust'))
            or old_cal['assignment_sha256'] != new_cal['assignment_sha256']
            or baseline['partition_sha256'] != candidate['partition_sha256']
            or baseline['supports'] != candidate['supports'] or _support(row) != _support(old_row)):
        raise ValueError(f'Incompatible original TabM reference for {dataset} seed {seed}')
    for before, after in zip(old_cal['clients'], new_cal['clients'], strict=True):
        for side in ('fit', 'holdout'):
            if before[side + '_source_indices_sha256'] != after[side + '_source_indices_sha256']:
                raise ValueError(f'Unmatched CAL {side} rows for {dataset} seed {seed}')
    suffix = Path(dataset) / 'grouped'
    audit = _read(output / 'experiment' / suffix / 'data_audit.json')
    if audit != _read(original / suffix / 'data_audit.json'):
        raise ValueError(f'Dataset audit differs for {dataset}')
    identities = audit.get('source_identity_sha256_by_role', {})
    if not audit.get('synthetic_only') and (not identities.get('val') or audit['role_rows']['val'] != row['metrics']['rows']):
        raise ValueError(f'Unverified VAL rows for {dataset}')
    suffix = suffix / 'main' / f'seed_{seed}' / 'result.json'
    return {'dataset': dataset, 'seed': seed, 'reference_result_sha256': _hash(original / suffix),
            'candidate_result_sha256': _hash(output / 'experiment' / suffix),
            'partition_sha256': candidate['partition_sha256'], 'cal_split_sha256': new_cal['assignment_sha256'],
            'val_source_identity_sha256': identities.get('val'), 'val_rows': row['metrics']['rows'],
            'target_far': old['target_far'], 'threshold_floor': old['threshold_floor'],
            'threshold_insufficient_policy': old['threshold_insufficient_policy']}


def _measurement(dataset, seed, detector, gate, stage, row, certificate):
    confusion, metrics = row['matrix'], row['metrics']
    return {
        'dataset': dataset, 'seed': seed, 'detector': detector, 'alarm_policy': gate, 'subtype_stage': stage,
        'class_cal_status': certificate.get('status', '') if gate == 'class_cal' else '',
        'val_rows': metrics['rows'], 'observed_classes': metrics['observed_classes'], 'defined_classes': metrics['defined_classes'],
        'false_alarms': sum(confusion[0][1:]), 'missed_attacks': sum(values[0] for values in confusion[1:]),
        'wrong_attack_subtypes': sum(sum(values[1:]) - values[i] for i, values in enumerate(confusion) if i),
        **{key: metrics[key] for key in METRICS},
    }


def _summary(dataset, branch, observations):
    detector, gate, stage = branch
    row = {'dataset': dataset, 'detector': detector, 'alarm_policy': gate, 'subtype_stage': stage,
           'seeds': len(observations)}
    for metric in METRICS:
        rates = [r[metric] for r in observations.values() if r[metric] is not None]
        stats = summarize(rates)
        row[f'{metric}_mean_pct'] = 100 * stats['mean'] if rates else None
        row[f'{metric}_sd_pp'] = 100 * stats['seed_sd'] if len(rates) > 1 else None
        if metric in ('far', 'fnr'):
            row[f'{metric}_max_pct'] = 100 * max(rates) if rates else None
            row[f'{metric}_seeds_above_1pct'] = sum(rate > ALARM_LIMIT for rate in rates)
    return row


def _paired(dataset, candidate, baseline, branches):
    a, b = branches[candidate], branches[baseline]
    if set(a) != set(b):
        raise ValueError(f'Unmatched comparison seeds for {dataset}')
    for metric in METRICS:
        values = [a[s][metric] - b[s][metric] for s in a if a[s][metric] is not None and b[s][metric] is not None]
        stats = summarize(values, bounds=(-1, 1))
        yield {'dataset': dataset, 'candidate': '/'.join(candidate), 'reference': '/'.join(baseline),
               'metric': metric, 'seeds': stats['seeds_with_metric'],
               **{f'{k}_pp': 100 * v if v is not None else None for k, v in stats.items() if k != 'seeds_with_metric'}}


def _summarize(root, datasets, seeds, smoke, original, sensitivity, *, primary):
    matrix = root / 'experiment'
    expected = {'replicates': len(datasets) * len(seeds), 'data_audits': len(datasets)}
    if _read(matrix / 'complete.json') != expected:
        raise ValueError('The dual-head study is missing planned cases')
    policies = MODES if sensitivity else {k: MODES[k] for k in ('fixed_070', 'scalar_cal')}
    by_seed, overview, contrasts, locks, matched, classes = [], [], [], [], [], []
    files = ['experiment/run.json', 'experiment/complete.json', 'stage_plan.json']
    if primary:
        files.extend(
            f'experiment/{name}.csv'
            for name in ('calibration', 'communication', 'confusion_counts', 'resources', 'timings')
        )
    for dataset in datasets:
        base = Path('experiment') / dataset / 'grouped'
        files.append((base / 'data_audit.json').as_posix())
        branches = {}
        for seed in seeds:
            case_root = base / 'main' / f'seed_{seed}'
            folder = root / case_root
            case, rows = _case(folder, dataset, seed)
            if case['settings']['portable_detector'] != 'binary_swa' or case['settings']['train_sampling'] != 'all_rows':
                raise ValueError(f'Unexpected detector or training role at {folder}')
            files.extend((case_root / filename).as_posix() for filename in ('result.json', 'cal_group_split.json'))
            cert = {}
            if sensitivity:
                cert_path = case_root / 'class_calibration.json'
                cert = _read(root / cert_path)
                digest = _hash(root / cert_path)
                if case['portable_class_cal']['sha256'] != digest:
                    raise ValueError(f'Changed threshold lock at {folder}')
                files.append(cert_path.as_posix())
                locks.append({'dataset': dataset, 'seed': seed, 'calibration_sha256': digest, 'selection_status': cert['status']})
            measurements = {('binary_swa', 'native_argmax', 'base'): rows['base']}
            for gate, names in policies.items():
                for name in names:
                    if (_support(rows[name]) != _support(rows[names[0]]) or
                            any(rows[name]['metrics'][m] != rows[names[0]]['metrics'][m] for m in ('far','fnr'))):
                        raise ValueError(f'Frozen alarm or class support differs at {folder}/{name}')
                for stage, name in zip(('base', 'local', 'federated'), names, strict=True):
                    measurements['binary_swa', gate, stage] = rows[name]
            if original is not None:
                ref, old_rows = _case(original / dataset / 'grouped' / 'main' / f'seed_{seed}', dataset, seed)
                matched.append(_matched_case(root, original, case, ref, rows['base_threshold'], old_rows['base_threshold'], dataset, seed))
                measurements['original', 'scalar_cal', 'base'] = old_rows['base_threshold']
                measurements['original', 'native_argmax', 'base'] = old_rows['base']
            for branch, row in measurements.items():
                observation = _measurement(dataset, seed, *branch, row, cert)
                by_seed.append(observation)
                branches.setdefault(branch, {})[seed] = observation
                for cl in row['per_class']:
                    classes.append({'dataset': dataset, 'seed': seed, 'detector': branch[0], 'alarm_policy': branch[1],
                                    'subtype_stage': branch[2], **cl})
        overview.extend(_summary(dataset, branch, observations) for branch, observations in branches.items())
        if original is not None:
            for gate in ('scalar_cal', 'native_argmax'):
                contrasts.extend(_paired(dataset, ('binary_swa', gate, 'base'), ('original', gate, 'base'), branches))
        base_gate = 'matched_scalar_cal' if sensitivity else 'scalar_cal'
        for gate in policies:
            if gate != base_gate:
                contrasts.extend(_paired(dataset, ('binary_swa', gate, 'federated'), ('binary_swa', base_gate, 'federated'), branches))
            contrasts.extend(_paired(dataset, ('binary_swa', gate, 'federated'), ('binary_swa', gate, 'local'), branches))
            contrasts.extend(_paired(dataset, ('binary_swa', gate, 'local'), ('binary_swa', gate, 'base'), branches))
    tables = {'metrics_by_seed.csv': by_seed, 'metrics_summary.csv': overview, 'paired_differences.csv': contrasts,
              'class_metrics_by_seed.csv': classes}
    if sensitivity:
        tables['calibration_locks.csv'] = locks
    if original is not None:
        tables['matched_reference.csv'] = matched
    for filename, records in tables.items():
        write_csv(root / filename, records)
    receipt = {
        'status': 'completed', 'datasets': list(datasets), 'seeds': list(seeds), 'smoke': smoke,
        'output': str(root),
        'scientific_status': ('primary_grouped_development_dual_head_tabm' if primary
                              else 'supplementary_grouped_development_alarm_comparison'),
        'execution_origin': 'current_fresh_training', 'reference_paired': original is not None,
        'threshold_sensitivity': sensitivity, 'role': 'grouped VAL', 'test_scored': False, 'test_evaluated': False,
        'trained_cases': expected['replicates'], 'artifacts': _bindings(root, [*tables, *files]),
        **(PRIMARY_METHOD if primary else {}),
    }
    _write(root / 'complete.json', receipt)
    return {**receipt, 'artifacts': {**receipt['artifacts'], 'complete.json': _hash(root / 'complete.json')}}


def _reference_bindings(folder, datasets, seeds, smoke):
    """Bind the original grouped benchmark's actual cases and source-row audits."""
    config = _read(folder / 'run.json')
    if (config['synthetic_only'] != smoke or config['seeds'] != list(seeds)
            or config.get('tabm_attention') or config.get('tabm_refinement')
            or config.get('tree_alarm_count') or config.get('portable_detector') not in (None, 'reference')):
        raise ValueError('The reference must be the original grouped TabM experiment with matching seeds')
    expected = sum(len(entry['scenarios']) * len(seeds) for entry in config['matrix'])
    if _read(folder / 'complete.json') != {'replicates': expected, 'data_audits': len(config['matrix'])}:
        raise ValueError('Reference benchmark has no matching completion receipt')
    names = ['run.json', 'complete.json']
    for name in datasets:
        base = f'{name}/grouped'
        names.append(f'{base}/data_audit.json')
        for seed in seeds:
            names.extend(f'{base}/main/seed_{seed}/{f}' for f in ('result.json', 'cal_group_split.json'))
    return _bindings(folder, names)


def run_alarm_comparison(output, project, *, device='auto', smoke=False, datasets=DATASETS, reference=None,
                         threshold_sensitivity=False, primary=False):
    """Run one dual-head model per seed, reusing its frozen scores for all prespecified decisions."""
    root, data_root = Path(output).resolve(), Path(project).resolve() / 'dataset'
    datasets, seeds = list(datasets), [SEEDS[0]] if smoke else list(SEEDS)
    if not datasets or len(datasets) != len(set(datasets)) or set(datasets) - set(DATASETS):
        raise ValueError('Select one or more distinct registered datasets')
    if device not in ('auto', 'cpu', 'cuda') or (smoke and device == 'cuda'):
        raise ValueError('Select an available device. Synthetic checks use CPU.')
    if root.is_relative_to(data_root) or data_root.is_relative_to(root):
        raise ValueError('The experiment output must be separate from the local datasets')
    original = Path(reference).resolve() if reference is not None else None
    if primary and (original is None or threshold_sensitivity or set(datasets) != set(DATASETS)):
        raise ValueError('The primary experiment requires all four datasets and the matched original reference')
    if original and (original.is_relative_to(root) or root.is_relative_to(original)):
        raise ValueError('Detector candidate and original reference require separate directories')
    device = 'cpu' if smoke else device
    options = grouped_validation_arguments(root / 'experiment', data_root, device, datasets, seeds, smoke=smoke)
    options.extend(('--portable-detector', 'binary_swa'))
    if threshold_sensitivity:
        options.append('--portable-class-cal')
    if original or not threshold_sensitivity:
        options.extend(['--threshold-insufficient-policy', 'disable'])
    source = Path(__file__).resolve().parent
    extra = ['revision_alarm_comparison.py', 'revision_v12.py']
    if threshold_sensitivity:
        extra.extend(('revision_portable_class_cal.py', 'revision_class_operating.py', 'revision_operating_selection.py'))
    root.mkdir(parents=True, exist_ok=True)
    with _output_lock(root, filename='.class_cal_lock'):
        sources = {**_source_hashes(), **_bindings(source, extra)}
        ref_hashes = _reference_bindings(original, datasets, seeds, smoke) if original else {}
        plan = {'schema': 'portable_alarm_comparison_v1', 'dataset_root': str(data_root), 'datasets': datasets,
                'device': device, 'seeds': seeds, 'smoke': smoke, 'reference': str(original) if original else None,
                'threshold_sensitivity': threshold_sensitivity, 'source_sha256': sources,
                'reference_sha256': ref_hashes, 'candidate_detector': 'binary_swa',
                'reference_detector': 'original' if original else None, 'comparison_policy': 'scalar_cal',
                **(PRIMARY_METHOD if primary else {})}
        plan_path = root / 'stage_plan.json'
        if plan_path.exists() and _read(plan_path) != plan:
            raise ValueError('Stage inputs or sources changed. Use a new output directory.')
        new_stage = not plan_path.exists()
        if new_stage:
            _write(plan_path, plan)
        completed = root / 'complete.json'
        if completed.is_file():
            saved = _read(completed)
            if _bindings(root, saved['artifacts'].keys()) != saved['artifacts']:
                raise ValueError('Completed alarm comparison artifacts changed')
            if not new_stage and 'reference_paired' in saved:
                return {**saved, 'artifacts': {**saved['artifacts'], 'complete.json': _hash(completed)}}
        if (root / 'experiment' / 'run.json').is_file():
            options.append('--resume')
        if run_matrix(options) != 0:
            raise RuntimeError('The grouped dual-head alarm comparison did not complete')
        if _bindings(source, sources.keys()) != sources or (original and _bindings(original, ref_hashes.keys()) != ref_hashes):
            raise RuntimeError('Source files or original benchmark artifacts changed during comparison')
        return _summarize(root, datasets, seeds, smoke, original, threshold_sensitivity, primary=primary)




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', choices=DATASETS, nargs='+', default=DATASETS)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--out', type=Path, help='New directory or an interrupted run to resume')
    parser.add_argument('--smoke', action='store_true', help='Use one synthetic seed on the CPU')
    args = parser.parse_args(argv)
    if len(set(args.dataset)) != len(args.dataset):
        parser.error('Supply each dataset once')
    if args.smoke and args.device == 'cuda':
        parser.error('The synthetic smoke runs on the CPU')
    if args.out:
        root = args.out.resolve()
        data = (ROOT / 'dataset').resolve()
        if root.is_relative_to(data) or data.is_relative_to(root):
            parser.error('Use an output directory separate from the datasets')
        root.mkdir(parents=True, exist_ok=True)
    else:
        parent = ROOT / 'outputs' / 'fedcaid_class_cal'
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(mkdtemp(prefix=f'run_{datetime.now():%Y%m%d_%H%M%S}_', dir=parent))
    run_alarm_comparison(root, ROOT, device=args.device, smoke=args.smoke, datasets=args.dataset, threshold_sensitivity=True)
    print(f'FedCAID | threshold experiment complete | {root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
