"""Matched NB15 subtype objectives with immutable TabM alarms and CAL guards.

Only new small heads are trained. The original multiclass head probabilities,
TabM probabilities, and FedAvg probabilities are bound inputs, not new results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from . import revision_next as native
from .revision_nb15_source import _bindings, _cases, _check_bindings, _source
from .revision_report import score, summarize, write_csv
from .revision_score_cache import load_score_cache
from .revision_shared_head import fit_shared_heads
from .revision_v12 import _hash, _output_lock, _read, _write

OBJECTIVES = ('multiclass', 'attack_only')
HEADS = ('head_local', 'head_donor_single', 'head_donor', 'head_federated')
METHODS = (*HEADS, 'fedavg_threshold')
RULES = ('legacy', 'source_protected')
SEEDS = native.SEEDS
MINIMUM_SEEDS = 5
TABLES = (
    'metrics',
    'class_metrics',
    'target_metrics',
    'subtype_audit',
    'subtype_transitions',
    'alerted_target_ranking',
    'guard_audit',
    'training',
    'communication',
)


def _selected_device(requested, original):
    selected = original if requested == 'auto' else requested
    if selected not in ('cpu', 'cuda'):
        raise ValueError('Select cpu, cuda or auto')
    if selected != original:
        raise ValueError('The new reference single-donor head must use the original head training device')
    if selected == 'cuda' and not torch.cuda.is_available():
        raise ValueError('The source head used CUDA; its matched targeted run requires CUDA')
    return selected


def _source_case_paths(source):
    return {int(row['seed']): Path(row['result']).resolve() for row in _cases(source).values()}


def _load_context(source, source_run, device):
    """Rebuild and verify the original grouped development routes."""
    settings = dict(source_run['settings'])
    settings['data_root'] = Path(settings['data_root'])
    cli = argparse.Namespace(**settings)
    cli.device = device
    data = native.load_revision_data('CIC-UNSW-NB15', cli.data_root, split_protocol='grouped')
    if data.audit != _read(source / 'CIC-UNSW-NB15' / 'data_audit.json'):
        raise ValueError('Targeted run data differs from its frozen source audit')
    return data, cli


def _case_context(data, cli, record, directory, source_run):
    identity = record['identity']
    target = data.names.index(identity['target_class'])
    parts, roster, policies, audit = native._design(native._scope_data(data), data.names, identity, target, None, cli)
    if native._comparable_score_route(audit) != native._comparable_score_route(record['audit']):
        raise ValueError('Reconstructed TRAIN/CAL/VAL route differs from cached case')
    if set(policies) != {'reserved'} or policies['reserved']['status'] != 'eligible':
        raise ValueError('Expected the original eligible reserved CAL policy')
    policy = policies['reserved']
    union = [{**binding, 'cal': np.unique(policy['roster'][i]['cal'])} for i, binding in enumerate(roster)]
    probabilities = load_score_cache(
        directory,
        record['score_cache'],
        identity,
        native._score_route(record['audit']),
        native._score_configuration(source_run),
        union,
        expected_modes={'head_local', 'head_donor', 'head_federated', 'base_threshold', 'fedavg_threshold'},
        expected_classes=data.names,
    )
    receiver = audit['receiver']
    if np.any(parts['train'][1][roster[receiver]['train']] == target) or np.any(
        parts['cal'][1][policy['roster'][receiver]['cal']] == target
    ):
        raise ValueError('The receiver target must be absent from TRAIN and entire CAL')
    donors = []
    for client, binding in enumerate(roster):
        if client == receiver:
            continue
        selected = binding['train']
        positive = selected[parts['train'][1][selected] == target]
        donors.append(
            {
                'client': client,
                'target_rows': len(positive),
                'target_groups': native._unique_rows(parts['train'][0], positive),
            }
        )
    chosen = min(donors, key=lambda item: (-item['target_groups'], item['client']))
    if chosen['target_groups'] < cli.min_donor_groups:
        raise ValueError('No eligible TRAIN-selected single donor')
    thresholds = {}
    for row in record['tables']['gates']:
        if row['mode'] == 'base_threshold' and row['cal_policy'] == 'reserved':
            if row['client'] in thresholds:
                raise ValueError('Source repeats a common-gate threshold')
            thresholds[row['client']] = float(row['threshold'])
    if set(thresholds) != set(range(len(roster))):
        raise ValueError('Source does not bind every common-gate threshold')
    return {
        'identity': identity,
        'parts': parts,
        'roster': roster,
        'policy': policy,
        'union': union,
        'audit': audit,
        'probabilities': probabilities,
        'receiver': receiver,
        'target': target,
        'thresholds': [thresholds[i] for i in range(len(roster))],
        'donor': chosen,
        'donor_support': donors,
    }


def _legacy_guard(base, probabilities, labels, target, groups, seed, cli):
    from .revision_subtype_guard import apply_cal_check, fit_subtype_guard

    count = int(groups.max()) + 1 if len(groups) else 0
    order = np.random.default_rng(seed).permutation(count)
    check = np.isin(groups, order[: count // 3])
    fitted = fit_subtype_guard(
        base[~check],
        probabilities[~check],
        labels[~check],
        (target,),
        max_rewrite_rate=cli.guard_rewrite_rate,
        min_negative_groups=cli.guard_min_negative_groups,
        group_ids=groups[~check],
    )
    guard, audit = apply_cal_check(
        base[check],
        probabilities[check],
        labels[check],
        (target,),
        fitted,
        min_check_groups=cli.guard_min_check_groups,
        check_tolerance=cli.guard_check_tolerance,
        group_ids=groups[check],
    )
    return guard, {
        'fitted': fitted,
        'check': audit,
        'fit_rows_sha256': sha256(np.flatnonzero(~check).astype('<i8').tobytes()).hexdigest(),
        'check_rows_sha256': sha256(np.flatnonzero(check).astype('<i8').tobytes()).hexdigest(),
    }


def _evaluate(context, data, cli, probabilities, objective):
    from sklearn.metrics import average_precision_score

    from .revision_controlled import development_cells
    from .revision_data import _feature_keys
    from .revision_fixed_subtype import subtype_transitions
    from .revision_source_guard import fit_source_guard, predict_source_guard
    from .revision_subtype_guard import guarded_subtype_predictions, target_attack_margin
    from .utils import hierarchical_decisions

    identity, parts, roster, union = (context[key] for key in ('identity', 'parts', 'roster', 'union'))
    receiver, target = context['receiver'], context['target']
    metadata = {**identity, 'cal_policy': 'reserved', 'objective': objective, 'role': 'val'}
    tables = {name: [] for name in TABLES}
    base_all = [
        hierarchical_decisions(p, threshold)
        for p, threshold in zip(context['probabilities']['base_threshold'], context['thresholds'], strict=True)
    ]
    base_predictions = [entry['predicted'][len(route['cal']) :] for entry, route in zip(base_all, union, strict=True)]
    attack_scores = [entry['attack_score'][len(route['cal']) :] for entry, route in zip(base_all, union, strict=True)]
    all_y = np.concatenate([parts['val'][1][binding['val']] for binding in roster])
    r = context['policy']['roster'][receiver]
    fit = context['policy']['splits'][receiver][0]
    cal_rows = r['cal'][fit]
    positions = np.searchsorted(union[receiver]['cal'], cal_rows)
    if not np.array_equal(union[receiver]['cal'][positions], cal_rows):
        raise ValueError('Receiver CAL guard rows are not aligned with cached scores')
    _, groups = np.unique(_feature_keys(parts['cal'][0][cal_rows]), return_inverse=True)
    groups = groups.reshape(-1)
    labels = parts['cal'][1][cal_rows]
    cal_base = base_all[receiver]['predicted'][positions]
    val_labels = parts['val'][1][roster[receiver]['val']]
    reference_alarm = base_predictions[receiver] > 0
    guards = {}

    def measure(rule, method, predictions, source_scores=None):
        meta = {**metadata, 'rule': rule, 'method': method}
        whole, classes, _ = score(all_y, np.concatenate(predictions), np.concatenate(attack_scores), data.names)
        tables['metrics'].append({**meta, **whole})
        tables['class_metrics'].extend({**meta, **row} for row in classes)
        target_row = development_cells(parts, roster, {method: predictions}, target, receiver, data.names)[0]
        receiver_attacks = val_labels > 0
        target_row.update(
            receiver_attack_support=int(receiver_attacks.sum()),
            receiver_missed_attacks=int(np.count_nonzero(receiver_attacks & (predictions[receiver] == 0))),
            receiver_fnr=(
                float(np.mean(predictions[receiver][receiver_attacks] == 0)) if receiver_attacks.any() else None
            ),
        )
        tables['target_metrics'].append({**meta, **target_row})
        for client, binding in enumerate(roster):
            if not np.array_equal(predictions[client] > 0, base_predictions[client] > 0):
                raise RuntimeError('A subtype policy changed the fixed binary alarms')
            eligible = (target,) if client == receiver else ()
            stats, transitions = subtype_transitions(
                parts['val'][1][binding['val']],
                base_predictions[client],
                predictions[client],
                data.names,
                eligible,
                subtype_scores=(
                    source_scores[client][len(union[client]['cal']) :] if source_scores is not None else None
                ),
            )
            audit = {
                **meta,
                'client': client,
                'base_alarm_mask_sha256': sha256((base_predictions[client] > 0).tobytes()).hexdigest(),
            }
            tables['subtype_audit'].append({**audit, **stats})
            tables['subtype_transitions'].extend({**audit, **row} for row in transitions)

    measure('base', 'base_threshold', base_predictions)
    for method in METHODS:
        source = probabilities[method]
        cal_scores = source[receiver][positions]
        val_scores = source[receiver][len(union[receiver]['cal']) :]
        legacy, audit = _legacy_guard(
            cal_base, cal_scores, labels, target, groups, identity['seed'] + 8123 + receiver, cli
        )
        protected = fit_source_guard(
            cal_base,
            cal_scores,
            labels,
            (target,),
            group_ids=groups,
            seed=identity['seed'] + 8123 + receiver,
            max_rewrite_rate=cli.guard_rewrite_rate,
            min_negative_groups=cli.guard_min_negative_groups,
            min_check_groups=cli.guard_min_check_groups,
            check_tolerance=cli.guard_check_tolerance,
        )
        guards[method] = {'legacy': legacy, 'legacy_audit': audit, 'source_protected': protected}
        for rule, receipt in (('legacy', legacy), ('source_protected', protected)):
            predictions = list(base_predictions)
            function = guarded_subtype_predictions if rule == 'legacy' else predict_source_guard
            predictions[receiver] = function(base_predictions[receiver], val_scores, (target,), receipt)
            measure(rule, method, predictions, source)
            tables['guard_audit'].append(
                {
                    **metadata,
                    'method': method,
                    'rule': rule,
                    'client': receiver,
                    'status': receipt.get('status', 'per_source_rules'),
                    'guard_receipt': json.dumps(receipt, sort_keys=True),
                    'cal_rows': len(cal_rows),
                    'cal_feature_groups': int(groups.max()) + 1,
                    'receiver_target_cal_rows': int(np.count_nonzero(labels == target)),
                }
            )
        truth = val_labels[reference_alarm] == target
        margin = target_attack_margin(val_scores[reference_alarm], target)
        tables['alerted_target_ranking'].append(
            {
                **metadata,
                'method': method,
                'alert_rows': int(reference_alarm.sum()),
                'target_support': int(truth.sum()),
                'target_margin_ap': float(average_precision_score(truth, margin)) if truth.any() else None,
                'target_probability_ap': (
                    float(average_precision_score(truth, val_scores[reference_alarm, target])) if truth.any() else None
                ),
            }
        )
    return tables, guards


def _verify_original(tables, original):
    """The cached multiclass scalar reference must reproduce its bound metrics."""
    aliases = {
        'head_local': 'guarded_local',
        'head_donor': 'guarded_donor',
        'head_federated': 'guarded_federated',
        'fedavg_threshold': 'guarded_fedavg_subtype',
        'base_threshold': 'base_threshold',
    }
    for name, fields in (
        ('metrics', ('macro_f1', 'far', 'fnr', 'ece')),
        ('target_metrics', ('precision', 'recall', 'f1', 'receiver_far')),
    ):
        before = {r['mode']: r for r in original['tables'][name]}
        for row in tables[name]:
            if row['rule'] not in ('legacy', 'base') or row['method'] not in aliases:
                continue
            source = before[aliases[row['method']]]
            for field in fields:
                if not np.isclose(row[field], source[field], atol=1e-12, rtol=0):
                    raise ValueError(f'Original fixed-alarm scalar reference was not reproduced: {name}/{field}')


def _train_case(context, data, cli, objective, destination):
    identity, parts, roster = (context[key] for key in ('identity', 'parts', 'roster'))
    args = native._args(identity, argparse.Namespace(**{**vars(cli), 'head_objective': objective}))
    train = [(parts['train'][0][r['train']], parts['train'][1][r['train']]) for r in roster]
    modes = ('head_donor_single',) if objective == 'multiclass' else HEADS
    fit = fit_shared_heads(
        train,
        len(data.names),
        args,
        receiver=context['receiver'],
        modes=modes,
        donor_subsets={'head_donor_single': (context['donor']['client'],)},
    )
    fresh, inference = native._head_outputs(fit['states'], parts, context['union'], args)
    probabilities = {**context['probabilities'], **fresh}
    tables, guards = _evaluate(context, data, cli, probabilities, objective)
    tables['training'].extend(
        {**identity, 'objective': objective, 'measurement_source': 'new_head_training', **r}
        for r in [*fit['performance'], *inference]
    )
    tables['communication'].extend(
        {**identity, 'objective': objective, 'measurement_source': 'new_head_training', **r}
        for r in fit['communication']
    )
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / 'trained_heads.partial.npz'
    np.savez_compressed(
        partial,
        **{
            f'{mode}_c{client}': weights
            for mode, states in fit['states'].items()
            for client, weights in enumerate(states)
        },
    )
    partial.replace(destination / 'trained_heads.npz')
    return {
        'tables': tables,
        'guards': guards,
        'model_spec': fit['model_spec'],
        'trained_modes': list(modes),
        'reused_head_modes': [m for m in HEADS if m not in modes],
        'trained_heads_sha256': _hash(destination / 'trained_heads.npz'),
    }


def _comparisons(tables, seeds):
    records = {}
    for table, scope, fields in (
        ('metrics', 'overall', ('macro_f1', 'far', 'fnr', 'ece')),
        ('target_metrics', 'receiver', ('precision', 'recall', 'f1', 'receiver_far', 'receiver_fnr')),
    ):
        for row in tables[table]:
            for metric in fields:
                key = (row['seed'], row['objective'], row['rule'], row['method'], scope, metric)
                if key in records:
                    raise ValueError('Duplicate targeted metric')
                records[key] = row[metric]
    pairs = []
    contrasts = [
        (
            'objective_effect_scalar',
            ('attack_only', 'legacy', 'head_federated'),
            ('multiclass', 'legacy', 'head_federated'),
        ),
        (
            'objective_effect_source_protected',
            ('attack_only', 'source_protected', 'head_federated'),
            ('multiclass', 'source_protected', 'head_federated'),
        ),
        (
            'combined_candidate',
            ('attack_only', 'source_protected', 'head_federated'),
            ('multiclass', 'legacy', 'head_federated'),
        ),
    ]
    for objective in OBJECTIVES:
        contrasts.append(
            (
                f'guard_effect_{objective}',
                (objective, 'source_protected', 'head_federated'),
                (objective, 'legacy', 'head_federated'),
            )
        )
        for rule in RULES:
            contrasts.append(
                (
                    f'{objective}_{rule}_full_minus_base',
                    (objective, rule, 'head_federated'),
                    (objective, 'base', 'base_threshold'),
                )
            )
            for label, candidate, reference in (
                ('full_minus_local', 'head_federated', 'head_local'),
                ('pool_minus_single', 'head_donor', 'head_donor_single'),
                ('full_minus_pool', 'head_federated', 'head_donor'),
                ('full_minus_fedavg_subtype', 'head_federated', 'fedavg_threshold'),
            ):
                contrasts.append(
                    (
                        f'{objective}_{rule}_{label}',
                        (objective, rule, candidate),
                        (objective, rule, reference),
                    )
                )
    groups = defaultdict(list)
    for label, candidate, reference in contrasts:
        for seed in seeds:
            for scope, metrics in (
                ('overall', ('macro_f1', 'far', 'fnr', 'ece')),
                ('receiver', ('precision', 'recall', 'f1', 'receiver_far', 'receiver_fnr')),
            ):
                for metric in metrics:
                    a, b = (records[seed, *branch, scope, metric] for branch in (candidate, reference))
                    delta = a - b if a is not None and b is not None else None
                    row = {
                        'contrast': label,
                        'seed': seed,
                        'scope': scope,
                        'metric': metric,
                        'candidate': a,
                        'reference': b,
                        'delta': delta,
                    }
                    pairs.append(row)
                    groups[label, scope, metric].append(delta)
    intervals = [
        {
            'contrast': key[0],
            'scope': key[1],
            'metric': key[2],
            **summarize(values, bounds=(-1, 1)),
            'interval_scope': f'unadjusted paired t95 across {len(seeds)} seeds sharing captures',
        }
        for key, values in groups.items()
    ]
    return pairs, intervals


def _development_criteria(tables, intervals, seeds):
    """Prespecified development checks; acceptance is separate from run completion."""
    selected = lambda row: (row['objective'], row['rule'], row['method']) == (
        'attack_only',
        'source_protected',
        'head_federated',
    )
    overall = sorted((row for row in tables['metrics'] if selected(row)), key=lambda row: row['seed'])
    receiver = sorted((row for row in tables['target_metrics'] if selected(row)), key=lambda row: row['seed'])
    primary = {(row['scope'], row['metric']): row for row in intervals if row['contrast'] == 'combined_candidate'}
    complete = len(seeds) >= MINIMUM_SEEDS and [row['seed'] for row in overall] == list(seeds) == [
        row['seed'] for row in receiver
    ]
    checks = [{'check': 'complete_five_seed_population', 'passed': complete}]
    for scope, rows, metric in (('overall', overall, 'far'), ('receiver', receiver, 'receiver_far')):
        observations = [
            {
                'seed': row['seed'],
                'value': row[metric],
                'passed': row[metric] is not None and row[metric] <= 0.01 + 1e-15,
            }
            for row in rows
        ]
        checks.append(
            {
                'check': f'{scope}_far_at_most_1pct_every_seed',
                'passed': complete and all(row['passed'] for row in observations),
                'limit': 0.01,
                'observations': observations,
            }
        )
    for scope, metric in (('overall', 'fnr'), ('receiver', 'receiver_fnr')):
        interval = primary[scope, metric]
        checks.append(
            {
                'check': f'{scope}_paired_fnr_upper95_at_most_1pp',
                'passed': (
                    complete
                    and interval['seeds_with_metric'] == len(seeds)
                    and interval['ci95_high'] is not None
                    and interval['ci95_high'] <= 0.01 + 1e-15
                ),
                'limit': 0.01,
                'interval': interval,
            }
        )
    for scope, metric in (('overall', 'macro_f1'), ('receiver', 'f1')):
        interval = primary[scope, metric]
        checks.append(
            {
                'check': f'{scope}_{metric}_paired_mean_nondecreasing',
                'passed': (
                    complete
                    and interval['seeds_with_metric'] == len(seeds)
                    and interval['mean'] is not None
                    and interval['mean'] >= -1e-15
                ),
                'limit': 0.0,
                'interval': interval,
            }
        )
    support = [
        {
            key: row.get(key)
            for key in (
                'seed',
                'support',
                'predicted',
                'true_positive',
                'precision',
                'recall',
                'f1',
                'receiver_benign_support',
                'receiver_attack_support',
            )
        }
        for row in receiver
    ]
    checks.append(
        {
            'check': 'receiver_target_support_every_seed',
            'passed': complete and all((row['support'] or 0) > 0 for row in support),
        }
    )
    failed = [row['check'] for row in checks if not row['passed']]
    return {
        'development_criteria_met': not failed,
        'failed_checks': failed,
        'checks': checks,
        'receiver_target_support_and_zero_results': support,
        'scope': (
            'development only; meeting criteria does not establish independent confirmation or significant'
            ' federated added value'
        ),
    }


@dataclass(frozen=True)
class _TargetedRun:
    """Separate the score producer provenance from the code fitting new heads."""

    output: Path
    source: Path
    source_run: dict
    cases: dict
    plan: dict


def _execution_code():
    """Bind every active local implementation used to reconstruct and fit heads."""
    names = ('revision_targeted_subtypes.py', 'revision_source_guard.py', 'revision_nb15_source.py')
    return {
        **native._source_receipt(),
        **{name: _hash(Path(__file__).with_name(name)) for name in names},
    }


def _separate_output(output, *inputs):
    """Keep generated evidence disjoint from source scores and raw files."""
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in inputs):
        raise ValueError('Targeted output must be disjoint from its score and dataset inputs')


def _pair_for_output(output, project, pair_source):
    """Resolve explicit source data only within this invocation's run directory."""
    if pair_source is None:
        return project / 'outputs' / 'v12_nb15_pair'
    pair = Path(pair_source).resolve(strict=True)
    if pair.parent != output.parent or pair == output:
        raise ValueError('Fresh NB15 score inputs must be a separate sibling inside this run')
    return pair


def _prepare_run(output, project, device, pair_source=None):
    """Check the original source receipts and select the matched execution setup."""
    output, project = Path(output).resolve(), Path(project).resolve(strict=True)
    pair = _pair_for_output(output, project, pair_source)
    _separate_output(output, pair, project / 'outputs' / 'v12_final_evidence')
    source_receipt, _, source_runs = _source(pair)
    source, source_run = pair / 'reference', source_runs['reference']
    if pair_source is not None and source_run['code_sha256'] != native._source_receipt():
        raise ValueError('Fresh NB15 scores must come from this run\'s current source code')
    _separate_output(output, Path(source_run['settings']['data_root']).resolve())
    cases = _source_case_paths(source)
    if len(cases) < MINIMUM_SEEDS:
        raise ValueError(f'Expected at least {MINIMUM_SEEDS} matched targeted seeds')
    selected_device = _selected_device(device, source_run['settings']['device'])
    plan = _targeted_plan(source, source_run, source_receipt, selected_device)
    if pair_source is not None:
        plan['score_source_scope'] = 'current_run_only'
    return _TargetedRun(output, source, source_run, cases, plan)


def _targeted_plan(source, source_run, source_receipt, device):
    """Pin both code lineages and the development-only comparison protocol."""
    return {
        'schema_version': 2,
        'source': source_receipt,
        'selected_source': str(source),
        'original_score_producer_sha256': source_run['code_sha256'],
        'current_reconstruction_and_fitting_sha256': _execution_code(),
        'objectives': list(OBJECTIVES),
        'rules': list(RULES),
        'methods': list(METHODS),
        'device': device,
        'torch_version': str(torch.__version__),
        'numpy_version': np.__version__,
        'single_donor_rule': 'max TRAIN target feature groups, tie lowest client ID',
        'detector_training': False,
        'test_evaluated': False,
        'source_reconstruction': (
            'Rebuild exact grouped roles from the score source audit. All fitting, scoring and selection '
            'exclude TEST. Verify score-cache metadata, class and row order, and legacy reference metrics.'
        ),
        'primary_contrast': ('combined_candidate; attack_only/source_protected/full minus multiclass/legacy/full'),
        'fixed_alarm_source': 'Original prior_adjusted TabM base_threshold probabilities and CAL thresholds',
    }


def _initialize_plan(run):
    """Create a single immutable plan or verify the one being resumed."""
    plan_path = run.output / 'targeted_plan.json'
    if plan_path.resolve() != plan_path:
        raise ValueError('Redirected targeted experiment plan')
    if plan_path.exists():
        if _read(plan_path) != run.plan:
            raise ValueError('Targeted source/configuration changed; preserve the output and use a new run directory')
    else:
        if any(path.name != '.v12_lock' for path in run.output.iterdir()):
            raise ValueError('Targeted output contains unbound files')
        _write(plan_path, run.plan)
    return plan_path


def _case_keys(cases):
    """List the reproducible five-by-two case inventory."""
    return {f'seed_{seed}/{objective}' for seed in cases for objective in OBJECTIVES}


def _read_state(path, cases):
    """Verify each previously published case before reusing it."""
    if path.resolve() != path:
        raise ValueError('Redirected targeted state')
    state = _read(path) if path.is_file() else {}
    if not isinstance(state, dict) or not state.keys() <= _case_keys(cases):
        raise ValueError('Unexpected targeted cases in the state snapshot')
    for key, bindings in state.items():
        child = path.parent / key
        if child.resolve() != child:
            raise ValueError(f'Redirected case directory: {child}')
        _check_bindings(child, bindings)
    return state


def _case_source(run, case_path):
    """Read one bound case result and check its hash before reconstructing rows."""
    inventory = run.plan['source']['artifacts']
    relative = case_path.relative_to(run.source.parent).as_posix()
    expected = inventory[relative]
    if case_path.resolve() != case_path or _hash(case_path) != expected:
        raise ValueError(f'NB15 reference case changed: {case_path}')
    return _read(case_path)


def _append_original_costs(tables, original, context, objective):
    """Identify archived multiclass costs separately from newly measured work."""
    for name, source in (('training', 'timings'), ('communication', 'communication')):
        tables[name].extend(
            {
                **context['identity'],
                'objective': objective,
                'measurement_source': 'original_multiclass_training',
                **row,
            }
            for row in original['tables'][source]
            if row.get('mode') in ('head_local', 'head_donor', 'head_federated')
        )


def _case_identity(context, original, objective, case_path):
    """Record the frozen comparators behind the current objectives."""
    return {
        'identity': context['identity'],
        'objective': objective,
        'source_case': str(case_path),
        'source_case_sha256': _hash(case_path),
        'source_cache_sha256': original['score_cache']['sha256'],
        'donor': context['donor'],
        'donor_support': context['donor_support'],
        'partition_sha256': context['policy']['partition_sha256'],
        'detector_training': False,
        'test_evaluated': False,
    }


def _stage_case(run, destination, context, original, case_path, data, cli, objective, plan_sha):
    """Write a complete case in a temporary sibling before publishing it."""
    if _execution_code() != run.plan['current_reconstruction_and_fitting_sha256']:
        raise ValueError('Code changed during targeted training; no mixed-version case is allowed')
    print(f"FedCAID | NB15 seed={context['identity']['seed']} {objective} | 开始训练子类型头", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.case-', dir=destination.parent) as scratch:
        staged = Path(scratch) / 'case'
        result = _train_case(context, data, cli, objective, staged)
        if objective == 'multiclass':
            _verify_original(result['tables'], original)
            _append_original_costs(result['tables'], original, context, objective)
        result.update(_case_identity(context, original, objective, case_path))
        _write(staged / 'result.json', result)
        receipt = {
            'targeted_plan_sha256': plan_sha,
            'case': destination.relative_to(run.output).as_posix(),
            'artifacts': _bindings(staged, ('result.json', 'trained_heads.npz')),
        }
        _write(staged / 'case_receipt.json', receipt)
        staged.rename(destination)
    print(f"FedCAID | NB15 seed={context['identity']['seed']} {objective} | 训练和评估完成", flush=True)


def _recover_case(destination, key, plan_sha):
    """Recover a fully published case after interruption before the state update."""
    if destination.resolve() != destination:
        raise ValueError(f'Redirected case directory: {destination}')
    receipt = _read(destination / 'case_receipt.json')
    if receipt['case'] != key or receipt['targeted_plan_sha256'] != plan_sha:
        raise ValueError(f'Published case does not belong to the current run: {destination}')
    if set(receipt['artifacts']) != {'result.json', 'trained_heads.npz'}:
        raise ValueError(f'Published case receipt is incomplete: {destination}')
    _check_bindings(destination, receipt['artifacts'])
    return _bindings(destination, (*receipt['artifacts'], 'case_receipt.json'))


def _recover_unrecorded(run, state, state_path, plan_sha):
    """Resume fully published cases without repeating their training."""
    for key in sorted(_case_keys(run.cases) - state.keys()):
        destination = run.output / key
        if destination.exists():
            state[key] = _recover_case(destination, key, plan_sha)
            _write(state_path, state)


def _train_seed(run, state, state_path, plan_sha, data, cli, seed):
    """Recover exact CAL/VAL rows and fit the missing objectives for one seed."""
    case_path = run.cases[seed]
    original = _case_source(run, case_path)
    context = _case_context(data, cli, original, case_path.parent, run.source_run)
    for objective in OBJECTIVES:
        key = f'seed_{seed}/{objective}'
        if key in state:
            continue
        destination = run.output / key
        if destination.resolve() != destination:
            raise ValueError(f'Redirected case directory: {destination}')
        _stage_case(run, destination, context, original, case_path, data, cli, objective, plan_sha)
        state[key] = _recover_case(destination, key, plan_sha)
        _write(state_path, state)


def _train_missing(run, state, state_path, plan_sha):
    """Load the real dataset once and execute all remaining five-seed cases."""
    pending = [seed for seed in sorted(run.cases) if any(f'seed_{seed}/{name}' not in state for name in OBJECTIVES)]
    if not pending:
        return
    print(f'FedCAID | 读取真实 NB15 数据，重建源对照的 TRAIN/CAL/VAL | 待训练种子 {pending}', flush=True)
    data, cli = _load_context(run.source, run.source_run, run.plan['device'])
    _write(run.output / 'data_audit.json', data.audit)
    print('FedCAID | NB15 源数据哈希与角色审计匹配 | 开始验证各随机种子分区', flush=True)
    for seed in pending:
        _train_seed(run, state, state_path, plan_sha, data, cli, seed)


def _write_tables(output, state, seeds):
    """Combine the case files and write matched result tables."""
    tables = {name: [] for name in TABLES}
    for key in sorted(state):
        for name, rows in _read(output / key / 'result.json')['tables'].items():
            tables[name].extend(rows)
    pairs, intervals = _comparisons(tables, seeds)
    for name, rows in tables.items():
        write_csv(output / f'{name}.csv', rows)
    write_csv(output / 'paired_by_seed.csv', pairs)
    write_csv(output / 'paired_intervals.csv', intervals)
    return tables, intervals


def _completion(run, state_path, plan_sha, state, new_cases):
    """Bind completed case evidence and give scientific criteria a separate status."""
    seeds = sorted(run.cases)
    tables, intervals = _write_tables(run.output, state, seeds)
    criteria = _development_criteria(tables, intervals, seeds)
    status = 'development_criteria_met' if criteria['development_criteria_met'] else 'development_criteria_not_met'
    names = (
        'targeted_plan.json',
        'targeted_state.json',
        'data_audit.json',
        'paired_by_seed.csv',
        'paired_intervals.csv',
        *(f'{name}.csv' for name in TABLES),
    )
    return {
        'schema_version': 2,
        'status': 'completed',
        'scientific_status': status,
        'output': str(run.output),
        'source': str(run.source),
        'cases': len(state),
        'seeds': sorted(run.cases),
        'detector_training': False,
        'new_head_training': True,
        'binary_alarm_masks_fixed': True,
        'test_evaluated': False,
        'independent_confirmation': False,
        'execution_origin': 'current_training' if new_cases else 'finalized_current_run',
        'trained_cases_this_invocation': new_cases,
        'reused_cases_this_invocation': len(state) - new_cases,
        'plan_sha256': plan_sha,
        'state_sha256': _hash(state_path),
        'primary_intervals': [row for row in intervals if row['contrast'] == 'combined_candidate'],
        **criteria,
        'artifacts': _bindings(run.output, names),
    }


def _resume_completion(path, state_path, plan_sha, state, cases):
    """Reuse immutable results from the current plan after validating all file hashes."""
    previous = _read(path)
    if (
        path.resolve() != path
        or previous['status'] != 'completed'
        or state.keys() != _case_keys(cases)
        or previous['plan_sha256'] != plan_sha
        or previous['state_sha256'] != _hash(state_path)
    ):
        raise ValueError('Targeted completion receipt changed')
    _check_bindings(path.parent, previous['artifacts'])
    previous.update(
        execution_origin='resumed_completed_run',
        trained_cases_this_invocation=0,
        reused_cases_this_invocation=len(state),
    )
    return previous


def _run_locked(run):
    """Resume or execute the declared cases under the exclusive experiment lock."""
    plan_path = _initialize_plan(run)
    plan_sha = _hash(plan_path)
    state_path, complete_path = (run.output / name for name in ('targeted_state.json', 'complete.json'))
    state = _read_state(state_path, run.cases)
    if complete_path.exists():
        result = _resume_completion(complete_path, state_path, plan_sha, state, run.cases)
    else:
        _recover_unrecorded(run, state, state_path, plan_sha)
        completed_before = len(state)
        _train_missing(run, state, state_path, plan_sha)
        if _execution_code() != run.plan['current_reconstruction_and_fitting_sha256']:
            raise ValueError('Code changed before the targeted completion receipt was written')
        result = _completion(run, state_path, plan_sha, state, len(state) - completed_before)
        _write(complete_path, result)
    return {**result, 'artifacts': {**result['artifacts'], 'complete.json': _hash(complete_path)}}


def run_targeted(output, project, *, device='auto', pair_source=None) -> dict:
    """Fit and evaluate real fixed-alarm NB15 heads in a fresh or resumable output."""
    run = _prepare_run(output, project, device, pair_source)
    run.output.mkdir(parents=True, exist_ok=True)
    if (run.output / '.v12_lock').resolve() != run.output / '.v12_lock':
        raise ValueError('Redirected targeted experiment lock')
    with _output_lock(run.output):
        return _run_locked(run)
