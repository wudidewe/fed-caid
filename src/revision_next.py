"""Run FedCAID matched TRAIN-missing heads, CAL coverage and scale experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score

from ._iov_layout import project_root
from .experiment_config import DATASET_PARAMETERS, DATASETS
from .revision_baseline import run_fedavg
from .revision_benign_cal import reserve_benign_cal
from .revision_binary_diagnostics import class_diagnostic_rows, summarize_binary_diagnostics, summarize_fnr_guard
from .revision_cal_split import InfeasibleDesign, cal_group_splits
from .revision_class_operating import attack_routes, class_alarm_thresholds, select_class_operating_point
from .revision_controlled import development_cells, plan_targets, remove_receiver_class
from .revision_data import _feature_keys, load_revision_data
from .revision_detector_objective import DETECTOR_OBJECTIVES, objective_settings
from .revision_fixed_subtype import eligible_missing_classes, fixed_subtype_predictions, subtype_transitions
from .revision_measurement import Resources
from .revision_operating_selection import DEFAULT_CANDIDATES, select_operating_point, split_operating_subsets
from .revision_report import METRICS, score, summarize, write_csv
from .revision_run import (
    _atomic_json,
    _cap_role,
    _gather_eval,
    _roster_hash,
    _source_hashes,
    _synthetic,
    _train_clients,
    allocate_clients,
)
from .revision_score_cache import load_score_cache, save_score_cache
from .revision_shared_head import (
    DONOR,
    FEDERATED,
    HEAD_OBJECTIVES,
    LOCAL,
    SINGLE_DONOR,
    fit_shared_heads,
    head_probabilities,
)
from .revision_subtype_guard import GUARDED_ROUTING
from .revision_supplement import FLOW_DATASETS, rebalance_scale
from .revision_target_scores import TargetScoreObserver
from .utils import empirical_alarm_threshold, hierarchical_decisions, sha

ROOT = project_root(__file__)
warnings.filterwarnings('ignore', message=r'The .* feature has just two bin edges.*')
POLICIES = ('natural', 'reserved')
SEEDS = (42, 43, 44, 45, 46)
CASE_ID = ('dataset', 'scenario', 'alpha', 'clients', 'target_class', 'seed')
CONDITIONS = CASE_ID[:-1] + ('cal_policy',)
TABLES = (
    'metrics',
    'class_metrics',
    'target_metrics',
    'target_scores',
    'gates',
    'coverage',
    'timings',
    'communication',
    'rounds',
    'confusion',
    'subtype_audit',
    'subtype_transitions',
)
GUARD_TABLES = ('subtype_guard_cal', 'alerted_target_ranking')
SELECTION_TABLES = ('operating_selection', 'operating_candidates', 'operating_reference_pairs')
SELECTING_POLICIES = ('cal-select', 'class-select')
CLASS_ALARM_ROUTING = 'local_tabm_attack_top1_v1'
CLASS_CANDIDATES = (0.01, 0.0075, 0.005)
FIXED_MODES = {
    LOCAL: 'fixed_local',
    FEDERATED: 'fixed_federated',
    DONOR: 'fixed_donor',
    SINGLE_DONOR: 'fixed_donor_single',
    'fedavg_threshold': 'fixed_fedavg_subtype',
}
FIXED_ROUTING = 'missing_attack_top1_v1'
GUARD_ROUTING = GUARDED_ROUTING
GUARD_MODES = {source: mode.replace('fixed_', 'guarded_', 1) for source, mode in FIXED_MODES.items()}
GUARD_TARGETS = {'CIC-IDS-2017': 'DoS Hulk', 'CIC-UNSW-NB15': 'Exploits'}
OPERATING_REFERENCE_FAR = 0.01
ORIGINAL_V11_EVALUATION_SOURCES = {
    'revision_next.py': 'b9f279e507cca94d9f58f536dd408e060598d6a10fdf58a12caf5285779d0b1e',
    'revision_external.py': 'fb2042f3d37fd57f040f16ac8b49681e0d64fcf6740af47b3a9c6dd3d4c919cc',
}
PRE_V12_SHARED_HEAD_SHA256 = '9faba2ca0b1ef0f358346ce1d72de1cec19595c8f419092dee5416409f7b2ed0'
V12_COMPATIBLE_SHARED_HEAD_SHA256 = 'a4de88679030aedeb0d565b56f1d8613bf33d44600b5a5a8ff54b570f0b518a8'
PRE_V12_EVALUATORS = {
    (
        '6c544b1d1697f9fcba8aeaca0b7a07920adf22f15a9660ccf5228836c0468d2d',
        'b9165e4dfb135a38d6662e38eab6aa8a117d68cecfc0baa0386ca0f9880695dc',
    ),
    (
        '8628f396587edf8198ed1368e8bf968634ed6758376fcf348ccdec96c178d14b',
        'b9165e4dfb135a38d6662e38eab6aa8a117d68cecfc0baa0386ca0f9880695dc',
    ),
    (
        'd9ff00ca95d2b834f99ee2486829f2eaeea656383ab14e7d4513687e8a165e64',
        'fbdbec72cdc03844884f1e57d0ba86306cfd1a02f021f5890eb9e9d331be281d',
    ),
}
PAIRINGS = {
    'aggregation': (FEDERATED, LOCAL),
    'aggregation_argmax': (FEDERATED + '_argmax', LOCAL + '_argmax'),
    'receiver_participation': (FEDERATED, DONOR),
    'against_local_detector': (FEDERATED, 'base_threshold'),
    'against_fedavg': (FEDERATED, 'fedavg_threshold'),
    'fixed_aggregation': ('fixed_federated', 'fixed_local'),
    'fixed_against_base': ('fixed_federated', 'base_threshold'),
    'fixed_receiver_participation': ('fixed_federated', 'fixed_donor'),
    'fixed_against_fedavg': ('fixed_federated', 'fixed_fedavg_subtype'),
    'guarded_aggregation': ('guarded_federated', 'guarded_local'),
    'guarded_against_base': ('guarded_federated', 'base_threshold'),
    'guarded_receiver_participation': ('guarded_federated', 'guarded_donor'),
    'guarded_against_fedavg': ('guarded_federated', 'guarded_fedavg_subtype'),
    'guarded_against_top1': ('guarded_federated', 'fixed_federated'),
    'guarded_donor_pool_vs_single': ('guarded_donor', 'guarded_donor_single'),
    'guarded_federated_vs_single': ('guarded_federated', 'guarded_donor_single'),
}
DEVELOPMENT_THRESHOLDS = {
    'minimum_target_f1_effect_pp': 1.0,
    'max_absolute_far': 0.01,
    'max_far_increase': 0.001,
    'max_fnr_increase': 0.005,
    'max_coverage_loss': 0.0,
    'minimum_attack_coverage': 1.0,
    'minimum_paired_seeds': 5,
}
FIXED_CRITERIA = {'minimum_f1_gain_over_base_pp': 1.0, 'maximum_macro_f1_loss_pp': 1.0}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _source_receipt():
    return {
        **_source_hashes(True),
        **{
            name: sha(Path(__file__).with_name(name))
            for name in (
                'revision_next.py',
                'revision_shared_head.py',
                'revision_benign_cal.py',
                'revision_external.py',
                'revision_target_scores.py',
                'revision_fixed_subtype.py',
                'revision_tabm_checkpoint.py',
                'revision_subtype_guard.py',
                'revision_score_cache.py',
                'revision_operating_selection.py',
                'revision_binary_diagnostics.py',
                'revision_class_operating.py',
                'revision_detector_objective.py',
            )
        },
    }


def _tables(cli):
    tables = TABLES + GUARD_TABLES if cli.guarded_subtypes else TABLES
    if cli.operating_guard:
        tables += ('binary_operating_guard', 'binary_diagnostics', 'binary_class_diagnostics')
    if cli.operating_policy in SELECTING_POLICIES:
        tables += SELECTION_TABLES
    return tables + ('replay_operating_points',) if cli.replay_from else tables


def _score_configuration(configuration):
    # Only post-scoring switches can vary during replay. Model options, code,
    # CAL routing and the source dataset audit have to match.
    excluded = {
        'base_cal_far',
        'guard_rewrite_rate',
        'guard_min_negative_groups',
        'save_scores',
        'save_row_evidence',
        'export_frozen',
        'replay_from',
        'guard_min_check_groups',
        'guard_check_tolerance',
        'max_operating_fnr_increase',
        'replay_device',
        'operating_policy',
        'operating_candidates',
        'operating_min_benign_rows',
        'operating_select_fraction',
        'operating_min_select_benign',
        'operating_min_select_attack',
        'operating_min_receiver_attack',
        'operating_min_attack_coverage',
        'operating_min_route_benign',
    }
    return {
        'code': configuration['code_sha256'],
        'settings': {k: v for k, v in configuration['settings'].items() if k not in excluded},
    }


def _score_route(audit):
    # Neither the selector's subdivisions of CAL holdout nor a failed
    # subdivision changes which rows were scored by the fitted models.
    return {
        **audit,
        'policies': {
            name: {k: v for k, v in detail.items() if k != 'operating_audit'}
            for name, detail in audit['policies'].items()
        },
    }


def _comparable_score_route(audit):
    """Ignore only donor qualification metadata when matching legacy score routes."""
    bound = _score_route(audit)
    intervention = bound.get('partition_intervention')
    if intervention is not None:
        superseded = {
            'cal_group_split',
            'donor_cal_fit_rows',
            'client_cal_fit_target_rows',
            'donor_cal_fit_minimum',
            'donor_cal_fit_by_policy',
        }
        bound['partition_intervention'] = {
            name: value for name, value in intervention.items() if name not in superseded
        }
    return bound


def _checked_replay_configuration(original, current):
    saved, expected = map(_score_configuration, (original, current))
    former_settings = {'detector_objective': 'prior_adjusted', **saved['settings']}
    current_settings = {'detector_objective': 'prior_adjusted', **expected['settings']}
    if saved['code'] == expected['code'] and former_settings == current_settings:
        return saved

    def remaining_code(binding, ignored):
        return {name: digest for name, digest in binding['code'].items() if name not in ignored}

    # The V12 additions leave the original local, pooled and full scoring branches intact.
    # Bind replay to the original archive receipt and only allow audited pre-V12 file digests.
    legacy_head = (
        original['version'] == current['version'] == 4
        and former_settings == current_settings
        and not current_settings.get('single_donor')
        and saved['code'].get('revision_shared_head.py') == PRE_V12_SHARED_HEAD_SHA256
        and expected['code'].get('revision_shared_head.py') == V12_COMPATIBLE_SHARED_HEAD_SHA256
    )
    if (
        legacy_head
        and tuple(saved['code'].get(name) for name in ('revision_next.py', 'revision_external.py'))
        in PRE_V12_EVALUATORS
    ):
        ignored = {
            'revision_next.py',
            'revision_external.py',
            'revision_shared_head.py',
            'revision_binary_diagnostics.py',
            'revision_class_operating.py',
        }
        if remaining_code(saved, ignored) == remaining_code(expected, ignored):
            return saved
    # Only admit the known original V11 fixed evaluator. Its schema-2 archive
    # stays bound to its original full receipt; the score-producing sources,
    # settings and ordered TRAIN/CAL/VAL routes still must match exactly.
    if (
        original['version'] == current['version'] == 4
        and 'operating_policy' not in original['settings']
        and 'operating_candidates' not in original['settings']
        and former_settings == current_settings
        and all(saved['code'].get(k) == v for k, v in ORIGINAL_V11_EVALUATION_SOURCES.items())
    ):
        evaluation_only = {
            *ORIGINAL_V11_EVALUATION_SOURCES,
            'revision_operating_selection.py',
            'revision_binary_diagnostics.py',
            'revision_class_operating.py',
        }
        if legacy_head:
            evaluation_only.add('revision_shared_head.py')
        if remaining_code(saved, evaluation_only) == remaining_code(expected, evaluation_only):
            return saved
    raise ValueError('Replay requires a supported schema-2 source with identical scoring settings and code')


def _run_version(cli):
    return 11 if cli.operating_guard else 10 if cli.guarded_subtypes else 9 if cli.fixed_subtypes else 8


def _runtime(cli):
    execution = cli.replay_device or cli.device
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'scikit_learn': sklearn.__version__,
        'torch': str(torch.__version__),
        'torch_cuda': torch.version.cuda,
        'torch_cpu_threads': torch.get_num_threads(),
        'training_device': cli.device,
        'execution_device': execution,
        'gpu': torch.cuda.get_device_name() if execution == 'cuda' and torch.cuda.is_available() else None,
    }


def _scope_data(data):
    x, y = data.parts['test']
    return {**data.parts, 'test': (x[:0], y[:0])}


def _label_blind_roster(parts, classes, clients, alpha, seed):
    # Reuse the historical TRAIN/CAL label-skew allocator. Supply empty VAL and
    # TEST to the allocator, then route the complete VAL population by permutation.
    x, y = parts['val']
    design_parts = {**parts, 'val': (x[:0], y[:0])}
    roster, repairs = allocate_clients(design_parts, classes, clients, alpha, seed)
    order = np.random.default_rng(seed + 2251).permutation(len(y))
    for binding, rows in zip(roster, np.array_split(order, clients), strict=True):
        if not len(rows):
            raise InfeasibleDesign('VAL has fewer rows than participating clients')
        binding['val'] = rows
    return roster, repairs


def _scale_specs(cli):
    counts = (3, 5) if cli.smoke else (3, 5, 10)
    result = [(f'scale_{budget}_c{n}', n, 0.8, budget) for budget in ('total', 'local') for n in counts]
    result.extend(
        (f'scale_alpha_{label}', 3, alpha, 'heterogeneity')
        for label, alpha in (('low', 0.3), ('reference', 0.8), ('high', 2.0))
    )
    return result


def _budget_parts(parts, clients, budget, seed, cli):
    if cli.smoke or budget is None:
        return parts
    train = cli.scale_train_total if budget != 'local' else cli.scale_train_per_client * clients
    cal = cli.scale_cal_total if budget != 'local' else cli.scale_cal_per_client * clients
    parts = _cap_role(parts, 'train', train, seed + 905, exact=True)
    return _cap_role(parts, 'cal', cal, seed + 906, exact=True)


def _unique_rows(x, rows):
    return int(len(np.unique(_feature_keys(x[rows])))) if len(rows) else 0


def _design(parts, names, identity, target, budget, cli):
    seed, clients, alpha = (identity[k] for k in ('seed', 'clients', 'alpha'))
    knobs = DATASET_PARAMETERS[identity['dataset']]
    parts = _budget_parts(parts, clients, budget, seed, cli)
    roster, repair_counts = _label_blind_roster(parts, len(names), clients, alpha, seed)
    quota_moves = rebalance_scale(roster, seed) if budget in ('total', 'local') else {}
    receiver = None if target is None else cli.receiver
    controlled = None
    if target is not None:
        roster, controlled = remove_receiver_class(
            parts,
            roster,
            target,
            receiver,
            seed,
            knobs['cal_fit_fraction'],
            'train_and_cal',
            donor_fit_min=None,
        )
        train_x, train_y = parts['train']
        donors = np.concatenate([r['train'] for i, r in enumerate(roster) if i != receiver])
        donor_positives = donors[train_y[donors] == target]
        distinct = _unique_rows(train_x, donor_positives)
        if distinct < cli.min_donor_groups:
            raise InfeasibleDesign(f'Donor TRAIN target groups {distinct} below declared {cli.min_donor_groups}')
        controlled['donor_train_target_feature_groups'] = distinct
        controlled['train_target_rows_by_client'] = [
            int((train_y[binding['train']] == target).sum()) for binding in roster
        ]
        if controlled['train_target_rows_by_client'][receiver] != 0:
            raise RuntimeError('Receiver target leaked into TRAIN')
        if getattr(cli, 'single_donor', False):
            support = []
            for client, binding in enumerate(roster):
                if client == receiver:
                    continue
                rows = binding['train']
                positives = rows[train_y[rows] == target]
                support.append(
                    {
                        'client': client,
                        'target_rows': len(positives),
                        'target_groups': _unique_rows(train_x, positives),
                    }
                )
            selected = min(support, key=lambda row: (-row['target_groups'], row['client']))
            controlled['single_donor'] = {
                'rule': 'max_train_target_feature_groups_then_lowest_client_id',
                'selected_client': selected['client'],
                'support': support,
            }
    cal_x, cal_y = parts['cal']
    if controlled is not None:
        controlled['donor_cal_fit_by_policy'] = {}
        controlled['absence_origin'] = (
            'target_removed'
            if any(controlled['roles'][role]['receiver_target_rows_before'] for role in ('train', 'cal'))
            else 'naturally_absent'
        )
    policies = {}
    for name in cli.cal_policies if target is not None else ('natural',):
        reservation = None
        try:
            routed = roster
            floor = cli.min_benign_rows if name == 'reserved' else 0
            if name == 'reserved':
                routed, reservation = reserve_benign_cal(cal_x, cal_y, roster, floor, seed)
            splits, audit = cal_group_splits(
                cal_x, cal_y, routed, knobs['cal_fit_fraction'], seed, min_holdout_benign=floor
            )
            operating_splits = operating_audit = None
            if cli.operating_policy in SELECTING_POLICIES:
                try:
                    operating_splits, operating_audit = split_operating_subsets(
                        cal_x,
                        cal_y,
                        routed,
                        splits,
                        cli.operating_select_fraction,
                        seed,
                        min_select_benign=cli.operating_min_select_benign,
                    )
                except InfeasibleDesign as error:
                    # Keep the valid original CAL and expose selector failure.
                    operating_audit = {'status': 'operating_split_unavailable', 'reason': str(error)}
            for receipt, binding, (_, hold) in zip(audit['clients'], routed, splits, strict=True):
                rows = binding['cal'][hold]
                receipt['holdout_benign_feature_groups'] = _unique_rows(cal_x, rows[cal_y[rows] == 0])
            if target is not None:
                if (cal_y[routed[receiver]['cal']] == target).any():
                    raise RuntimeError('Receiver target leaked into CAL')
                donor_rows = np.concatenate(
                    [r['cal'][fit] for i, (r, (fit, _)) in enumerate(zip(routed, splits, strict=True)) if i != receiver]
                )
                positive = donor_rows[cal_y[donor_rows] == target]
                audit['donor_cal_fit_target_rows'] = len(positive)
                audit['donor_cal_fit_target_feature_groups'] = _unique_rows(cal_x, positive)
                controlled['donor_cal_fit_by_policy'][name] = {
                    'cal_assignment_sha256': audit['assignment_sha256'],
                    'target_rows': len(positive),
                    'target_feature_groups': audit['donor_cal_fit_target_feature_groups'],
                    'minimum_feature_groups': cli.min_donor_groups,
                }
                if audit['donor_cal_fit_target_feature_groups'] < cli.min_donor_groups:
                    raise InfeasibleDesign('Donor CAL-fit target feature groups below declared minimum')
            policies[name] = {
                'status': 'eligible',
                'roster': routed,
                'splits': splits,
                'group_audit': audit,
                'reservation': reservation,
                'partition_sha256': _roster_hash(routed),
            }
            if cli.operating_policy in SELECTING_POLICIES:
                policies[name].update(operating_splits=operating_splits, operating_audit=operating_audit)
        except InfeasibleDesign as error:
            policies[name] = {
                'status': 'insufficient_or_infeasible_cal',
                'reason': str(error),
                'reservation': reservation,
            }
    receipt = {
        'route_sha256': _roster_hash(roster),
        'train_rows': len(parts['train'][1]),
        'cal_rows': len(cal_y),
        'val_rows': len(parts['val'][1]),
        'test_evaluated': False,
        'allocation_repairs': repair_counts,
        'quota_moves': quota_moves,
        'budget_policy': budget,
        'train_rows_by_client': [len(r['train']) for r in roster],
        'original_cal_rows_by_client': [len(r['cal']) for r in roster],
        'receiver': receiver,
        'partition_intervention': controlled,
        'policies': {
            name: {k: v for k, v in policy.items() if k not in ('roster', 'splits', 'operating_splits')}
            for name, policy in policies.items()
        },
    }
    return parts, roster, policies, receipt


def _args(identity, cli):
    return argparse.Namespace(
        **{
            **DATASET_PARAMETERS[identity['dataset']],
            'seed': identity['seed'],
            'device': cli.device,
            'detector_objective': getattr(cli, 'detector_objective', 'prior_adjusted'),
            'rounds': cli.rounds,
            'base_epochs': cli.detector_epochs,
            'head_epochs': cli.head_epochs,
            'head_lr': cli.head_lr,
            'head_batch_size': cli.head_batch_size,
            'clip': cli.head_clip,
            'head_objective': cli.head_objective,
            'batch_size': 64 if cli.smoke else DATASET_PARAMETERS[identity['dataset']]['batch_size'],
            'target_far': DEVELOPMENT_THRESHOLDS['max_absolute_far'],
            'threshold_floor': cli.threshold_floor,
            'ple_boundary_policy': 'clip_to_train_edges',
        }
    )


def _head_outputs(states, parts, union, args):
    result, timings = {}, []
    for method, weights in states.items():
        start = time.perf_counter()
        result[method] = [
            head_probabilities(w, _gather_eval(parts, r, ('cal', 'val')), device='cpu', batch_size=args.head_batch_size)
            for w, r in zip(weights, union, strict=True)
        ]
        resident = None
        if len(union[0]['val']):
            row = parts['val'][0][union[0]['val'][:1]]
            head_probabilities(weights[0], row, device='cpu')
            measurements = []
            for _ in range(20):
                tick = time.perf_counter()
                p = head_probabilities(weights[0], row, device='cpu')
                hierarchical_decisions(p, args.threshold_floor)
                measurements.append((time.perf_counter() - tick) * 1000)
            resident = {
                'single_request_including_state_binding_median_ms': float(np.median(measurements)),
                'single_request_including_state_binding_p95_ms': float(np.quantile(measurements, 0.95)),
            }
        timings.append(
            {
                'mode': method,
                'stage': 'head_cal_val_prediction_and_inference_probe',
                'elapsed_seconds': time.perf_counter() - start,
                'prediction_rows': sum(len(r['cal']) + len(r['val']) for r in union),
                **(resident or {}),
            }
        )
    return result, timings


def _detector_outputs(
    parts,
    roster,
    names,
    features,
    args,
    train,
    *,
    fixed=False,
    checkpoint=None,
    receiver=None,
    include_fedavg=True,
):
    canonical, timings = [], []

    def capture(index, node):
        from .revision_tabm_checkpoint import save_tabm_checkpoint, tabm_node_predictor

        start = time.perf_counter()
        predict = tabm_node_predictor(node, batch_size=args.batch_size)
        # Score CAL and VAL separately, with the same CPU predictor used for sealed inference.
        outputs = [predict(parts[role][0][roster[index][role]]) for role in ('cal', 'val')]
        canonical.append(np.concatenate(outputs))
        timings.append(
            {
                'mode': 'base_threshold',
                'stage': 'canonical_tabm_cal_val_cpu',
                'client': index,
                'elapsed_seconds': time.perf_counter() - start,
                'prediction_rows': sum(len(p) for p in outputs),
            }
        )
        if checkpoint is not None and index == receiver:
            save_tabm_checkpoint(checkpoint, node, batch_size=args.batch_size, inference_device='cpu')

    with TemporaryDirectory(prefix='fedcaid_v8_local_') as scratch:
        inputs, times, _ = _train_clients(
            parts, roster, names, features, args, Path(scratch), on_fitted_client=capture if fixed else None
        )
    result = {
        'base_threshold': (
            canonical if fixed else [np.concatenate((x.cal_probabilities, x.eval_probabilities)) for x in inputs]
        )
    }
    for item in times:
        item.pop('ple_stability_private', None)
    del inputs
    times.extend(timings)
    if not include_fedavg:
        return result, times, None
    fed = run_fedavg(train, [_gather_eval(parts, r, ('cal', 'val')) for r in roster], len(names), args)
    result['fedavg_threshold'] = fed['probabilities']
    return (
        result,
        [*times, {'mode': 'fedavg_threshold', 'stage': 'train_and_score_fedavg', **fed['performance']}],
        {'mode': 'fedavg_threshold', **fed['communication']},
    )


def _gate_threshold(probabilities, scalar, attack_thresholds):
    return scalar if attack_thresholds is None else class_alarm_thresholds(probabilities, attack_thresholds)


def _guard_calibration(
    parts,
    roster,
    splits,
    union,
    probabilities,
    thresholds,
    base,
    target,
    receiver,
    args,
    cli,
    results,
    class_thresholds=None,
):
    from .revision_subtype_guard import apply_cal_check, audit_subtype_guard, fit_subtype_guard

    cal_x, cal_y = parts['cal']
    guards = {source: [] for source in GUARD_MODES if source in probabilities}
    for i, (binding, (fit, _), merged) in enumerate(zip(roster, splits, union, strict=True)):
        eligible = _eligible_for_client(
            parts, probabilities['base_threshold'][i].shape[1], binding, i, receiver, target
        )
        rows = binding['cal'][fit]
        ordered = np.searchsorted(merged['cal'], rows)
        if not np.array_equal(merged['cal'][ordered], rows):
            raise RuntimeError('CAL-fit guard scores differ from the recorded route')
        base_p = probabilities['base_threshold'][i][ordered]
        attack_thresholds = class_thresholds[i] if class_thresholds is not None else None
        original = hierarchical_decisions(
            base_p, _gate_threshold(base_p, thresholds['base_threshold'][i], attack_thresholds)
        )['predicted']
        _, group_ids = np.unique(_feature_keys(cal_x[rows]), return_inverse=True)
        group_ids = group_ids.reshape(-1)
        # A fresh local group split inside CAL-fit leaves the CAL-holdout used
        # to choose the binary operating point out of the guard check.
        order = np.random.default_rng(args.seed + 8123 + i).permutation(int(group_ids.max()) + 1 if len(rows) else 0)
        checking = np.isin(group_ids, order[: len(order) // 3])
        fitting = ~checking
        for source, fitted in guards.items():
            scores = probabilities[source][i][ordered]
            guard = fit_subtype_guard(
                original[fitting],
                scores[fitting],
                cal_y[rows[fitting]],
                eligible,
                max_rewrite_rate=cli.guard_rewrite_rate,
                min_negative_groups=cli.guard_min_negative_groups,
                group_ids=group_ids[fitting],
            )
            effective = guard
            if eligible:
                check_inputs = (original[checking], scores[checking], cal_y[rows[checking]], eligible, guard)
                if cli.operating_guard:
                    effective, check = apply_cal_check(
                        *check_inputs,
                        min_check_groups=cli.guard_min_check_groups,
                        check_tolerance=cli.guard_check_tolerance,
                        group_ids=group_ids[checking],
                    )
                else:
                    check = audit_subtype_guard(*check_inputs, group_ids=group_ids[checking])
                results['subtype_guard_cal'].append(
                    {
                        **base,
                        'role': 'cal',
                        'client': i,
                        'mode': GUARD_MODES[source],
                        'eligible_classes': list(eligible),
                        'effective_status': effective['status'],
                        'effective_threshold': effective['threshold'],
                        'guard_fit_ids_sha256': hashlib.sha256(rows[fitting].astype('<i8').tobytes()).hexdigest(),
                        'guard_check_ids_sha256': hashlib.sha256(rows[checking].astype('<i8').tobytes()).hexdigest(),
                        **{f'fit_{k}': v for k, v in guard.items()},
                        **{f'check_{k}': v for k, v in check.items()},
                    }
                )
            fitted.append(effective)
    if receiver is not None:
        from .revision_subtype_guard import target_attack_margin

        for source in guards:
            for i, (binding, (fit, _), merged) in enumerate(zip(roster, splits, union, strict=True)):
                if i == receiver:
                    continue
                rows = binding['cal'][fit]
                rows = rows[cal_y[rows] == target]
                positions = np.searchsorted(merged['cal'], rows)
                marg = target_attack_margin(probabilities[source][i][positions], target)
                base_p = probabilities['base_threshold'][i][positions]
                attack_thresholds = class_thresholds[i] if class_thresholds is not None else None
                alerted = hierarchical_decisions(
                    base_p, _gate_threshold(base_p, thresholds['base_threshold'][i], attack_thresholds)
                )['attack_alert']
                cutoff = guards[source][receiver]['threshold']
                results['subtype_guard_cal'].append(
                    {
                        **base,
                        'role': 'cal',
                        'client': i,
                        'mode': GUARD_MODES[source],
                        'eligible_classes': [target],
                        'donor_positive_rows': len(rows),
                        'donor_positive_groups': _unique_rows(cal_x, rows),
                        'donor_tabm_alerted_positives': int(alerted.sum()),
                        'donor_tabm_alerted_margin_pass': (
                            int((alerted & (marg > cutoff)).sum()) if cutoff is not None else None
                        ),
                    }
                )
    return guards


def _eligible_for_client(parts, names_count, binding, i, receiver, target):
    if receiver is not None:
        return (target,) if i == receiver else ()
    return eligible_missing_classes(parts['train'][1][binding['train']], names_count)


def _fixed_policy(
    parts,
    names,
    roster,
    union,
    probabilities,
    decisions,
    forecasts,
    thresholds,
    enabled,
    results,
    base,
    target,
    receiver,
    states,
    args,
    evidence,
    guards=None,
):
    from hashlib import sha256

    original = decisions['base_threshold']
    gate_rows = {r['client']: r for r in results['gates'] if r['mode'] == 'base_threshold'}
    variants = [(source, mode, None) for source, mode in FIXED_MODES.items()]
    if guards is not None:
        from .revision_subtype_guard import guarded_subtype_predictions

        variants.extend((source, GUARD_MODES[source], fitted) for source, fitted in guards.items())
    for source, mode, fitted in variants:
        if source not in probabilities:
            continue
        decisions[mode] = []
        forecasts[mode] = forecasts['base_threshold']
        thresholds[mode] = thresholds['base_threshold']
        enabled[mode] = enabled['base_threshold']
        for i, binding in enumerate(roster):
            val = binding['val']
            y = parts['val'][1][val]
            scores = probabilities[source][i][len(union[i]['cal']) :]
            eligible = _eligible_for_client(parts, len(names), binding, i, receiver, target)
            route = fixed_subtype_predictions if fitted is None else guarded_subtype_predictions
            route_options = () if fitted is None else (fitted[i],)
            predicted = route(original[i], scores, eligible, *route_options)
            mask = original[i] > 0
            if not np.array_equal(mask, predicted > 0):
                raise RuntimeError('Fixed subtype branch changed a binary alarm')
            decisions[mode].append(predicted)
            gate = gate_rows[i]
            results['gates'].append({**gate, 'mode': mode, 'decision_source': 'base_threshold'})
            mask_hash = sha256(mask.tobytes()).hexdigest()
            stats, rows = subtype_transitions(y, original[i], predicted, names, eligible, subtype_scores=scores)
            meta = {
                **base,
                'mode': mode,
                'client': i,
                'eligible_classes': list(eligible),
                'base_alarm_mask_sha256': mask_hash,
            }
            results['subtype_audit'].append({**meta, **stats})
            results['subtype_transitions'].extend({**meta, **r} for r in rows)
            if evidence is not None:
                prefix = f"{base['cal_policy']}_c{i}_"
                if prefix + 'val_ids' not in evidence:
                    evidence.update(
                        {
                            prefix + 'val_ids': val,
                            prefix + 'label': y,
                            prefix + 'base_alert_mask': mask,
                            prefix + 'base_prediction': original[i],
                            prefix + 'base_attack_score': forecasts['base_threshold'][i],
                            prefix + 'eligible_classes': np.asarray(eligible, dtype=np.int64),
                        }
                    )
                evidence[prefix + mode + '_prediction'] = predicted
                if fitted is None and (receiver is None or i == receiver):
                    evidence[prefix + mode + '_subtype_scores'] = scores
            if source in states and eligible and mask.any():
                start = time.perf_counter()
                alert_scores = head_probabilities(
                    states[source][i],
                    parts['val'][0][val[mask]],
                    device='cpu',
                    batch_size=args.head_batch_size,
                )
                routed = route(original[i][mask], alert_scores, eligible, *route_options)
                elapsed = time.perf_counter() - start
                if not np.array_equal(routed, predicted[mask]):
                    raise RuntimeError('Alert-only subtype predictions differ from full-row scoring')
                results['timings'].append(
                    {
                        **base,
                        'mode': mode,
                        'client': i,
                        'stage': 'fixed_subtype_alert_only_cpu',
                        'elapsed_seconds': elapsed,
                        'prediction_rows': int(mask.sum()),
                        'eval_rows': len(y),
                    }
                )


def _measure_policy(
    identity,
    policy_name,
    policy,
    args,
    cli,
    parts,
    names,
    union,
    probabilities,
    target,
    receiver,
    states=None,
    evidence=None,
):
    base = {**identity, 'cal_policy': policy_name, 'role': 'val'}
    roster, splits = policy['roster'], policy['splits']
    results = {name: [] for name in _tables(cli)}
    decisions, forecasts, thresholds, enabled = {}, {}, {}, {}
    reference_thresholds = []
    operating = None
    class_routed = cli.operating_policy == 'class-select'
    if cli.operating_policy in SELECTING_POLICIES:
        if 'base_threshold' not in probabilities:
            raise RuntimeError('Operating selection requires the common TabM base_threshold scores')
        operating_splits = policy.get('operating_splits')
        if operating_splits is None:
            operating = {
                'schema': 'cal_operating_selection_v1',
                'status': 'no_feasible_operating_point',
                'reason': 'operating_threshold_split_unavailable',
                'split_reason': (policy.get('operating_audit') or {}).get('reason'),
                'selected_cal_far': None,
                'reference_cal_far': OPERATING_REFERENCE_FAR,
                'candidate_grid': list(cli.operating_candidates),
                'thresholds': [],
                'reference_thresholds': [],
                'candidates': [],
                'support': {},
                'constraints': {},
                'selection_sha256': None,
            }
        else:
            selection_clients = []
            references = []
            for i, (r, merged, (_, hold)) in enumerate(zip(roster, union, splits, strict=True)):
                ordered = np.searchsorted(merged['cal'], r['cal'][hold])
                if not np.array_equal(merged['cal'][ordered], r['cal'][hold]):
                    raise RuntimeError('CAL probability route differs from union prediction')
                op = operating_splits[i]
                labels = parts['cal'][1][r['cal'][hold]]
                probabilities_for_client = probabilities['base_threshold'][i][ordered]
                scores = np.clip(np.asarray(probabilities_for_client[:, 1:], dtype=np.float64).sum(axis=1), 0.0, 1.0)
                reference, _ = empirical_alarm_threshold(
                    probabilities_for_client,
                    labels,
                    OPERATING_REFERENCE_FAR,
                    args.threshold_floor,
                    cli.min_benign_rows,
                    'disable',
                )
                references.append(reference)
                selection_client = {
                    'threshold_fit_scores': scores[op['threshold_fit']],
                    'threshold_fit_labels': labels[op['threshold_fit']],
                    'operating_select_scores': scores[op['operating_select']],
                    'operating_select_labels': labels[op['operating_select']],
                    'receiver': i == receiver,
                }
                if class_routed:
                    routes = attack_routes(probabilities_for_client)
                    selection_client.update(
                        threshold_fit_routes=routes[op['threshold_fit']],
                        operating_select_routes=routes[op['operating_select']],
                    )
                selection_clients.append(selection_client)
            selector, options = (
                (
                    select_class_operating_point,
                    {'classes': len(names), 'min_route_benign': cli.operating_min_route_benign},
                )
                if class_routed
                else (select_operating_point, {'pooled_fallback': False})
            )
            operating = selector(
                selection_clients,
                candidates=cli.operating_candidates,
                max_far=DEVELOPMENT_THRESHOLDS['max_absolute_far'],
                reference_far=OPERATING_REFERENCE_FAR,
                max_fnr_increase=cli.max_operating_fnr_increase,
                min_benign_rows=cli.operating_min_benign_rows,
                min_select_benign_rows=cli.operating_min_select_benign,
                min_select_attack_rows=cli.operating_min_select_attack,
                min_receiver_attack_rows=cli.operating_min_receiver_attack,
                min_attack_coverage=cli.operating_min_attack_coverage,
                floor=args.threshold_floor,
                reference_thresholds=references,
                **options,
            )
        group_audit = policy.get('operating_audit') or {}
        selection_base = {**base, 'role': 'cal'}
        results['operating_selection'].append(
            {
                **selection_base,
                'status': operating['status'],
                'reason': operating.get('reason'),
                'split_reason': operating.get('split_reason'),
                'selected_cal_far': operating.get('selected_cal_far'),
                'reference_cal_far': operating.get('reference_cal_far'),
                'candidate_grid': list(operating.get('candidate_grid', [])),
                'thresholds': list(operating.get('thresholds', [])),
                'class_thresholds': operating.get('class_thresholds'),
                'alarm_routing': CLASS_ALARM_ROUTING if class_routed else 'scalar',
                'reference_thresholds': list(operating.get('reference_thresholds', [])),
                'selection_sha256': operating.get('selection_sha256'),
                'receiver_client': operating.get('receiver_client'),
                'cal_group_assignment_sha256': group_audit.get('assignment_sha256'),
                'group_overlap': group_audit.get('global_exact_feature_overlap'),
                'reference_scope': 'original_v11_parent_holdout; CAL comparisons are development diagnostics',
                'constraints': operating.get('constraints'),
                'support': operating.get('support'),
            }
        )
        for candidate in operating.get('candidates', []):
            overall, receiver_row = candidate['overall'], candidate['receiver'] or {}
            results['operating_candidates'].append(
                {
                    **selection_base,
                    'candidate_cal_far': candidate['candidate_cal_far'],
                    'feasible': candidate['feasible'],
                    'reason': candidate['reason'],
                    'overall_far': overall.get('far'),
                    'overall_fnr': overall.get('fnr'),
                    'overall_fnr_delta': (
                        overall['fnr'] - candidate['reference_fnr']
                        if overall['fnr'] is not None and candidate['reference_fnr'] is not None
                        else None
                    ),
                    'overall_alert_coverage': overall.get('alert_coverage'),
                    'reference_far': candidate.get('reference_far'),
                    'reference_fnr': candidate.get('reference_fnr'),
                    'reference_overall_far_within_nominal': candidate.get('reference_overall_far_within_nominal'),
                    'reference_receiver_far': receiver_row.get('reference_far'),
                    'reference_receiver_far_within_nominal': candidate.get('reference_receiver_far_within_nominal'),
                    'receiver_far': receiver_row.get('far'),
                    'receiver_fnr': receiver_row.get('fnr'),
                    'receiver_fnr_delta': receiver_row.get('fnr_delta'),
                    'class_thresholds': candidate.get('class_thresholds'),
                    'disabled_receiver_class_routes': receiver_row.get('disabled_class_routes'),
                    **{f'check_{name}': value for name, value in candidate['checks'].items()},
                }
            )
    class_thresholds = (operating.get('class_thresholds') or None) if operating is not None else None
    prediction_started = time.perf_counter()
    for method, output in probabilities.items():
        argmax = {'base_threshold': 'base', 'fedavg_threshold': 'fedavg'}.get(method, method + '_argmax')
        decisions[method], decisions[argmax] = [], []
        forecasts[method], forecasts[argmax] = [], []
        thresholds[method], enabled[method] = [], []
        for i, (r, merged, (_, hold), p) in enumerate(zip(roster, union, splits, output, strict=True)):
            ordered = np.searchsorted(merged['cal'], r['cal'][hold])
            if not np.array_equal(merged['cal'][ordered], r['cal'][hold]):
                raise RuntimeError('CAL probability route differs from union prediction')
            y_cal = parts['cal'][1][r['cal'][hold]]
            op = (
                policy['operating_splits'][i]
                if operating is not None and policy.get('operating_splits') is not None
                else None
            )
            far_target = cli.base_cal_far if cli.fixed_subtypes and method == 'base_threshold' else args.target_far
            if method == 'base_threshold' and operating is not None:
                if operating['status'] == 'selected':
                    threshold = float(operating['thresholds'][i])
                    record = {
                        'threshold': threshold,
                        'target_far': operating['selected_cal_far'],
                        'calibration_rule': 'cal_operating_selection',
                        'operating_selection_status': 'selected',
                        'selection_sha256': operating['selection_sha256'],
                        'benign_holdout_rows': int((y_cal[op['threshold_fit']] == 0).sum()),
                        'cal_fit_rows': len(op['threshold_fit']),
                        'cal_holdout_rows': len(hold),
                        'threshold_fit_rows': len(op['threshold_fit']),
                        'operating_select_rows': len(op['operating_select']),
                        'threshold_floor': args.threshold_floor,
                    }
                else:
                    threshold, original = empirical_alarm_threshold(
                        p[ordered],
                        y_cal,
                        OPERATING_REFERENCE_FAR,
                        args.threshold_floor,
                        cli.min_benign_rows,
                        'disable',
                    )
                    record = {
                        **original,
                        'operating_selection_status': 'no_feasible_operating_point',
                        'selection_reason': operating.get('reason'),
                        'fixed_reference_fallback': True,
                    }
            else:
                threshold, record = empirical_alarm_threshold(
                    p[ordered], y_cal, far_target, args.threshold_floor, cli.min_benign_rows, 'disable'
                )
            val_p = p[len(merged['cal']) :]
            if len(val_p) != len(r['val']):
                raise RuntimeError('VAL scores and row identities differ')
            attack_thresholds = (
                class_thresholds[i] if method == 'base_threshold' and class_thresholds is not None else None
            )
            prediction = hierarchical_decisions(val_p, _gate_threshold(val_p, threshold, attack_thresholds))
            if attack_thresholds is not None:
                record.update(
                    class_thresholds=attack_thresholds,
                    alarm_routing=CLASS_ALARM_ROUTING,
                    threshold_role='V11 scalar reference; class thresholds used for prediction',
                )
            decisions[method].append(prediction['predicted'])
            decisions[argmax].append(val_p.argmax(axis=1))
            forecasts[method].append(prediction['attack_score'])
            forecasts[argmax].append(prediction['attack_score'])
            thresholds[method].append(threshold)
            live = any(t < 1 for t in attack_thresholds) if attack_thresholds is not None else threshold < 1
            enabled[method].append(live)
            y_val = parts['val'][1][r['val']]
            if cli.operating_guard and method == 'base_threshold':
                if operating is not None and operating.get('reference_thresholds'):
                    reference = float(operating['reference_thresholds'][i])
                else:
                    reference, _ = empirical_alarm_threshold(
                        p[ordered],
                        y_cal,
                        OPERATING_REFERENCE_FAR,
                        args.threshold_floor,
                        cli.min_benign_rows,
                        'disable',
                    )
                reference_thresholds.append(reference)
                attack = y_val > 0
                n_attack = int(attack.sum())
                proposed_missed = int(np.count_nonzero(attack & ~prediction['attack_alert']))
                reference_missed = int(np.count_nonzero(attack & (prediction['attack_score'] <= reference)))
                results['binary_operating_guard'].append(
                    {
                        **base,
                        'scope': 'receiver' if i == receiver else 'client',
                        'client': i,
                        'attack_support': n_attack,
                        'candidate_missed': proposed_missed,
                        'reference_missed': reference_missed,
                        'candidate_threshold': threshold,
                        **({'candidate_class_thresholds': attack_thresholds} if attack_thresholds is not None else {}),
                        'reference_threshold': reference,
                        'candidate_fnr': proposed_missed / n_attack if n_attack else None,
                        'reference_fnr': reference_missed / n_attack if n_attack else None,
                        'fnr_delta': (proposed_missed - reference_missed) / n_attack if n_attack else None,
                        'candidate_cal_far': (
                            operating.get('selected_cal_far') if operating is not None else cli.base_cal_far
                        ),
                        'reference_cal_far': OPERATING_REFERENCE_FAR,
                        'operating_selection_status': operating.get('status') if operating is not None else None,
                        'selection_sha256': operating.get('selection_sha256') if operating is not None else None,
                    }
                )
            results['gates'].append(
                {
                    **base,
                    'mode': method,
                    'client': i,
                    **record,
                    'gate_enabled': live,
                    'val_rows': len(y_val),
                    'val_attack_rows': int((y_val > 0).sum()),
                    'holdout_benign_feature_groups': policy['group_audit']['clients'][i][
                        'holdout_benign_feature_groups'
                    ],
                }
            )
    results['timings'].append(
        {
            **base,
            'stage': 'threshold_and_decision_all_modes',
            'elapsed_seconds': time.perf_counter() - prediction_started,
        }
    )
    if reference_thresholds:
        guard_rows = results['binary_operating_guard']
        total = sum(row['attack_support'] for row in guard_rows)
        proposed = sum(row['candidate_missed'] for row in guard_rows)
        reference = sum(row['reference_missed'] for row in guard_rows)
        guard_rows.append(
            {
                **base,
                'scope': 'overall',
                'attack_support': total,
                'candidate_missed': proposed,
                'reference_missed': reference,
                'candidate_fnr': proposed / total if total else None,
                'reference_fnr': reference / total if total else None,
                'fnr_delta': (proposed - reference) / total if total else None,
                'candidate_cal_far': (operating.get('selected_cal_far') if operating is not None else cli.base_cal_far),
                'reference_cal_far': OPERATING_REFERENCE_FAR,
                'operating_selection_status': operating.get('status') if operating is not None else None,
                'selection_sha256': operating.get('selection_sha256') if operating is not None else None,
            }
        )
    guards = (
        _guard_calibration(
            parts,
            roster,
            splits,
            union,
            probabilities,
            thresholds,
            base,
            target,
            receiver,
            args,
            cli,
            results,
            class_thresholds,
        )
        if cli.guarded_subtypes
        else None
    )
    if cli.fixed_subtypes:
        _fixed_policy(
            parts,
            names,
            roster,
            union,
            probabilities,
            decisions,
            forecasts,
            thresholds,
            enabled,
            results,
            base,
            target,
            receiver,
            states,
            args,
            evidence,
            guards,
        )
    if cli.operating_guard:
        diagnostic_modes = [mode for mode in ('fixed_federated', 'guarded_federated') if mode in decisions] or [
            'base_threshold'
        ]
        for final_mode in diagnostic_modes:
            for i, (r, merged, (_, hold)) in enumerate(zip(roster, union, splits, strict=True)):
                op = policy['operating_splits'][i] if policy.get('operating_splits') is not None else None
                hold_rows = r['cal'][hold]
                ordered = np.searchsorted(merged['cal'], hold_rows)
                base_probabilities = probabilities['base_threshold'][i]
                cal_p = base_probabilities[ordered]
                cal_scores = np.clip(np.asarray(cal_p[:, 1:], dtype=np.float64).sum(axis=1), 0.0, 1.0)
                cal_labels = parts['cal'][1][hold_rows]
                val_p = base_probabilities[len(merged['cal']) :]
                val_scores = forecasts['base_threshold'][i]
                threshold = float(thresholds['base_threshold'][i])
                attack_thresholds = class_thresholds[i] if class_thresholds is not None else None
                fit_at = op['threshold_fit'] if op is not None else slice(None)
                select_at = op['operating_select'] if op is not None else slice(0, 0)
                reference = float(reference_thresholds[i]) if reference_thresholds else None
                diagnostic = summarize_binary_diagnostics(
                    cal_scores,
                    cal_labels,
                    threshold,
                    threshold_fit_scores=cal_scores[fit_at],
                    threshold_fit_labels=cal_labels[fit_at],
                    operating_select_scores=cal_scores[select_at],
                    operating_select_labels=cal_labels[select_at],
                    cal_thresholds=_gate_threshold(cal_p, threshold, attack_thresholds),
                    threshold_fit_thresholds=_gate_threshold(cal_p[fit_at], threshold, attack_thresholds),
                    operating_select_thresholds=_gate_threshold(cal_p[select_at], threshold, attack_thresholds),
                    val_thresholds=_gate_threshold(val_p, threshold, attack_thresholds),
                    val_scores=val_scores,
                    val_labels=parts['val'][1][r['val']],
                    reference_threshold=reference,
                    target=target,
                    min_benign_rows=cli.operating_min_benign_rows if op is not None else cli.min_benign_rows,
                    val_predictions=decisions[final_mode][i],
                )
                diagnostic_id = {
                    **base,
                    'mode': final_mode,
                    'client': i,
                    'receiver': i == receiver,
                    'threshold': threshold,
                    'reference_threshold': reference,
                    'operating_selection_status': operating.get('status') if operating is not None else None,
                    'alarm_routing': CLASS_ALARM_ROUTING if attack_thresholds is not None else 'scalar',
                    'class_thresholds': attack_thresholds,
                }
                results['binary_diagnostics'].append(
                    {
                        **diagnostic_id,
                        'class_names': json.dumps({str(j): name for j, name in enumerate(names)}),
                        **diagnostic,
                    }
                )
                results['binary_class_diagnostics'].extend(
                    {**diagnostic_id, **row} for row in class_diagnostic_rows(diagnostic, names)
                )
    y = np.concatenate([parts['val'][1][r['val']] for r in roster])
    if operating is not None:
        _compare_operating_reference(
            parts,
            names,
            roster,
            splits,
            union,
            probabilities,
            reference_thresholds,
            decisions,
            forecasts,
            results,
            base,
            target,
            receiver,
            args,
            cli,
            y,
            operating,
        )
    for method, chosen in decisions.items():
        summary, by_class, confusion = score(y, np.concatenate(chosen), np.concatenate(forecasts[method]), names)
        results['metrics'].append({**base, 'mode': method, **summary})
        results['class_metrics'].extend({**base, 'mode': method, **row} for row in by_class)
        for j, counts in enumerate(confusion):
            results['confusion'].extend(
                {**base, 'mode': method, 'actual': names[j], 'predicted': label, 'count': count}
                for label, count in zip(names, counts, strict=True)
                if count
            )
    attack_support = int((y > 0).sum())
    for method, gate in enabled.items():
        muted = [g for g in results['gates'] if g['mode'] == method and not g['gate_enabled']]
        off_rows = sum(g['val_rows'] for g in muted)
        off_attacks = sum(g['val_attack_rows'] for g in muted)
        results['coverage'].append(
            {
                **base,
                'mode': method,
                'enabled_clients': sum(gate),
                'enabled_client_fraction': sum(gate) / len(gate),
                'val_rows': len(y),
                'enabled_row_fraction': (len(y) - off_rows) / len(y),
                'attack_support': attack_support,
                'attacks_on_disabled_clients': off_attacks,
                'enabled_attack_fraction': (
                    (attack_support - off_attacks) / attack_support if attack_support else None
                ),
            }
        )
    if target is not None:
        r, merged = roster[receiver], union[receiver]
        receiver_labels = parts['val'][1][r['val']]
        receiver_x = parts['val'][0]
        target_groups = _unique_rows(receiver_x, r['val'][receiver_labels == target])
        observer = TargetScoreObserver(target, receiver, receiver_labels)
        for method in (LOCAL, FEDERATED):
            # Observer uses the conditional-adapter branch names internally.
            observer(
                'conditional_local' if method == LOCAL else 'conditional_full',
                receiver,
                probabilities[method][receiver][len(merged['cal']) :],
            )
        results['target_scores'].extend(
            {**base, **x} for x in observer.summarize(decisions[LOCAL][receiver], decisions[FEDERATED][receiver])
        )
        cells = development_cells(parts, roster, decisions, target, receiver, names)
        for cell in cells:
            method = cell['mode']
            source = next(
                (k for k in FIXED_MODES if method in (FIXED_MODES[k], GUARD_MODES[k])),
                method.removesuffix('_argmax'),
            )
            if source == 'base':
                source = 'base_threshold'
            if source == 'fedavg':
                source = 'fedavg_threshold'
            target_p = probabilities[source][receiver][len(merged['cal']) :, target]
            truth = receiver_labels == target
            cell['target_feature_groups'] = target_groups
            cell['average_precision'] = float(average_precision_score(truth, target_p)) if truth.any() else None
            cell['gate_enabled'] = enabled[method][receiver] if method in enabled else None
            if method in (*FIXED_MODES.values(), *GUARD_MODES.values()):
                cell['average_precision_scope'] = 'standalone source target score on all receiver VAL; diagnostic only'
            if cli.fixed_subtypes:
                alerted = decisions['base_threshold'][receiver] > 0
                matching = truth & alerted
                cell['base_alert_target_support'] = int(matching.sum())
                cell['base_alert_target_recall_ceiling'] = float(matching.sum() / truth.sum()) if truth.any() else None
                cell['target_recall_among_base_alerted'] = (
                    float((matching & (decisions[method][receiver] == target)).sum() / matching.sum())
                    if matching.any()
                    else None
                )
            results['target_metrics'].append({**base, **cell})
        if guards is not None:
            from .revision_subtype_guard import target_attack_margin

            truth = receiver_labels == target
            alerted = decisions['base_threshold'][receiver] > 0
            positive = truth[alerted]
            for source in guards:
                p = probabilities[source][receiver][len(merged['cal']) :][alerted]
                margin = target_attack_margin(p, target)
                results['alerted_target_ranking'].append(
                    {
                        **base,
                        'mode': source,
                        'alert_rows': int(alerted.sum()),
                        'target_support': int(positive.sum()),
                        'target_margin_ap': (
                            float(average_precision_score(positive, margin)) if positive.any() else None
                        ),
                        'target_score_ap': (
                            float(average_precision_score(positive, p[:, target])) if positive.any() else None
                        ),
                    }
                )
    return results, thresholds, enabled, guards, reference_thresholds, operating


def _compare_operating_reference(
    parts,
    names,
    roster,
    splits,
    union,
    probabilities,
    reference_thresholds,
    decisions,
    forecasts,
    results,
    base,
    target,
    receiver,
    args,
    cli,
    y,
    operating,
):
    """Compare guarded federation on candidate and original V11 alert masks."""
    from .revision_subtype_guard import guarded_subtype_predictions

    ref_guards = _guard_calibration(
        parts,
        roster,
        splits,
        union,
        probabilities,
        {'base_threshold': reference_thresholds},
        base,
        target,
        receiver,
        args,
        cli,
        {'subtype_guard_cal': []},
    )
    operating['reference_subtype_guard'] = ref_guards[FEDERATED][receiver]
    reference = []
    for i, route in enumerate(union):
        start = len(route['cal'])
        base_pred = hierarchical_decisions(probabilities['base_threshold'][i][start:], reference_thresholds[i])[
            'predicted'
        ]
        eligible = (target,) if i == receiver else ()
        reference.append(
            guarded_subtype_predictions(
                base_pred, probabilities[FEDERATED][i][start:], eligible, ref_guards[FEDERATED][i]
            )
        )
    candidate = decisions['guarded_federated']
    truth = parts['val'][1][roster[receiver]['val']]

    def target_f1(predicted):
        tp = int(np.count_nonzero((truth == target) & (predicted == target)))
        denominator = int(np.count_nonzero(truth == target) + np.count_nonzero(predicted == target))
        return 2 * tp / denominator if denominator else None

    def macro_f1(predicted):
        summary, _, _ = score(y, np.concatenate(predicted), np.concatenate(forecasts['base_threshold']), names)
        return summary['macro_f1']

    for scope, metric, a, b in (
        ('receiver', 'f1', target_f1(candidate[receiver]), target_f1(reference[receiver])),
        ('overall', 'macro_f1', macro_f1(candidate), macro_f1(reference)),
    ):
        results['operating_reference_pairs'].append(
            {
                **base,
                'scope': scope,
                'metric': metric,
                'candidate': a,
                'reference': b,
                'delta': a - b if a is not None and b is not None else None,
                'selection_status': operating['status'],
                'reference_role': 'original_v11_fixed_1pct',
            }
        )


def _run_case(directory, identity, design, data, cli, screens, configuration, replay_directory):
    parts, roster, policies, audit = design
    valid = {k: p for k, p in policies.items() if p['status'] == 'eligible'}
    empty = {name: [] for name in _tables(cli)}
    if not valid:
        return {'identity': identity, 'status': 'no_eligible_cal_policy', 'audit': audit, 'tables': empty}
    receiver = audit['receiver']
    target = data.names.index(identity['target_class']) if receiver is not None else None
    args = _args(identity, cli)
    union = []
    for i, binding in enumerate(roster):
        rows = np.unique(np.concatenate([p['roster'][i]['cal'] for p in valid.values()]))
        union.append({**binding, 'cal': rows})
    checkpoint = (
        directory / 'tabm_receiver.pt'
        if cli.fixed_subtypes and cli.export_frozen and cli.primary_cal_policy in valid and receiver is not None
        else None
    )
    source_record = None
    with Resources('cpu') if cli.guarded_subtypes else nullcontext() as whole:
        if replay_directory is None:
            parts_train = parts['train']
            train = [(parts_train[0][r['train']], parts_train[1][r['train']]) for r in roster]
            with Resources(args.device) as resources:
                donor = (audit.get('partition_intervention') or {}).get('single_donor')
                donor_options = {'donor_subsets': {SINGLE_DONOR: (donor['selected_client'],)}} if donor else {}
                fit = fit_shared_heads(train, len(data.names), args, receiver=receiver, **donor_options)
                predictions, inference = _head_outputs(fit['states'], parts, union, args)
            empty['timings'].extend({**identity, **r} for r in [*fit['performance'], *inference])
            empty['communication'].extend({**identity, **r} for r in fit['communication'])
            empty['rounds'].extend({**identity, **r} for r in fit['rounds'])
            empty['timings'].append({**identity, 'stage': 'head_all_modes_profile', **resources.result})
            if (receiver is not None and not cli.head_only) or cli.fixed_subtypes:
                with Resources(args.device) if cli.guarded_subtypes else nullcontext() as detector:
                    baseline, times, comm = _detector_outputs(
                        parts,
                        union,
                        data.names,
                        data.features,
                        args,
                        train,
                        fixed=cli.fixed_subtypes,
                        checkpoint=checkpoint,
                        receiver=receiver,
                        include_fedavg=receiver is not None,
                    )
                if detector is not None:
                    empty['timings'].append({**identity, 'stage': 'tabm_and_fedavg_profile', **detector.result})
                predictions.update(baseline)
                empty['timings'].extend({**identity, **r} for r in times)
                if comm is not None:
                    empty['communication'].append({**identity, **comm})
        else:
            source_record = _read(replay_directory / 'result.json')
            if source_record['identity'] != identity or _comparable_score_route(
                source_record['audit']
            ) != _comparable_score_route(audit):
                raise ValueError('Replay case identity or CAL/VAL route differs')
            if not source_record.get('score_cache'):
                raise ValueError(f'Replay source has no private scores: {replay_directory}')
            expected = {LOCAL, FEDERATED, 'base_threshold'}
            if receiver is not None:
                expected.update((DONOR, 'fedavg_threshold'))
                if getattr(cli, 'single_donor', False):
                    expected.add(SINGLE_DONOR)
            predictions = load_score_cache(
                replay_directory,
                source_record['score_cache'],
                identity,
                _score_route(source_record['audit']),
                configuration['replay_score_binding'],
                union,
                expected_modes=expected,
                expected_classes=data.names,
            )
            fit = {'model_spec': source_record['model_spec'], 'states': {}}
            empty['communication'].extend(
                {**row, 'measurement_source': 'original_training_run'}
                for row in source_record['tables']['communication']
            )
        outcome = _evaluate_case(
            directory,
            identity,
            data,
            cli,
            screens,
            configuration,
            parts,
            audit,
            valid,
            args,
            union,
            target,
            receiver,
            fit,
            predictions,
            empty,
            checkpoint,
            replay_directory,
            source_record,
        )
        if source_record is not None:
            _replay_operating_points(identity, empty, source_record, cli)
    if whole is not None:
        accelerator = {
            key: max((r[key] for r in empty['timings'] if r.get(key) is not None), default=None)
            for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')
        }
        empty['timings'].append({**identity, 'stage': 'case_end_to_end_profile', **whole.result, **accelerator})
    return outcome


def _evaluate_case(
    directory,
    identity,
    data,
    cli,
    screens,
    configuration,
    parts,
    audit,
    valid,
    args,
    union,
    target,
    receiver,
    fit,
    predictions,
    empty,
    checkpoint,
    replay_directory,
    source_record,
):
    cache_receipt = (
        save_score_cache(
            directory,
            identity,
            _score_route(audit),
            _score_configuration(configuration),
            union,
            predictions,
            class_names=data.names,
        )
        if cli.save_scores
        else None
    )
    artifacts = {}
    evidence = {} if cli.fixed_subtypes and cli.save_row_evidence else None
    for name, policy in valid.items():
        result, thresholds, enabled, guards, reference_thresholds, operating_selection = _measure_policy(
            identity,
            name,
            policy,
            args,
            cli,
            parts,
            data.names,
            union,
            predictions,
            target,
            receiver,
            states=fit['states'],
            evidence=evidence,
        )
        for table, rows in result.items():
            empty[table].extend(rows)
        export_ready = operating_selection is None or operating_selection.get('status') == 'selected'
        if (
            cli.export_frozen
            and name == cli.primary_cal_policy
            and not cli.smoke
            and receiver is not None
            and export_ready
        ):
            from .revision_external import frozen_selection_digest, save_frozen_head

            if source_record is None and _source_receipt() != configuration['code_sha256']:
                raise RuntimeError(
                    'Source code changed after run start; refusing to export frozen receipts from mixed'
                    ' source versions'
                )
            screen = screens()
            donor = (audit.get('partition_intervention') or {}).get('single_donor')
            keep = (LOCAL, FEDERATED, DONOR, SINGLE_DONOR) if donor else (LOCAL, FEDERATED, DONOR)
            states = {m: fit['states'][m][receiver] for m in keep} if source_record is None else None
            if source_record is not None:
                inherited = source_record.get('frozen_head_artifacts', {}).get(name)
                if not inherited:
                    raise ValueError('Replay export requires a frozen checkpoint in the source case')
                for path_key, hash_key in (
                    ('file', 'sha256'),
                    ('receipt', 'receipt_sha256'),
                    ('tabm_checkpoint', 'tabm_checkpoint_sha256'),
                ):
                    if sha(Path(inherited[path_key])) != inherited[hash_key]:
                        raise ValueError(f'Replay frozen artifact hash differs: {path_key}')
                inherited_info = _read(Path(inherited['receipt']))
                if inherited_info['development_screen']['source_files_sha256'] != screen['source_files_sha256']:
                    raise ValueError('Replay frozen screen identifies different source files')
                expected_case = {k: identity[k] for k in ('dataset', 'seed', 'target_class')}
                expected_case.update(receiver=receiver, class_order=list(data.names), feature_order=list(data.features))
                if (
                    any(inherited_info.get(k) != v for k, v in expected_case.items())
                    or inherited_info['settings'].get('route_sha256') != audit['route_sha256']
                    or inherited_info['settings'].get('cal_partition_sha256') != policy['partition_sha256']
                ):
                    raise ValueError('Replay frozen checkpoint has a different case or CAL/VAL route')
                with np.load(Path(inherited['file']), allow_pickle=False) as saved:
                    states = {m: saved[m] for m in keep}
                copyfile(Path(inherited['tabm_checkpoint']), checkpoint)
            invariant = {
                **{k: identity[k] for k in ('scenario', 'alpha', 'clients', 'target_class')},
                'cal_policy': name,
                'head_rounds': args.rounds,
                'head_epochs': args.head_epochs,
                'head_lr': args.head_lr,
                'head_batch_size': args.head_batch_size,
                'training_device': args.device,
                'head_objective': args.head_objective,
                'detector_objective': args.detector_objective,
                'detector_objective_settings': objective_settings(args.detector_objective),
                'head_clip': args.clip,
                'secure_scale': args.secure_scale,
                'threshold_floor': args.threshold_floor,
                'target_far': args.target_far,
                'min_benign_rows': cli.min_benign_rows,
                'source_sha256': _source_receipt(),
            }
            frozen_options = {}
            if cli.fixed_subtypes:
                base_value = thresholds['base_threshold'][receiver]
                is_muted = not enabled['base_threshold'][receiver]
                routing = GUARD_ROUTING if guards is not None else FIXED_ROUTING
                frozen_options['fixed_gate'] = {
                    'checkpoint': str(checkpoint),
                    'threshold': base_value,
                    'gate_disabled': is_muted,
                    'routing': routing,
                }
                if guards is not None:
                    frozen_options['fixed_gate']['subtype_guards'] = {m: guards[m][receiver] for m in keep}
                if reference_thresholds:
                    frozen_options['fixed_gate'].update(
                        reference_threshold=reference_thresholds[receiver],
                        max_operating_fnr_increase=cli.max_operating_fnr_increase,
                    )
                    invariant.update(
                        guard_min_check_groups=cli.guard_min_check_groups,
                        guard_check_tolerance=cli.guard_check_tolerance,
                        operating_reference_cal_far=OPERATING_REFERENCE_FAR,
                    )
                if cli.operating_guard and operating_selection is not None:
                    group = policy.get('operating_audit') or {}
                    operating_meta = {
                        'schema': operating_selection['schema'],
                        'status': operating_selection['status'],
                        'reason': operating_selection.get('reason'),
                        'selection_rule': operating_selection['selection_rule'],
                        'candidate_grid': list(operating_selection.get('candidate_grid', [])),
                        'constraints': operating_selection.get('constraints', {}),
                        'selected_cal_far': operating_selection.get('selected_cal_far'),
                        'reference_cal_far': operating_selection.get('reference_cal_far'),
                        'receiver_client': receiver,
                        'receiver_threshold': float(base_value),
                        'receiver_reference_threshold': (
                            float(reference_thresholds[receiver]) if reference_thresholds else None
                        ),
                        'thresholds': [float(value) for value in operating_selection.get('thresholds', [])],
                        'reference_thresholds': [
                            float(value) for value in operating_selection.get('reference_thresholds', [])
                        ],
                        'cal_group_assignment_sha256': group.get('assignment_sha256'),
                        'group_overlap': group.get('global_exact_feature_overlap'),
                        'reference_subtype_guard': operating_selection['reference_subtype_guard'],
                        'selection_sha256': None,
                        'support': operating_selection.get('support', {}),
                    }
                    if cli.operating_policy == 'class-select':
                        operating_meta.update(
                            class_thresholds=operating_selection['class_thresholds'],
                            class_routing=CLASS_ALARM_ROUTING,
                        )
                        invariant.update(class_routing=CLASS_ALARM_ROUTING)
                    operating_meta['selection_sha256'] = frozen_selection_digest(operating_meta)
                    frozen_options['fixed_gate']['operating_selection'] = operating_meta
                    invariant.update(
                        operating_candidates=list(operating_selection.get('candidate_grid', [])),
                        operating_constraints=operating_selection.get('constraints', {}),
                        operating_select_fraction=cli.operating_select_fraction,
                        operating_min_benign_rows=cli.operating_min_benign_rows,
                        operating_min_select_benign=cli.operating_min_select_benign,
                        operating_min_select_attack=cli.operating_min_select_attack,
                        operating_min_receiver_attack=cli.operating_min_receiver_attack,
                        operating_min_attack_coverage=cli.operating_min_attack_coverage,
                    )
                invariant.update(
                    routing=routing,
                    base_cal_far=cli.base_cal_far,
                    base_inference_device='cpu',
                    base_batch_size=args.batch_size,
                )
            artifacts[name] = save_frozen_head(
                directory / 'head_frozen.npz',
                dataset=identity['dataset'],
                names=data.names,
                features=data.features,
                seed=identity['seed'],
                receiver=receiver,
                target_class=identity['target_class'],
                vectors=states,
                thresholds={m: thresholds['base_threshold' if cli.fixed_subtypes else m][receiver] for m in keep},
                development_screen=screen,
                settings={
                    'protocol': invariant,
                    'route_sha256': audit['route_sha256'],
                    **({'single_donor_client': donor['selected_client']} if donor else {}),
                    'cal_partition_sha256': policy['partition_sha256'],
                    'gate_disabled': {
                        m: not enabled['base_threshold' if cli.fixed_subtypes else m][receiver] for m in keep
                    },
                },
                **frozen_options,
            )
    evidence_receipt = None
    if evidence is not None:
        path = directory / 'fixed_subtype_rows_private.npz'
        temp = directory / 'fixed_subtype_rows_partial.npz'
        np.savez_compressed(temp, **evidence)
        temp.replace(path)
        evidence_receipt = {
            'file': path.name,
            'sha256': sha(path),
            'class_order': list(data.names),
            'val_identity_scope': 'row positions within ordered audited VAL role',
            'route_sha256': audit['route_sha256'],
            'routing': GUARD_ROUTING if cli.guarded_subtypes else FIXED_ROUTING,
        }
    return {
        'identity': identity,
        'status': 'evaluated',
        'audit': audit,
        'model_spec': fit['model_spec'],
        'tables': empty,
        'frozen_head_artifacts': artifacts,
        'fixed_subtype_evidence': evidence_receipt,
        'score_cache': cache_receipt,
    }


def _record_case(rows, outcome, fields):
    for table, values in outcome['tables'].items():
        rows[table].extend({**fields, **v} for v in values)


def _contrast(rows, groups, criteria, primary_cal_policy):
    comparisons, by_group, intervals, flags = [], defaultdict(list), [], []
    sources = {
        'overall': ('metrics', METRICS),
        'receiver': ('target_metrics', ('f1', 'precision', 'recall', 'average_precision', 'receiver_far')),
        'coverage': (
            'coverage',
            ('enabled_client_fraction', 'enabled_row_fraction', 'enabled_attack_fraction'),
        ),
        'alerted_ranking': ('alerted_target_ranking', ('target_margin_ap', 'target_score_ap')),
    }
    for scope, (table, metrics) in sources.items():
        observations = defaultdict(dict)
        for r in rows.get(table, []):
            key = tuple(r[k] for k in (*CONDITIONS, 'seed'))
            observations[key][r['mode']] = r
        for key, observed in observations.items():
            condition = dict(zip(CONDITIONS, key[:-1], strict=True))
            contrasts = dict(PAIRINGS)
            if scope == 'overall':
                contrasts['local_gate_vs_raw'] = ('base_threshold', 'base')
            for contrast, (candidate, reference) in contrasts.items():
                if not {candidate, reference} <= observed.keys():
                    continue
                for metric in metrics:
                    a, b = observed[candidate][metric], observed[reference][metric]
                    delta = a - b if a is not None and b is not None else None
                    comparisons.append(
                        {
                            **condition,
                            'seed': key[-1],
                            'scope': scope,
                            'contrast': contrast,
                            'metric': metric,
                            'candidate': a,
                            'reference': b,
                            'delta': delta,
                        }
                    )
                    by_group[(*key[:-1], scope, contrast, metric)].append(comparisons[-1])
    index = {}
    for key, pair in by_group.items():
        stats = summarize([r['delta'] for r in pair], bounds=(-1, 1))
        values = [r['candidate'] for r in pair if r['candidate'] is not None]
        entry = {
            **dict(zip((*CONDITIONS, 'scope', 'contrast', 'metric'), key, strict=True)),
            'planned_seeds': len(groups[key[:-3]]),
            'observed_seeds': len(pair),
            'candidate_max': max(values, default=None),
            'candidate_min': min(values, default=None),
            **stats,
        }
        intervals.append(entry)
        index[key] = entry
    for key, statuses in groups.items():
        identity = dict(zip(CONDITIONS, key, strict=True))
        if identity['target_class'] is None:
            continue
        f1 = index.get((*key, 'receiver', 'aggregation', 'f1'))
        far = index.get((*key, 'receiver', 'aggregation', 'receiver_far'))
        overall_far = index.get((*key, 'overall', 'aggregation', 'far'))
        fnr = index.get((*key, 'overall', 'aggregation', 'fnr'))
        coverage = index.get((*key, 'coverage', 'aggregation', 'enabled_attack_fraction'))
        enough = len(statuses) >= criteria['minimum_paired_seeds'] and all(
            v is not None and v['seeds_with_metric'] == len(statuses) for v in (f1, far, overall_far, fnr, coverage)
        )
        receiver_gates = [
            r['gate_enabled']
            for r in rows['target_metrics']
            if tuple(r[k] for k in CONDITIONS) == key and r['mode'] in (LOCAL, FEDERATED)
        ]
        gate_ok = len(receiver_gates) == len(statuses) * 2 and all(receiver_gates)
        meets = (
            enough
            and gate_ok
            and f1['ci95_low'] > 0
            and f1['mean'] >= criteria['minimum_target_f1_effect_pp'] / 100
            and all(
                v['candidate_max'] <= criteria['max_absolute_far'] and v['ci95_high'] <= criteria['max_far_increase']
                for v in (far, overall_far)
            )
            and fnr['ci95_high'] <= criteria['max_fnr_increase']
            and coverage['candidate_min'] >= criteria['minimum_attack_coverage']
            and coverage['ci95_low'] >= -criteria['max_coverage_loss']
        )
        flags.append(
            {
                **identity,
                'primary_cal_policy': identity['cal_policy'] == primary_cal_policy,
                'planned_seeds': len(statuses),
                'all_receiver_gates_enabled': gate_ok,
                'target_f1_delta_mean': f1['mean'] if f1 else None,
                'target_f1_delta_ci95_low': f1['ci95_low'] if f1 else None,
                'target_f1_delta_ci95_high': f1['ci95_high'] if f1 else None,
                'max_receiver_far': far['candidate_max'] if far else None,
                'max_overall_far': overall_far['candidate_max'] if overall_far else None,
                'overall_fnr_delta_ci95_high': fnr['ci95_high'] if fnr else None,
                'min_enabled_attack_fraction': coverage['candidate_min'] if coverage else None,
                'status': (
                    'insufficient_paired_support'
                    if not enough
                    else 'development_signal_requires_external_confirmation'
                    if meets
                    else 'criterion_not_met'
                ),
            }
        )
    return comparisons, intervals, flags


def _fixed_assessment(
    intervals,
    groups,
    criteria,
    primary_cal_policy,
    *,
    prefix='fixed',
    cal_check=(),
    operating=None,
    selection=None,
    nonregression=None,
):
    index = {tuple(r[k] for k in (*CONDITIONS, 'scope', 'contrast', 'metric')): r for r in intervals}
    calibration = defaultdict(list)
    for row in cal_check:
        if row['mode'] == 'guarded_federated' and 'fit_status' in row:
            calibration[tuple(row[k] for k in CONDITIONS)].append(row)
    checks = []
    for key, statuses in groups.items():
        if key[4] is None:
            continue
        selection_rows = (selection or {}).get(key, [])
        selection_failures = sum(row.get('status') != 'selected' for row in selection_rows)
        selection_missing = max(0, len(statuses) - len(selection_rows))
        selection_required = selection is not None
        chosen = {row['seed']: row.get('selected_cal_far') for row in selection_rows if 'seed' in row}
        selected_rates = {rate for rate in chosen.values() if rate is not None}
        selection_complete = not selection_required or (not selection_failures and not selection_missing)

        def interval(scope, contrast, metric):
            return index.get((*key, scope, contrast, metric), {})

        aggregation = interval('receiver', f'{prefix}_aggregation', 'f1')
        against_base = interval('receiver', f'{prefix}_against_base', 'f1')
        against_donor = interval('receiver', f'{prefix}_receiver_participation', 'f1')
        macro = interval('overall', f'{prefix}_against_base', 'macro_f1')
        far = interval('overall', f'{prefix}_against_base', 'far')
        fnr = interval('overall', f'{prefix}_against_base', 'fnr')
        receiver_far = interval('receiver', f'{prefix}_against_base', 'receiver_far')
        coverage = interval('coverage', f'{prefix}_against_base', 'enabled_attack_fraction')
        all_stats = (aggregation, against_base, macro, far, fnr, receiver_far, coverage)
        complete = (
            len(statuses) >= criteria['minimum_paired_seeds']
            and all(r.get('seeds_with_metric') == len(statuses) for r in all_stats)
            and selection_complete
        )
        identical = all(
            r.get('mean') == 0 and r.get('seed_sd') in (0, None) and r.get('seeds_with_metric') == len(statuses)
            for r in (far, fnr, receiver_far, coverage)
        )
        gain_local = (
            complete
            and aggregation['ci95_low'] > 0
            and aggregation['mean'] >= criteria['minimum_target_f1_effect_pp'] / 100
        )
        gain_base = (
            complete
            and against_base['ci95_low'] > 0
            and against_base['mean'] >= FIXED_CRITERIA['minimum_f1_gain_over_base_pp'] / 100
        )
        macro_guard = complete and macro['ci95_low'] >= -FIXED_CRITERIA['maximum_macro_f1_loss_pp'] / 100
        coverage_guard = complete and coverage['candidate_min'] >= criteria['minimum_attack_coverage']
        usable = complete and identical and gain_local and gain_base and macro_guard and coverage_guard
        max_far = max(
            (r['candidate_max'] for r in (far, receiver_far) if r.get('candidate_max') is not None),
            default=None,
        )
        absolute = (
            max_far is not None
            and max_far <= criteria['max_absolute_far']
            and all(r.get('seeds_with_metric') == len(statuses) for r in (far, receiver_far))
        )
        operating_stats = {} if operating is None else operating.get(key, {})
        fnr_guard = (
            True
            if operating is None
            else bool(operating_stats) and all(v for k, v in operating_stats.items() if k.endswith('_guard_met'))
        )
        reference_stats = {} if nonregression is None else nonregression[key]
        reference_guard = nonregression is None or reference_stats['subtype_nonregression_met']
        status = (
            'no_feasible_operating_point'
            if selection_required and selection_failures
            else (
                'insufficient_paired_support'
                if not complete
                else (
                    'subtype_gain_operating_fnr_not_met'
                    if usable and absolute and not fnr_guard
                    else (
                        'subtype_gain_reference_regression'
                        if usable and absolute and not reference_guard
                        else (
                            'development_signal_requires_external_confirmation'
                            if usable and absolute
                            else 'subtype_gain_absolute_far_not_met'
                            if usable
                            else 'criterion_not_met'
                        )
                    )
                )
            )
        )
        audit = calibration[key]
        checks.append(
            {
                **dict(zip(CONDITIONS, key, strict=True)),
                'primary_cal_policy': key[-1] == primary_cal_policy,
                'planned_seeds': len(statuses),
                'status': status,
                'binary_metrics_unchanged': identical,
                'subtype_signal': usable,
                'paired_support_met': complete,
                'gain_over_local_met': gain_local,
                'gain_over_base_met': gain_base,
                'macro_f1_guard_met': macro_guard,
                'enabled_attack_coverage_met': coverage_guard,
                'positive_interval_vs_donor': (against_donor.get('ci95_low') or 0) > 0,
                'operating_selection_required': selection_required,
                'operating_selection_complete': selection_complete,
                'operating_selection_failures': selection_failures,
                'operating_selection_missing': selection_missing,
                'selected_cal_far': (
                    next(iter(selected_rates)) if selection_complete and len(selected_rates) == 1 else None
                ),
                'selected_cal_far_by_seed': json.dumps(chosen, sort_keys=True) if selection_required else None,
                **(
                    {
                        'cal_check_seeds_with_negative_groups': sum(r['check_negative_groups'] > 0 for r in audit),
                        'cal_check_seeds_above_budget': sum(r['check_meets_empirical_budget'] is False for r in audit),
                        'cal_guard_disabled_seeds': sum(
                            r.get('effective_status', r['fit_status']) != 'calibrated' for r in audit
                        ),
                        'cal_check_fallback_seeds': sum(
                            r['fit_status'] == 'calibrated' and r.get('effective_status', 'calibrated') != 'calibrated'
                            for r in audit
                        ),
                    }
                    if prefix == 'guarded'
                    else {}
                ),
                **operating_stats,
                **reference_stats,
                'observed_absolute_far_met': absolute,
                'max_receiver_far': receiver_far.get('candidate_max'),
                'max_overall_far': far.get('candidate_max'),
                'min_enabled_attack_fraction': coverage.get('candidate_min'),
                **{
                    f'{label}_{field}': record.get(field)
                    for label, record in (
                        ('f1_vs_local', aggregation),
                        ('f1_vs_base', against_base),
                        ('f1_vs_donor', against_donor),
                        ('macro_f1_vs_base', macro),
                    )
                    for field in ('mean', 'ci95_low', 'ci95_high')
                },
            }
        )
    return checks


def _operating_results(rows, conditions, cli):
    paired = defaultdict(list)
    for row in rows:
        if row['scope'] in ('overall', 'receiver'):
            paired[(tuple(row[k] for k in CONDITIONS), row['scope'])].append(row)
    answer = {}
    for condition in conditions:
        scopes = ('overall', 'receiver') if condition[4] is not None else ('overall',)
        answer[condition] = {
            f'{scope}_{name}': value
            for scope in scopes
            for name, value in summarize_fnr_guard(
                paired[condition, scope], cli.seeds, cli.max_operating_fnr_increase
            ).items()
        }
    return answer


def _reference_results(rows, conditions, seeds):
    by_condition = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in CONDITIONS)
        metric_rows = by_condition[key].setdefault(row['metric'], {})
        if row['seed'] in metric_rows:
            raise ValueError('Duplicate operating reference metric for seed')
        metric_rows[row['seed']] = row['delta']
    report = {}
    for key in conditions:
        results = {}
        for metric in ('f1', 'macro_f1'):
            observed = by_condition[key].get(metric, {})
            stats = summarize([observed.get(seed) for seed in seeds], bounds=(-1, 1))
            results.update({f'reference_{metric}_{field}': value for field, value in stats.items()})
            results[f'reference_{metric}_nonregression_met'] = (
                stats['seeds_with_metric'] == len(seeds) and stats['mean'] is not None and stats['mean'] >= -1e-12
            )
        results['subtype_nonregression_met'] = all(
            results[f'reference_{m}_nonregression_met'] for m in ('f1', 'macro_f1')
        )
        report[key] = results
    return report


def _replay_operating_points(identity, empty, original, cli):
    """Compare common operating points with matched models and row assignments."""
    kept = ('base_threshold', *GUARD_MODES.values())
    for table, scope, metrics in (
        ('metrics', 'overall', ('far', 'fnr', 'alert_rate', 'macro_f1')),
        ('coverage', 'overall', ('enabled_attack_fraction',)),
        (
            'target_metrics',
            'receiver',
            ('receiver_far', 'base_alert_target_recall_ceiling', 'precision', 'recall', 'f1'),
        ),
    ):
        before = {(r['cal_policy'], r['mode']): r for r in original['tables'][table] if r['mode'] in kept}
        for row in empty[table]:
            key = (row['cal_policy'], row['mode'])
            if key not in before:
                continue
            for metric in metrics:
                old, new = before[key][metric], row[metric]
                empty['replay_operating_points'].append(
                    {
                        **identity,
                        'cal_policy': key[0],
                        'mode': key[1],
                        'scope': scope,
                        'metric': metric,
                        'source': old,
                        'replay': new,
                        'delta': new - old if new is not None and old is not None else None,
                        'replay_cal_far': cli.base_cal_far,
                        'replay_guard_rewrite_rate': cli.guard_rewrite_rate,
                    }
                )


def _coverage_changes(rows):
    by_scope = defaultdict(dict)
    for r in rows['coverage']:
        by_scope[tuple(r[k] for k in CASE_ID) + (r['mode'],)][r['cal_policy']] = r
    metrics = {(tuple(r[k] for k in CASE_ID) + (r['mode'],), r['cal_policy']): r for r in rows['metrics']}
    changes = []
    for key, policies in by_scope.items():
        if not set(POLICIES) <= policies.keys():
            continue
        a, b = (policies[p] for p in POLICIES)
        observed = {
            f'{policy}_{metric}': metrics[key, policy][metric]
            for policy in POLICIES
            for metric in ('far', 'fnr', 'alert_rate', 'ece', 'macro_f1')
        }
        changes.append(
            {
                **dict(zip((*CASE_ID, 'mode'), key, strict=True)),
                **observed,
                'natural_enabled_row_fraction': a['enabled_row_fraction'],
                'reserved_enabled_row_fraction': b['enabled_row_fraction'],
                'natural_attacks_on_disabled_clients': a['attacks_on_disabled_clients'],
                'reserved_attacks_on_disabled_clients': b['attacks_on_disabled_clients'],
            }
        )
    return changes


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=('train-absence', 'scale'), default='train-absence')
    parser.add_argument('--fixed-subtypes', action='store_true', help='Use common TabM alarms for subtype comparisons')
    parser.add_argument('--guarded-subtypes', action='store_true', help='Add the V10 CAL-only subtype margin guard')
    parser.add_argument(
        '--operating-guard', action='store_true', help='Enable V11 CAL-check fallback and FNR criterion'
    )
    parser.add_argument(
        '--operating-policy',
        choices=('fixed', *SELECTING_POLICIES),
        default='fixed',
        help='Use the original CAL threshold or explicitly evaluate a CAL-only candidate',
    )
    parser.add_argument('--guard-min-check-groups', type=int)
    parser.add_argument('--guard-check-tolerance', type=float, default=0.02)
    parser.add_argument('--max-operating-fnr-increase', type=float, default=0.01)
    parser.add_argument(
        '--operating-candidates',
        nargs='+',
        type=float,
        help='Nominal CAL FAR grid; selected loosest feasible candidate',
    )
    parser.add_argument(
        '--operating-min-benign-rows',
        type=int,
        default=200,
        help='Minimum threshold-fit benign rows for a client-local 1% quantile',
    )
    parser.add_argument(
        '--operating-select-fraction',
        type=float,
        default=0.5,
        help='Fraction of CAL holdout groups assigned to operating-select',
    )
    parser.add_argument('--operating-min-select-benign', type=int, default=30)
    parser.add_argument('--operating-min-select-attack', type=int, default=20)
    parser.add_argument('--operating-min-receiver-attack', type=int, default=10)
    parser.add_argument('--operating-min-attack-coverage', type=float, default=0.5)
    parser.add_argument(
        '--operating-min-route-benign',
        type=int,
        help='Minimum benign rows in both CAL subsets for an individual class route',
    )
    parser.add_argument('--guard-rewrite-rate', type=float, default=0.05)
    parser.add_argument('--guard-min-negative-groups', type=int)
    parser.add_argument('--head-objective', choices=HEAD_OBJECTIVES, default='multiclass')
    parser.add_argument(
        '--detector-objective',
        choices=DETECTOR_OBJECTIVES,
        default='prior_adjusted',
        help='TRAIN-only TabM objective; fit and evaluate in a separate matched run',
    )
    parser.add_argument('--all-targets', action='store_true', help='Use the full six-target V10 design')
    parser.add_argument(
        '--save-scores',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Keep private aligned CAL and VAL scores (default on for V10 controlled runs)',
    )
    parser.add_argument('--replay-from', type=Path, help='Re-evaluate completed V10 score cache without model fitting')
    parser.add_argument(
        '--base-cal-far',
        type=float,
        default=0.01,
        help='Prospective CAL quantile target for common TabM alarms',
    )
    parser.add_argument(
        '--save-row-evidence', action='store_true', help='Keep private aligned subtype scores and alarm masks'
    )
    parser.add_argument('--dataset', nargs='+', choices=DATASETS)
    parser.add_argument('--target', action='append', help='Exact target name; use with a single dataset')
    parser.add_argument('--target-count', type=int, default=3)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--receiver', type=int, default=0)
    parser.add_argument('--cal-policies', nargs='+', choices=POLICIES)
    parser.add_argument('--primary-cal-policy', choices=POLICIES)
    parser.add_argument('--min-benign-rows', type=int)
    parser.add_argument('--min-donor-groups', type=int, default=5)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--head-epochs', type=int)
    parser.add_argument('--detector-epochs', type=int)
    parser.add_argument('--head-lr', type=float, default=0.05)
    parser.add_argument('--head-clip', type=float, default=1.0)
    parser.add_argument('--head-batch-size', type=int, default=4096)
    parser.add_argument('--threshold-floor', type=float, default=0.5)
    parser.add_argument('--scale-train-total', type=int, default=100_000)
    parser.add_argument('--scale-cal-total', type=int, default=16_500)
    parser.add_argument('--scale-train-per-client', type=int, default=33_000)
    parser.add_argument('--scale-cal-per-client', type=int, default=5_500)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Synthetic inputs, no research evidence')
    parser.add_argument('--check-data', action='store_true', help='Real partition/policy preflight; no training')
    parser.add_argument('--head-only', action='store_true', help='Pilot only; omit TabM detector and FedAvg')
    parser.add_argument(
        '--single-donor',
        action='store_true',
        default=argparse.SUPPRESS,
        help='Add the TRAIN-selected single-donor development control',
    )
    parser.add_argument(
        '--export-frozen',
        action='store_true',
        help='Export primary-policy receiver head states for external evaluation',
    )
    parser.add_argument('--resume', action='store_true')
    return parser


def _normalize(parser, cli):
    scale = cli.task == 'scale'
    if cli.operating_guard:
        cli.guarded_subtypes = True
    if cli.operating_candidates is None:
        cli.operating_candidates = list(
            CLASS_CANDIDATES if cli.operating_policy == 'class-select' else DEFAULT_CANDIDATES
        )
    if cli.operating_min_route_benign is None:
        cli.operating_min_route_benign = 3 if cli.smoke else 30
    if cli.guarded_subtypes:
        cli.fixed_subtypes = True
    if cli.save_scores is None:
        cli.save_scores = cli.guarded_subtypes and not (scale or cli.replay_from or cli.check_data)
    if cli.guard_min_negative_groups is None:
        cli.guard_min_negative_groups = 3 if cli.smoke else 30
    if cli.guard_min_check_groups is None:
        cli.guard_min_check_groups = 3 if cli.smoke else 30
    if cli.smoke:
        cli.operating_min_benign_rows = min(cli.operating_min_benign_rows, 8)
        cli.operating_min_select_benign = min(cli.operating_min_select_benign, 3)
        cli.operating_min_select_attack = min(cli.operating_min_select_attack, 3)
        cli.operating_min_receiver_attack = min(cli.operating_min_receiver_attack, 3)
    cli.dataset = cli.dataset or (
        ['CIC-IDS-2017'] if cli.smoke else ['CIC-UNSW-NB15'] if scale else list(FLOW_DATASETS)
    )
    cli.seeds = cli.seeds or ([42] if cli.smoke else list(SEEDS))
    if cli.cal_policies is None:
        cli.cal_policies = ['natural'] if scale else list(POLICIES)
    cli.primary_cal_policy = cli.primary_cal_policy or ('reserved' if 'reserved' in cli.cal_policies else 'natural')
    cli.min_benign_rows = (8 if cli.smoke else 1000) if cli.min_benign_rows is None else cli.min_benign_rows
    cli.rounds = (1 if cli.smoke else 3) if cli.rounds is None else cli.rounds
    cli.head_epochs = (cli.rounds if cli.smoke else max(5, cli.rounds)) if cli.head_epochs is None else cli.head_epochs
    cli.detector_epochs = (
        (cli.rounds if cli.smoke else max(5, cli.rounds)) if cli.detector_epochs is None else cli.detector_epochs
    )
    if (
        len(set(cli.dataset)) != len(cli.dataset)
        or len(set(cli.seeds)) != len(cli.seeds)
        or any(not 0 <= s <= 2**32 - 1 for s in cli.seeds)
    ):
        parser.error('Use distinct datasets and nonnegative uint32 seeds')
    if (
        len(set(cli.cal_policies)) != len(cli.cal_policies)
        or cli.primary_cal_policy not in cli.cal_policies
        or not 0 <= cli.receiver < 3
        or cli.target_count < 1
        or cli.min_benign_rows < 1
        or cli.min_donor_groups < 1
        or cli.rounds < 1
        or cli.head_epochs < cli.rounds
        or cli.detector_epochs < cli.rounds
        or cli.head_batch_size < 1
        or not np.isfinite(cli.head_lr)
        or cli.head_lr <= 0
        or not np.isfinite(cli.head_clip)
        or cli.head_clip <= 0
        or not 0 <= cli.threshold_floor < 1
        or min(cli.scale_train_total, cli.scale_cal_total, cli.scale_train_per_client, cli.scale_cal_per_client) < 20
    ):
        parser.error('Invalid receiver, training budget or calibration options')
    if cli.target and (len(cli.dataset) != 1 or scale):
        parser.error('An explicit target is supported only for one dataset with train-absence')
    if cli.operating_policy in SELECTING_POLICIES and (not cli.operating_guard or scale or cli.base_cal_far != 0.01):
        parser.error(
            'CAL selection requires V11 train-absence and uses --operating-candidates instead of' ' --base-cal-far'
        )
    if (
        not 0 <= cli.guard_rewrite_rate < 1
        or cli.guard_min_negative_groups < 1
        or not 0 <= cli.guard_check_tolerance < 1
        or cli.guard_min_check_groups < 1
        or not 0 <= cli.max_operating_fnr_increase <= 1
        or not 0 < cli.operating_select_fraction < 1
        or not cli.operating_candidates
        or len(set(cli.operating_candidates)) != len(cli.operating_candidates)
        or any(not 0 < value < 1 for value in cli.operating_candidates)
        or cli.operating_min_select_benign < 1
        or cli.operating_min_select_attack < 1
        or cli.operating_min_receiver_attack < 1
        or cli.operating_min_benign_rows < 1
        or cli.operating_min_route_benign < 1
        or not 0 <= cli.operating_min_attack_coverage <= 1
        or (cli.all_targets and (not cli.guarded_subtypes or cli.target or scale))
        or (not cli.guarded_subtypes and (cli.save_scores or cli.replay_from or cli.head_objective != 'multiclass'))
    ):
        parser.error('Invalid V10 guard budget, objective or target options')
    if (cli.replay_from and (cli.check_data or cli.save_scores)) or (cli.check_data and cli.save_scores):
        parser.error('Replay uses existing scores and cannot save another cache or run preflight')
    if scale and (cli.cal_policies != ['natural'] or cli.export_frozen):
        parser.error('Scale uses natural CAL to preserve per-client quotas and does not export a receiver head')
    if not scale and not cli.smoke and not set(cli.dataset) <= set(FLOW_DATASETS):
        parser.error('Use traffic datasets for controlled target absence')
    if cli.fixed_subtypes and cli.head_only:
        parser.error('Fixed subtype comparison requires the TabM detector')
    if not 0 < cli.base_cal_far < 1 or not cli.fixed_subtypes and (cli.base_cal_far != 0.01 or cli.save_row_evidence):
        parser.error('Common CAL target and private row evidence require fixed subtype evaluation')
    if cli.export_frozen and (cli.head_only or cli.smoke or cli.check_data):
        parser.error('Export requires a real matched training run')
    if getattr(cli, 'single_donor', False) and (scale or not cli.guarded_subtypes):
        parser.error('The single-donor control requires guarded train-absence evaluation')
    if getattr(cli, 'single_donor', False) and cli.export_frozen and cli.operating_policy != 'cal-select':
        parser.error('Single-donor freezing requires scalar CAL selection')
    if cli.resume and not cli.out:
        parser.error('--resume requires the explicit original --out directory')
    cli.replay_device = None
    if cli.replay_from:
        configured = _read(cli.replay_from / 'run.json')['settings']
        cli.device = configured['device']
        cli.replay_device = 'cpu'
    elif cli.device == 'auto':
        cli.device = 'cpu' if cli.smoke or not torch.cuda.is_available() else 'cuda'
    if cli.device == 'cuda' and not cli.replay_device and not cli.check_data and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this Python environment')
    cli.data_root = cli.data_root.resolve()
    version = _run_version(cli)
    root = (
        cli.out or ROOT / 'outputs' / f'revision_v{version}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
    ).resolve()
    if cli.replay_from:
        cli.replay_from = cli.replay_from.resolve()
        if root.is_relative_to(cli.replay_from) or cli.replay_from.is_relative_to(root):
            parser.error('Replay must write to a separate output tree')
    if root == cli.data_root or root.is_relative_to(cli.data_root):
        parser.error('Output must be outside raw dataset storage')
    return root


def _screen_provider(root, dataset, data, parts):
    checked = None

    def screen():
        nonlocal checked
        if checked is None:
            from .revision_external import load_development_screen, prepare_development_screen

            path = root / dataset / 'development_screen.npy'
            expected = data.audit['source_files_sha256']
            receipt = path.with_suffix('.json')
            if path.exists() and not receipt.is_file():
                path.unlink()
            checked = (
                load_development_screen(path) if path.exists() else prepare_development_screen(path, parts, expected)
            )
            if checked['source_files_sha256'] != expected:
                raise ValueError('Development screen source differs from the pinned dataset')
        return checked

    return screen


def _case_output(directory, identity, design, data, cli, screens, configuration, replay_directory):
    if _source_receipt() != configuration['code_sha256']:
        raise ValueError('Source files changed during the run; restart with a new output directory')
    path = directory / 'result.json'
    if path.is_file():
        record = _read(path)
        if record['identity'] != identity or record['audit'] != design[-1]:
            raise ValueError(f'Reconstructed case configuration or route differs from {path}')
        evidence = record.get('fixed_subtype_evidence')
        if evidence is not None and sha(directory / evidence['file']) != evidence['sha256']:
            raise ValueError(f'Private row evidence differs from {path}')
        cache = record.get('score_cache')
        if cache is not None and sha(directory / cache['file']) != cache['sha256']:
            raise ValueError(f'Private score cache differs from {path}')
        for frozen in record.get('frozen_head_artifacts', {}).values():
            if 'tabm_checkpoint' in frozen and sha(Path(frozen['tabm_checkpoint'])) != frozen['tabm_checkpoint_sha256']:
                raise ValueError(f'Frozen TabM evidence differs from {path}')
            if (
                sha(Path(frozen['file'])) != frozen['sha256']
                or sha(Path(frozen['receipt'])) != frozen['receipt_sha256']
            ):
                raise ValueError(f'Frozen model evidence differs from {path}')
            if _read(Path(frozen['receipt']))['development_screen'] != screens():
                raise ValueError(f'Frozen development screen differs from {path}')
        return record
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        'head_frozen.npz',
        'head_frozen.json',
        'tabm_receiver.pt',
        'fixed_subtype_rows_partial.npz',
        'fixed_subtype_rows_private.npz',
        'private_cal_val_scores_partial.npz',
        'private_cal_val_scores.npz',
    ):
        unfinished = directory / name
        if unfinished.is_file():
            unfinished.unlink()
    if cli.check_data:
        record = {
            'identity': identity,
            'status': 'preflight_only',
            'audit': design[-1],
            'tables': {name: [] for name in _tables(cli)},
        }
    else:
        record = _run_case(directory, identity, design, data, cli, screens, configuration, replay_directory)
    if _source_receipt() != configuration['code_sha256']:
        raise ValueError('Source files changed while evaluating a case; no result was committed')
    _atomic_json(path, record)
    return record


def _operating_criteria(cli):
    """Describe the applied detector guard and matched reference."""
    criteria = {
        'reference_cal_far': OPERATING_REFERENCE_FAR,
        'max_fnr_increase': cli.max_operating_fnr_increase,
        'scope': 'receiver and overall',
        'criterion': '95% paired seed interval upper bound <= max_fnr_increase',
        'guard_check_rule': 'prespecified observed CAL-check rate and minimum support, else revert to TabM',
    }
    if cli.operating_policy in SELECTING_POLICIES:
        criteria.update(
            reference_scope='original V11 parent holdout may include operating-select CAL rows',
            nonregression='matched reference guarded head; target F1 and macro F1 paired mean deltas >= 0',
        )
    return criteria


def _benchmark_configuration(cli, version):
    """Record the normalized settings, source identity and scientific scope."""
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(cli).items()
        if key not in ('resume', 'out')
    }
    configuration = {
        'version': version - 7,
        'settings': settings,
        'code_sha256': _source_receipt(),
        'fixed_subtype_criteria': FIXED_CRITERIA if cli.fixed_subtypes else None,
        'development_criteria': DEVELOPMENT_THRESHOLDS,
        'evaluation_scope': 'grouped benchmark VAL; target selection uses TRAIN/CAL; no TEST prediction',
        'interval_scope': 'paired t intervals across seeds sharing source captures; unadjusted',
    }
    if cli.operating_guard:
        configuration['operating_guard_criteria'] = _operating_criteria(cli)
    return configuration


def _replay_case_receipts(source, complete):
    """Return the per-policy result hashes listed by a completed source run."""
    receipts = {}
    with (source / 'case_status.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in (*CASE_ID, 'cal_policy'))
            if key in receipts:
                raise ValueError('Replay source repeats a case policy')
            receipts[key] = row['result_sha256']
    if len(receipts) != complete['planned_case_policies']:
        raise ValueError('Replay source case list differs from its completion receipt')
    return receipts


def _bind_replay_source(parser, cli, configuration, version):
    """Bind the replay settings and case list before starting evaluation."""
    source = cli.replay_from
    if not source:
        return {}
    original = _read(source / 'run.json')
    marker = _read(source / 'complete.json')
    if (
        original['version'] != configuration['version']
        or marker['mode'] != 'evaluation'
        or not original['settings']['save_scores']
    ):
        parser.error(f'Replay requires freshly generated V{version} scores with identical scoring settings and code')
    configuration['replay_score_binding'] = _checked_replay_configuration(original, configuration)
    if cli.export_frozen and not original['settings']['export_frozen']:
        parser.error('Replay export requires an original run with --export-frozen')
    configuration['replay_source_run_sha256'] = sha(source / 'run.json')
    configuration['replay_source_complete_sha256'] = sha(source / 'complete.json')
    configuration['replay_source_cases_sha256'] = sha(source / 'case_status.csv')
    return _replay_case_receipts(source, marker)


def _initialize_run(parser, root, cli, configuration):
    """Create or verify experiment receipts before writing new outputs."""
    if cli.resume:
        if _read(root / 'run.json') != configuration:
            parser.error('Resume requires the same source files, settings and dataset root')
    else:
        root.mkdir(parents=True, exist_ok=False)
        _atomic_json(root / 'run.json', configuration)
    if cli.operating_guard and not (root / 'runtime.json').is_file():
        _atomic_json(root / 'runtime.json', _runtime(cli))


def _write_dataset_audit(root, dataset, data, replay_from):
    """Write the current dataset audit after checking resume and replay identities."""
    data_dir = root / dataset
    data_dir.mkdir(exist_ok=True)
    audit_path = data_dir / 'data_audit.json'
    if audit_path.is_file() and _read(audit_path) != data.audit:
        raise ValueError(f'Source audit changed for {dataset}')
    if replay_from and _read(replay_from / dataset / 'data_audit.json') != data.audit:
        raise ValueError(f'Replay data or preparation receipts changed for {dataset}')
    _atomic_json(audit_path, data.audit)
    return data_dir


def _dataset_plan(cli, dataset, data, parts):
    """Choose the ordered scenarios and prescribed target classes for this dataset."""
    if cli.task == 'scale':
        return _scale_specs(cli), {'targets': [{'slot': 0, 'class_index': None, 'class': None}]}
    explicit = cli.target
    if cli.guarded_subtypes and not cli.smoke and not cli.all_targets and not cli.target:
        explicit = [GUARD_TARGETS[dataset]]
    targets = plan_targets(parts, data.names, 1 if cli.smoke else cli.target_count, explicit)
    return [('train_absence', 3, 0.3, None)], targets


def _ordered_designs(plan, targets, seeds):
    """Yield workloads in the declared order of scenario, target and seed."""
    for scenario, clients, alpha, budget in plan:
        for target in targets['targets']:
            for seed in seeds:
                yield scenario, clients, alpha, budget, target, seed


def _resolve_case_design(parts, names, identity, target_index, budget, cli):
    """Preserve unavailable designs as explicit per-case outcomes."""
    try:
        return _design(parts, names, identity, target_index, budget, cli)
    except InfeasibleDesign as error:
        audit = {'status': 'unavailable_design', 'reason': str(error), 'policies': {}}
        return None, None, {}, audit


def _replay_case_directory(cli, relative, identity, replay_receipts):
    """Locate and verify an original result before reusing its scores."""
    if cli.replay_from is None:
        return None
    ancestor = cli.replay_from / relative
    case = tuple(str(identity[key]) if identity[key] is not None else '' for key in CASE_ID)
    digest = sha(ancestor / 'result.json')
    if any(replay_receipts.get((*case, policy)) != digest for policy in cli.cal_policies):
        raise ValueError('Replay source case result differs from its manifest')
    return ancestor


def _policy_workload(identity, name, outcome, path, check_data):
    """Construct the published status for one case and calibration policy."""
    audit = outcome['audit']
    detail = audit['policies'].get(name, {})
    donor = (audit.get('partition_intervention') or {}).get('single_donor')
    status = detail.get('status', audit.get('status', 'unavailable_design'))
    if status == 'eligible':
        status = 'preflight_eligible' if check_data else 'evaluated'
    return {
        **identity,
        'cal_policy': name,
        'status': status,
        'reason': detail.get('reason', audit.get('reason')),
        'receiver': audit.get('receiver'),
        'partition_sha256': detail.get('partition_sha256'),
        **_donor_workload(donor),
        **_output_workload(outcome, name, path),
    }


def _donor_workload(donor):
    """Add the selected donor details when a single donor participates."""
    if donor:
        return {'single_donor_client': donor['selected_client'], 'single_donor_selection_rule': donor['rule']}
    return {}


def _output_workload(outcome, name, path):
    """Record paths and result integrity for the current case policy."""
    return {
        'result': str(path),
        'frozen_head': outcome.get('frozen_head_artifacts', {}).get(name, {}).get('file'),
        'fixed_subtype_evidence': (outcome.get('fixed_subtype_evidence') or {}).get('file'),
        'score_cache': (outcome.get('score_cache') or {}).get('file'),
        'result_sha256': sha(path),
    }


def _record_workloads(tables, outcome, identity, directory, policy_names, check_data):
    """Record policy and metric outcomes, then return the last policy status."""
    rows, workloads, statuses = tables
    path = directory / 'result.json'
    for name in policy_names:
        workload = _policy_workload(identity, name, outcome, path, check_data)
        workloads.append(workload)
        statuses[tuple(workload[key] for key in CONDITIONS)].append(workload['status'])
    _record_case(rows, outcome, identity)
    return workloads[-1]['status']


def _evaluate_dataset(root, dataset, cli, configuration, replay_receipts, tables):
    """Load one dataset, bind its audit and evaluate its ordered benchmark cases."""
    data = (
        _synthetic(dataset, 'grouped')
        if cli.smoke
        else load_revision_data(dataset, cli.data_root, split_protocol='grouped')
    )
    parts = _scope_data(data)
    data_dir = _write_dataset_audit(root, dataset, data, cli.replay_from)
    screens = _screen_provider(root, dataset, data, parts)
    plan, targets = _dataset_plan(cli, dataset, data, parts)
    _atomic_json(data_dir / 'targets.json', targets)
    for scenario, clients, alpha, budget, target, seed in _ordered_designs(plan, targets, cli.seeds):
        identity = dict(zip(CASE_ID, (dataset, scenario, alpha, clients, target['class'], seed), strict=True))
        relative = Path(dataset) / scenario / f"target_{target['slot']}" / f'seed_{seed}'
        design = _resolve_case_design(parts, data.names, identity, target['class_index'], budget, cli)
        ancestor = _replay_case_directory(cli, relative, identity, replay_receipts)
        directory = root / relative
        outcome = _case_output(directory, identity, design, data, cli, screens, configuration, ancestor)
        names = cli.cal_policies if budget is None else ['natural']
        last_status = _record_workloads(tables, outcome, identity, directory, names, cli.check_data)
        print(f"{dataset} {scenario} {target['class'] or '-'} seed={seed} {last_status}", flush=True)


def _run_datasets(root, cli, configuration, replay_receipts):
    """Accumulate results and condition statuses for all selected datasets."""
    tables = ({name: [] for name in _tables(cli)}, [], defaultdict(list))
    for dataset in cli.dataset:
        _evaluate_dataset(root, dataset, cli, configuration, replay_receipts, tables)
    return tables


def _write_replay_intervals(root, observations):
    """Summarize replay deltas within the fixed experiment conditions."""
    if not observations:
        return
    grouped = defaultdict(list)
    fields = (*CONDITIONS, 'mode', 'scope', 'metric', 'replay_cal_far', 'replay_guard_rewrite_rate')
    for row in observations:
        grouped[tuple(row[key] for key in fields)].append(row['delta'])
    write_csv(
        root / 'replay_operating_intervals.csv',
        [
            {**dict(zip(fields, key, strict=True)), **summarize(values, bounds=(-1, 1))}
            for key, values in grouped.items()
        ],
    )


def _write_condition_table(root, name, values):
    """Publish one decision or reference summary per matched condition."""
    write_csv(
        root / f'{name}.csv',
        [{**dict(zip(CONDITIONS, key, strict=True)), **value} for key, value in values.items()],
    )


def _write_operating_tables(root, cli, rows, statuses):
    """Export binary guard results and selected-point nonregression evidence."""
    operating, selection, nonregression = None, None, None
    if cli.operating_guard:
        operating = _operating_results(rows['binary_operating_guard'], statuses, cli)
        _write_condition_table(root, 'binary_operating_decisions', operating)
    if cli.operating_policy in SELECTING_POLICIES:
        selection = defaultdict(list)
        for row in rows['operating_selection']:
            selection[tuple(row[key] for key in CONDITIONS)].append(row)
        nonregression = _reference_results(rows['operating_reference_pairs'], statuses, cli.seeds)
        _write_condition_table(root, 'operating_reference_intervals', nonregression)
    return operating, selection, nonregression


def _write_subtype_tables(root, cli, rows, statuses, intervals, point_results):
    """Record fixed and eligible guarded subtype assessments."""
    operating, selection, nonregression = point_results
    common = (intervals, statuses, DEVELOPMENT_THRESHOLDS, cli.primary_cal_policy)
    fixed = _fixed_assessment(*common, selection=selection)
    if fixed:
        write_csv(root / 'fixed_subtype_decisions.csv', fixed)
    if cli.guarded_subtypes:
        guarded = _fixed_assessment(
            *common,
            prefix='guarded',
            cal_check=rows['subtype_guard_cal'],
            operating=operating,
            selection=selection,
            nonregression=nonregression,
        )
        if guarded:
            write_csv(root / 'guarded_subtype_decisions.csv', guarded)


def _legacy_flags(flags, operating_guard):
    """Mark legacy comparisons explicitly when using the operating guard."""
    if not operating_guard:
        return flags
    scope = 'legacy full-minus-local contrast; not the V11 guarded/operating primary decision'
    return [{**row, 'decision_scope': scope} for row in flags]


def _write_development_tables(root, cli, rows, statuses):
    """Export development comparisons after collecting all matched cases."""
    pairs, intervals, flags = _contrast(rows, statuses, DEVELOPMENT_THRESHOLDS, cli.primary_cal_policy)
    point_results = _write_operating_tables(root, cli, rows, statuses)
    if cli.fixed_subtypes:
        _write_subtype_tables(root, cli, rows, statuses, intervals, point_results)
    tables = (
        ('paired_by_seed', pairs),
        ('paired_intervals', intervals),
        ('development_decisions', _legacy_flags(flags, cli.operating_guard)),
        ('cal_coverage_changes', _coverage_changes(rows)),
    )
    for name, values in tables:
        if values:
            write_csv(root / f'{name}.csv', values)


def _write_benchmark_tables(root, cli, rows, workloads, statuses):
    """Write analysis in the order required by the benchmark receipts."""
    for table, items in rows.items():
        if items:
            write_csv(root / f'{table}.csv', items)
    if cli.replay_from:
        _write_replay_intervals(root, rows['replay_operating_points'])
    write_csv(root / 'case_status.csv', workloads)
    if not cli.check_data:
        _write_development_tables(root, cli, rows, statuses)


def _selection_outcome(cli, rows):
    """Report whether operating-point selection applies and how often it failed."""
    applicable = cli.operating_policy in SELECTING_POLICIES and not cli.check_data
    failures = sum(row['status'] != 'selected' for row in rows['operating_selection']) if applicable else None
    return applicable, failures


def _completed_workloads(cli, workloads):
    """Count the recorded case-policy statuses for the completion receipt."""
    unavailable = [item for item in workloads if item['status'] not in ('evaluated', 'preflight_eligible')]
    return {
        'task': cli.task,
        'mode': 'preflight' if cli.check_data else 'evaluation',
        'planned_case_policies': len(workloads),
        'eligible_preflight': sum(item['status'] == 'preflight_eligible' for item in workloads),
        'evaluated': sum(item['status'] == 'evaluated' for item in workloads),
        'unavailable': len(unavailable),
        'unavailable_by_status': dict(Counter(item['status'] for item in unavailable)),
        'unavailable_by_reason': dict(Counter(item['reason'] or item['status'] for item in unavailable)),
    }


def _completion_receipt(cli, rows, workloads):
    """Describe the exact executed experiment and its evidence limits."""
    applicable, failures = _selection_outcome(cli, rows)
    return {
        **_completed_workloads(cli, workloads),
        'smoke': cli.smoke,
        'head_only': cli.head_only,
        'fixed_subtypes': cli.fixed_subtypes,
        'guarded_subtypes': cli.guarded_subtypes,
        'score_replay': cli.replay_from is not None,
        'operating_guard': cli.operating_guard,
        'operating_policy': cli.operating_policy,
        'operating_selection_applicable': applicable,
        'operating_selection_failures': failures,
        'independent_confirmation': False,
        'test_evaluated': False,
    }


def _announce_operating_policy(cli):
    """Explain the unchanged behavior of the fixed operating rule."""
    if cli.operating_guard and cli.operating_policy not in SELECTING_POLICIES:
        print(
            'V11 NOTE: --operating-policy fixed reproduces the original V11 threshold; '
            'the CAL-only selector is not evaluated. Use --operating-policy cal-select '
            'for NB15 operating-point selection.',
            flush=True,
        )


def main(argv=None):
    """Evaluate the declared benchmark and return its process exit status."""
    parser = _parser()
    cli = parser.parse_args(argv)
    root = _normalize(parser, cli)
    if cli.smoke:
        torch.set_num_threads(1)
    version = _run_version(cli)
    _announce_operating_policy(cli)
    configuration = _benchmark_configuration(cli, version)
    replay_receipts = _bind_replay_source(parser, cli, configuration, version)
    _initialize_run(parser, root, cli, configuration)
    print(f'V{version} OUTPUT: {root}', flush=True)
    rows, workloads, statuses = _run_datasets(root, cli, configuration, replay_receipts)
    _write_benchmark_tables(root, cli, rows, workloads, statuses)
    _atomic_json(root / 'complete.json', _completion_receipt(cli, rows, workloads))
    print(f'V{version} COMPLETE: {root}', flush=True)
    return 0
