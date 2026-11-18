"""Train and evaluate matched NB15 detectors with a frozen CAL operating rule."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import revision_next as native
from . import utils as core
from .privacy_run import PrivateClient
from .revision_binary_diagnostics import _class_loss, _low_far, _rates
from .revision_fresh_source import SEEDS, _common_switches
from .revision_report import summarize, write_csv
from .revision_v12 import _bindings, _hash, _write
from .utils import empirical_alarm_threshold, hierarchical_decisions

DATASET = 'CIC-UNSW-NB15'
TARGET = 'Exploits'
SYNTHETIC_TARGET = 'ATTACK_1'
OUTPUT_CONTAINER = 'fedcaid_nb15_low_far'
EVAL_ROLES = ('cal', 'val')
WEIGHT_FIT_MODES = {'frozen_head': 'frozen', 'full_finetune': 'full'}
SOURCE_MODULES = (
    'revision_fresh_source.py',
    'revision_nb15_low_far.py',
)

TAIL_FRACTION = 0.01
EPOCHS = 2
LEARNING_RATE = 5e-4
BCE_WEIGHT = 0.25
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0

ARMS = ('reference', 'frozen_head', 'full_finetune')
SCOPES = ('receiver', 'overall')
FAR_CAP = 0.01
FNR_INCREASE_LIMIT = 0.01
MIN_CAL_BENIGN_ROWS = 100
METRICS = ('val_far', 'val_fnr', 'val_macro_f1', 'val_binary_ece_10', 'val_tpr_at_far_1pct_diagnostic')
PAIRS = (('frozen_head', 'reference'), ('full_finetune', 'reference'), ('full_finetune', 'frozen_head'))
CLASS_FIELDS = ('support', 'predicted', 'true_positive', 'binary_recall', 'precision', 'recall', 'f1')


class BinaryAttackScorer(nn.Module):
    """Add a zero-initialized binary readout to the shared TabM score."""

    def __init__(self, reference_model, mode, device):
        super().__init__()
        self.detector = deepcopy(reference_model).to(device)
        self.detector.requires_grad_(mode == 'full')
        width = self.detector.base.output.in_features
        # Construct parameters directly so adding the branch does not change RNG.
        self.weight = nn.Parameter(torch.zeros(width, device=device))
        self.bias = nn.Parameter(torch.zeros((), device=device))
        # Disable dropout in both arms so only trainable parameter scope differs.
        self.eval()

    def forward(self, x):
        """Return attack log odds of the mean ensemble attack probability."""
        detector = self.detector
        base = detector.base
        features = base.num_module(detector.prepare_ple_input(x)).flatten(1)
        crossed = detector.apply_cross(features)
        hidden = base.backbone(base.ensemble_view(crossed))
        logits = base.output(hidden)
        attack = logits[..., 1:].logsumexp(dim=-1)
        member_odds = attack - logits[..., 0]
        log_attack = F.logsigmoid(member_odds).logsumexp(dim=1)
        log_benign = F.logsigmoid(-member_odds).logsumexp(dim=1)
        correction = hidden.mean(dim=1) @ self.weight + self.bias
        return log_attack - log_benign + correction


def _feature_matrix(x, width, batch_size):
    """Check feature shape and the batch size before taking TRAIN or score slices."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != width or x.dtype.kind not in 'fiu':
        raise ValueError('Expected numeric rows in the reference TabM feature order')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Low-FAR batch size must be a positive integer')
    return x


def _batch_tensor(x, transform, device):
    """Apply only the supplied TRAIN-fitted transform, if one is needed."""
    if transform is not None:
        x = transform(x)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class LowFARFit:
    """Trained binary scoring arm and its budget and objective receipt."""

    scorer: BinaryAttackScorer
    transform: object
    device: torch.device
    batch_size: int
    receipt: dict

    @torch.no_grad()
    def score(self, x):
        """Score features without consulting or fitting with evaluation labels."""
        x = _feature_matrix(x, self.scorer.detector.input_features, self.batch_size)
        output = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self.batch_size):
            rows = slice(start, start + self.batch_size)
            batch = _batch_tensor(x[rows], self.transform, self.device)
            output[rows] = self.scorer(batch).double().sigmoid().cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError('Low-FAR TabM produced nonfinite scores')
        return output


def _tail_loss(logits, attack, benign_count):
    """Rank attacks above the largest 1% of benign TRAIN scores in the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, attack.to(logits.dtype))
    if benign_count == 0 or benign_count == len(attack):
        return BCE_WEIGHT * bce, 0
    tail_size = ceil(TAIL_FRACTION * benign_count)
    tail = logits[~attack].topk(tail_size).values
    attack_scores = logits[attack]
    ranking = F.softplus(tail[:, None] - attack_scores[None, :]).mean()
    return ranking + BCE_WEIGHT * bce, tail_size * len(attack_scores)


def _train_epoch(model, parameters, optimizer, features, attack, order, config):
    """Fit one common TRAIN permutation and count scored pairs and steps."""
    transform, device, batch_size = config
    monitor = core.TrainingFiniteMonitor(device, 'NB15 low-FAR detector')
    steps, pairs = 0, 0
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        known_attack = attack[rows]
        benign_count = len(rows) - int(known_attack.sum())
        x = _batch_tensor(features[rows], transform, device)
        y = torch.as_tensor(known_attack, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, pair_count = _tail_loss(model(x), y, benign_count)
        loss.backward()
        monitor.observe(loss, nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP))
        optimizer.step()
        pairs += pair_count
        steps += 1
    monitor.check()
    return steps, pairs


def _parameter_audit(reference, scorer):
    """Verify which original TabM tensors and readout parameters changed."""
    current = dict(scorer.detector.named_parameters())
    changed = [
        name
        for name, before in reference.named_parameters()
        if not torch.equal(before.detach().to(current[name].device), current[name].detach())
    ]
    changed_readout = bool(torch.count_nonzero(scorer.weight)) or bool(scorer.bias.detach() != 0)
    return changed, changed_readout


def _training_inputs(reference, x, y, mode, batch_size):
    """Require one real binary training population and the reference class order."""
    if not isinstance(reference, core.CrossTabM) or mode not in {'frozen', 'full'}:
        raise ValueError('Low-FAR fitting requires frozen or full CrossTabM')
    x = _feature_matrix(x, reference.input_features, batch_size)
    y = np.asarray(y)
    if (
        y.shape != (len(x),)
        or y.dtype.kind not in 'iu'
        or (y < 0).any()
        or (y >= reference.base.output.out_features).any()
        or not (y == 0).any()
        or not (y > 0).any()
    ):
        raise ValueError('Low-FAR fitting needs BENIGN and attack labels from TRAIN')
    return x, y > 0


def _fit_receipt(mode, seed, x, batch_size, steps, pairs, updated, readout, parameters):
    """Record the fixed fit budget, scoring geometry and effective updates."""
    return {
        'mode': mode,
        'seed': seed,
        'training_role': 'TRAIN',
        'train_rows': len(x),
        'train_rows_visited': EPOCHS * len(x),
        'epochs': EPOCHS,
        'batch_size': batch_size,
        'optimizer_steps': steps,
        'tail_pairs': pairs,
        'trainable_parameters': sum(p.numel() for p in parameters),
        'base_parameters_updated': updated,
        'binary_readout_updated': readout,
        'base_score': 'log odds of mean ensemble attack probability plus zero-start linear readout',
        'loss': 'pairwise softplus over top benign TRAIN scores plus weighted BCE',
        'benign_tail_fraction': TAIL_FRACTION,
        'bce_weight': BCE_WEIGHT,
        'adamw_learning_rate': LEARNING_RATE,
        'adamw_weight_decay': WEIGHT_DECAY,
        'gradient_clip_norm': GRADIENT_CLIP,
        'dropout_during_fine_tuning': False,
    }


def fit_low_far(reference_model, train_x, train_y, *, mode, seed, device, transform=None, batch_size=2048):
    """Clone a same-run reference and fit an isolated frozen or full binary arm."""
    x, attack = _training_inputs(reference_model, train_x, train_y, mode, batch_size)
    device = torch.device(device)
    scorer = BinaryAttackScorer(reference_model, mode, device)
    parameters = [p for p in scorer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    total_steps, total_pairs = 0, 0
    config = transform, device, batch_size
    for _ in range(EPOCHS):
        steps, pairs = _train_epoch(scorer, parameters, optimizer, x, attack, rng.permutation(len(x)), config)
        total_steps += steps
        total_pairs += pairs
    updated, readout = _parameter_audit(reference_model, scorer)
    if (mode == 'frozen' and updated) or (mode == 'full' and not updated) or not readout or not total_pairs:
        raise RuntimeError('The low-FAR update did not match the declared training scope')
    receipt = _fit_receipt(mode, seed, x, batch_size, total_steps, total_pairs, updated, readout, parameters)
    return LowFARFit(scorer, transform, device, batch_size, receipt)


def _validate_role(labels, probabilities, classes):
    """Require valid aligned class probabilities and integer class indices."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or not labels.size
        or labels.dtype.kind not in 'ui'
        or (labels < 0).any()
        or (labels >= classes).any()
        or probabilities.shape != (len(labels), classes)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError('NB15 probabilities must align with nonempty CAL or VAL class labels')
    return labels, probabilities


def _binary_ece(scores, labels):
    """Return row-weighted binary ECE over ten equal-width probability bins."""
    bins = np.minimum((scores * 10).astype(int), 9)
    score_mass = np.bincount(bins, weights=scores, minlength=10)
    attack_mass = np.bincount(bins, weights=labels > 0, minlength=10)
    return float(np.abs(score_mass - attack_mass).sum() / len(labels))


def _decide(p, labels, threshold):
    """Apply the experiment's strict binary gate and retain subtype outputs."""
    summary = hierarchical_decisions(p, threshold)
    return {'labels': labels, **summary}


def _pool(records):
    """Concatenate row-aligned decisions with their frozen per-client gates."""
    return {key: np.concatenate([record[key] for record in records]) for key in records[0]}


def _binary_metrics(decision, *, ranking=False):
    """Measure the locked gate and optionally add diagnostic score ranking."""
    labels, scores = decision['labels'], decision['attack_score']
    measurements = _rates(labels, decision['attack_alert'])
    measurements['binary_ece_10'] = _binary_ece(scores, labels)
    if ranking:
        diagnostic = _low_far(scores, labels)
        measurements.update(
            tpr_at_far_1pct_diagnostic=diagnostic['tpr_at_far_1pct'],
            standardized_partial_auc_at_far_1pct_diagnostic=diagnostic['standardized_partial_auc_at_far_1pct'],
            empirical_far_resolution=diagnostic['empirical_far_resolution'],
        )
    return measurements


def _client_results(labels, probabilities, classes):
    """Fit a strict threshold from CAL benign scores and apply it to VAL."""
    cal_y, cal_p = _validate_role(labels[0], probabilities[0], classes)
    val_y, val_p = _validate_role(labels[1], probabilities[1], classes)
    threshold, receipt = empirical_alarm_threshold(
        cal_p, cal_y, FAR_CAP, floor=0, min_benign_rows=MIN_CAL_BENIGN_ROWS, insufficient_policy='disable'
    )
    decisions = {
        'cal': _decide(cal_p, cal_y, threshold),
        'val': _decide(val_p, val_y, threshold),
    }
    return receipt, decisions


def _role_metrics(decisions):
    """Prefix CAL and VAL metrics and keep ROC statistics descriptive."""
    return {
        f'{role}_{metric}': value
        for role, decision in decisions.items()
        for metric, value in _binary_metrics(decision, ranking=role == 'val').items()
    }


def _class_rows(identity, val, names):
    """Preserve unsupported classes and all supported zero-recall classes."""
    for index, name in enumerate(names):
        truth = val['labels'] == index
        binary_correct = val['attack_alert'] if index else ~val['attack_alert']
        metrics = _class_loss(truth, binary_correct, val['predicted'] == index)
        yield {
            **identity,
            'class_index': index,
            'class_name': name,
            **{key: metrics[key] for key in CLASS_FIELDS},
        }


def _classification_metrics(rows):
    """Summarize classification over observed true labels for this VAL scope."""
    supported = [row for row in rows if row['support']]
    total = sum(row['support'] for row in supported)
    return {
        'val_accuracy': sum(row['true_positive'] for row in supported) / total,
        'val_balanced_accuracy': sum(row['recall'] for row in supported) / len(supported),
        'val_macro_f1': sum(row['f1'] for row in supported) / len(supported),
        'val_weighted_f1': sum(row['f1'] * row['support'] for row in supported) / total,
        'observed_val_classes': len(supported),
        'defined_classes': len(rows),
    }


def _scope_results(identity, clients, names):
    """Pool counts using each client's own threshold and report class outcomes."""
    decisions = {role: _pool([client[1][role] for client in clients]) for role in ('cal', 'val')}
    classes = list(_class_rows(identity, decisions['val'], names))
    return {
        **identity,
        **_role_metrics(decisions),
        **_classification_metrics(classes),
        'cal_clients_with_insufficient_support': sum(c[0]['alert_gate_disabled_for_support'] for c in clients),
        'val_ranking_scope': 'receiver_raw_scores' if identity['scope'] == 'receiver' else 'pooled_raw_scores',
    }, classes


def _evaluate_arm(seed, names, receiver, label_rows, probabilities, arm):
    """Report per-client thresholds and aggregate receiver and overall outcomes."""
    clients = [
        _client_results(labels, scores, len(names)) for labels, scores in zip(label_rows, probabilities, strict=True)
    ]
    identity = {'seed': seed, 'arm': arm, 'receiver_client': receiver}
    client_rows = [
        {**identity, 'client': index, **receipt, **_role_metrics(decisions)}
        for index, (receipt, decisions) in enumerate(clients)
    ]
    scoped = [
        _scope_results({**identity, 'scope': scope}, selected, names)
        for scope, selected in (('receiver', clients[receiver : receiver + 1]), ('overall', clients))
    ]
    return [point for point, _ in scoped], client_rows, [row for _, per_class in scoped for row in per_class]


def _difference(value, reference):
    """Keep paired differences undefined when either side lacks support."""
    return value - reference if value is not None and reference is not None else None


def _class_differences(rows):
    """Add supported-class recall pairs without inventing a promotion margin."""
    reference = {(row['scope'], row['class_index']): row for row in rows if row['arm'] == 'reference'}
    return [
        {
            **row,
            **{
                key + '_delta_vs_reference': _difference(row[key], reference[(row['scope'], row['class_index'])][key])
                for key in ('binary_recall', 'recall', 'f1')
            },
        }
        for row in rows
    ]


def evaluate_seed(seed, names, receiver, labels_by_client, probabilities_by_arm):
    """Evaluate one seed with identical roles and one CAL policy for all arms."""
    if (
        len(names) < 2
        or len(set(names)) != len(names)
        or not 0 <= receiver < len(labels_by_client)
        or set(probabilities_by_arm) != set(ARMS)
    ):
        raise ValueError('Expected aligned NB15 class names, receiver and three arms')
    results = [_evaluate_arm(seed, names, receiver, labels_by_client, probabilities_by_arm[arm], arm) for arm in ARMS]
    points, clients, classes = ([row for member in table for row in member] for table in zip(*results, strict=True))
    return {'working_points': points, 'client_working_points': clients, 'per_class': _class_differences(classes)}


def _index_points(rows, seeds):
    """Require one row for every seed, scope and arm before computing effects."""
    expected = {(seed, scope, arm) for seed in seeds for scope in SCOPES for arm in ARMS}
    index = {(row['seed'], row['scope'], row['arm']): row for row in rows}
    if len(seeds) != len(set(seeds)) or set(index) != expected or len(rows) != len(expected):
        raise ValueError('NB15 paired results are incomplete or ambiguous')
    return index


def _paired_stats(index, seeds, scope, candidate, reference):
    """Summarize within-seed changes for detectors evaluated on matching rows."""
    return {
        metric: summarize(
            [
                _difference(index[seed, scope, candidate][metric], index[seed, scope, reference][metric])
                for seed in seeds
            ],
            bounds=(-1, 1),
        )
        for metric in METRICS
    }


def _absolute_checks(index, seeds, scope, candidate):
    """Check per-seed FAR and reject unavailable or entirely silent gates."""
    rows = [index[seed, scope, candidate] for seed in seeds]
    values = [row['val_far'] for row in rows if row['val_far'] is not None]
    return {
        'max_far': max(values) if values else None,
        'seeds_above_far_cap': sum(value > FAR_CAP for value in values),
        'far_pass': len(values) == len(seeds) and all(value <= FAR_CAP for value in values),
        'cal_support_pass': all(row['cal_clients_with_insufficient_support'] == 0 for row in rows),
        'nonzero_attack_detection_each_seed': all(row['val_true_positives'] > 0 for row in rows),
    }


def _scope_check(index, seeds, scope, candidate, stats):
    """Assess binary and classification guards on one evaluation population."""
    fnr, macro = stats['val_fnr'], stats['val_macro_f1']
    return {
        **_absolute_checks(index, seeds, scope, candidate),
        'paired_fnr_pass': fnr['seeds_with_metric'] == len(seeds)
        and fnr['ci95_high'] is not None
        and fnr['ci95_high'] <= FNR_INCREASE_LIMIT,
        'macro_f1_mean_non_decrease': macro['seeds_with_metric'] == len(seeds)
        and macro['mean'] is not None
        and macro['mean'] >= 0,
    }


def _candidate_checks(index, intervals, seeds, candidate):
    """Record each candidate independently without automatically promoting it."""
    checks = {
        scope: _scope_check(index, seeds, scope, candidate, intervals[candidate + '_minus_reference'][scope])
        for scope in SCOPES
    }
    flags = (
        'far_pass',
        'cal_support_pass',
        'nonzero_attack_detection_each_seed',
        'paired_fnr_pass',
        'macro_f1_mean_non_decrease',
    )
    passes = all(record[flag] for record in checks.values() for flag in flags) if len(seeds) >= 5 else None
    return {'scopes': checks, 'development_guard_pass': passes}


def _paired_intervals(index, seeds):
    """Return all prespecified contrasts across the two evaluation scopes."""
    return {
        f'{candidate}_minus_{reference}': {
            scope: _paired_stats(index, seeds, scope, candidate, reference) for scope in SCOPES
        }
        for candidate, reference in PAIRS
    }


def assess_results(rows, seeds):
    """Compute paired intervals and preregistered development-only checks."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('NB15 assessment requires a nonempty seed plan')
    index = _index_points(rows, seeds)
    intervals = _paired_intervals(index, seeds)
    return {
        'scope': 'grouped same-source development',
        'seeds': list(seeds),
        'calibration_rule': 'per-client benign CAL higher quantile with zero floor and strict score > threshold',
        'selection_on_val': False,
        'test_evaluated': False,
        'automatic_promotion': False,
        'far_cap': FAR_CAP,
        'paired_fnr_ci95_upper_limit': FNR_INCREASE_LIMIT,
        'min_cal_benign_rows': MIN_CAL_BENIGN_ROWS,
        'paired_intervals': intervals,
        'candidates': {arm: _candidate_checks(index, intervals, seeds, arm) for arm in ARMS[1:]},
    }


def _configuration(project, output, device, smoke):
    """Start with the same controlled-absence design as the main NB15 evidence."""
    parser = native._parser()
    cli = parser.parse_args(
        [
            *_common_switches(project / 'dataset', device),
            '--dataset',
            DATASET,
            '--operating-policy',
            'fixed',
            '--detector-objective',
            'prior_adjusted',
            '--out',
            str(output),
            *(('--smoke',) if smoke else ()),
        ]
    )
    native._normalize(parser, cli)
    return cli


def _source_snapshot(project):
    """Bind original and candidate source without reading old result directories."""
    source = Path(__file__).resolve().parent
    return {
        **native._source_receipt(),
        **_bindings(source, SOURCE_MODULES),
        'main.py': _hash(project / 'main.py'),
    }


def _allocate_output(project, container=OUTPUT_CONTAINER, *, output=None):
    """Keep every experiment invocation separate from all previous outputs."""
    if output is not None:
        target = Path(output).absolute()
        if target.resolve() != target:
            raise ValueError('The NB15 stage output cannot be redirected')
        target.mkdir(exist_ok=False)
        return target
    root = project / 'outputs' / container
    if root.resolve() != root:
        raise ValueError('The NB15 output container must stay inside the project')
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime('run_%Y%m%d_%H%M%S_')
    return Path(mkdtemp(prefix=prefix, dir=root))


def _prepare_seed(data, cli, seed):
    """Reuse grouped allocation, controlled absence and reserved CAL holdouts."""
    name = SYNTHETIC_TARGET if cli.smoke else TARGET
    target = data.names.index(name)
    identity = dict(zip(native.CASE_ID, (DATASET, 'train_absence', 0.3, 3, name, seed), strict=True))
    parts, roster, policies, audit = native._design(
        native._scope_data(data),
        data.names,
        identity,
        target,
        None,
        cli,
    )
    policy = policies['reserved']
    if policy['status'] != 'eligible':
        raise ValueError(f"Reserved CAL unavailable for seed {seed}: {policy.get('reason')}")
    return parts, roster, policy, audit, native._args(identity, cli)


def _client_rows(roster, policy, client):
    """Use TRAIN as routed and only the original group-disjoint CAL holdout."""
    original, routed = roster[client], policy['roster'][client]
    if not np.array_equal(original['train'], routed['train']):
        raise ValueError('CAL reservation changed the client TRAIN allocation')
    _, holdout = policy['splits'][client]
    return {'train': original['train'], 'cal': routed['cal'][holdout], 'val': original['val']}


def _joint_probabilities(reference, attack_probability):
    """Keep the reference attack subtype ordering while replacing the alarm score."""
    dtype = np.float64
    scores = np.asarray(attack_probability, dtype=dtype)
    if scores.shape != (len(reference),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('The candidate must return one finite attack probability per row')
    attack = np.asarray(reference[:, 1:], dtype=dtype)
    total = attack.sum(axis=1, keepdims=True)
    conditional = np.full_like(attack, 1 / attack.shape[1])
    np.divide(attack, total, out=conditional, where=total > 0)
    return np.column_stack((1 - scores, scores[:, None] * conditional))


def _candidate(fit, node):
    """Score CAL and VAL with a trained branch, retaining the reference subtype rule."""
    return tuple(_joint_probabilities(node.probabilities[role], fit.score(node.parts[role][0])) for role in EVAL_ROLES)


def _fit_reference_client(data, parts, rows, args, client, scratch):
    """Fit the original local detector and score only the supplied evaluation roles."""
    local = {role: parts[role] for role in rows}
    client_id = f'client_{client}'
    print(f'FedCAID | NB15 seed={args.seed} {client_id} | 训练参考 TabM', flush=True)
    return PrivateClient(client_id, local, rows, data.names, data.features, args, scratch / client_id)


def _candidate_seed(seed, client, *, independent_client_rng):
    if independent_client_rng:
        return [seed, client]
    # Retain the candidate permutations used by the manuscript reference run.
    return seed + client


def _train_client(data, parts, rows, args, client, scratch, *, independent_client_rng):
    """Fit a fresh local reference and both candidates on its same TRAIN rows."""
    node = _fit_reference_client(data, parts, rows, args, client, scratch)
    client_id = node.client_id
    scores = {'reference': tuple(node.probabilities[role] for role in EVAL_ROLES)}
    receipt = {'client': client, 'reference': node.performance, 'candidates': {}}
    seed = _candidate_seed(args.seed, client, independent_client_rng=independent_client_rng)
    for arm, mode in WEIGHT_FIT_MODES.items():
        print(f'FedCAID | NB15 seed={args.seed} {client_id} | {arm}', flush=True)
        fit = fit_low_far(
            node.model,
            *node.parts['train'],
            mode=mode,
            seed=seed,
            device=args.device,
            batch_size=args.batch_size,
        )
        scores[arm] = _candidate(fit, node)
        receipt['candidates'][arm] = fit.receipt
        del fit
    return tuple(node.parts[role][1] for role in EVAL_ROLES), scores, receipt


def _run_seed(data, cli, seed):
    """Keep only aggregated evidence after the matched scoring of all clients."""
    parts, roster, policy, audit, args = _prepare_seed(data, cli, seed)
    labels, scores, receipts = [], defaultdict(list), []
    with TemporaryDirectory(prefix='fedcaid_nb15_low_far_') as temporary:
        for client in range(len(roster)):
            rows = _client_rows(roster, policy, client)
            truth, probabilities, receipt = _train_client(
                data,
                parts,
                rows,
                args,
                client,
                Path(temporary),
                independent_client_rng=cli.independent_client_rng,
            )
            labels.append(truth)
            receipts.append(receipt)
            for arm, values in probabilities.items():
                scores[arm].append(values)
    tables = evaluate_seed(seed, data.names, audit['receiver'], labels, dict(scores))
    return tables, audit, receipts


def _collect_seeds(output, project, cli, seeds, data, source):
    """Write per-seed outcomes before aggregating paired metrics."""
    tables = defaultdict(list)
    for seed in seeds:
        if _source_snapshot(project) != source:
            raise ValueError('NB15 source changed during the run')
        current, audit, training = _run_seed(data, cli, seed)
        _write(output / f'seed_{seed}.json', {'seed': seed, 'audit': audit, 'training': training, 'tables': current})
        for name, rows in current.items():
            tables[name].extend(rows)
    if _source_snapshot(project) != source:
        raise ValueError('NB15 source changed during the run')
    return tables


def _write_plan(output, project, cli, seeds, source):
    """Declare the controlled development scope and independent-run provenance."""
    _write(
        output / 'plan.json',
        {
            'dataset': DATASET,
            'protocol': 'grouped',
            'scenario': 'train_absence',
            'seeds': list(seeds),
            'device': cli.device,
            'data_root': str(project / 'dataset'),
            'target': SYNTHETIC_TARGET if cli.smoke else TARGET,
            'clients': 3,
            'alpha': 0.3,
            'synthetic_only': cli.smoke,
            'source_sha256': source,
            'previous_run_inputs_allowed': False,
            'test_evaluated': False,
            'selection_on_val': False,
            'reference_epochs': cli.detector_epochs,
            'candidate_update_epochs': EPOCHS,
            'candidate_train_benign_tail_fraction': TAIL_FRACTION,
            'candidate_rng': 'seed_sequence_by_seed_and_client' if cli.independent_client_rng else 'reference_seed_sum',
            'arms': list(ARMS),
            'subtype_rule': 'common freshly trained reference subtype',
        },
    )


def _publish(output, tables, seeds):
    """Publish working points, class metrics and a sealed development decision."""
    for name, rows in tables.items():
        write_csv(output / f'{name}.csv', rows)
    assessment = assess_results(tables['working_points'], seeds)
    _write(output / 'assessment.json', assessment)
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != 'complete.json')
    _write(output / 'complete.json', {'status': 'completed', 'artifacts': _bindings(output, names)})
    print(f"FedCAID | NB15 专项结果（{len(seeds)} 个种子） | {output / 'assessment.json'}", flush=True)


def run_nb15_low_far(project, *, output=None, device='auto', smoke=False, independent_client_rng=False):
    """Run a fresh NB15 comparison and return its output directory.

    Set independent_client_rng for experiments with separate candidate RNG
    streams. The default reproduces the reference candidate permutations.
    """
    if type(independent_client_rng) is not bool:
        raise ValueError('independent_client_rng must be a boolean')
    project = Path(project).resolve(strict=True)
    output = _allocate_output(project, output=output)
    cli = _configuration(project, output, device, smoke)
    cli.independent_client_rng = independent_client_rng
    seeds, source = SEEDS[:1] if smoke else SEEDS, _source_snapshot(project)
    _write_plan(output, project, cli, seeds, source)
    print(f'FedCAID | NB15 低误报专项 | {output}', flush=True)
    if smoke:
        torch.set_num_threads(1)
    data = (
        native._synthetic(DATASET, 'grouped')
        if smoke
        else native.load_revision_data(
            DATASET,
            cli.data_root,
            split_protocol='grouped',
        )
    )
    _write(output / 'data_audit.json', data.audit)
    tables = _collect_seeds(output, project, cli, seeds, data, source)
    _publish(output, tables, seeds)
    return output
