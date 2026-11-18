"""Additional paired experiments on the unchanged detector pipeline."""

from collections import defaultdict
from time import perf_counter

import numpy as np
from sklearn.linear_model import SGDClassifier

from .utils import empirical_alarm_threshold, hierarchical_decisions

FLOW_DATASETS = ('CIC-IDS-2017', 'CIC-UNSW-NB15')
CAN_DATASETS = ('CIC-IoV-2024', 'Car-Hacking')
SCALE_DATASETS = ('Car-Hacking', 'CIC-UNSW-NB15')
PROFILES = (
    'core',
    'transfer',
    'controlled_transfer',
    'temporal',
    'quantization',
    'quantization_grouped',
    'scale',
    'all',
)
TRANSFER_MODES = ('local_rounds', 'local_only', 'full_global', 'full')
TRANSFER_PAIRS = {
    'full_global_minus_local_rounds': ('full_global', 'local_rounds'),
    'full_minus_local_only': ('full', 'local_only'),
}
MIN_DONOR_ROWS = 5
MIN_DONOR_TRAIN_ROWS = 5
REPAIR_FIXED = 'transfer_repair_fixed_gate'
REPAIR_UNION = 'transfer_repair_union'
REPAIR_RESCUE = 'transfer_repair_rescue'
REPAIR_MODES = (REPAIR_FIXED, REPAIR_UNION, REPAIR_RESCUE)
REPAIR_PAIRS = {
    'repair_fixed_minus_base_threshold': (REPAIR_FIXED, 'base_threshold'),
    'repair_union_minus_base_threshold': (REPAIR_UNION, 'base_threshold'),
    'repair_union_minus_full': (REPAIR_UNION, 'full'),
    'repair_rescue_minus_base_threshold': (REPAIR_RESCUE, 'base_threshold'),
    'repair_rescue_minus_full': (REPAIR_RESCUE, 'full'),
}
SENSITIVITY_POLICIES = ((0.5, 1.0), (0.25, 1.0), (1.0, 1.0), (0.5, 0.25), (0.5, 0.5))
TRANSFER_SCORE_METRICS = {
    'delta_logp_mean': (-np.inf, np.inf),
    'delta_logp_median': (-np.inf, np.inf),
    'logp_increase_rate': (0, 1),
    'logp_decrease_rate': (0, 1),
    'delta_attack_rank_mean': (-np.inf, np.inf),
    'attack_rank_improved_rate': (0, 1),
    'delta_attack_log_margin_mean': (-np.inf, np.inf),
    'delta_top1_rate': (-1, 1),
    'delta_alert_rate': (-1, 1),
    'delta_recall': (-1, 1),
}
TRANSFER_SCORE_STATE_FIELDS = (
    'local_alert_rate',
    'federated_alert_rate',
    'local_top1_rate',
    'federated_top1_rate',
    'local_top1_no_alert_rate',
    'federated_top1_no_alert_rate',
    'local_alert_wrong_subtype_rate',
    'federated_alert_wrong_subtype_rate',
    'local_recall',
    'federated_recall',
)


def policy_specs(args):
    """Return the configured gate and residual fractions with suffixes for any additional sensitivity runs."""
    selected = (args.repair_gate_target_fraction, args.repair_residual_budget_fraction)
    policies = [selected]
    if getattr(args, 'repair_sensitivity', False):
        policies.extend(pair for pair in SENSITIVITY_POLICIES if pair != selected)
    return [(g, r, '' if i == 0 else f'__g{g:g}_r{r:g}'.replace('.', 'p')) for i, (g, r) in enumerate(policies)]


def repair_base(mode):
    """Return the base repair mode for the supplied mode name, or None when it is unrelated."""
    return next((base for base in REPAIR_MODES if mode == base or mode.startswith(base + '__g')), None)


def repair_conditions(mode, settings):
    """Return the gate and residual fractions for a repair mode using its suffix or the supplied settings."""
    base = repair_base(mode)
    if base is None:
        return {}
    suffix = mode[len(base) :]
    if suffix:
        gate, residual = suffix[3:].split('_r', 1)
        return {
            'repair_gate_target_fraction': float(gate.replace('p', '.')),
            'repair_residual_budget_fraction': float(residual.replace('p', '.')),
        }
    return {
        name: settings.get(name, default)
        for name, default in (('repair_gate_target_fraction', 0.5), ('repair_residual_budget_fraction', 1.0))
    }


def repair_pairs_for_modes(modes):
    """Return baseline repair contrasts and matching comparisons for the supplied sensitivity modes."""
    pairs = dict(REPAIR_PAIRS)
    for mode in modes:
        base = repair_base(mode)
        if base is None or mode == base:
            continue
        suffix = mode[len(base) :]
        for contrast, (left, right) in REPAIR_PAIRS.items():
            if left == base:
                label, reference = contrast.split('_minus_', 1)
                pairs[f'{label}{suffix}_minus_{reference}'] = (mode, right)
    return pairs


def supplement_cases(datasets, profile, *, transfer_clients=3, transfer_alpha=0.3, controlled_count=3):
    """Build supported cases without extending time ordering to other sources."""
    if type(transfer_clients) is not int or transfer_clients < 3:
        raise ValueError('Secure transfer needs at least three clients')
    if not np.isfinite(transfer_alpha) or transfer_alpha <= 0:
        raise ValueError('Transfer alpha must be finite and positive')
    cases = defaultdict(list)
    for dataset in datasets:
        if profile == 'controlled_transfer' and dataset in FLOW_DATASETS:
            if type(controlled_count) is not int or controlled_count < 1:
                raise ValueError('Controlled target count must be positive')
            cases[dataset, 'grouped'].extend(
                (f'controlled_transfer_{i}', transfer_clients, transfer_alpha) for i in range(controlled_count)
            )
        if profile in ('core', 'transfer', 'all') and dataset in FLOW_DATASETS:
            cases[dataset, 'grouped'].append(('supplement_transfer', transfer_clients, transfer_alpha))
        if profile in ('core', 'temporal', 'all') and dataset == 'CIC-UNSW-NB15':
            cases[dataset, 'temporal'].append(('supplement_temporal', 3, 0.8))
        if profile in ('quantization', 'all') and dataset in CAN_DATASETS:
            cases[dataset, 'frozen'].append(('supplement_quantization', 3, 0.8))
        if profile == 'quantization_grouped' and dataset in CAN_DATASETS:
            cases[dataset, 'grouped'].append(('supplement_quantization', 3, 0.8))
        if profile in ('scale', 'all') and dataset in SCALE_DATASETS:
            for budget in ('total', 'local'):
                cases[dataset, 'grouped'].extend(
                    (f'supplement_scale_{budget}_c{clients}', clients, 0.8) for clients in (3, 5, 10)
                )
    return dict(cases)


def rebalance_scale(roster, seed):
    """Enforce equal local TRAIN and CAL quotas by deterministic transfers."""
    rng = np.random.default_rng(seed + 741)
    moves = {}
    for role in ('train', 'cal'):
        sizes = [len(client[role]) for client in roster]
        quotient, remainder = divmod(sum(sizes), len(roster))
        quotas = [quotient + (i < remainder) for i in range(len(roster))]
        surplus = [client[role][quota:] for client, quota in zip(roster, quotas, strict=True)]
        pool = rng.permutation(np.concatenate(surplus))
        offset = 0
        for client, quota in zip(roster, quotas, strict=True):
            kept = client[role][:quota]
            deficit = quota - len(kept)
            client[role] = rng.permutation(np.concatenate((kept, pool[offset : offset + deficit])))
            offset += deficit
        if offset != len(pool):
            raise RuntimeError('Scale quota assignment lost data')
        moves[role] = offset
    return moves


class AttackSubtypeHead:
    """Local linear classifier of attack subtypes, fitted only on TRAIN."""

    def __init__(self, x, y, seed):
        attack = y > 0
        labels = y[attack]
        classes, counts = np.unique(labels, return_counts=True)
        self.model = None
        self.classes = classes
        self.training_rows = int(len(labels))
        if len(classes) >= 2:
            weights = np.minimum(np.sqrt(counts.max() / counts), 8.0)
            sample_weights = weights[np.searchsorted(classes, labels)]
            sample_weights /= sample_weights.mean()
            self.model = SGDClassifier(
                loss='log_loss',
                max_iter=5,
                tol=None,
                random_state=seed,
                average=True,
            ).fit(x[attack], labels, sample_weight=sample_weights)

    def predict(self, x, fallback):
        """Preserve original subtype when the local head cannot be fitted."""
        if self.model is None or not len(x):
            return fallback
        return self.model.predict(x)


def add_subtype_mode(predictions, scores, diagnostics, subtype_guesses):
    """Reuse the exact binary gate of base_threshold."""
    reference, mode = 'base_threshold', 'base_threshold_subtype'
    started = perf_counter()
    predictions[mode] = [
        np.where(guess > 0, subtype, 0) for guess, subtype in zip(predictions[reference], subtype_guesses, strict=True)
    ]
    scores[mode] = scores[reference]
    diagnostics['calibration'].extend(
        [{**row, 'mode': mode} for row in diagnostics['calibration'] if row['mode'] == reference]
    )
    return {'stage': 'subtype_gate', 'mode': mode, 'elapsed_seconds': perf_counter() - started}


def subtype_diagnostics(labels, predicted, original_subtype, new_subtype, names):
    """Separate failure to alert from subtype errors on development VAL."""
    result = []
    for label, name in enumerate(names[1:], 1):
        mask = labels == label
        alarm = predicted[mask] > 0
        original = original_subtype[mask] == label
        candidate = new_subtype[mask] == label
        result.append(
            {
                'class': name,
                'rows': int(mask.sum()),
                'missed_alerts': int((~alarm).sum()),
                'original_wrong_given_alert': int((alarm & ~original).sum()),
                'candidate_wrong_given_alert': int((alarm & ~candidate).sum()),
                'original_correct_ignoring_gate': int(original.sum()),
                'candidate_correct_ignoring_gate': int(candidate.sum()),
            }
        )
    return result


def missing_class_repair(client, holdout, shared_cal, shared_eval, base_decisions, eligible, args):
    """Return decisions, binary probability forecasts by mode, and CAL status.

    Probability forecasts select the local or shared normalized model output by
    the corresponding routing rule. They are forecasts for ECE evaluation, with
    no fitted probability calibration or calibration guarantee. The FAR gates
    still use subtype evidence and the original CAL-only decision rules.
    """
    gate_fraction = getattr(args, 'repair_gate_target_fraction', 0.5)
    residual_fraction = getattr(args, 'repair_residual_budget_fraction', 1.0)
    union_fallback = getattr(args, 'repair_union_fallback', 'local')
    if union_fallback not in ('local', 'base'):
        raise ValueError('Repair union fallback must be \'local\' or \'base\'')
    if not np.isfinite(gate_fraction) or not 0 < gate_fraction <= 1:
        raise ValueError('Repair gate target fraction must be in (0, 1]')
    if not np.isfinite(residual_fraction) or not 0 <= residual_fraction <= 1:
        raise ValueError('Repair residual budget fraction must be in [0, 1]')
    provenance = {
        'repair_gate_target_fraction': gate_fraction,
        'repair_residual_budget_fraction': residual_fraction,
        'repair_union_fallback': union_fallback,
        'source_mode': 'base_threshold',
        'donor_kind': 'full_weight_fedavg',
        'score_kind': 'selected_branch_binary_probability',
    }
    labels = client.cal_labels[holdout]
    local_cal = client.cal_probabilities[holdout]

    def donor_evidence(p):
        subtype = 1 + p[:, 1:].argmax(axis=1)
        return subtype, np.where(eligible[subtype], p[np.arange(len(p)), subtype], 0.0)

    _, donor_cal = donor_evidence(shared_cal[holdout])
    donor_subtype, donor_score = donor_evidence(shared_eval)
    donor_budget = args.target_far * gate_fraction
    min_benign = getattr(args, 'threshold_min_benign_rows', 0)
    insufficient_policy = getattr(args, 'threshold_insufficient_policy', 'floor')
    donor_threshold, donor_status = empirical_alarm_threshold(
        np.column_stack((1 - donor_cal, donor_cal)),
        labels,
        donor_budget,
        0.0,
        min_benign,
        'disable',
    )
    donor_alert = donor_score > donor_threshold
    fixed = np.where(donor_alert & (base_decisions > 0), donor_subtype, base_decisions)
    base_threshold, fixed_status = empirical_alarm_threshold(
        local_cal,
        labels,
        args.target_far,
        args.threshold_floor,
        min_benign,
        insufficient_policy,
    )
    fixed_status.update(
        mode=REPAIR_FIXED,
        threshold_source='unchanged base_threshold alarm',
        donor_threshold=donor_threshold,
        eligible_donor_classes=int(eligible.sum()),
        cal_fit_rows=len(client.cal_labels) - len(holdout),
        cal_holdout_rows=len(holdout),
        score_notes=(
            'Local sum of attack probabilities on every row; subtype replacement '
            'preserves local attack mass. No fitted probability calibration.'
        ),
        **provenance,
    )

    local_budget = args.target_far * (1 - gate_fraction) if eligible.any() else args.target_far
    if local_budget == 0:
        # A zero allocation disables the branch, including scores above the CAL maximum.
        local_threshold, local_status = 1.0, {'calibration_far': 0.0 if (labels == 0).any() else None}
    else:
        local_threshold, local_status = empirical_alarm_threshold(
            local_cal,
            labels,
            local_budget,
            args.threshold_floor,
            min_benign,
            insufficient_policy,
        )
    local = hierarchical_decisions(client.eval_probabilities, local_threshold)
    benign = labels == 0
    positive = labels > 0

    def _cal_fnr(alerts):
        return float((positive & ~alerts).sum() / positive.sum()) if positive.any() else None

    cal_base_alert = hierarchical_decisions(local_cal, base_threshold)['attack_alert']
    cal_local = hierarchical_decisions(local_cal, local_threshold)['attack_alert']
    unalerted_scores = donor_cal[benign & ~cal_base_alert]
    remaining_base_budget = max(0, int(args.target_far * benign.sum()) - int((cal_base_alert & benign).sum()))

    def residual_threshold(budget):
        # Strict comparison handles ties conservatively at the budget boundary.
        if not benign.any() or benign.sum() < min_benign or budget <= 0:
            return 1.0
        if budget >= len(unalerted_scores):
            return 0.0
        boundary = len(unalerted_scores) - budget - 1
        return float(np.partition(unalerted_scores, boundary)[boundary])

    union_threshold = donor_threshold
    if union_fallback == 'base':
        union_threshold = max(donor_threshold, residual_threshold(remaining_base_budget))
    union_alert = donor_score > union_threshold
    union_fallback_alert = cal_base_alert if union_fallback == 'base' else cal_local
    cal_union = union_fallback_alert | (donor_cal > union_threshold)
    union = np.where(union_alert, donor_subtype, base_decisions if union_fallback == 'base' else local['predicted'])
    status = {
        **provenance,
        'mode': REPAIR_UNION,
        'threshold_source': (
            'CAL holdout, residual benign FAR budget'
            if union_fallback == 'base'
            else 'CAL holdout, split benign FAR budget'
        ),
        'cal_fit_rows': len(client.cal_labels) - len(holdout),
        'cal_holdout_rows': len(holdout),
        'benign_holdout_rows': int(benign.sum()),
        'target_far': args.target_far,
        'threshold': base_threshold if union_fallback == 'base' else local_threshold,
        'threshold_floor': args.threshold_floor,
        'donor_threshold': union_threshold,
        'local_target_far': args.target_far if union_fallback == 'base' else local_budget,
        'remaining_base_benign_budget': remaining_base_budget,
        'donor_support_disabled': donor_status['alert_gate_disabled_for_support'],
        'donor_target_far': donor_budget if eligible.any() else 0.0,
        'local_calibration_far': (
            fixed_status['calibration_far'] if union_fallback == 'base' else local_status['calibration_far']
        ),
        'donor_calibration_far': (float((donor_cal[benign] > union_threshold).mean()) if benign.any() else None),
        'calibration_far': float(cal_union[benign].mean()) if benign.any() else None,
        'cal_fnr_base': _cal_fnr(cal_base_alert),
        'cal_fnr_local': _cal_fnr(cal_local),
        'cal_fnr_union': _cal_fnr(cal_union),
        'no_benign_fallback': not bool(benign.any()),
        'formal_conformal_guarantee': False,
        'calibration_rule': (
            'base-preserving donor union; every base alarm is retained'
            if union_fallback == 'base'
            else 'strict union of two benign CAL quantile gates'
        ),
        'eligible_donor_classes': int(eligible.sum()),
        'score_notes': (
            'Shared sum of attack probabilities where the donor gate fires; local sum '
            'elsewhere. Branch selection does not guarantee probability calibration.'
        ),
    }
    available_remaining = remaining_base_budget
    remaining = int(np.floor(residual_fraction * available_remaining))
    # The historical policy is retained for old run configurations.
    if union_fallback == 'local' and benign.any() and benign.sum() >= min_benign and residual_fraction > 0:
        if remaining >= len(unalerted_scores):
            rescue_threshold = 0.0
        else:
            boundary = len(unalerted_scores) - remaining - 1
            rescue_threshold = float(np.partition(unalerted_scores, boundary)[boundary])
    else:
        rescue_threshold = residual_threshold(remaining)
    rescue_alert = (base_decisions == 0) & (donor_score > rescue_threshold)
    rescue = np.where(rescue_alert, donor_subtype, fixed)
    cal_rescue = cal_base_alert | (donor_cal > rescue_threshold)
    rescue_status = {
        **fixed_status,
        'mode': REPAIR_RESCUE,
        'threshold_source': 'base CAL alarm with residual-budget rescue',
        'rescue_threshold': rescue_threshold,
        'remaining_benign_cal_false_alerts': remaining,
        'available_benign_cal_false_alerts': available_remaining,
        'calibration_rule': 'preserve base alerts and spend residual benign CAL budget',
        'calibration_far': float(cal_rescue[benign].mean()) if benign.any() else None,
        'cal_fnr_base': _cal_fnr(cal_base_alert),
        'cal_fnr_rescue': _cal_fnr(cal_rescue),
        'score_notes': (
            'Shared sum of attack probabilities only for added rescue alerts; local sum '
            'for existing alerts and all other rows. No fitted probability calibration.'
        ),
    }
    # Each q is the attack mass of a selected normalized branch distribution,
    # never the maximum of two unrelated evidence scores. Fixed subtype changes
    # can redistribute local attack mass without changing its binary forecast.
    local_probability = local['attack_score']
    shared_probability = np.clip(shared_eval[:, 1:].sum(axis=1), 0.0, 1.0)
    probabilities = {
        REPAIR_FIXED: local_probability,
        REPAIR_UNION: np.where(union_alert, shared_probability, local_probability),
        REPAIR_RESCUE: np.where(rescue_alert, shared_probability, local_probability),
    }
    return (
        {REPAIR_FIXED: fixed, REPAIR_UNION: union, REPAIR_RESCUE: rescue},
        probabilities,
        (fixed_status, status, rescue_status),
    )


def transfer_rows(parts, roster, fit_support, names, predictions, unseen):
    """Evaluate predefined receiver/class cells on all routed unique TEST rows."""
    train_y, test_y = parts['train'][1], parts['test'][1]
    fits = np.asarray(fit_support)
    donor = fits.sum(axis=0) - fits
    repair_modes = tuple(mode for mode in predictions if mode.startswith(REPAIR_MODES))
    repair = bool(repair_modes)
    modes = (*TRANSFER_MODES, 'base_threshold', *repair_modes) if repair else TRANSFER_MODES
    modes = tuple(
        dict.fromkeys(
            (
                *modes,
                *(
                    mode
                    for mode in predictions
                    if mode in ('fedavg', 'fedavg_threshold') or mode.startswith('conditional_')
                ),
            )
        )
    )
    train_total = np.bincount(train_y, minlength=len(names)) if repair else None
    result = []
    for client_index, client in enumerate(roster):
        train_counts = np.bincount(train_y[client['train']], minlength=len(names))
        selected = unseen[client['test']]
        labels = test_y[client['test'][selected]]
        for label, name in enumerate(names[1:], 1):
            if fits[client_index, label] or donor[client_index, label] < MIN_DONOR_ROWS:
                continue
            true = labels == label
            if not true.any():
                continue
            support = int(true.sum())
            for mode in modes:
                guessed = predictions[mode][client_index][len(client['val']) :][selected] == label
                predicted = int(guessed.sum())
                true_positive = int((guessed & true).sum())
                row = {
                    'client': f'client_{client_index}',
                    'class': name,
                    'mode': mode,
                    'local_train_rows': int(train_counts[label]),
                    'local_cal_fit_rows': int(fits[client_index, label]),
                    'other_cal_fit_rows': int(donor[client_index, label]),
                    'transfer_scope': ('train_and_fit_absent' if train_counts[label] == 0 else 'fit_only_absent'),
                    'test_support': support,
                    'predicted': predicted,
                    'true_positive': true_positive,
                    'recall': true_positive / support,
                    'precision': true_positive / predicted if predicted else None,
                    'f1': 2 * true_positive / (support + predicted),
                }
                if repair:
                    other_train = int(train_total[label] - train_counts[label])
                    row.update(
                        other_train_rows=other_train,
                        repair_eligible=bool(train_counts[label] == 0 and other_train >= MIN_DONOR_TRAIN_ROWS),
                    )
                result.append(row)
    return result


class TransferScoreDiagnostics:
    """Collect evaluation statistics on the same inputs as the hard transfer analysis."""

    def __init__(self, parts, roster, names, unseen):
        if len(names) < 3:
            raise ValueError('Transfer score diagnostics need at least two attack classes')
        self.class_ids = {name: i for i, name in enumerate(names)}
        self.receivers = {f'client_{i}': i for i in range(len(roster))}
        self.samples = []
        for client in roster:
            test = client['test']
            labels = parts['test'][1][test]
            selected = unseen[test] & (labels > 0)
            eval_rows = len(client['val']) + np.flatnonzero(selected)
            self.samples.append((eval_rows, labels[selected]))
        self.scores = {}

    def observe(self, mode, client, probabilities, predicted):
        """Store compact vectors for novel attack rows while probabilities are available."""
        if mode not in TRANSFER_MODES:
            return
        eval_rows, labels = self.samples[client]
        attack = probabilities[eval_rows, 1:]
        rows, target = np.arange(len(labels)), labels - 1
        true_score = attack[rows, target]
        earlier = np.arange(attack.shape[1]) < target[:, None]
        higher = (attack > true_score[:, None]) | ((attack == true_score[:, None]) & earlier)
        rank = 1 + higher.sum(axis=1)
        attack[rows, target] = 0
        floor = np.finfo(np.float64).tiny
        logp = np.log(np.maximum(true_score, floor))
        margin = logp - np.log(np.maximum(attack.max(axis=1), floor))
        decisions = predicted[eval_rows]
        self.scores[mode, client] = (logp, rank, margin, decisions > 0, decisions == labels)

    def paired_rows(self, transfer):
        """Report same-input, same-budget changes without exporting per-input values."""
        pairs = {local: (contrast, fed) for contrast, (fed, local) in TRANSFER_PAIRS.items()}
        result = []
        for row in transfer:
            local = row['mode']
            if local not in pairs:
                continue
            contrast, fed = pairs[local]
            client = self.receivers[row['client']]
            label = self.class_ids[row['class']]
            selected = self.samples[client][1] == label
            if int(selected.sum()) != row['test_support']:
                raise ValueError('Transfer score and hard-decision support differ')
            left = [values[selected] for values in self.scores[fed, client]]
            right = [values[selected] for values in self.scores[local, client]]
            logp, rank, margin = (a - b for a, b in zip(left[:3], right[:3], strict=True))
            fed_alert, fed_correct = left[3:]
            local_alert, local_correct = right[3:]
            fed_top1, local_top1 = left[1] == 1, right[1] == 1
            result.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            'client',
                            'class',
                            'transfer_scope',
                            'test_support',
                            'local_train_rows',
                            'local_cal_fit_rows',
                            'other_cal_fit_rows',
                        )
                    },
                    'contrast': contrast,
                    'delta_logp_mean': float(logp.mean()),
                    'delta_logp_median': float(np.median(logp)),
                    'logp_increase_rate': float((logp > 0).mean()),
                    'logp_decrease_rate': float((logp < 0).mean()),
                    'delta_attack_rank_mean': float(rank.mean()),
                    'attack_rank_improved_rate': float((rank < 0).mean()),
                    'delta_attack_log_margin_mean': float(margin.mean()),
                    'delta_top1_rate': float(fed_top1.mean() - local_top1.mean()),
                    'delta_alert_rate': float(fed_alert.mean() - local_alert.mean()),
                    'delta_recall': float(fed_correct.mean() - local_correct.mean()),
                    'local_alert_rate': float(local_alert.mean()),
                    'federated_alert_rate': float(fed_alert.mean()),
                    'local_top1_rate': float(local_top1.mean()),
                    'federated_top1_rate': float(fed_top1.mean()),
                    'local_top1_no_alert_rate': float((local_top1 & ~local_alert).mean()),
                    'federated_top1_no_alert_rate': float((fed_top1 & ~fed_alert).mean()),
                    'local_alert_wrong_subtype_rate': float((local_alert & ~local_top1).mean()),
                    'federated_alert_wrong_subtype_rate': float((fed_alert & ~fed_top1).mean()),
                    'local_recall': float(local_correct.mean()),
                    'federated_recall': float(fed_correct.mean()),
                }
            )
        return result
