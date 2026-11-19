"""Diagnose the frozen Hulk alarm and run a prespecified CAL threshold replay."""

import csv
import json
from pathlib import Path
from statistics import fmean, stdev

import matplotlib
import numpy as np
from scipy.stats import t
from sklearn.metrics import average_precision_score, roc_auc_score

from .revision_hulk_2018_cal import load_original_calibration
from .revision_hulk_2018_features import write_feature_audit
from .revision_hulk_2018 import SEEDS, sha256_file
from .revision_external import _frozen_models, _load_capture, _plan, _read_screen
from .revision_operating_selection import _fit_threshold
from .revision_report import score
from .revision_shared_head import head_probabilities
from .revision_subtype_guard import guarded_subtype_predictions
from .utils import hierarchical_decisions

matplotlib.use('Agg')
from matplotlib import pyplot as plt


REFERENCE_FAR = 0.01
MIN_RECALL_GAIN = 0.05
MIN_SEEDS = 3
FAR_GRID = np.linspace(0, REFERENCE_FAR, 101)
METHODS = ('base_threshold', 'head_local', 'head_donor', 'head_federated')
POLICIES = ('frozen', 'cal_1pct_with_floor', 'cal_1pct_without_floor')
METHOD_LABELS = {
    'base_threshold': 'Frozen TabM',
    'head_local': 'Local subtype head',
    'head_donor': 'Donor pool',
    'head_federated': 'Federated subtype head',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pilot(pilot):
    """Verify the existing seal, inputs and exact-overlap screen before replay."""
    original = read_json(pilot / 'score/seal.json')
    current, infos = _plan([case['file'] for case in original['cases']], original['manifest'])
    if current != original or current['seeds'] != SEEDS or len(current['captures']) != 1:
        raise ValueError('The archived one-capture pilot differs from its original seal.')
    screen = infos[0]['development_screen']
    x, y, _, audit = _load_capture(
        current['captures'][0], infos[0], _read_screen(screen),
        set(screen['development_group_ids']), current['exact_overlap_policy'],
    )
    if audit['groups'] != 1 or audit['rows_flagged_by_development_screen']:
        raise ValueError('The pilot capture support differs from its verified scope.')
    return current, infos, x, y, audit


def recall_curve(scores, y, index, benign_scores, thresholds):
    """Count strict exceedances at prespecified benign FAR budgets."""
    attack = np.sort(scores[y == index])
    far = (len(benign_scores) - np.searchsorted(benign_scores, thresholds, side='right')) / len(benign_scores)
    coverage = (len(attack) - np.searchsorted(attack, thresholds, side='right')) / len(attack)
    return far, coverage


def diagnose_seed(info, detector, x, y, hulk, slow, output):
    """Retain frozen scores and compute descriptive, single-capture operating curves."""
    seed = info['seed']
    cutoff = info['fixed_gate']['threshold']
    decision = hierarchical_decisions(detector(x), cutoff)
    scores, subtype, flagged = (decision[key] for key in ('attack_score', 'subtype', 'attack_alert'))
    np.save(output / f'attack_scores_seed_{seed}.npy', scores, allow_pickle=False)
    benign = np.sort(scores[y == 0])
    index = np.ceil((len(benign) - 1) * (1 - FAR_GRID)).astype(int)
    thresholds = benign[index]
    curve_rows, diagnostics, quantiles = [], [], []
    for cls in (0, hulk, slow):
        mask = y == cls
        p05, p50, p95 = np.quantile(scores[mask], [0.05, 0.5, 0.95])
        quantiles.append({
            'seed': seed, 'class': info['class_order'][cls], 'rows': int(mask.sum()),
            'q05': p05, 'median': p50, 'q95': p95,
        })
        if cls == 0:
            continue
        far, recall = recall_curve(scores, y, cls, benign, thresholds)
        curve_rows.extend({
            'seed': seed, 'class': info['class_order'][cls], 'far_budget': cap,
            'observed_far': f, 'alert_recall': r,
        } for cap, f, r in zip(FAR_GRID, far, recall, strict=True))
        relevant = (y == 0) | mask
        truth, ranking = mask[relevant], scores[relevant]
        diagnostics.append({
            'seed': seed, 'class': info['class_order'][cls], 'frozen_threshold': cutoff,
            'frozen_far': float(flagged[y == 0].mean()),
            'frozen_alert_recall': float(flagged[mask].mean()),
            'empirical_far_at_1pct_budget': far[-1],
            'empirical_recall_at_1pct_budget': recall[-1],
            'recall_difference': float(recall[-1] - flagged[mask].mean()),
            'average_precision_vs_benign': average_precision_score(truth, ranking),
            'standardized_partial_auc_at_1pct': roc_auc_score(truth, ranking, max_fpr=REFERENCE_FAR),
        })
    return (scores, subtype), curve_rows, diagnostics, quantiles


def align_with_pilot(pilot, diagnostics):
    """Compare regenerated frozen operating points with the archived scoring."""
    with (pilot / 'score/result/fixed_alarm_transitions.csv').open(encoding='utf-8', newline='') as handle:
        archived = {
            (int(row['seed']), row['class']): float(row['recall_ceiling'])
            for row in csv.DictReader(handle)
            if row['row_type'] == 'class_summary' and row['mode'] == 'head_federated'
            and row['class'] in ('DoS Hulk', 'DoS Slowhttptest')
        }
    for row in diagnostics:
        key = (row['seed'], row['class'])
        if row['frozen_alert_recall'] != archived[key]:
            raise ValueError(f'Regenerated frozen pilot score differs from its receipt: {key}')


def plot_curves(output, curve_rows, diagnostics):
    """Plot the two class-specific curves with archived operating points."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65), layout='constrained', sharey=True)
    for ax, attack in zip(axes, ('DoS Hulk', 'DoS Slowhttptest'), strict=True):
        for seed in SEEDS:
            rows = [r for r in curve_rows if r['seed'] == seed and r['class'] == attack]
            line, = ax.plot([100*r['observed_far'] for r in rows],
                            [100*r['alert_recall'] for r in rows], label=str(seed), linewidth=1.3)
            orig, = [r for r in diagnostics if r['seed'] == seed and r['class'] == attack]
            ax.scatter(100*orig['frozen_far'], 100*orig['frozen_alert_recall'],
                       color=line.get_color(), edgecolor='white', linewidth=0.4, s=28, zorder=4)
        ax.set(title=attack, xlabel='False alarm rate (%)', xlim=(0, 1), ylim=(0, 102))
        ax.grid(alpha=0.18)
    axes[0].set_ylabel('Binary alarm recall (%)')
    axes[0].legend(title='Seed', frameon=False, loc='upper left', fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'low_far_curves.{suffix}', dpi=350)
    plt.close(fig)


def comparison_row(info, policy, cutoff, y, predictions, scores, hulk, slow, method):
    metrics, classes, _ = score(y, predictions, scores, info['class_order'])
    alerts = predictions > 0
    target = classes[hulk]
    return {
        'seed': info['seed'], 'policy': policy, 'method': method, 'threshold': cutoff,
        'far': metrics['far'], 'fnr': metrics['fnr'], 'macro_f1': metrics['macro_f1'],
        'hulk_alarm_recall': float(alerts[y == hulk].mean()),
        'slowhttp_alarm_recall': float(alerts[y == slow].mean()),
        'hulk_precision': target['precision'], 'hulk_recall': target['recall'], 'hulk_f1': target['f1'],
        'slowhttp_to_hulk_rows': int(np.sum((y == slow) & (predictions == hulk))),
    }


def compare_thresholds(models, cached, calibrations, x, y, hulk, slow):
    """Fit the two source CAL thresholds and replay each frozen subtype head."""
    results, cal_rows = [], []
    for (info, states, detector), (scores, subtype) in zip(models, cached, strict=True):
        seed = info['seed']
        fixed = info['fixed_gate']
        fit_x, fit_y = calibrations[seed]
        cal_scores = hierarchical_decisions(detector(fit_x))['attack_score']
        operating = fixed['operating_selection']
        support = operating['constraints']['min_benign_rows']
        cutoffs = {'frozen': fixed['threshold']}
        for policy, floor in (('cal_1pct_with_floor', 0.5), ('cal_1pct_without_floor', 0.0)):
            cutoff, record = _fit_threshold(cal_scores, fit_y, REFERENCE_FAR, floor, support, pooled_fallback=False)
            if record['alert_gate_disabled_for_support']:
                raise ValueError(f'Insufficient archived CAL threshold-fit support for seed {seed}.')
            cutoffs[policy] = cutoff
            cal_rows.append({
                'seed': seed, 'cal_role': '2017 receiver CAL threshold fit', 'policy': policy,
                'target_far': REFERENCE_FAR, 'floor': floor, 'threshold': cutoff,
                'frozen_selected_threshold': fixed['threshold'],
                'benign_rows': int(np.sum(fit_y == 0)), 'cal_far': record['calibration_far'],
            })
        if not np.isclose(cutoffs['cal_1pct_with_floor'], fixed['threshold'], rtol=0, atol=1e-8):
            raise ValueError(f'Reconstructed 1% CAL threshold does not reproduce the archived selection: seed {seed}.')
        base = {policy: np.where(scores > cutoff, subtype, 0) for policy, cutoff in cutoffs.items()}
        for method in METHODS:
            probabilities = (None if method == 'base_threshold' else
                             head_probabilities(states[method], x, device='cpu',
                                                batch_size=info['settings']['protocol']['head_batch_size']))
            for policy, cutoff in cutoffs.items():
                predictions = base[policy]
                if probabilities is not None:
                    predictions = guarded_subtype_predictions(
                        predictions, probabilities, fixed['eligible_attack_indices'], fixed['subtype_guards'][method]
                    )
                if not np.array_equal(predictions > 0, base[policy] > 0):
                    raise RuntimeError('A subtype head changed the binary alarm mask.')
                results.append(comparison_row(info, policy, cutoff, y, predictions, scores, hulk, slow, method))
        print(f'Seed {seed}: CAL thresholds and all three masks replayed.', flush=True)
    return cal_rows, results


def paired_comparison(rows):
    """Summarize within-seed differences on this single external capture."""
    lookup = {(r['seed'], r['policy'], r['method']): r for r in rows}
    pairs = []
    for method in METHODS:
        for key in ('far', 'fnr', 'hulk_alarm_recall', 'hulk_f1', 'macro_f1', 'slowhttp_to_hulk_rows'):
            delta = [lookup[(seed, POLICIES[2], method)][key] - lookup[(seed, POLICIES[1], method)][key]
                     for seed in SEEDS]
            center, sd = fmean(delta), stdev(delta)
            half = t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta))
            pairs.append({'method': method, 'contrast': 'without_floor_minus_with_floor', 'metric': key,
                          'seeds': len(delta), 'mean': center, 'sd': sd,
                          'nominal_ci95_low': center - half, 'nominal_ci95_high': center + half})
    return pairs


def run_hulk_2018_diagnostics(output, pilot, frozen_source, project):
    """Run the complete frozen diagnosis, including a conditional threshold replay."""
    plan, infos, x, y, audit = load_pilot(pilot)
    names = infos[0]['class_order']
    hulk, slow = (names.index(label) for label in ('DoS Hulk', 'DoS Slowhttptest'))
    output.mkdir(exist_ok=False)
    write_json(output / 'plan.json', {
        'pilot': str(pilot), 'pilot_seal_sha256': sha256_file(pilot / 'score/seal.json'),
        'frozen_source_run': str(frozen_source), 'seeds': SEEDS, 'support': audit,
        'scope': 'exploratory diagnosis of one previously examined public capture',
        'empirical_far_budget': REFERENCE_FAR, 'minimum_recall_gain': MIN_RECALL_GAIN,
        'minimum_seeds_for_replay': MIN_SEEDS, 'threshold_fit_role': '2017 receiver CAL threshold fit',
        'calibration_rule': 'higher quantile at 1% FAR; strict exceedance; same original CAL rows',
        'subtype_replay': 'original frozen guards; rewrite budgets evaluated on the original mask only',
        'analysis_source_sha256': {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (Path(__file__).name, 'revision_hulk_2018_cal.py', 'revision_hulk_2018_features.py')
        },
    })
    print(f'Diagnostic outputs: {output}', flush=True)
    np.save(output / 'labels.npy', y, allow_pickle=False)
    models = _frozen_models(plan, infos, fixed=True)
    cached, curve_rows, diagnostics, quantiles = [], [], [], []
    for info, _, detector in models:
        cache, curves, summaries, sample = diagnose_seed(info, detector, x, y, hulk, slow, output)
        cached.append(cache)
        curve_rows.extend(curves)
        diagnostics.extend(summaries)
        quantiles.extend(sample)
        print(f'Seed {info["seed"]}: frozen alarm scores and low FAR curves complete.', flush=True)
    align_with_pilot(pilot, diagnostics)
    write_csv(output / 'score_quantiles.csv', quantiles)
    write_csv(output / 'score_diagnostics.csv', diagnostics)
    write_csv(output / 'low_far_curves.csv', curve_rows)
    plot_curves(output, curve_rows, diagnostics)

    hulk_rows = [row for row in diagnostics if row['class'] == 'DoS Hulk']
    passing = sum(row['recall_difference'] >= MIN_RECALL_GAIN for row in hulk_rows)
    replay = passing >= MIN_SEEDS
    print(f'Low FAR gate: {passing}/{len(SEEDS)} seeds; threshold replay = {replay}.', flush=True)
    calibrations, development = load_original_calibration(project, frozen_source, infos)
    write_feature_audit(output, x, y, *development, infos[0]['feature_order'], names, development_role='grouped_val')
    del development

    result_rows = []
    if replay:
        cal_rows, result_rows = compare_thresholds(models, cached, calibrations, x, y, hulk, slow)
        write_csv(output / 'calibration.csv', cal_rows)
        write_csv(output / 'threshold_replay.csv', result_rows)
        write_csv(output / 'paired_replay.csv', paired_comparison(result_rows))

    lines = [
        '# Frozen Hulk alarm diagnosis', '',
        'Exploratory analysis of one previously examined CSE-CIC-IDS2018 capture.',
        'All records and all five original seeds were retained.', '',
        '| Diagnostic | Value |', '| --- | ---: |',
        f'| Frozen Hulk alarm recall, seed mean | {100*fmean(r["frozen_alert_recall"] for r in hulk_rows):.3f}% |',
        f'| Descriptive Hulk alarm recall at FAR at most 1%, seed mean | '
        f'{100*fmean(r["empirical_recall_at_1pct_budget"] for r in hulk_rows):.3f}% |',
        f'| Seeds with at least 5 points of descriptive recall gain | {passing} of {len(SEEDS)} |',
        f'| Prespecified 2017 CAL threshold replay | {"completed" if replay else "gate not met"} |', '',
        'The descriptive curve uses 2018 labels to measure separability. All replay thresholds are set on 2017 CAL.',
        'The replay holds the subtype guards fixed. Their original check concerns the frozen alarm mask.', '',
        '[Feature comparison](feature_audit.csv) · [Score diagnostics](score_diagnostics.csv) · [Low FAR curves](low_far_curves.pdf)',
    ]
    if replay:
        lines.extend(('', '| Policy | FAR (%) | FNR (%) | Hulk alarm recall (%) | Federated Hulk F1 (%) |',
                      '| --- | ---: | ---: | ---: | ---: |'))
        for policy in POLICIES:
            rows = [r for r in result_rows if r['policy'] == policy and r['method'] == 'head_federated']
            values = [f'{100*fmean(r[k] for r in rows):.3f}' for k in ('far', 'fnr', 'hulk_alarm_recall', 'hulk_f1')]
            lines.append('| ' + ' | '.join((policy, *values)) + ' |')
        lines.append('[Per-seed threshold replay](threshold_replay.csv) · [Paired seed differences](paired_replay.csv)')
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    receipt = {
        'status': 'completed', 'replay': replay, 'passing_seeds': passing,
        'seed_count': len(SEEDS), 'rows': len(y),
        'summary_path': str(output / 'SUMMARY.md'),
        'artifacts_sha256': {
            path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / 'complete.json', receipt)
    return receipt
