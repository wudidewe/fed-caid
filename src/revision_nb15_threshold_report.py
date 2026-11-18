"""Summarize NB15 threshold and weight experiments without mixing their inputs."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from .revision_fresh_report import _artifact_path, _verified_files
from .revision_v12 import _hash, _read, _write

THRESHOLD_ARMS = {
    'reference_current': '原 CAL 1.0%',
    'matched_cal_1pct': '分组重划 CAL 1.0%',
    'conservative_cal_0_9pct': '分组重划 CAL 0.9%',
}
WEIGHT_ARMS = {
    'reference': '参考 TabM',
    'frozen_head': '冻结骨干',
    'full_finetune': '全量微调',
}
TABLES = ('working_points', 'client_working_points', 'per_class')
SCOPES = {'receiver': '接收端', 'overall': '整体'}
CLASSES = ('Exploits', 'Analysis')


def _number(value):
    return None if value in (None, '') else float(value)


def _percent(value, places=2):
    return '—' if value is None else f'{100 * value:.{places}f}%'


def _points(value):
    return '—' if value is None else f'{100 * value:+.2f}'


def _mean(rows, field):
    values = [_number(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _table(headers, rows):
    """Build a compact Markdown table from already formatted fields."""
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *('| ' + ' | '.join(map(str, row)) + ' |' for row in rows),
    ]


def _operating_cells(arm, scope, rows, cap):
    rates = [_number(row['val_far']) for row in rows]
    measured = [rate for rate in rates if rate is not None]
    exceeding = sum(rate > cap for rate in measured)
    return (
        arm,
        scope,
        _percent(_mean(rows, 'val_far'), 3),
        _percent(max(measured, default=None), 3),
        f'{exceeding}/{len(measured)}',
        _percent(_mean(rows, 'val_fnr')),
        _percent(_mean(rows, 'val_macro_f1')),
        _percent(_mean(rows, 'val_binary_ece_10'), 3),
    )


def _operating_table(rows, arms, cap):
    return _table(
        ('分支', '范围', '平均 FAR', '最高 FAR', '超限种子', '平均 FNR', '平均 macro-F1', '二元 ECE'),
        (
            _operating_cells(label, scope_label, selected, cap)
            for arm, label in arms.items()
            for scope, scope_label in SCOPES.items()
            if (selected := [row for row in rows if row['arm'] == arm and row['scope'] == scope])
        ),
    )


def _paired_interval(entry):
    point, lower, upper = (entry.get(key) for key in ('mean', 'ci95_low', 'ci95_high'))
    return f'{_points(point)} [{_points(lower)}, {_points(upper)}]'


def _paired_table(assessment, arms):
    return _table(
        ('配对对照', '范围', 'ΔFAR', 'ΔFNR', 'Δmacro-F1'),
        (
            (
                f'{arms[candidate]} − {arms[reference]}',
                SCOPES[scope],
                *(_paired_interval(record.get(metric, {})) for metric in ('val_far', 'val_fnr', 'val_macro_f1')),
            )
            for contrast, scopes in assessment['paired_intervals'].items()
            for candidate, reference in [contrast.split('_minus_')]
            for scope, record in scopes.items()
        ),
    )


def _verdict(value):
    return '待评估' if value is None else ('通过' if value else '未通过')


def _cal_check(candidate):
    check = candidate.get('cal_check')
    if check is None:
        return '—'
    return f"{_verdict(check['pass'])}（{check['clients_passing']}/{check['clients_expected']}）"


def _guard_table(assessment, arms):
    return _table(
        ('候选', '范围', '逐种子 FAR', '配对 FNR', '平均 macro-F1', '独立 CAL 检查', '联合验收'),
        (
            (
                arms[arm],
                SCOPES[scope],
                _verdict(record['far_pass']),
                _verdict(record['paired_fnr_pass']),
                _verdict(record['macro_f1_mean_non_decrease']),
                _cal_check(candidate),
                _verdict(candidate['development_guard_pass']),
            )
            for arm, candidate in assessment['candidates'].items()
            for scope, record in candidate['scopes'].items()
        ),
    )


def _support_range(rows):
    sizes = [int(row['support']) for row in rows]
    low, high = min(sizes), max(sizes)
    return str(low) if low == high else f'{low}–{high}'


def _class_table(rows, arms):
    return _table(
        ('接收端类别', '分支', '逐种子支持数', '平均二元告警召回', '平均子类召回', '平均子类 F1'),
        (
            (
                name,
                label,
                _support_range(selected),
                *(_percent(_mean(selected, metric)) for metric in ('binary_recall', 'recall', 'f1')),
            )
            for name in CLASSES
            for arm, label in arms.items()
            if (
                selected := [
                    row
                    for row in rows
                    if row['scope'] == 'receiver' and row['class_name'] == name and row['arm'] == arm
                ]
            )
        ),
    )


def _threshold_table(rows):
    indexed = {(int(row['seed']), int(row['client']), row['arm']): row for row in rows}
    cases = sorted({(seed, client) for seed, client, _ in indexed})
    return _table(
        ('种子', '客户端', *THRESHOLD_ARMS.values()),
        (
            (seed, client, *(str(indexed[seed, client, arm]['threshold']) for arm in THRESHOLD_ARMS))
            for seed, client in cases
        ),
    )


def _feasibility_section(assessment):
    evidence = assessment.get('receiver_threshold_only_feasibility')
    if evidence is None:
        return []
    values = evidence['per_seed_lower_bounds']
    return [
        '',
        '## 接收端阈值可行性诊断',
        '',
        '对各种子 VAL 分数回顾性计算 FAR 不超过上限时可达到的最低 FNR。' '这一诊断仅用于确定当前分数的可行范围，所有正式阈值由 CAL 确定。',
        '',
        *_table(
            ('种子', '相对参考规则的最低 FNR 增量（百分点）'), ((row['seed'], _points(row['min_val_fnr_increment'])) for row in values)
        ),
        '',
        f"平均增量下界为 {_points(evidence['mean_increment_lower_bound'])} 个百分点。"
        f"联合约束下配对 95% 区间上界的下界为 {_points(evidence['paired_upper_95_lower_bound'])} 个百分点，"
        f"预设上限为 {_points(evidence['fnr_budget'])} 个百分点。",
        '在本批 VAL 分数上，单靠调整接收端阈值无法同时满足两项门槛。'
        if evidence['joint_fnr_and_far_impossible_on_this_val']
        else '这一最低界限保留了满足两项门槛的可能性。',
    ]


def _common_sections(tables, assessment, arms):
    return [
        '## VAL 工作点',
        '',
        *_operating_table(tables['working_points'], arms, assessment['far_cap']),
        '',
        '## 配对效果及 95% 区间（百分点）',
        '',
        *_paired_table(assessment, arms),
        '',
        '## 开发验收',
        '',
        *_guard_table(assessment, arms),
        '',
        '## 重点少数类',
        '',
        *_class_table(tables['per_class'], arms),
    ]


def write_threshold_report(output, tables, assessment):
    """Write the current run's threshold working points and exact cutoff summary."""
    output = Path(output)
    lines = [
        '# NB15 分组 CAL 阈值专项',
        '',
        '同一轮新训练的参考 TabM 分数用于三个阈值分支。' '两条分组重划规则共享 CAL 拟合与检查分组，VAL 衡量锁定工作点。' '证据范围是同源分组开发数据。',
        '',
        *_common_sections(tables, assessment, THRESHOLD_ARMS),
        '',
        '## 各种子和客户端的锁定阈值',
        '',
        *_threshold_table(tables['client_working_points']),
        *_feasibility_section(assessment),
        '',
        '[逐种子结果](working_points.csv) · [客户端 CAL 检查和 VAL 指标](client_working_points.csv)'
        ' · [全部逐类结果](per_class.csv) · [开发验收与配对区间](assessment.json)',
    ]
    return _write_markdown(output, lines)


def _weighted_client_ranking(clients):
    per_seed = defaultdict(list)
    for row in clients:
        rank_tpr = _number(row['val_tpr_at_far_1pct_diagnostic'])
        attacks = int(row['val_attack_rows'])
        if rank_tpr is not None and attacks:
            per_seed[row['arm'], row['seed']].append((rank_tpr, attacks))
    weighted = defaultdict(list)
    for (arm, _), pairs in per_seed.items():
        weighted[arm].append(sum(tpr * attacks for tpr, attacks in pairs) / sum(n for _, n in pairs))
    return {arm: mean(values) for arm, values in weighted.items()}


def _weight_ranking_table(tables):
    weighted = _weighted_client_ranking(tables['client_working_points'])
    return _table(
        ('分支', '接收端', '逐客户端按攻击支持加权', '整体直接合并分数'),
        (
            (
                label,
                _percent(
                    _mean([row for row in points if row['scope'] == 'receiver'], 'val_tpr_at_far_1pct_diagnostic')
                ),
                _percent(weighted.get(arm)),
                _percent(_mean([row for row in points if row['scope'] == 'overall'], 'val_tpr_at_far_1pct_diagnostic')),
            )
            for arm, label in WEIGHT_ARMS.items()
            for points in [[row for row in tables['working_points'] if row['arm'] == arm]]
        ),
    )


def _write_markdown(output, lines):
    summary = output / 'SUMMARY.md'
    with summary.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return summary


def _read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _weight_evidence(source):
    """Read the original result tables only after verifying bound artifacts."""
    complete = _artifact_path(source, 'complete.json')
    digest = _hash(complete)
    receipt = _read(complete)
    bindings = receipt.get('artifacts', {})
    required = {'assessment.json', 'plan.json', *(f'{name}.csv' for name in TABLES)}
    if receipt.get('status') != 'completed' or not required <= bindings.keys():
        raise ValueError('Expected a finalized NB15 weight experiment')
    paths = {name: path for name, path, _ in _verified_files(source, bindings)}
    tables = {name: _read_csv(paths[f'{name}.csv']) for name in TABLES}
    assessment, plan = _read(paths['assessment.json']), _read(paths['plan.json'])
    if plan['dataset'] != 'CIC-UNSW-NB15' or list(plan['arms']) != list(WEIGHT_ARMS):
        raise ValueError('Unexpected NB15 weight experiment plan')
    for _ in _verified_files(source, {'complete.json': digest, **bindings}):
        pass
    return tables, assessment, {'complete.json': digest, **bindings}


def write_weight_experiment_report(source, output):
    """Verify the earlier weight run and publish a separate reporting-only appendix."""
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.resolve() != output or output.is_relative_to(source):
        raise ValueError('The weight report requires a new directory outside the source run')
    tables, assessment, hashes = _weight_evidence(source)
    lines = [
        '# NB15 低误报权重专项 · 核验报告',
        '',
        f'原始运行 {source.name}。本报告由已完成结果生成，训练用例数为 0。',
        '冻结头与全量微调使用匹配更新预算，各自相对参考分支的差异也包括附加训练。',
        '',
        *_common_sections(tables, assessment, WEIGHT_ARMS),
        '',
        '## VAL 分数排序诊断（1% FAR 下的攻击召回）',
        '',
        *_weight_ranking_table(tables),
        '',
        '整体直接合并原始分数受客户端分数尺度影响。' '逐客户端指标各自在本地达到 FAR 上限，按攻击支持数在种子内汇总，再对种子取平均。',
        '',
        '[来源和 SHA-256 回执](source_evidence.json)',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.nb15-weights-', dir=output.parent) as temporary:
        report = Path(temporary) / 'report'
        report.mkdir()
        _write_markdown(report, lines)
        _write(
            report / 'source_evidence.json',
            {
                'source_run_directory': str(source),
                'source_sha256': hashes,
                'report_module_sha256': _hash(Path(__file__)),
                'trained_cases': 0,
            },
        )
        report.rename(output)
    return output / 'SUMMARY.md'
