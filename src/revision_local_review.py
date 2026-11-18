"""Assemble an auditable review package for the local-data, single-host setup."""

import csv
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from .revision_local_holdout import holdout_rows
from .revision_local_sources import copy_legacy_evidence, copy_official_reports
from .revision_local_transport import run_local_transport
from .revision_report import write_csv
from .revision_v12 import _bindings, _read, _write

MAIN_REPORT = 'sources/full_review'
IOV_REPORT = 'sources/iov_diagnosis'
MAIN_TABLES = f'{MAIN_REPORT}/tables/benchmark_matrix'
IOV_TABLES = f'{IOV_REPORT}/tables/iov_diagnosis'
TEMPORAL = 'history/nb15_temporal'
SCALE = 'history/scale_can'
TRANSPORT = 'transport/local_transport_summary.csv'
IOV_FIXED_PAIR = f'{IOV_TABLES}/iov_fixed_mask_paired.csv'
IOV_MODES = ('base_threshold', 'local_only', 'full')
PAIR_NAMES = {
    'full_minus_local_only': '完整 K−1 − 本地适配',
    'full_minus_fedavg_threshold': '完整 K−1 − 全权重 FedAvg',
    'guarded_aggregation': '完整共享头 − 本地共享头',
    'guarded_receiver_participation': '完整共享头 − 多供体',
}


def _rows(root, relative):
    """Read a copied source table, leaving an unavailable source explicit."""
    path = root / relative
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _available(root, relative):
    """Resolve a link only when this package contains its evidence."""
    return relative if (root / relative).is_file() else ''


def _effect(row, scope, source):
    """Preserve the measured interval and its actual reference."""
    return {
        'dataset': row['dataset'],
        'protocol': scope,
        'target_class': row.get('target_class', ''),
        'endpoint': row['metric'],
        'contrast': row['contrast'],
        'comparison': PAIR_NAMES[row['contrast']],
        'seeds': row.get('n', row.get('seeds_with_metric', '')),
        'mean': row['mean'],
        'ci95_low': row['ci95_low'],
        'ci95_high': row['ci95_high'],
        'source_artifact': source,
    }


def _effects(root):
    """Keep original benchmark and controlled shared-head effects separate."""
    source = f'{MAIN_TABLES}/paper_paired.csv'
    for row in _rows(root, source):
        if row['contrast'] in PAIR_NAMES and row['metric'] == 'macro_f1':
            yield _effect(row, '原始 K−1 · 同源分组 VAL', source)
    source = f'{MAIN_REPORT}/paired_evidence.csv'
    for row in _rows(root, source):
        if (
            (row['stage'], row['branch']) in {('nb15_pair', 'reference'), ('hulk_attribution', 'attribution')}
            and row['contrast'] in {'guarded_aggregation', 'guarded_receiver_participation'}
            and (row['scope'], row['metric']) == ('receiver', 'f1')
        ):
            yield _effect(row, '共享头 · 受控缺类 VAL', source)


def _sum(rows, field):
    """Add counts over the declared, disjoint client row assignments."""
    return sum(int(row[field]) for row in rows)


def _iov_seed(seed, mode, clients, comparisons):
    """Report row burden, group concentration and the directional branch change."""
    false_rows = _sum(clients, 'false_alarm_rows')
    benign_rows = _sum(clients, 'benign_rows')
    largest = max(int(row['top_false_alarm_group_rows']) for row in clients)
    return {
        'seed': seed,
        'mode': mode,
        'false_alarm_rows': false_rows,
        'benign_rows': benign_rows,
        'row_far': false_rows / benign_rows if benign_rows else None,
        'false_alarm_client_group_occurrences': _sum(clients, 'false_alarm_groups'),
        'largest_client_false_alarm_group_rows': largest,
        'largest_client_group_share_of_seed_false_alarms': largest / false_rows if false_rows else None,
        'comparison_reference': comparisons[0]['reference'] if comparisons else '',
        'benign_alert_rows_added': _sum(comparisons, 'benign_alert_rows_added') if comparisons else None,
        'benign_alert_rows_removed': _sum(comparisons, 'benign_alert_rows_removed') if comparisons else None,
        'source_artifact': f'{IOV_TABLES}/iov_operating_diagnostics.csv',
        'comparison_artifact': f'{IOV_TABLES}/iov_alarm_overlap.csv',
    }


def _iov_attribution(root):
    """Summarize each IoV seed without counting repeated groups as unique across clients."""
    operating = [row for row in _rows(root, f'{IOV_TABLES}/iov_operating_diagnostics.csv') if row['role'] == 'val']
    overlap = [row for row in _rows(root, f'{IOV_TABLES}/iov_alarm_overlap.csv') if row['role'] == 'val']
    references = {'local_only': 'base_threshold', 'full': 'local_only'}
    for seed in sorted({row['seed'] for row in operating}):
        for mode in IOV_MODES:
            clients = [row for row in operating if (row['seed'], row['mode']) == (seed, mode)]
            if {row['client'] for row in clients} != {'0', '1', '2'} or len(clients) != 3:
                raise ValueError('The IoV source lacks a complete seed/mode/client comparison')
            pairs = [
                row
                for row in overlap
                if (row['seed'], row['candidate'], row['reference']) == (seed, mode, references.get(mode))
            ]
            if mode in references and (len(pairs) != 3 or {row['client'] for row in pairs} != {'0', '1', '2'}):
                raise ValueError('The IoV source lacks the matched alarm comparisons')
            yield _iov_seed(seed, mode, clients, pairs)


def _temporal_evidence(root):
    """Pass the separate, locally pinned historical time-split source to the audit."""
    paths = [f'{TEMPORAL}/{name}' for name in ('run.json', 'data_audit.json', 'metrics_by_seed.csv')]
    if not all((root / path).is_file() for path in paths):
        return None
    return {
        'run': _read(root / paths[0]),
        'audit': _read(root / paths[1]),
        'metrics': _rows(root, paths[2]),
        'source': paths[2],
    }


def _holdouts(root):
    """Link individual source audits under the published source report."""
    report = MAIN_REPORT if (root / MAIN_REPORT / 'tables').is_dir() else IOV_REPORT
    rows = holdout_rows(root / report / 'tables', _temporal_evidence(root))
    for row in rows:
        if row['evidence_scope'] == 'current_grouped' and row['source_artifact']:
            row['source_artifact'] = f"{report}/{row['source_artifact']}"
    return rows


def _source_entry(root, issue, evidence, scope, limit):
    """Record available files without implying evaluation or approval."""
    path = _available(root, evidence) if evidence else ''
    return {
        'issue': issue,
        'status': 'evidence_available' if path else 'not_available_with_current_inputs',
        'evaluation_scope': scope,
        'source_artifact': path,
        'claim_limit': limit,
    }


def _evidence_index(root):
    """Map the main review questions to source-bound results and spoken limits."""
    definitions = (
        ('四数据集与类别覆盖', f'{MAIN_REPORT}/benchmark_operating_points.csv', '同源分组 VAL', '按实际覆盖类报告'),
        ('聚合贡献与基线', 'aggregation_effects.csv', '五种子配对', '分别报告本地适配、FedAvg 和共享头对照'),
        ('IoV 误报', 'iov_seed_attribution.csv', '同源分组 VAL', '原工作点诊断，候选阈值未评价'),
        ('IoV 固定基础告警', IOV_FIXED_PAIR, '同源分组 CAL/VAL', '沿用基础阈值，分别报告误报、漏报和逐类变化'),
        ('NB15 联合工作点', f'{MAIN_REPORT}/tables/nb15_pair/binary_pair_status.json', '受控缺类开发评估', '按实际 FAR、FNR 和宏 F1 判据报告'),
        ('留出集来源', 'holdout_assessment.csv', '角色与历史评分审计', '未评分状态不等于从未用于开发'),
        ('NB15 时间留出', f'{TEMPORAL}/metrics_by_seed.csv', '已评分的同源跨时间补充', '历史探索结果'),
        ('规模、多 α 与 CAN 量化', f'{SCALE}/archive_provenance.json', '历史工作负载与版本', '固定总量与固定每客户端量分列'),
        ('单机通信', TRANSPORT, '单机 TCP 回环、合成模型更新', '时延和实际应用字节不包含训练与推理'),
        ('真实独立采集', '', '当前工作流未新增独立评价', '跨采集泛化仍未验证'),
        ('物理 LAN', '', '当前仅有单机环境', '跨主机延迟和吞吐未验证'),
    )
    return [_source_entry(root, *row) for row in definitions]


def _nb15_limit(root):
    """Read the original joint binary verdict rather than assuming failure."""
    path = root / MAIN_REPORT / 'tables/nb15_pair/binary_pair_status.json'
    if not path.is_file():
        return 'NB15 二元判据在当前汇总缺少已核验的回执。'
    if _read(path)['combined_acceptance']:
        return 'NB15 二元联合判据在开发 VAL 达标。与子类型收益联合陈述仍需绑定同一个报警配置。'
    return 'NB15 候选未达到本轮 FAR、FNR 与宏 F1 联合判据。受控 Exploits 收益按同告警对照单列。'


def _aggregation_limit(effects):
    """Qualify the current full-versus-local macro-F1 result."""
    native = [row for row in effects if row['contrast'] == 'full_minus_local_only']
    supported = [row['dataset'] for row in native if row['ci95_low'] and float(row['ci95_low']) > 0]
    if supported:
        return '原始 K−1 完整联邦相对本地适配的正向配对区间仅见于 ' + '、'.join(supported) + '。'
    if native:
        return '四数据集主对照尚未显示原始 K−1 完整联邦相对本地适配有稳定宏 F1 增益。'
    return '当前汇总缺少已核验的原始 K−1 配对结果。'


def _limitations(root, effects):
    """State the manuscript boundaries that available data cannot resolve."""
    return [
        '五种子区间反映共享采集上的配对变异。Hulk 子类型证据来自受控缺类的同源分组 VAL。',
        _aggregation_limit(effects),
        _nb15_limit(root),
        'IoV 在原阈值下出现高频正常输入误报。行 FAR 用于衡量告警负担，客户端内的精确输入组用于解释集中度。',
        'CAN 的宏 F1 按实际支持类解释。各类的零支持和零召回分别保留。',
        '时间留出来自同一份本地基准，历史评分按探索性证据引用。证据包不重新评分 TEST。',
        '单机回环验证使用合成更新。安全结论限于可信宿主和两个诚实且不串通的收集器的实现模型。',
        '外部采集和真实 LAN 在本地资源范围外。公开 CI、固定发布版本、永久归档和许可仍需发布时核验。',
    ]


def _markdown_table(headers, rows):
    """Escape table cells while keeping links local to the package."""

    def line(cells):
        return '| ' + ' | '.join(str(value).replace('|', r'\|').replace('\n', ' ') for value in cells) + ' |'

    return [line(headers), line(('---',) * len(headers)), *map(line, rows), '']


def _effect_lines(effects):
    """Render paired effects without converting an interval crossing zero to a success."""
    return [
        (
            item['dataset'],
            item['protocol'],
            item['target_class'] or '总体',
            '宏 F1' if item['endpoint'] == 'macro_f1' else '目标类 F1',
            item['comparison'],
            item['seeds'],
            f"{float(item['mean']):+.5f} [{float(item['ci95_low']):+.5f}, {float(item['ci95_high']):+.5f}]",
        )
        for item in effects
        if all(item[key] for key in ('mean', 'ci95_low', 'ci95_high'))
    ]


def _percent(value):
    """Preserve undefined rates for a population without benign rows."""
    return '—' if value is None else f'{value:.3%}'


def _iov_cal_note(root):
    """Show the effective calibration support instead of only the row count."""
    cal = [row for row in _rows(root, f'{IOV_TABLES}/iov_operating_diagnostics.csv') if row['role'] == 'cal_holdout']
    if not cal:
        return []
    thresholds = [float(row['threshold']) for row in cal]
    attacks = [int(row['attack_groups']) for row in cal]
    normal = [int(row['benign_groups']) for row in cal]
    binding = sum(row['cal_floor_binding'] == 'True' for row in cal)
    return [
        f'CAL 严格阈值范围 {min(thresholds):g}–{max(thresholds):g}'
        f'，下限生效 {binding}/{len(cal)}。每个客户端有 {min(normal)}–{max(normal)} 个正常组'
        f'和 {min(attacks)}–{max(attacks)} 个攻击组。',
        '',
    ]


def _iov_lines(rows, root):
    """Show the user-visible burden and the paired direction in every seed."""
    if not rows:
        return []
    table = (
        (
            row['seed'],
            row['mode'],
            row['false_alarm_rows'],
            _percent(row['row_far']),
            row['false_alarm_client_group_occurrences'],
            row['largest_client_false_alarm_group_rows'],
            row['benign_alert_rows_added'] if row['comparison_reference'] else '—',
        )
        for row in rows
    )
    return [
        '## IoV 误报归因',
        '',
        '组数按客户端出现次数计。最大误报组按单客户端统计。新增行以 CSV 中的参考分支为准。',
        '',
        *_iov_cal_note(root),
        *_markdown_table(('种子', '分支', '误报行', '行 FAR', '误报组出现次数', '最大组误报行', '新增行'), table),
    ]


def _transport_lines(root):
    """Show actual application payload separately from logical field elements."""
    rows = _rows(root, TRANSPORT)
    if not rows:
        return []
    table = (
        (
            row['dimension'],
            row['clients'],
            row['measured_rounds'],
            f"{float(row['round_median_ms']):.3f}",
            f"{float(row['round_p95_ms']):.3f}",
            f"{float(row['numeric_share_payload_bytes_per_round']):,.0f}",
            f"{float(row['tcp_application_mean_bytes_per_round']):,.0f}",
        )
        for row in rows
    )
    return [
        '## 单机 TCP 回环',
        '',
        '每轮处理合成更新。计时和字节数只覆盖已归档的协议测量。',
        '',
        *_markdown_table(('更新维度', '客户端', '轮数', '中位数 ms', 'P95 ms', '逻辑字节/轮', 'TCP 应用字节/轮'), table),
    ]


def _holdout_lines(rows):
    """Expose the rare-class support and prior use of each retained test view."""
    table = (
        (
            row['dataset'],
            row['protocol'],
            row['role'],
            row.get('attack_support', '—'),
            row.get('missing_attack_classes', '') or '—',
            row['scored_in_current_run'],
            row['scored_in_verified_history'],
        )
        for row in rows
        if row['role'] != 'VAL'
    )
    return [
        '## 留出集与攻击支持',
        '',
        'TEST 的支持按行计，TEST_EXACT_UNIQUE 按精确输入组计。历史状态为 unknown 时保留未知。',
        '',
        *_markdown_table(('数据集', '协议', '评价视图', '攻击支持', '缺失攻击类', '原运行评分', '历史评分'), table),
    ]


def _overview(root, effects, iov, holdouts, index):
    """Build one manuscript-facing summary from measured and unavailable evidence."""
    links = (
        (
            row['issue'],
            row['evaluation_scope'],
            row['claim_limit'],
            f"[产物]({row['source_artifact']})" if row['source_artifact'] else '当前不可用',
        )
        for row in index
    )
    return '\n'.join(
        [
            '# FedCAID 本地实验与审稿证据',
            '',
            '证据包按已核验的运行与历史归档分别汇总。训练和新评分案例数均为零。',
            '',
            '## 本地证据清单',
            '',
            *_markdown_table(('项目', '评价范围', '解读口径', '证据'), links),
            '## 匹配对照的配对差值',
            '',
            *_markdown_table(('数据集', '工作流', '目标', '指标', '对照', '种子数', '差值及 95% 区间'), _effect_lines(effects)),
            *_iov_lines(iov, root),
            *_transport_lines(root),
            *_holdout_lines(holdouts),
            '## 论文局限性',
            '',
            *[f'- {item}' for item in _limitations(root, effects)],
            '',
            '[来源及回执](source_manifest.json) · [交付核验](complete.json)',
            '',
        ]
    )


def _write_tables(working):
    """Export derived tables using only verified copies in this package."""
    effects = list(_effects(working))
    iov = list(_iov_attribution(working))
    holdouts = _holdouts(working)
    write_csv(working / 'aggregation_effects.csv', effects)
    write_csv(working / 'iov_seed_attribution.csv', iov)
    write_csv(working / 'holdout_assessment.csv', holdouts)
    index = _evidence_index(working)
    write_csv(working / 'evidence_index.csv', index)
    (working / 'SUMMARY.md').write_text(_overview(working, effects, iov, holdouts, index), encoding='utf-8')


def _source_records(working, project):
    """Bind existing evidence without training, scoring or opening endpoints."""
    outputs = project / 'outputs'
    modules = (
        'revision_local_review.py',
        'revision_local_sources.py',
        'revision_local_holdout.py',
        'revision_local_transport.py',
    )
    return {
        'generator_sha256': _bindings(Path(__file__).parent, modules),
        'official_runs': copy_official_reports(outputs, working / 'sources'),
        'historical_results': copy_legacy_evidence(outputs, working / 'history'),
        'single_host_measurement': run_local_transport(working / 'transport', project),
    }


def _publish(output):
    """Commit the evidence by publishing the complete marker last."""
    names = (p.relative_to(output).as_posix() for p in sorted(output.rglob('*')) if p.is_file())
    receipt = {
        'status': 'completed',
        'workflow': 'local_evidence',
        'trained_cases': 0,
        'new_test_scoring': False,
        'new_network_measurement': False,
        'reviewer_acceptance': 'not_assessed',
        'artifacts': _bindings(output, names),
    }
    _write(output / 'complete.json', receipt)
    return {**receipt, 'output': str(output)}


def run_local_review(project):
    """Build a new local evidence package without modifying the source experiments."""
    project = Path(project).resolve()
    container = project / 'outputs' / 'fedcaid_local_evidence'
    if container.resolve() != container:
        raise ValueError('The local review output must stay in its declared directory')
    container.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=datetime.now().strftime('review_%Y%m%d_%H%M%S_'), dir=container)) / 'report'
    output.mkdir()
    try:
        _write(output / 'source_manifest.json', _source_records(output, project))
        _write_tables(output)
        return _publish(output)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _write(output / 'complete.json', {'status': 'failed', 'reason': str(error), 'trained_cases': 0})
        raise
