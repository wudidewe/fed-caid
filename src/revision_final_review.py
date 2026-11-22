"""Start a new, isolated FedCAID review experiment and report its evidence."""

from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from .revision_alarm_comparison import PRIMARY_METHOD, run_alarm_comparison
from .revision_benchmark_matrix import run_benchmark_matrix
from .revision_donor_check_replay import run_donor_check_replay
from .revision_external_preflight import preflight_hulk_declaration, run_verified_hulk
from .revision_fresh_report import run_fresh_report
from .revision_fresh_source import run_fresh_hulk_attribution, run_fresh_iov_diagnosis, run_fresh_nb15_pair
from .revision_full_matrix import run_can_quantization, run_kminus1_transfer, run_nb15_temporal, run_scale_matrix
from .revision_lan_evidence import preflight_lan, run_lan, run_loopback
from .revision_nb15_low_far import run_nb15_low_far
from .revision_nb15_threshold import run_nb15_threshold
from .revision_targeted_subtypes import run_targeted
from .revision_v12 import _bindings, _completion_summary, _hash, _read, _write

CURRENT_REVIEW = 'fedcaid_review'
SCHEMA_VERSION = 9
REVIEW_STATUSES = {'completed', 'pending', 'failed'}
STAGE_LABELS = {
    'loopback': '单机 TCP 回环与聚合校验（180 轮）',
    'benchmark_matrix': '原版 TabM 与 FedAvg 参考实验（分组 VAL，五种子）',
    'alarm_comparison': '双头 TabM 加标量 CAL 主实验（四数据集，分组 VAL，五种子）',
    'iov_diagnosis': 'CIC-IoV-2024 固定基础告警对照（五种子）',
    'lan': '真实 LAN 通信测量',
    'nb15_pair': 'NB15 TabM 参考与难正常样本候选（五种子，新训练）',
    'targeted': 'NB15 固定告警子类型训练对照（5 种子 × 2 目标）',
    'donor_check': 'NB15 供体 CAL 分数断点与接收端检查（五种子）',
    'hulk_attribution': 'Hulk 匹配联邦对照（五种子，新训练）',
    'external': 'Hulk V7 独立采集确认',
    'nb15_low_far': 'NB15 冻结骨干与全量微调（五种子）',
    'nb15_threshold': 'NB15 分组 CAL 阈值对照（五种子）',
    'scale_matrix': '规模、训练预算与多 α（两套流量数据，五种子）',
    'can_quantization': '两套 CAN 的原始 K-1 量化对照（五种子）',
    'nb15_temporal': 'NB15 同源按时间留出（精确输入清除，五种子）',
    'kminus1_transfer': '两套流量数据的原始 K-1 缺类迁移与分数对照（五种子）',
    'report': '汇总本轮实验',
}
IOV_WORKFLOW = 'iov_diagnosis'
FULL_WORKFLOW = 'full_review'


def _without_artifacts(record):
    return {key: value for key, value in record.items() if key != 'artifacts'}


def _new_run_directory(project, output, device):
    """Allocate an empty child directory atomically for each invocation."""
    destination = project / 'outputs' / CURRENT_REVIEW if output is None else Path(output).absolute()
    if destination.resolve() != destination or destination.parent != project / 'outputs':
        raise ValueError('Use a non-redirected run container directly inside outputs')
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Use auto, cpu, or cuda for the fresh run')
    destination.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime('run_%Y%m%d_%H%M%S_')
    return Path(mkdtemp(prefix=prefix, dir=destination))


def _check_input(manifest, check):
    """Check a declaration and bind its bytes without reading capture contents."""
    try:
        manifest = Path(manifest).resolve()
        original = _hash(manifest) if manifest.is_file() else None
        result = check(manifest)
        if result['status'] == 'ready':
            if original is None or _hash(manifest) != original:
                raise ValueError('Run declaration changed during startup preflight')
            result = {**result, 'manifest_sha256': original}
        return {'manifest': str(manifest), **result}
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {'manifest': str(manifest), 'status': 'failed', 'reason': str(error)}


def check_optional_inputs(hulk_manifest, lan_manifest):
    """Check both optional declarations without training or connecting."""
    return {
        'external': _check_input(hulk_manifest, preflight_hulk_declaration),
        'lan': _check_input(lan_manifest, preflight_lan),
    }


def _make_plan(project, output, manifests, device, smoke):
    """Record the current code and forecast dependencies before running any stage."""
    source = Path(__file__).resolve().parent
    entry = source.parent / 'main.py'
    workflow = FULL_WORKFLOW if manifests else IOV_WORKFLOW
    stages = list(STAGE_LABELS) if manifests else [IOV_WORKFLOW, 'report']
    return {
        'schema_version': SCHEMA_VERSION,
        'project': str(project),
        'run_directory': str(output),
        'requested_device': device,
        'workflow': workflow,
        'synthetic_only': smoke,
        'manifests': [str(path) for path in manifests],
        'input_preflight': check_optional_inputs(*manifests) if manifests else {},
        'stages': stages,
        **({**PRIMARY_METHOD, 'primary_stage': 'alarm_comparison',
            'reference_detector': 'original_tabm', 'reference_stage': 'benchmark_matrix',
            'comparison_alarm_policy': 'fixed_070'}
           if manifests else {}),
        'source_sha256': _bindings(
            source, sorted(path.relative_to(source).as_posix() for path in source.rglob('*.py'))
        ),
        'entry_sha256': {entry.name: _hash(entry)} if entry.is_file() else {},
        'previous_run_inputs_allowed': False,
        'intermediate_artifacts': 'freshly produced in this run directory',
    }


def _stage_note(record):
    """Summarize execution provenance and the available scientific verdict."""
    detail = record.get('reason') or record.get('scientific_status') or record.get('assessment')
    if isinstance(detail, dict):
        detail = detail.get('scientific_status') or detail.get('status')
    binary = record.get('binary_assessment', {})
    detail = detail or binary.get('status')
    if 'combined_acceptance' in binary:
        joint = '通过' if binary['combined_acceptance'] else '未通过'
        detail = f'{detail}，FAR/FNR 与宏 F1 联合判据{joint}'
    notes = [str(value) for value in (record.get('execution_origin'), detail) if value]
    return f" ({', '.join(notes)})" if notes else ''


def _run_stage(name, path, operation):
    """Execute one stage in its assigned directory and record a terminal status."""
    print(f'FedCAID | {STAGE_LABELS[name]}', flush=True)
    started = perf_counter()
    try:
        if path.resolve() != path:
            raise ValueError(f'Redirected stage output: {path}')
        result = operation(path)
        if result.get('status') not in REVIEW_STATUSES:
            raise ValueError('Unrecognized stage status')
        record = {'output': str(path), **result}
        if Path(record['output']).resolve() != path:
            raise ValueError(f'Stage referred to results outside its current-run directory: {path}')
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, SystemExit) as error:
        record = {'status': 'failed', 'reason': str(error), 'output': str(path)}
    record['elapsed_seconds'] = round(perf_counter() - started, 3)
    print(
        f"FedCAID | {STAGE_LABELS[name]} | {record['status']}{_stage_note(record)} | {record['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return record


def _after(state, key, operation):
    """A blocked stage may only wait for its declared current-run dependency."""
    if state[key]['status'] != 'completed':
        return {'status': 'pending', 'reason': f'本轮 {STAGE_LABELS[key]} 尚未完成'}
    return operation()


def _with_declaration(checked, operation):
    """Run an optional stage with the declaration bound before training."""
    if checked['status'] != 'ready':
        return checked.copy()
    path = Path(checked['manifest'])
    if _hash(path) != checked['manifest_sha256']:
        raise ValueError(f'Run declaration changed after startup: {path}')
    return operation(path)


def _nb15_stage(runner, project, output, device, *, smoke=False):
    """Bind a focused comparison to its assigned full-review stage."""
    if runner(project, output=output, device=device, smoke=smoke) != output:
        raise ValueError('NB15 study wrote outside its full-review stage')
    marker = output / 'complete.json'
    result = _read(marker)
    if result['status'] != 'completed' or not result['artifacts']:
        raise ValueError('NB15 study has no completed artifact receipt')
    return {
        'status': 'completed',
        'execution_origin': 'current_fresh_training',
        'scientific_status': _nb15_status(_read(output / 'assessment.json')),
        'artifacts': {**result['artifacts'], marker.name: _hash(marker)},
    }


def _nb15_status(assessment):
    """Read either the per-candidate weight verdicts or the threshold verdict."""
    if 'candidates' in assessment and 'development_guard_pass' not in assessment:
        verdicts = [row['development_guard_pass'] for row in assessment['candidates'].values()]
    else:
        verdicts = [assessment['development_guard_pass']]
    if any(value is True for value in verdicts):
        return 'development_criteria_met'
    if any(value is False for value in verdicts):
        return 'development_criteria_not_met'
    return 'development_criteria_not_assessed'


def _operations(project, output, preflight, state, device):
    """Build stage functions using only outputs under this invocation's directory."""
    pair, hulk = output / 'nb15_pair', output / 'hulk_attribution'
    data = project / 'dataset'
    return {
        'loopback': run_loopback,
        'benchmark_matrix': lambda path: run_benchmark_matrix(path, project, device=device),
        'alarm_comparison': lambda path: _after(
            state,
            'benchmark_matrix',
            lambda: run_alarm_comparison(
                path, project, device=device, reference=output / 'benchmark_matrix', primary=True,
            ),
        ),
        'iov_diagnosis': lambda path: run_fresh_iov_diagnosis(path, project, device=device),
        'lan': lambda path: _with_declaration(preflight['lan'], lambda manifest: run_lan(path, manifest)),
        'nb15_pair': lambda path: run_fresh_nb15_pair(path, project, device=device),
        'targeted': lambda path: _after(
            state, 'nb15_pair', lambda: run_targeted(path, project, device=device, pair_source=pair)
        ),
        'donor_check': lambda path: _after(
            state,
            'targeted',
            lambda: run_donor_check_replay(path, project, state['targeted'], pair_source=pair),
        ),
        'hulk_attribution': lambda path: run_fresh_hulk_attribution(path, project, device=device),
        'external': lambda path: _with_declaration(
            preflight['external'],
            lambda manifest: _after(state, 'hulk_attribution', lambda: run_verified_hulk(path, hulk, manifest)),
        ),
        'nb15_low_far': lambda path: _nb15_stage(run_nb15_low_far, project, path, device),
        'nb15_threshold': lambda path: _nb15_stage(run_nb15_threshold, project, path, device),
        'scale_matrix': lambda path: run_scale_matrix(path, data, device=device),
        'can_quantization': lambda path: run_can_quantization(path, data, device=device),
        'nb15_temporal': lambda path: run_nb15_temporal(path, data, device=device),
        'kminus1_transfer': lambda path: run_kminus1_transfer(path, data, device=device),
        'report': lambda path: run_fresh_report(path, state),
    }


def _verify_source(plan):
    """Reject source edits once a new invocation has recorded its plan."""
    source = Path(__file__).resolve().parent
    entry, modules = plan['entry_sha256'], plan['source_sha256']
    if _bindings(source.parent, entry) != entry or _bindings(source, modules) != modules:
        raise ValueError('Source changed mid-run. Start a new experiment to use the edited code.')


def _run_stages(project, output, state_path, device, plan):
    """Persist stage outcomes incrementally and reject mid-run source changes."""
    state = {}
    operations = _planned_operations(project, output, plan, state, device)
    if list(operations) != plan['stages']:
        raise ValueError('The planned and executable stages differ')
    for name, operation in operations.items():
        _verify_source(plan)
        state[name] = _run_stage(name, output / name, operation)
        _write(state_path, state)
    _verify_source(plan)
    return state


def _planned_operations(project, output, plan, state, device):
    """Keep the focused and complete workflows separate while sharing the report."""
    if plan['workflow'] == IOV_WORKFLOW:
        return {
            IOV_WORKFLOW: lambda path: run_fresh_iov_diagnosis(
                path, project, device=device, smoke=plan['synthetic_only'],
            ),
            'report': lambda path: run_fresh_report(path, state),
        }
    return _operations(project, output, plan['input_preflight'], state, device)


def _write_completion(path, plan, state_path, state):
    """Save execution status and return a failure code for unfinished required work."""
    summary = _completion_summary(state, _read(plan)['stages'])
    optional_pending = summary['optional_pending_steps']
    required_pending = summary['required_pending_steps']
    _write(
        path,
        {
            'schema_version': SCHEMA_VERSION,
            **summary,
            'run_directory': str(path.parent),
            'previous_run_inputs_allowed': False,
            'plan_sha256': _hash(plan),
            'state_sha256': _hash(state_path),
            'steps': {name: _without_artifacts(result) for name, result in state.items()},
        },
    )
    print(f"FedCAID | {summary['status']} | {path}", flush=True)
    if optional_pending:
        print(f"FedCAID | Pending optional evidence | {', '.join(optional_pending)}", flush=True)
    if required_pending:
        print(f"FedCAID | Pending required stages | {', '.join(required_pending)}", flush=True)
    if state.get('report', {}).get('status') == 'completed':
        print(f"FedCAID | 本轮报告 | {path.parent / 'report' / 'SUMMARY.md'}", flush=True)
    return 2 if summary['failed_steps'] or required_pending else 0


def _print_preflight(checks):
    """Show readiness and declaration paths before the first stage starts."""
    for stage, checked in checks.items():
        reason = checked.get('reason')
        suffix = f' ({reason})' if reason else ''
        print(f"FedCAID | 启动预检 | {STAGE_LABELS[stage]} | {checked['status']}{suffix}", flush=True)
        print(f"FedCAID | 声明文件 | {checked['manifest']}", flush=True)


def _execute_plan(project, output, manifests, device, *, smoke=False):
    """Persist the plan and preserve a receipt when execution is interrupted."""
    plan_path, state_path, summary = (
        output / name for name in ('final_plan.json', 'final_state.json', 'final_complete.json')
    )
    plan = _make_plan(project, output, manifests, device, smoke)
    _write(plan_path, plan)
    _write(summary, {'schema_version': SCHEMA_VERSION, 'status': 'running', 'plan_sha256': _hash(plan_path)})
    print(f'FedCAID | 新运行目录 | {output}', flush=True)
    _print_preflight(plan['input_preflight'])
    try:
        state = _run_stages(project, output, state_path, device, plan)
    except (KeyboardInterrupt, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        _write_abort(summary, plan_path, error)
        raise
    return _write_completion(summary, plan_path, state_path, state)


def _write_abort(summary, plan_path, error):
    """Record an interrupted or invalid fresh invocation."""
    _write(
        summary,
        {
            'schema_version': SCHEMA_VERSION,
            'status': 'aborted',
            'run_directory': str(summary.parent),
            'reason': str(error) or type(error).__name__,
            'plan_sha256': _hash(plan_path),
        },
    )


def _complete_hulk_2018(output, source, frozen, project):
    """Save supplementary evidence apart from the sealed benchmark report."""
    from .revision_hulk_2018 import run_hulk_2018

    print(f'FedCAID | CSE-CIC-IDS2018 exploratory evaluation | {output}', flush=True)
    receipt = run_hulk_2018(output, source, frozen, project)
    print(f"FedCAID | CSE-CIC-IDS2018 completed | {receipt['summary_path']}", flush=True)
    return 0


def run_hulk_2018_review(project, source_run, source_csv):
    """Run one new pilot using an explicitly selected finalized review run."""
    from .revision_hulk_2018 import validate_hulk_2018_source
    from .revision_local_sources import _official_kind

    project, source_run = Path(project).resolve(), Path(source_run).resolve(strict=True)
    source = validate_hulk_2018_source(source_csv)
    if _official_kind(source_run) != FULL_WORKFLOW:
        raise ValueError('Select a completed full review run with verified frozen Hulk models.')
    run = _new_run_directory(project, project / 'outputs/hulk_2018_review', 'cpu')
    return _complete_hulk_2018(run / 'hulk_2018', source, source_run / 'hulk_attribution', project)


def run_final_review(project, output=None, *, hulk_manifest=None, lan_manifest=None,
                     hulk_2018_source=None, device='auto'):
    """Run the four-dataset review and the declared optional 2018 study."""
    project = Path(project).resolve()
    source = None
    if hulk_2018_source is not None:
        from .revision_hulk_2018 import validate_hulk_2018_source

        source = validate_hulk_2018_source(hulk_2018_source)
    output = _new_run_directory(project, output, device)
    manifests = (
        Path(hulk_manifest or project / 'external' / 'hulk_manifest.json').resolve(),
        Path(lan_manifest or project / 'external' / 'lan_manifest.json').resolve(),
    )
    status = _execute_plan(project, output, manifests, device)
    if status == 0 and source is not None:
        return _complete_hulk_2018(output / 'hulk_2018', source, output / 'hulk_attribution', project)
    return status


def run_iov_review(project, output=None, *, device='auto', smoke=False):
    """Start a focused IoV run without reading any older experimental outputs."""
    project = Path(project).resolve()
    output = _new_run_directory(project, output, device)
    return _execute_plan(project, output, (), device, smoke=smoke)
