"""Start the FedCAID experiment suite from a downloaded source checkout."""

import argparse
import json
import sys
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from src import (
    revision_benchmark_matrix,
    revision_final_review,
    revision_fresh_report,
    revision_full_matrix,
    revision_lan_evidence,
    revision_local_review,
    revision_nb15_low_far,
    revision_nb15_threshold,
    revision_next,
    revision_protocol_validation,
    revision_resume_review,
)
from src.dataset_preparation import raw_source_paths
from src.experiment_config import DATASETS

PROJECT = Path(__file__).resolve().parent


def _parse_args(argv):
    """Read optional diagnostics and the LAN worker settings."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    _add_modes(parser)
    parser.add_argument('--hulk-manifest', type=Path)
    parser.add_argument('--hulk-2018-source', type=Path,
                        help='Optional original CSE-CIC-IDS2018 CSV for full review or --hulk-2018')
    parser.add_argument('--lan-manifest', type=Path)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--sessions', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument(
        '--independent-client-rng',
        action='store_true',
        help='Use separate candidate RNG streams in a new --nb15-low-far run',
    )
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _add_modes(parser):
    """Register mutually exclusive run modes on the one public entry."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full-review', action='store_true', help='Run the full review benchmark')
    mode.add_argument('--hulk-2018', type=Path, metavar='RUN',
                      help='Run the 2018 study with a completed full review run and its frozen Hulk models')
    mode.add_argument('--iov-review', action='store_true', help='Run the fresh five-seed fixed-alarm IoV ablation')
    mode.add_argument('--nb15-low-far', action='store_true', help='Run the five-seed NB15 TabM weight comparison')
    mode.add_argument('--nb15-threshold', action='store_true', help='Run the five-seed NB15 CAL-threshold comparison')
    mode.add_argument('--local-evidence', action='store_true', help='Summarize verified local evidence')
    mode.add_argument('--refresh-report', type=Path, metavar='RUN', help='Regenerate a verified report only')
    mode.add_argument('--resume', metavar='RUN', help='Resume the supplementary suffix of a compatible run')
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--demo', action='store_true', help='Run one synthetic CPU example')
    mode.add_argument('--smoke', action='store_true', help='Run the synthetic CPU experiment')
    mode.add_argument('--collector', action='store_true', help='Serve one declared LAN collector')


def _validate_args(parser, args):
    """Reject manifests on modes that do not use them."""
    if args.collector and args.port is None:
        parser.error('--collector requires --port')
    if args.independent_client_rng and not args.nb15_low_far:
        parser.error('--independent-client-rng requires --nb15-low-far')
    local_mode = (
        args.collector
        or args.iov_review
        or args.nb15_low_far
        or args.nb15_threshold
        or args.local_evidence
        or args.resume is not None
        or args.refresh_report is not None
        or args.smoke
        or args.demo
    )
    if (local_mode or args.hulk_2018) and (args.hulk_manifest is not None or args.lan_manifest is not None):
        parser.error('Manifest options are only used by full review or preflight')
    if (local_mode or args.preflight_only) and args.hulk_2018_source is not None:
        parser.error('--hulk-2018-source requires full review or --hulk-2018')


def _dataset_files(datasets):
    """Find expected CSVs before starting stages that require real data."""
    paths = tuple(path for name in datasets for path in raw_source_paths(name, PROJECT / 'dataset'))
    missing = tuple(path for path in paths if not path.is_file())
    return len(paths) - len(missing), missing


def _require_data(missing):
    """List missing paths in the layout the user needs to populate."""
    if missing:
        names = '\n'.join(f'  {path.relative_to(PROJECT)}' for path in missing)
        raise FileNotFoundError(
            f'Experiment data is missing. See dataset/README.md for these input files:\n{names}\n'
            'Use python main.py --demo for a synthetic installation example.'
        )


def _full_review(args):
    """Start the full review once all required original inputs are present."""
    _, missing = _dataset_files(DATASETS)
    _require_data(missing)
    from src.revision_hulk_2018 import DEFAULT_CAPTURE

    source = args.hulk_2018_source
    if source is None and (PROJECT / DEFAULT_CAPTURE).exists():
        source = PROJECT / DEFAULT_CAPTURE
    optional = {'hulk_2018_source': source} if source is not None else {}
    return revision_final_review.run_final_review(
        PROJECT, hulk_manifest=args.hulk_manifest, lan_manifest=args.lan_manifest, **optional
    )


def _hulk_2018_review(args):
    """Replay the optional 2018 study with explicitly selected completed source models."""
    from src.revision_hulk_2018 import DEFAULT_CAPTURE

    _require_data(_dataset_files(('CIC-IDS-2017',))[1])
    source = args.hulk_2018_source or PROJECT / DEFAULT_CAPTURE
    return revision_final_review.run_hulk_2018_review(PROJECT, args.hulk_2018, source)


def _demo():
    """Run one synthetic case and leave a receipt in the downloaded checkout."""
    container = PROJECT / 'outputs' / 'fedcaid_demo'
    if container.resolve() != container:
        raise ValueError('Synthetic output must stay inside the checkout')
    container.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='run_', dir=container)) / 'benchmark'
    print('FedCAID | synthetic_only=True | CPU example for installation verification', flush=True)
    result = _smoke_case(output, ['--dataset', 'CIC-IDS-2017', '--single-donor'])
    print(f"FedCAID | 合成示例回执 | {output / 'complete.json'}", flush=True)
    return result


def _check_manifests(args):
    """Check both optional inputs without scoring or connecting."""
    declarations = PROJECT / 'external'
    checks = revision_final_review.check_optional_inputs(
        args.hulk_manifest or declarations / 'hulk_manifest.json',
        args.lan_manifest or declarations / 'lan_manifest.json',
    )
    print(json.dumps({'hulk': checks['external'], 'lan': checks['lan']}, ensure_ascii=False, indent=2))
    return 2 if any(item['status'] == 'failed' for item in checks.values()) else 0


def _smoke_case(output, switches):
    """Verify one synthetic native training case and its completion record."""
    result = revision_next.main(
        [
            '--smoke',
            '--device',
            'cpu',
            '--operating-guard',
            '--seeds',
            '42',
            *switches,
            '--out',
            str(output),
        ]
    )
    receipt = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
    if not receipt['smoke'] or receipt['test_evaluated'] or not receipt['evaluated']:
        raise RuntimeError(f'Synthetic experiment did not complete: {output.name}')
    return result


def _check_iov_smoke(root):
    """Verify that focused diagnostics preserve the reference branch evaluations."""
    (fresh,) = (root / 'outputs' / 'fedcaid_review').glob('run_*')
    case = Path('CIC-IoV-2024/grouped/main/seed_42/result.json')
    modes = {'base_threshold', 'local_only', 'full'}
    original = json.loads((root / 'benchmark_matrix' / case).read_text(encoding='utf-8'))
    focused = json.loads((fresh / 'iov_diagnosis' / case).read_text(encoding='utf-8'))
    expected = {row['mode']: row for row in original['evaluations'] if row['mode'] in modes}
    observed = {row['mode']: row for row in focused['evaluations']}
    if set(expected) != modes or any(expected[mode] != observed[mode] for mode in modes):
        raise RuntimeError('IoV diagnostics changed the reference evaluation')
    _check_fixed_iov_smoke(observed)


def _check_fixed_iov_smoke(observed):
    """Independently check native VAL binary metrics for the fixed controls."""
    base = observed['base_threshold']['metrics']
    for mode in ('fixed_base_local', 'fixed_base_full'):
        actual = observed[mode]['metrics']
        if any(base[metric] != actual[metric] for metric in ('far', 'fnr', 'ece', 'alert_rate')):
            raise RuntimeError(f'IoV fixed alarm control changed a baseline binary metric: {mode}')


def _smoke_supplementary(root):
    """Exercise all local supplementary entry points with temporary inputs."""
    revision_lan_evidence.run_loopback(root / 'loopback')
    for name, runner in (
        ('scale_matrix', revision_full_matrix.run_scale_matrix),
        ('can_quantization', revision_full_matrix.run_can_quantization),
        ('nb15_temporal', revision_full_matrix.run_nb15_temporal),
        ('kminus1_transfer', revision_full_matrix.run_kminus1_transfer),
    ):
        runner(root / name, root / 'dataset', device='cpu', smoke=True)


def _smoke_nb15(root):
    """Exercise the two bundled NB15 studies and their result receipts."""
    for name, runner in (
        ('nb15_low_far', revision_nb15_low_far.run_nb15_low_far),
        ('nb15_threshold', revision_nb15_threshold.run_nb15_threshold),
    ):
        revision_final_review._nb15_stage(runner, PROJECT, root / name, 'cpu', smoke=True)


def _smoke():
    """Exercise all current native paths and the focused report on synthetic CPU data."""
    print('FedCAID | synthetic_only=True | CPU smoke suite for installation verification', flush=True)
    with TemporaryDirectory(prefix='fedcaid_smoke_') as temporary:
        root = Path(temporary).resolve()
        result = _smoke_case(root / 'reference', ['--dataset', 'CIC-IDS-2017', '--single-donor'])
        if result:
            return result
        result = _smoke_case(root / 'candidate', ['--dataset', 'CIC-UNSW-NB15', '--detector-objective', 'hard_benign'])
        return result or _smoke_suite(root)


def _smoke_suite(root):
    """Exercise the benchmark and each supplementary experiment after the base cases."""
    revision_benchmark_matrix.run_benchmark_matrix(root / 'benchmark_matrix', PROJECT, device='cpu', smoke=True)
    result = revision_final_review.run_iov_review(root, device='cpu', smoke=True)
    if result:
        return result
    _check_iov_smoke(root)
    _smoke_supplementary(root)
    _smoke_nb15(root)
    return 0


def _collector(args):
    """Run one LAN collector on the address selected for this host."""
    revision_protocol_validation.serve_collector(args.bind, args.port, sessions=args.sessions, timeout=args.timeout)
    return 0


def _refresh_report(source):
    """Write a derived report to a separate directory attached to one sealed run."""
    destination = PROJECT / 'outputs' / 'fedcaid_review_reports'
    if destination.resolve() != destination:
        raise ValueError('The report output container must stay in the current checkout')
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix='refresh_', dir=destination))
    receipt = revision_fresh_report.regenerate_verified_report(source, root / 'report')
    print(f"FedCAID | Source run status | {receipt['status']}", flush=True)
    print(f"FedCAID | 从回执刷新报告 | {root / 'report' / 'SUMMARY.md'}", flush=True)
    return 2 if receipt['failed_steps'] or receipt['required_pending_steps'] else 0


def _run_local_evidence():
    """Build the current single-host evidence package from finalized sources."""
    receipt = revision_local_review.run_local_review(PROJECT)
    print(f"FedCAID | 本地证据汇总 | {Path(receipt['output']) / 'SUMMARY.md'}", flush=True)
    return 0


def _run_utility(args):
    """Dispatch modes that do not start a new real benchmark."""
    if args.demo:
        return _demo()
    if args.smoke:
        return _smoke()
    if args.collector:
        return _collector(args)
    if args.preflight_only:
        return _check_manifests(args)
    if args.hulk_2018 is not None:
        return _hulk_2018_review(args)
    if args.refresh_report is not None:
        return _refresh_report(args.refresh_report)
    if args.local_evidence:
        return _run_local_evidence()
    if args.resume:
        return revision_resume_review.resume_review(PROJECT, args.resume)
    return None


def _run_focused(args):
    """Dispatch one real study after validating its dataset layout."""
    if args.iov_review:
        _require_data(_dataset_files(('CIC-IoV-2024',))[1])
        return revision_final_review.run_iov_review(PROJECT)
    _require_data(_dataset_files(('CIC-UNSW-NB15',))[1])
    if args.nb15_threshold:
        revision_nb15_threshold.run_nb15_threshold(PROJECT)
    elif args.independent_client_rng:
        revision_nb15_low_far.run_nb15_low_far(PROJECT, independent_client_rng=True)
    else:
        revision_nb15_low_far.run_nb15_low_far(PROJECT)
    return 0


def _run(args):
    """Run an explicit mode or resolve the no-argument launch."""
    result = _run_utility(args)
    if result is not None:
        return result
    if args.iov_review or args.nb15_low_far or args.nb15_threshold:
        return _run_focused(args)
    return _full_review(args)


def _missing_import(error):
    """Show setup guidance when dependencies are missing from the interpreter."""
    print(f'FedCAID 导入失败：{error}', file=sys.stderr, flush=True)
    print(f'请安装依赖："{sys.executable}" -m pip install -r "{PROJECT / "requirements.txt"}"', file=sys.stderr)
    return 2


def main(argv=None):
    """Start the configured workflow and return a process exit code."""
    args = _parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print('FedCAID | 已停止。运行目录内保留已写入的回执。', flush=True)
        return 130
    except ImportError as error:
        return _missing_import(error)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f'FedCAID 运行失败：{error}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(errors='backslashreplace')
    raise SystemExit(main())
