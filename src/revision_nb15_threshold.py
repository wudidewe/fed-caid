"""Compare predeclared NB15 alarm rules on freshly fitted reference detectors."""

from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from scipy.stats import t
import torch

from . import revision_next as native
from . import utils as core
from .revision_nb15_low_far import (
    DATASET,
    SEEDS,
    SYNTHETIC_TARGET,
    TARGET,
    _allocate_output,
    _client_rows,
    _configuration,
    _fit_reference_client,
    _prepare_seed,
    _source_snapshot,
)
from .revision_nb15_threshold_evaluation import (
    FNR_INCREASE_LIMIT,
    assess_results,
    evaluate_locked_seed,
    freeze_thresholds,
)
from .revision_nb15_threshold_report import write_threshold_report
from .revision_operating_selection import split_operating_subsets
from .revision_report import write_csv
from .revision_v12 import _bindings, _hash, _write


OUTPUT_CONTAINER = "fedcaid_nb15_threshold"
SOURCE_MODULES = (
    "revision_nb15_threshold.py",
    "revision_nb15_threshold_evaluation.py",
    "revision_nb15_threshold_report.py",
    "revision_operating_selection.py",
    "utils.py",
)
CAL_CHECK_FRACTION = 0.5


def _sources(project):
    """Bind the ordinary reference training and the new threshold policy code."""
    return {**_source_snapshot(project), **_bindings(Path(__file__).resolve().parent, SOURCE_MODULES)}


def _fit_clients(data, parts, routes, args, temporary):
    """Use the local training population and CAL without giving the fit any VAL rows."""
    nodes = []
    for client, rows in enumerate(routes):
        training_and_cal = {role: rows[role] for role in ("train", "cal")}
        node = _fit_reference_client(data, parts, training_and_cal, args, client, temporary)
        del node.parts["train"]
        nodes.append(node)
    return nodes


def _score_val(node, features, indices, args):
    """Generate VAL scores only after the CAL threshold receipt exists."""
    model = node.model.to(args.device)
    try:
        return core.probabilities(
            core.log_prediction(model, node.transform(features[indices]), args.device, args.batch_size)
        )
    finally:
        model.cpu()


def _lock_cal(output, seed, nodes, subsets, split_audit, classes):
    """Persist thresholds from the disjoint CAL assignment before accessing VAL."""
    path = output / f"threshold_lock_{seed}.json"
    truth = [node.parts["cal"][1] for node in nodes]
    probabilities = [node.probabilities["cal"] for node in nodes]
    lock = freeze_thresholds(truth, probabilities, subsets, classes)
    _write(path, {"seed": seed, "locked_before_val_scoring": True, "thresholds": lock, "split": split_audit})
    return lock, _hash(path)


def _run_seed(output, data, cli, seed):
    """Apply all policies to identical scores after writing the CAL-only threshold lock."""
    parts, roster, policy, audit, args = _prepare_seed(data, cli, seed)
    routes = [_client_rows(roster, policy, client) for client in range(len(roster))]
    subsets, split_audit = split_operating_subsets(
        *parts["cal"], policy["roster"], policy["splits"], CAL_CHECK_FRACTION, seed,
        min_select_benign=0 if cli.smoke else 100,
    )
    with TemporaryDirectory(prefix="fedcaid_nb15_threshold_") as temporary:
        nodes = _fit_clients(data, parts, routes, args, Path(temporary))
        lock, lock_hash = _lock_cal(output, seed, nodes, subsets, split_audit, len(data.names))
        print(f"FedCAID | NB15 seed={seed} | CAL 阈值已锁定，开始 VAL", flush=True)
        val_x, val_y = parts["val"]
        scores = [_score_val(node, val_x, rows["val"], args) for node, rows in zip(nodes, routes, strict=True)]
        truth = [val_y[rows["val"]] for rows in routes]
        tables = evaluate_locked_seed(seed, data.names, audit["receiver"], truth, scores, lock)
    _write(output / f"seed_{seed}.json", {"seed": seed, "design": audit, "threshold_lock_sha256": lock_hash})
    return tables


def _collect(output, project, data, cli, seeds, source):
    """Collect all prespecified outcomes without evaluating any archived scores."""
    tables = defaultdict(list)
    for seed in seeds:
        if _sources(project) != source:
            raise ValueError("NB15 threshold experiment code changed during evaluation")
        for name, rows in _run_seed(output, data, cli, seed).items():
            tables[name].extend(rows)
    if _sources(project) != source:
        raise ValueError("NB15 threshold experiment code changed during evaluation")
    return tables


def _receiver_feasibility(rows, seeds):
    """Bound the paired FNR upper endpoint for any FAR-compliant receiver threshold."""
    reference = {r["seed"]: r for r in rows if r["arm"] == "reference_current" and r["scope"] == "receiver"}
    if len(seeds) != len(SEEDS) or any(
        reference[seed]["val_tpr_at_far_1pct_diagnostic"] is None for seed in seeds
    ):
        return None
    minimum = {
        seed: 1 - reference[seed]["val_tpr_at_far_1pct_diagnostic"] - reference[seed]["val_fnr"]
        for seed in seeds
    }
    mean_min, max_min = mean(minimum.values()), max(minimum.values())
    # With a fixed mean m and one difference >= max_min, SE >= (max_min - m)/(n-1) when m <= max_min.
    n = len(seeds)
    critical = float(t.ppf(0.975, n - 1))
    bound = mean_min + critical / (n - 1) * (max_min - mean_min)
    # By Cauchy-Schwarz the t upper endpoint also bounds any subset mean
    # when the t multiplier on centered L2 dominates the subset weights.
    for count in range(2, n):
        if critical / sqrt(n * (n - 1)) >= sqrt(1 / count - 1 / n):
            bound = max(bound, mean(sorted(minimum.values())[-count:]))
    return {
        "diagnostic_only": True, "selection_on_val": False, "fnr_budget": FNR_INCREASE_LIMIT,
        "per_seed_lower_bounds": [{"seed": seed, "min_val_fnr_increment": minimum[seed]} for seed in seeds],
        "mean_increment_lower_bound": mean_min, "paired_upper_95_lower_bound": bound,
        "joint_fnr_and_far_impossible_on_this_val": bound > FNR_INCREASE_LIMIT,
    }


def _write_plan(output, cli, seeds, source):
    """Declare score isolation and the complete fresh-run workload."""
    _write(output / "plan.json", {
        "dataset": DATASET, "target": SYNTHETIC_TARGET if cli.smoke else TARGET,
        "scenario": "train_absence", "protocol": "grouped", "clients": 3, "alpha": 0.3,
        "data_root": str(cli.data_root), "device": cli.device, "seeds": list(seeds),
        "synthetic_only": cli.smoke, "previous_run_inputs_allowed": False,
        "training": "fresh original prior-adjusted TabM on TRAIN; no candidate weight updates",
        "reference_epochs": cli.detector_epochs, "cal_check_fraction": CAL_CHECK_FRACTION,
        "arm_targets": {"reference_current": 0.01, "matched_cal_1pct": 0.01, "conservative_cal_0_9pct": 0.009},
        "locks_written_before_val_scoring": True, "selection_on_val": False,
        "test_evaluated": False, "source_sha256": source,
        "temporal_followup": "eligible only after the primary candidate passes grouped VAL",
    })


def _publish(output, tables, seeds):
    """Seal tables, locked thresholds, feasibility diagnosis and the report."""
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    assessment = assess_results(tables, seeds)
    assessment["receiver_threshold_only_feasibility"] = _receiver_feasibility(tables["working_points"], seeds)
    assessment["temporal_followup"] = (
        "eligible_for_separate_same_source_evaluation" if assessment["development_guard_pass"] is True
        else "not_triggered_by_grouped_development"
    )
    _write(output / "assessment.json", assessment)
    write_threshold_report(output, tables, assessment)
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "complete.json")
    _write(output / "complete.json", {"status": "completed", "artifacts": _bindings(output, names)})
    print(f"FedCAID | NB15 阈值专项完成 | {output / 'SUMMARY.md'}", flush=True)


def run_nb15_threshold(project, *, output=None, device="auto", smoke=False):
    """Execute only the prespecified threshold comparison in a new output directory."""
    project = Path(project).resolve(strict=True)
    output = _allocate_output(project, OUTPUT_CONTAINER, output=output)
    cli = _configuration(project, output, device, smoke)
    seeds, source = SEEDS[:1] if smoke else SEEDS, _sources(project)
    _write_plan(output, cli, seeds, source)
    print(f"FedCAID | NB15 阈值专项 | {output}", flush=True)
    if smoke:
        torch.set_num_threads(1)
    data = native._synthetic(DATASET, "grouped") if smoke else native.load_revision_data(
        DATASET, cli.data_root, split_protocol="grouped",
    )
    _write(output / "data_audit.json", data.audit)
    _publish(output, _collect(output, project, data, cli, seeds, source), seeds)
    return output
