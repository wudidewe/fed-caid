"""Validate the fixed-alarm NB15 evidence used by the targeted experiment."""

import csv
import math
from pathlib import Path

from .revision_detector_objective import HARD_BENIGN, objective_settings
from .revision_report import summarize, write_csv
from .revision_v12 import _hash, _read


STAGES = {"reference": "prior_adjusted", "candidate": HARD_BENIGN}
HISTORICAL_STAGES = {**STAGES, "candidate": "class_balanced"}
SEEDS = {str(seed) for seed in range(42, 47)}
CASE_FIELDS = ("dataset", "scenario", "alpha", "clients", "target_class", "cal_policy", "seed")
NB15_CASE = ("CIC-UNSW-NB15", "train_absence", "0.3", "3", "Exploits", "reserved")
AUDIT = Path("CIC-UNSW-NB15/data_audit.json")
PAIR_RECEIPTS = ("v12_complete.json", "v12_plan.json", "v12_state.json")
MATCHED_SETTING_EXCLUSIONS = {"detector_objective", "save_scores", "save_row_evidence", "export_frozen"}
TARGET_FAR = 0.01
MAX_FNR_INCREASE = 0.01
BINARY_OUTPUTS = ("binary_pair_status.json", "binary_pair_by_seed.csv")


def _inside(root, name):
    """Resolve an existing artifact without leaving its receipt directory."""
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact leaves its recorded directory: {name}")
    return path


def _bindings(root, names):
    """Bind an artifact inventory to its SHA-256 content digests."""
    return {name: _hash(_inside(root, name)) for name in sorted(names)}


def _check_bindings(root, bindings):
    """Reject an empty or changed artifact inventory."""
    if not bindings or _bindings(root, bindings) != bindings:
        raise ValueError(f"Saved artifacts changed: {root}")


def _files(root):
    """List the saved files in one experiment case."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _checked_case(root, row):
    """Check one source result and identify its prespecified case."""
    key = tuple(row[field] for field in CASE_FIELDS)
    if row["status"] != "evaluated" or key[:-1] != NB15_CASE:
        raise ValueError(f"Expected an evaluated NB15 Exploits case: {root}")
    path = _inside(root, row["result"])
    if _hash(path) != row["result_sha256"]:
        raise ValueError(f"Case result changed: {path}")
    return key


def _cases(root):
    """Verify all five result records and return them by case identity."""
    cases = {}
    with (root / "case_status.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = _checked_case(root, row)
            if key in cases:
                raise ValueError(f"Duplicate NB15 case: {root}")
            cases[key] = row
    if len(cases) != len(SEEDS) or {key[-1] for key in cases} != SEEDS:
        raise ValueError("The NB15 source requires all five prespecified seeds (42-46)")
    return cases


def _original_pair(root):
    """Read the V12 plan and state after checking their completion receipt."""
    marker = _read(root / PAIR_RECEIPTS[0])
    identity = (marker.get("mode"), marker.get("scope"), marker.get("status"), marker.get("test_evaluated"))
    if identity != ("nb15-pair", "development", "completed", False):
        raise ValueError("Expected the completed real V12 NB15 pair, without TEST evaluation")
    for name, field in zip(PAIR_RECEIPTS[1:], ("plan_sha256", "state_sha256"), strict=True):
        if _hash(_inside(root, name)) != marker[field]:
            raise ValueError(f"Original NB15 receipt changed: {name}")
    return _read(root / PAIR_RECEIPTS[1]), _read(root / PAIR_RECEIPTS[2])


def _run_receipt(root, objective):
    """Require a real fixed-policy score export for the requested detector objective."""
    run, done = _read(root / "run.json"), _read(root / "complete.json")
    settings = run["settings"]
    valid_run = (
        run["version"] == 4
        and settings["detector_objective"] == objective
        and settings["operating_policy"] == "fixed"
        and settings["save_scores"] is True
    )
    expected = {
        "mode": "evaluation",
        "score_replay": False,
        "test_evaluated": False,
        "evaluated": len(SEEDS),
        "smoke": False,
    }
    if not valid_run or any(done.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Expected five real fixed-policy {objective} source cases")
    return run


def _case_artifacts(root, row):
    """Verify the score cache referenced by an already checked result row."""
    case_path = _inside(root, row["result"])
    cache = _read(case_path).get("score_cache") or {}
    if cache.get("schema_version") != 2 or cache.get("file") != "private_cal_val_scores.npz":
        raise ValueError(f"Missing schema-2 private score cache: {case_path}")
    cache_path = _inside(root, case_path.parent / cache["file"])
    if _hash(cache_path) != cache["sha256"]:
        raise ValueError(f"Original NB15 score cache changed: {cache_path}")
    return case_path, cache_path


def _stage(source, name, objective, bindings):
    """Verify one objective's published tables, cases and private scores."""
    root = _inside(source, name)
    if not {"run.json", "complete.json", "case_status.csv"} <= bindings.keys():
        raise ValueError(f"Original NB15 stage receipt is incomplete: {root}")
    _check_bindings(root, bindings)
    run = _run_receipt(root, objective)
    cases = _cases(root)
    paths = [root / path for path in bindings] + [root / AUDIT]
    for row in cases.values():
        paths.extend(_case_artifacts(root, row))
    return run, cases, [path.relative_to(source).as_posix() for path in paths]


def _verify_pair(source, runs, cases):
    """Require both objectives to use the same software, partition and data audit."""
    reference, candidate = cases["reference"], cases["candidate"]
    if reference.keys() != candidate.keys() or any(
        reference[key][field] != candidate[key][field]
        for key in reference
        for field in ("partition_sha256", "receiver", "status")
    ):
        raise ValueError("NB15 cases or row partitions differ")
    settings = [
        {key: value for key, value in runs[name]["settings"].items() if key not in MATCHED_SETTING_EXCLUSIONS}
        for name in STAGES
    ]
    if settings[0] != settings[1] or runs["reference"]["code_sha256"] != runs["candidate"]["code_sha256"]:
        raise ValueError("Source objectives were not run with matching settings and code")
    if _read(_inside(source, Path("reference") / AUDIT)) != _read(_inside(source, Path("candidate") / AUDIT)):
        raise ValueError("Source objectives used different data audits")


def _source(source):
    """Return the verified artifact inventory, original plan and objective receipts."""
    source = Path(source).resolve(strict=True)
    plan, state = _original_pair(source)
    names = list(PAIR_RECEIPTS)
    if "detector_objectives" in plan:
        evidence = _read(source / PAIR_RECEIPTS[0]).get("binary_artifacts", {})
        if set(evidence) != set(BINARY_OUTPUTS):
            raise ValueError("Missing paired NB15 binary assessment")
        _check_bindings(source, evidence)
        names.extend(evidence)
    runs, cases = {}, {}
    for name, objective in _plan_objectives(plan).items():
        runs[name], cases[name], artifacts = _stage(source, name, objective, state[name])
        names.extend(artifacts)
    _verify_pair(source, runs, cases)
    return {"path": str(source), "artifacts": _bindings(source, names)}, plan, runs


def _plan_objectives(plan):
    """Keep old class-balanced receipts readable through the explicit evidence API."""
    objectives = plan.get("detector_objectives", HISTORICAL_STAGES)
    if objectives not in (STAGES, HISTORICAL_STAGES):
        raise ValueError("Unexpected NB15 detector objective assignment")
    return objectives


def _development_rows(path, mode):
    """Read one fixed-mode VAL table with only the five prespecified NB15 cases."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if (row["role"], row["mode"]) == ("val", mode)]
    if {row["seed"] for row in selected} != SEEDS or any(
        tuple(row[field] for field in CASE_FIELDS[:-1]) != NB15_CASE for row in selected
    ):
        raise ValueError(f"Unexpected paired NB15 development cases: {path}")
    return selected


def _base_alarm(row, clients):
    """Require equal fixed-head and base binary error rates at the operating point."""
    benign_rows = sum(int(client["val_benign_rows"]) for client in clients)
    attack_rows = sum(int(client["val_attack_rows"]) for client in clients)
    far = sum(int(client["val_false_positives"]) for client in clients) / benign_rows
    fnr = sum(int(client["val_false_negatives"]) for client in clients) / attack_rows
    if not math.isclose(far, float(row["far"])) or not math.isclose(fnr, float(row["fnr"])):
        raise ValueError(f"Fixed head and base detector binary counts disagree: seed {row['seed']}")
    return {
        "far": far,
        "fnr": fnr,
        "ece": float(row["ece"]),
        "macro_f1": float(row["macro_f1"]),
        "benign_rows": benign_rows,
        "attack_rows": attack_rows,
    }


def _optional_rate(row, field):
    """Preserve undefined CAL or continuous-score diagnostics."""
    return float(row[field]) if row[field] else None


def _receiver_alarm(clients):
    """Report the fixed receiver alert and keep continuous-score diagnostics separate."""
    [row] = [client for client in clients if client["receiver"] == "True"]
    return {
        "receiver": int(row["client"]),
        "far": float(row["val_far"]),
        "fnr": float(row["val_fnr"]),
        "benign_rows": int(row["val_benign_rows"]),
        "attack_rows": int(row["val_attack_rows"]),
        "cal_threshold": float(row["threshold"]),
        "cal_far": _optional_rate(row, "cal_far"),
        "cal_tpr_at_far_1pct": _optional_rate(row, "cal_tpr_at_far_1pct"),
        "val_tpr_at_far_1pct_diagnostic": _optional_rate(row, "val_tpr_at_far_1pct"),
    }


def _binary_outcomes(root):
    """Collect the receiver and pooled base alert outcomes on the same VAL rows."""
    metrics = _development_rows(root / "metrics.csv", "base_threshold")
    diagnostics = _development_rows(root / "binary_diagnostics.csv", "fixed_federated")
    by_case = {}
    for row in metrics:
        seed = row["seed"]
        clients = [client for client in diagnostics if client["seed"] == seed]
        if seed in by_case or sorted(int(client["client"]) for client in clients) != list(range(3)):
            raise ValueError(f"Invalid paired NB15 client inventory: {root}")
        by_case[seed] = {"overall": _base_alarm(row, clients), "receiver": _receiver_alarm(clients)}
    return by_case


def _paired_binary_rows(root):
    """Build a paired receiver and overall table from the two fixed-policy fits."""
    reference, candidate = (_binary_outcomes(root / name) for name in STAGES)
    paired = []
    for seed in sorted(SEEDS):
        for scope in ("overall", "receiver"):
            before, after = reference[seed][scope], candidate[seed][scope]
            for key in ("benign_rows", "attack_rows", "receiver"):
                if before.get(key) != after.get(key):
                    raise ValueError(f"Unmatched NB15 binary support for seed {seed}, {scope}")
            paired.append({
                "seed": int(seed),
                "scope": scope,
                **{f"reference_{key}": value for key, value in before.items()},
                **{f"candidate_{key}": value for key, value in after.items()},
                "delta_fnr": after["fnr"] - before["fnr"],
                "delta_far": after["far"] - before["far"],
                "delta_macro_f1": after["macro_f1"] - before["macro_f1"] if scope == "overall" else None,
            })
    return paired


def _binary_summary(rows):
    """Compare candidate FAR and paired FNR against prespecified constraints."""
    far = [row["candidate_far"] for row in rows]
    fnr_delta = summarize([row["delta_fnr"] for row in rows], bounds=(-1, 1))
    return {
        "candidate_far_max": max(far),
        "candidate_far_seeds_over_limit": sum(value > TARGET_FAR for value in far),
        "reference_far_seeds_over_limit": sum(row["reference_far"] > TARGET_FAR for row in rows),
        "candidate_minus_reference_far": summarize([row["delta_far"] for row in rows], bounds=(-1, 1)),
        "candidate_minus_reference_fnr": fnr_delta,
        "far_pass": all(value <= TARGET_FAR for value in far),
        "fnr_pass": fnr_delta["ci95_high"] <= MAX_FNR_INCREASE,
    }


def _classification_summary(rows):
    """Check pooled matched-row macro-F1 without tuning on VAL."""
    delta = summarize([row["delta_macro_f1"] for row in rows if row["scope"] == "overall"], bounds=(-1, 1))
    return {
        "mode": "base_threshold",
        "scope": "overall",
        "candidate_minus_reference_macro_f1": delta,
        "minimum_paired_mean_delta": 0.0,
        "pass": delta["mean"] >= 0,
    }


def binary_pair_assessment(root, states, plan):
    """Publish a binary development result without using VAL to pick a threshold."""
    runs = {name: _run_receipt(root / name, objective) for name, objective in _plan_objectives(plan).items()}
    _verify_pair(root, runs, {name: _cases(root / name) for name in states})
    rows = _paired_binary_rows(root)
    table = {
        scope: _binary_summary([row for row in rows if row["scope"] == scope])
        for scope in ("overall", "receiver")
    }
    passed = all(value["far_pass"] and value["fnr_pass"] for value in table.values())
    classification = _classification_summary(rows)
    write_csv(root / BINARY_OUTPUTS[1], rows)
    return {
        "status": "development_binary_criteria_met" if passed else "development_binary_criteria_not_met",
        "combined_acceptance": passed and classification["pass"],
        "classification_guard": classification,
        "evaluation_scope": "grouped benchmark VAL, fixed scalar thresholds fitted on CAL only",
        "test_evaluated": False,
        "selection_on_val": False,
        "candidate_settings": objective_settings(runs["candidate"]["settings"]["detector_objective"]),
        "criteria": {"max_far_each_seed": TARGET_FAR, "max_paired_fnr_increase_ci95_upper": MAX_FNR_INCREASE},
        "scopes": table,
    }
