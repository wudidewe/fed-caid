"""Export review evidence using only receipts produced inside one fresh run."""

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from shutil import copyfile
from statistics import mean
from tempfile import TemporaryDirectory

from .revision_benchmark_matrix import benchmark_paper_rows
from .revision_iov_diagnostics import CLASS_FILE as IOV_CLASS_FILE
from .revision_iov_diagnostics import PAIRED_FILE as IOV_PAIRED_FILE
from .revision_iov_diagnostics import STATUS_FILE as IOV_STATUS_FILE
from .revision_nb15_review_tables import BINARY_TABLE, CLASS_TABLE, FRONTIER_TABLE, write_nb15_review_tables
from .revision_v12 import _hash, _read, _write


STAGE_LABELS = {
    "benchmark_matrix": "Four-dataset benchmark matrix",
    "iov_diagnosis": "CIC-IoV-2024 working-point diagnosis",
    "scale_matrix": "Client scale and Dirichlet heterogeneity",
    "can_quantization": "CAN adapter quantization",
    "nb15_temporal": "NB15 exact-purged later-day evaluation",
    "loopback": "Same-host TCP communication measurement",
    "kminus1_transfer": "Original K−1 missing-class transfer",
    "nb15_pair": "NB15 matched detector and calibration inputs",
    "nb15_low_far": "NB15 frozen backbone and full weight tuning",
    "nb15_threshold": "NB15 grouped CAL threshold comparison",
    "targeted": "NB15 fixed-alarm subtype comparison",
    "donor_check": "NB15 donor CAL threshold replay",
    "hulk_attribution": "Hulk frozen attribution",
    "external": "Hulk independent capture confirmation",
    "lan": "LAN transport measurement",
}
STAGE_STATUSES = {"completed", "pending", "failed"}
REVIEW_DOCUMENTS = {"data_audit.json", "binary_pair_status.json", IOV_STATUS_FILE}
STAGE_COVERAGE = {
    "nb15_pair": ("CIC-UNSW-NB15", "Matched TabM detector objectives", "Grouped benchmark VAL"),
    "nb15_low_far": ("CIC-UNSW-NB15", "Frozen backbone and full weight tuning", "Grouped development VAL"),
    "nb15_threshold": ("CIC-UNSW-NB15", "Matched CAL threshold rules", "Grouped development VAL"),
    "targeted": ("CIC-UNSW-NB15", "Fixed-alarm subtype protections", "Grouped benchmark VAL"),
    "donor_check": (
        "CIC-UNSW-NB15",
        "Donor CAL feasibility, receiver check and replay",
        "Grouped benchmark VAL",
    ),
    "hulk_attribution": (
        "CIC-IDS-2017",
        "Controlled Hulk absence and matched subtype controls",
        "Grouped benchmark VAL",
    ),
    "external": ("CIC-IDS-2017", "Frozen Hulk models on independent capture", "Declared capture holdout"),
    "lan": ("Transport", "Collector communication at 3, 5 and 10 clients", "Declared LAN endpoints"),
    "iov_diagnosis": ("CIC-IoV-2024", "Working-point and feature-group support audit", "Grouped CAL and VAL"),
    "scale_matrix": (
        "CIC-IDS-2017 and CIC-UNSW-NB15",
        "Client scale and Dirichlet heterogeneity",
        "Grouped benchmark VAL, 3/5/10 clients and multiple α",
    ),
    "nb15_temporal": (
        "CIC-UNSW-NB15",
        "Same-source later-day benchmark, previously explored evaluation roles",
        "Chronological split with exact-input purge",
    ),
    "loopback": (
        "Transport",
        "Synthetic numeric aggregation at 3, 5 and 10 clients",
        "Same-host TCP loopback",
    ),
    "kminus1_transfer": (
        "CIC-IDS-2017 and CIC-UNSW-NB15",
        "Original K−1 transfer with controlled missing-class cells",
        "Grouped benchmark and retrospective TEST",
    ),
}
OUTSIDE_COVERAGE = (
    ("CIC-IoV-2024", "Shared subtype head", "CAN benchmark"),
    ("Car-Hacking", "Shared subtype head", "CAN benchmark"),
    ("Four-dataset matrix", "All-dataset frozen-role baseline comparison", "Frozen benchmark roles"),
)
MATRIX_DATASETS = ("CIC-IDS-2017", "CIC-UNSW-NB15", "CIC-IoV-2024", "Car-Hacking")
CAN_DATASETS = ("CIC-IoV-2024", "Car-Hacking")
ABLATION_TABLE = "benchmark_ablation_effects.csv"
CLASS_SUPPORT_TABLE = "benchmark_class_support.csv"
STAGE_TIMING_TABLE = "stage_wall_time.csv"
ABLATION_METRICS = ("macro_f1", "far", "fnr")
ABLATION_CONTRASTS = {
    "local_only_minus_base_threshold": "Local adapter − base, each with CAL threshold",
    "full_minus_local_only": "Federated adapter − matched local adapter",
    "full_minus_without_prox": "Federated adapter with − without proximal penalty",
    "full_minus_without_threshold": "Federated CAL threshold − fixed 0.5",
    "local_only_minus_local_fixed": "Local CAL threshold − fixed 0.5",
}
RARE_CLASS_MAX_SUPPORT = 100
OPTIONAL_PEERS = {
    "benchmark_matrix",
    "iov_diagnosis",
    "scale_matrix",
    "can_quantization",
    "nb15_temporal",
    "loopback",
    "kminus1_transfer",
    "nb15_low_far",
    "nb15_threshold",
}
AUXILIARY_STAGES = {"scale_matrix", "can_quantization", "nb15_temporal", "loopback", "kminus1_transfer"}
AUXILIARY_TABLES = {
    "scale_matrix": {
        "metrics.csv": "Scale and heterogeneity absolute grouped VAL metrics",
        "paired_intervals.csv": "Scale and heterogeneity matched effects",
        "timings.csv": "Scale and heterogeneity single-host time and memory measurements",
        "communication.csv": "Scale and heterogeneity logical model payload",
        "case_status.csv": "Scale and heterogeneity run coverage by case",
        "coverage.csv": "Scale and heterogeneity operating and CAL coverage",
    },
    "can_quantization": {
        "quantization_diagnostics.csv": "Original CAN adapter quantization mechanics",
        "metrics_by_seed.csv": "CAN detection metrics by seed and evaluation role",
        "paired_differences.csv": "Matched CAN quantization effects",
        "partition_support.csv": "CAN per-class support",
    },
    "nb15_temporal": {
        "metrics_by_seed.csv": "NB15 later-day benchmark metrics by evaluation role",
        "paired_differences.csv": "NB15 later-day matched effects",
        "per_class.csv": "NB15 later-day per-class precision, recall, F1 and support",
        "data_audit.json": "NB15 later-day chronological role and exact-input purge audit",
    },
    "loopback": {
        "loopback_summary.csv": "Same-host TCP timing, logical payload and actual application bytes",
        "loopback_plan.json": "Same-host TCP measurement settings",
    },
    "kminus1_transfer": {
        "transfer_score_deltas.csv": "Original K−1 per-cell score and rank changes",
        "transfer_score_intervals.csv": "Original K−1 paired score and rank effects",
        "train_budget_comparison.csv": "Original K−1 fixed and full training budget comparisons",
        "transfer_cells.csv": "Original K−1 missing-class cell measurements",
        "transfer_matrix.csv": "Original K−1 missing-class case inventory",
        "metrics_by_seed.csv": "Original K−1 transfer performance by seed and role",
        "paired_differences.csv": "Original K−1 transfer matched detection effects",
    },
}
NATIVE_BRANCHES = {
    "nb15_pair": ("reference", "candidate"),
    "hulk_attribution": ("attribution",),
}
CONDITION_FIELDS = ("dataset", "scenario", "alpha", "clients", "target_class", "cal_policy")
OPERATING_FIELDS = (
    "status",
    "observed_absolute_far_met",
    "max_receiver_far",
    "max_overall_far",
    "selected_cal_far",
    "receiver_operating_fnr_delta_ci95_high",
    "overall_operating_fnr_delta_ci95_high",
    "cal_guard_disabled_seeds",
    "cal_check_fallback_seeds",
)
NATIVE_CONTRASTS = {
    "guarded_against_base": "Federated − base TabM",
    "guarded_aggregation": "Federated − local subtype",
    "guarded_receiver_participation": "Federated − donor pool",
    "guarded_donor_pool_vs_single": "Donor pool − single donor",
    "guarded_against_fedavg": "Federated − full-weight FedAvg subtype",
}
PAIR_METRICS = {("receiver", "f1"), ("overall", "macro_f1")}
PAIR_FIELDS = ("scope", "metric", "seeds_with_metric", "mean", "ci95_low", "ci95_high")
HIGHLIGHTED_PAIRS = {
    ("nb15_pair", "guarded_aggregation", "receiver", "f1"),
    ("hulk_attribution", "guarded_aggregation", "receiver", "f1"),
    ("hulk_attribution", "guarded_receiver_participation", "receiver", "f1"),
    ("hulk_attribution", "guarded_against_fedavg", "overall", "macro_f1"),
    ("targeted", "combined_candidate", "receiver", "f1"),
    ("donor_check", "guard_vs_legacy", "receiver", "f1"),
}
BRANCHES = {
    "base_threshold": ("Base TabM", ""),
    "guarded_local": ("Local subtype", "head_local"),
    "guarded_donor_single": ("Single donor", "head_donor_single"),
    "guarded_donor": ("Donor pool", "head_donor"),
    "guarded_federated": ("Complete federated", "head_federated"),
    "guarded_fedavg_subtype": ("Full-weight FedAvg subtype", "fedavg_threshold"),
}
BRANCH_SOURCE = "tables/hulk_attribution/attribution"
BRANCH_INPUTS = (
    "case_status",
    "metrics",
    "target_metrics",
    "binary_diagnostics",
    "subtype_audit",
    "communication",
)
PAYLOAD_FIELDS = (
    "round_payload_bytes",
    "training_payload_bytes",
    "receiver_delivery_payload_bytes",
    "total_payload_bytes",
    "feature_map_setup_payload_bytes",
    "shared_preprocessing_setup_payload_bytes",
)

SCOPE = """## Scope

- Each experimental artifact below belongs to the named run. Where evaluated, the NB15
  subtype and donor-CAL stages share the run's detector baseline and scores.
- Five seed runs on shared captures quantify development variability. Only the
  separately declared independent capture stage can confirm performance on new captures.
- Subtype replay holds TabM binary alarms fixed. Its changes affect subtype labels.
  Read the absolute FAR and scientific criteria alongside the paired subtype effects.
- The low-FAR criteria include observed absolute FAR and the paired FNR guard in
  both receiver and overall populations. On a fixed alarm mask, any target F1 change
  comes from subtype assignment. Receiver participation is assessed against the donor pool.
- Logical model payload describes numeric exchanges. Measured LAN payload and time
  require a separate transport run. Same-host TCP records describe synthetic aggregation
  on one computer. The later-day NB15 benchmark uses the original data source and
  previously explored evaluation roles.
- Stage completion records execution. Consult the scientific status and coverage for each result.
"""


def _artifact_path(root, name):
    """Reject absolute, traversal, and redirected bindings before reading them."""
    relative = Path(name)
    if relative.is_absolute() or relative.anchor or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Expected a relative artifact inside this run: {name}")
    path = root / relative
    if path.resolve() != path:
        raise ValueError(f"Redirected artifact is outside its declared location: {path}")
    return path


def _stage_root(run_root, stage, record):
    """Bind each stage to the directory assigned by the current run."""
    expected = run_root / stage
    declared = Path(record.get("output", expected))
    root = Path(record.get("artifact_root", declared))
    if declared != expected or root.resolve() != root or not root.is_relative_to(expected):
        raise ValueError(f"Stage {stage} must stay under its assigned current-run directory")
    return root


def _copy_table(source, relative, expected, working):
    """Copy bound table bytes without modifying labels, precision or original results."""
    destination = working / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, destination)
    if _hash(destination) != expected:
        raise ValueError(f"Table changed while the report was copied: {source}")


def _collect_stage(run_root, stage, record, working):
    """Verify one stage and export its bound tables and data role audits."""
    if not isinstance(record, Mapping) or record.get("status") not in STAGE_STATUSES:
        raise ValueError(f"Unknown execution status for stage {stage}")
    root = _stage_root(run_root, stage, record)
    bindings = record.get("artifacts", {})
    if record["status"] != "completed":
        return {}, []
    if not bindings:
        raise ValueError(f"Completed stage {stage} has no artifact bindings")
    tables = []
    for name, path, expected in _verified_files(root, bindings):
        if _export_for_review(stage, Path(name)):
            relative = Path("tables") / stage / name
            _copy_table(path, relative, expected, working)
            tables.append(relative.as_posix())
    return {str(root / name): digest for name, digest in bindings.items()}, sorted(tables)


def _verified_files(root, bindings):
    """Verify bound source file contents and paths before any use."""
    for name, expected in bindings.items():
        path = _artifact_path(root, name)
        if _hash(path) != expected:
            raise ValueError(f"Current-run artifact changed: {path}")
        yield name, path, expected


def _export_for_review(stage, artifact):
    """Keep both NB15 objective receipts alongside measurements and audits."""
    if artifact.suffix.lower() == ".csv" or artifact.name in REVIEW_DOCUMENTS:
        return True
    if stage in {"nb15_low_far", "nb15_threshold"} and artifact.name in {
        "assessment.json", "plan.json", "complete.json", "SUMMARY.md",
    }:
        return True
    if stage == "external" and artifact.as_posix() == "result/complete.json":
        return True
    if stage in AUXILIARY_STAGES and artifact.name in {
        "run.json",
        "complete.json",
        "stage_complete.json",
        "loopback_plan.json",
    }:
        return True
    return stage == "nb15_pair" and artifact.as_posix() in {"reference/run.json", "candidate/run.json"}


def _rows(root, source):
    """Read the rows of one exported, hash-checked table."""
    path = root / source
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def _selected_fields(row, fields):
    """Preserve the original values and precision of selected columns."""
    return {field: row.get(field, "") for field in fields}


def _decisions(working):
    """Read recorded guarded decisions from the primary CAL policy only."""
    for stage, branches in NATIVE_BRANCHES.items():
        for branch in branches:
            source = f"tables/{stage}/{branch}/guarded_subtype_decisions.csv"
            for row in _rows(working, source):
                if row.get("primary_cal_policy", "").lower() == "true" and row.get("status"):
                    yield {
                        "stage": stage,
                        "branch": branch,
                        "detector_objective": _objective(working, stage, branch),
                        **_selected_fields(row, CONDITION_FIELDS + OPERATING_FIELDS),
                        "artifact": source,
                    }


def _objective(working, stage, branch):
    """Identify the objective from the branch receipt that was hash checked."""
    path = working / "tables" / stage / branch / "run.json"
    return _read(path).get("settings", {}).get("detector_objective", "") if path.is_file() else ""


def _matches(row, decision):
    """Keep comparisons within the decision's recorded experimental condition."""
    return not decision or all(row.get(key, "") == decision[key] for key in CONDITION_FIELDS)


def _paired_rows(working, stage, source, contrasts, decision=None):
    """Export specified paired comparisons, including all observed seed counts."""
    context = _selected_fields(decision or {}, ("branch", "detector_objective", *CONDITION_FIELDS))
    for row in _rows(working, source):
        contrast = row.get("contrast")
        if contrast not in contrasts or (row.get("scope"), row.get("metric")) not in PAIR_METRICS:
            continue
        if _matches(row, decision):
            yield {
                "stage": stage,
                **context,
                "contrast": contrast,
                "comparison": contrasts[contrast],
                **_selected_fields(row, PAIR_FIELDS),
                "artifact": source,
            }


def _paired(working, decisions):
    """Collect native controls and the pre-existing targeted and donor contrasts."""
    for decision in decisions:
        stage = decision["stage"]
        source = f"tables/{stage}/{decision['branch']}/paired_intervals.csv"
        yield from _paired_rows(working, stage, source, NATIVE_CONTRASTS, decision)
    yield from _paired_rows(
        working,
        "targeted",
        "tables/targeted/paired_intervals.csv",
        {"combined_candidate": "Protected attack-only − legacy multiclass"},
    )
    yield from _paired_rows(
        working,
        "donor_check",
        "tables/donor_check/summary.csv",
        {
            "guard_vs_legacy": "Donor-checked federated − legacy federated",
            "aggregation_matched_donor_cal": "Donor-checked federated − matched local",
            "receiver_participation_matched_donor_cal": "Donor-checked federated − matched donor pool",
        },
    )


def _branch_inputs(working, decision):
    """Read hash-checked records belonging to one Hulk condition."""
    condition = CONDITION_FIELDS[:-1]
    records = {}
    for name in BRANCH_INPUTS:
        rows = _rows(working, f"{BRANCH_SOURCE}/{name}.csv")
        records[name] = [
            row
            for row in rows
            if all(row.get(field) == decision[field] for field in condition)
            and (name == "communication" or row.get("cal_policy") == decision["cal_policy"])
            and row.get("role", "val") == "val"
        ]
    return records


def _mode_rows(rows, seed):
    """Require one measurement per seed and mode."""
    indexed = {}
    for row in rows:
        if row["seed"] != seed:
            continue
        mode = row["mode"]
        if mode in indexed:
            raise ValueError(f"Repeated Hulk mode {mode} for seed {seed}")
        indexed[mode] = row
    return indexed


def _mask_audit(rows, case):
    """Verify that every guarded branch inherited the same per-client TabM mask."""
    expected = set(BRANCHES) - {"base_threshold"}
    by_client = {}
    for row in rows:
        if row["seed"] != case["seed"] or row["mode"] not in expected:
            continue
        modes = by_client.setdefault(row["client"], {})
        if row["mode"] in modes:
            raise ValueError("Repeated Hulk alarm mask audit")
        modes[row["mode"]] = row["base_alarm_mask_sha256"]
    if len(by_client) != int(case["clients"]):
        raise ValueError("Incomplete Hulk alarm mask audit")
    for modes in by_client.values():
        if set(modes) != expected or "" in modes.values() or len(set(modes.values())) != 1:
            raise ValueError("Hulk branch alarm masks are not matched")
    return by_client[case["receiver"]]["guarded_federated"]


def _branch_metrics(overall, target, binary):
    """Extract receiver target performance and measured binary alarm rates."""
    if float(target["receiver_far"]) != float(binary["val_far"]):
        raise ValueError("Hulk receiver FAR differs across matched branches")
    return {
        **{f"target_{field}": target[field] for field in ("support", "precision", "recall", "f1")},
        "overall_macro_f1": overall["macro_f1"],
        "receiver_far": target["receiver_far"],
        "receiver_fnr": binary["val_fnr"],
        "overall_far": overall["far"],
        "overall_fnr": overall["fnr"],
    }


def _branch_seed(records, case):
    """Join complete branch measurements and logical payload on the same seed."""
    seed = case["seed"]
    overall, target, communication = (
        _mode_rows(records[name], seed) for name in ("metrics", "target_metrics", "communication")
    )
    binary = [row for row in records["binary_diagnostics"] if row["client"] == case["receiver"]]
    reference = _mode_rows(binary, seed)["guarded_federated"]
    mask = _mask_audit(records["subtype_audit"], case)
    context = _selected_fields(case, (*CONDITION_FIELDS, "seed", "receiver", "partition_sha256"))
    for mode, (label, payload_mode) in BRANCHES.items():
        if target[mode]["client"] != f"client_{case['receiver']}":
            raise ValueError("Hulk target metrics do not belong to the recorded receiver")
        metrics = _branch_metrics(overall[mode], target[mode], reference)
        baseline = overall["base_threshold"]
        if any(overall[mode][key] != baseline[key] for key in ("far", "fnr")):
            raise ValueError("Hulk overall FAR or FNR differs across matched branches")
        if target[mode]["support"] != target["base_threshold"]["support"]:
            raise ValueError("Hulk receiver target support differs across matched branches")
        yield {
            **context,
            "mode": mode,
            "branch": label,
            "receiver_alarm_mask_sha256": mask,
            **metrics,
            **_branch_payload(communication, payload_mode),
        }


def _branch_payload(communication, mode):
    """Expose training, delivery and separately recorded preprocessing bytes."""
    if not mode:
        return {**dict.fromkeys(PAYLOAD_FIELDS, 0), "payload_scope": "Local TabM, no model exchange"}
    row = communication[mode]
    return {**_selected_fields(row, PAYLOAD_FIELDS), "payload_scope": row["scope"]}


def _branch_rows(working, decisions):
    """Export complete per-seed VAL results from the primary Hulk CAL policy."""
    for decision in decisions:
        if decision["stage"] != "hulk_attribution":
            continue
        records = _branch_inputs(working, decision)
        if not all(records.values()):
            continue
        for case in records["case_status"]:
            if case["status"] == "evaluated":
                yield from _branch_seed(records, case)


def _audit_rows(working, tables):
    """Summarize bound exact-input comparisons and link the underlying role audit."""
    for stage, sources in tables.items():
        for source in sources:
            if Path(source).name == "data_audit.json":
                yield _role_audit_summary(stage, source, _read(working / source))


def _role_audit_summary(stage, source, audit):
    """Retain role size, within-CAL repetition and the audited overlap scope."""
    exact = audit.get("exact_feature_audit", {})
    overlap = exact.get("cross_role_exact_overlap", {})
    cal_repeats = exact.get("within_role_repeats", {}).get("cal", {})
    return {
        "stage": stage,
        "evaluation_scope": audit.get("evaluation_scope", ""),
        **_selected_fields(audit.get("role_rows", {}), ("train", "cal", "val", "test")),
        "cal_repeated_rows": cal_repeats.get("rows_beyond_one_per_distinct_input", ""),
        "audited_role_pairs": len(overlap),
        "max_unique_shared_inputs": max(
            (row["unique_shared_inputs"] for row in overlap.values()), default=""
        ),
        "scope_limit": exact.get("scope_limit", ""),
        "artifact": source,
    }


def _write_table(path, rows):
    """Write a compact evidence index only when its source was collected."""
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _science(record):
    """Extract the recorded assessment without interpreting execution as success."""
    assessment = record.get("scientific_status", record.get("assessment"))
    if isinstance(assessment, Mapping):
        assessment = assessment.get("scientific_status", assessment.get("status", "see stage receipt"))
    statuses = [str(assessment)] if assessment else []
    binary = record.get("binary_assessment", {})
    if binary.get("status"):
        statuses.append(f"binary — {binary['status']}")
    if "combined_acceptance" in binary:
        verdict = "criteria met" if binary["combined_acceptance"] else "criteria not met"
        statuses.append(f"binary and macro-F1 — {verdict}")
    native = {(row["branch"], row["status"]) for row in record.get("native_development_decisions", [])}
    statuses.extend(f"{branch} — {status}" for branch, status in sorted(native))
    return " · ".join(statuses) or "not assessed"


def _cell(value):
    """Keep arbitrary status text within one Markdown table cell."""
    return " ".join(str(value).splitlines()).replace("|", r"\|")


def _status_lines(stages):
    """Keep stage completion and scientific assessment in separate columns."""
    lines = ["| Experiment | Execution | Scientific status | Note |", "| --- | --- | --- | --- |"]
    for stage, record in stages.items():
        cells = (STAGE_LABELS[stage], record["status"], _science(record), record.get("reason", ""))
        lines.append("| " + " | ".join(map(_cell, cells)) + " |")
    return lines


def _heldout_scored(stage, record):
    """Separate benchmark TEST scoring from independent capture scoring."""
    if stage in {"lan", "loopback"}:
        return "not applicable"
    key = "scoring" if stage == "external" else "test_evaluated"
    default = "no" if record["status"] == "pending" else "not recorded"
    return {True: "yes", False: "no"}.get(record.get(key), default)


def _scope_note(stage, record):
    """Preserve blockers and distinguish reported transport from host verification."""
    if stage == "lan" and record["status"] == "completed":
        if record.get("independent_physical_hosts_verified") is False:
            return "Operator-declared endpoints. Independent physical host verification unavailable."
    if stage == "loopback" and record["status"] == "completed":
        return "Timing covers synthetic aggregation transport on one host."
    if stage == "nb15_temporal" and record["status"] == "completed":
        return "The dated holdout shares the benchmark source and has been explored previously."
    if stage == "can_quantization" and record["status"] == "completed":
        return "Previously explored same-source benchmark TEST roles are reported separately from VAL."
    if stage == "kminus1_transfer" and record["status"] == "completed":
        return "Controlled absence and retrospective TEST describe the grouped benchmark source."
    return record.get("reason", "")


def _coverage_record(dataset, evidence, protocol, status, heldout_scored, note):
    """Keep coverage comparable across datasets, protocols and transport."""
    return dict(
        dataset=dataset,
        evidence=evidence,
        protocol=protocol,
        execution=status,
        heldout_scored=heldout_scored,
        note=note,
    )


def _coverage(stages):
    """List performed work, conditional stages and work outside this run."""
    matrix = stages.get("benchmark_matrix")
    for dataset in MATRIX_DATASETS:
        yield _coverage_record(
            dataset,
            "Dirichlet label skew, 3 clients, α=0.8, base TabM / K−1 adapter / full-weight FedAvg",
            "Grouped benchmark VAL",
            matrix["status"] if matrix else "outside run scope",
            _heldout_scored("benchmark_matrix", matrix) if matrix else "not run",
            matrix.get("reason", "") if matrix else "",
        )
    yield from _can_coverage(stages.get("can_quantization"))
    for stage, scope in STAGE_COVERAGE.items():
        record = stages.get(stage)
        if record is None:
            if stage != "iov_diagnosis":
                yield _coverage_record(*scope, "outside run scope", "not run", "")
            continue
        yield _coverage_record(
            *scope, record["status"], _heldout_scored(stage, record), _scope_note(stage, record)
        )
    for scope in OUTSIDE_COVERAGE:
        yield _coverage_record(*scope, "outside run scope", "not run", "")


def _can_coverage(record):
    """Expose both CAN dataset scopes while retaining stage-specific scoring status."""
    for dataset in CAN_DATASETS:
        yield _coverage_record(
            dataset,
            "Original K−1 adapter quantization",
            "Grouped and frozen benchmark splits",
            record["status"] if record else "outside run scope",
            _heldout_scored("can_quantization", record) if record else "not run",
            _scope_note("can_quantization", record) if record else "",
        )


def _coverage_lines(rows):
    """Display the coverage checklist using current stage receipts only."""
    return [
        "## Evidence coverage",
        "",
        "| Dataset or area | Evidence | Protocol | This run | Held-out scoring |",
        "| --- | --- | --- | --- | --- |",
        *[
            "| "
            + " | ".join(
                _cell(row[key]) for key in ("dataset", "evidence", "protocol", "execution", "heldout_scored")
            )
            + " |"
            for row in rows
        ],
        "",
        (
            "Held-out scoring denotes benchmark TEST or the declared external capture, as applicable. "
            "Temporal TEST is same-source and previously explored. "
            "See the [coverage ledger](evidence_coverage.csv) for stage notes."
        ),
        "",
    ]


def _summary(run_root, stages, tables, evidence):
    """Render current-run statuses and an index of verified tables."""
    lines = [
        "# FedCAID fresh-run evidence",
        "",
        f"Run `{run_root.name}`",
        "",
        *evidence["refresh_note"],
        *_status_lines(stages),
        "",
        SCOPE.strip(),
        "",
        *_coverage_lines(evidence["coverage"]),
        *_evidence_lines(evidence, stages),
        "## Verified tables and role audits",
        "",
    ]
    for stage, paths in tables.items():
        if paths:
            lines.extend([f"### {STAGE_LABELS[stage]}", ""])
            lines.extend(f"- [{Path(path).name}]({path})" for path in paths)
            lines.append("")
    lines.extend([
        "[Execution details](stage_status.json) · [Verified stage artifact hashes](evidence_plan.json)", ""
    ])
    return "\n".join(lines)


def _evidence_lines(evidence, stages):
    """Link machine-readable summaries and state the scope of their intervals."""
    links = [
        ("decisions", "Operating decisions", "operating_decisions.csv"),
        ("paired", "Paired contrasts", "paired_evidence.csv"),
        ("branches", "Matched Hulk branches by seed", "matched_branch_metrics.csv"),
        ("audits", "Data role audits", "data_role_audit.csv"),
        ("run_plan_sha256", "Run plan and code hashes", evidence["run_plan_link"]),
        ("matrix", "Four-dataset operating points and class scope", "benchmark_operating_points.csv"),
        ("ablation", "Paired ablation effects in percentage points", ABLATION_TABLE),
        ("class_support", "All defined classes with support and zero-recall counts", CLASS_SUPPORT_TABLE),
        ("timing", "Recorded stage elapsed time and its scope", STAGE_TIMING_TABLE),
        ("continuation", "Verified continuation plan", "source_continuation_plan.json"),
    ]
    selected = [f"[{label}]({path})" for key, label, path in links if evidence[key]]
    selected.extend(evidence["matrix_links"])
    selected.extend(evidence["nb15_links"])
    selected.extend(evidence["iov_links"])
    selected.extend(evidence["auxiliary_links"])
    if not selected:
        return []
    return [
        "## Key evidence",
        "",
        *_matrix_lines(evidence["matrix"]),
        *_ablation_lines(evidence["ablation"]),
        *_class_support_lines(evidence["class_support"]),
        *_iov_lines(evidence["iov"]),
        *_iov_fixed_lines(evidence["iov_fixed_status"], evidence["iov_classes"]),
        *_nb15_binary_lines(stages.get("nb15_pair", {}).get("binary_assessment", {})),
        *_nb15_donor_lines(evidence["nb15_donor"]),
        *_paired_lines(evidence["paired"]),
        *_branch_lines(evidence["branches"]),
        *_external_lines(stages.get("external", {})),
        *_auxiliary_lines(stages),
        *_loopback_lines(evidence["loopback_rows"]),
        *_timing_lines(evidence["timing"]),
        *[f"- {link}" for link in selected],
        "",
    ]


def _auxiliary_lines(stages):
    """Show what newly run auxiliary experiments cover and their inference limits."""
    purposes = {
        "scale_matrix": (
            "Grouped VAL records separate workload and Dirichlet settings. Time and memory describe "
            "single-host benchmark cases."
        ),
        "can_quantization": (
            "The original K−1 adapter compares quantization settings. Keep encoding mechanics "
            "and detection outcomes by evaluation role separate."
        ),
        "nb15_temporal": (
            "The exact-input-purged later-day TEST is a previously explored part of the same "
            "benchmark source."
        ),
        "kminus1_transfer": (
            "Original K−1 score and rank effects complement class predictions under controlled absence. "
            "Retrospective TEST remains scoped to the same benchmark source."
        ),
    }
    selected = []
    for stage, purpose in purposes.items():
        record = stages.get(stage)
        if record and record["status"] == "completed":
            cases = f" {record['cases']} completed cases." if type(record.get("cases")) is int else ""
            selected.append(f"| {STAGE_LABELS[stage]} | {_cell(purpose + cases)} |")
    if not selected:
        return []
    return ["### Auxiliary benchmark scope", "", "| Experiment | Scope |", "| --- | --- |", *selected, ""]


def _loopback_lines(rows):
    """Show measured TCP application bytes separately from logical numeric payload."""
    if not rows:
        return []
    confirmed = all(
        row.get("all_plaintext_quantized_sums_equal", "").lower() == "true"
        and row.get("all_clients_accepted", "").lower() == "true"
        for row in rows
    )
    parity = (
        ["Every measured round matched the plaintext quantized sum and all clients accepted.", ""]
        if confirmed
        else []
    )
    return [
        "### Same-host TCP aggregation",
        "",
        (
            "Synthetic updates on one physical host. Round timing covers aggregation transport. "
            "Detector training and inference are outside the measurement."
        ),
        "",
        *parity,
        (
            "| Dimensions | Clients | Rounds | Median ms | P95 ms | Logical bytes/round | TCP application"
            " bytes/round |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_loopback_line(row) for row in rows],
        "",
    ]


def _loopback_line(row):
    """Format one declared loopback condition from the verified summary."""
    cells = (
        row["dimension"],
        row["clients"],
        row["measured_rounds"],
        f"{float(row['round_median_ms']):.3f}",
        f"{float(row['round_p95_ms']):.3f}",
        f"{int(row['numeric_share_payload_bytes_per_round']):,}",
        f"{float(row['tcp_application_mean_bytes_per_round']):,.1f}",
    )
    return "| " + " | ".join(map(_cell, cells)) + " |"


def _matrix_lines(rows):
    """Show the operating and class-coverage limits of the compact baseline."""
    if not rows:
        return []
    fedavg = {row["dataset"]: row for row in rows if row["mode"] == "fedavg_threshold"}
    return [
        "### Four-dataset K−1 adapter operating points",
        "",
        (
            "Grouped VAL with Dirichlet label skew, three clients and α=0.8. "
            "Macro-F1 averages observed true classes. Payload counts logical bytes per round."
        ),
        "",
        (
            "| Dataset | Observed / defined | Zero VAL support | Supported classes with zero recall in any"
            " seed | Full macro-F1 | Full FAR max | Seeds above FAR cap | Bytes full / FedAvg |"
        ),
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        *[_matrix_line(row, fedavg[row["dataset"]]) for row in rows if row["mode"] == "full"],
        "",
    ]


def _matrix_line(row, fedavg):
    """Format absolute performance without masking classes with no observations."""
    bytes_key = "round_payload_bytes_mean"
    cells = (
        row["dataset"],
        f"{row['observed_classes']}/{row['defined_classes']}",
        row["zero_support_classes"].replace(" | ", ", ") or "None",
        row["zero_recall_classes_any_seed"].replace(" | ", ", ") or "None",
        f"{float(row['macro_f1_mean']):.4f}",
        f"{row['far_max']:.3%}" if row["far_max"] is not None else "Unavailable",
        f"{row['far_seeds_over_cap']}/{row['far_observed_seeds']}",
        f"{row[bytes_key]:,.0f} / {fedavg[bytes_key]:,.0f}",
    )
    return "| " + " | ".join(map(_cell, cells)) + " |"


def _ablation_lines(rows):
    """Show paired effects using percentage points for all three metrics."""
    if not rows:
        return []
    return [
        "### Matched K−1 adapter ablations",
        "",
        "Grouped development VAL. Each cell shows the mean paired change [95% interval] in percentage points.",
        "Positive FAR and FNR changes mean more errors. The CSV records paired seed counts for each metric.",
        "",
        "| Dataset | Comparison | Δ macro-F1 (pp) | Δ FAR (pp) | Δ FNR (pp) |",
        "| --- | --- | ---: | ---: | ---: |",
        *[_ablation_line(row) for row in rows],
        "",
    ]


def _ablation_line(row):
    """Format one ablation under a single grouped design."""
    changes = (_ablation_interval(row, metric) for metric in ABLATION_METRICS)
    cells = (row["dataset"], row["comparison"], *changes)
    return "| " + " | ".join(map(_cell, cells)) + " |"


def _ablation_interval(row, metric):
    """Show missing estimates and intervals separately from a real zero."""
    average, low, high = (row[f"{metric}_{field}_pp"] for field in ("mean", "ci95_low", "ci95_high"))
    if average is None:
        return "Unavailable"
    bounds = f"[{low:+.4f}, {high:+.4f}]" if low is not None and high is not None else "[unavailable]"
    return f"{average:+.4f} {bounds}"


def _class_support_lines(rows):
    """Show sparse classes and keep the full set in the companion CSV."""
    if not rows:
        return []
    shown = [row for row in rows if row["support_max"] <= RARE_CLASS_MAX_SUPPORT or row["zero_recall_seeds"]]
    return [
        "### Sparse classes and zero recall in the K−1 benchmark",
        "",
        f"Full adapter on grouped VAL. Shown when support per seed never exceeds {RARE_CLASS_MAX_SUPPORT} "
        "or a supported seed has zero recall. The CSV includes every defined class.",
        "P / R / F1 are means across seeds with positive support and defined values. Support is a per-seed range.",
        "",
        "| Dataset | Class | VAL support | Supported seeds | Zero-recall seeds | P / R / F1 (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *[_class_support_line(row) for row in shown],
        "",
    ]


def _class_support_line(row):
    """Display unobserved classes alongside observed classes with zero recall."""
    low, high = row["support_min"], row["support_max"]
    support = f"{low:,}" if low == high else f"{low:,}–{high:,}"
    metrics = (row[f"{metric}_mean"] for metric in ("precision", "recall", "f1"))
    displayed = " / ".join("—" if value is None else f"{value * 100:.2f}" for value in metrics)
    cells = (row["dataset"], row["class"], support, row["supported_seeds"], row["zero_recall_seeds"], displayed)
    return "| " + " | ".join(map(_cell, cells)) + " |"


def _timing_lines(rows):
    """Distinguish per-attempt durations from whole-experiment timing."""
    if not rows:
        return []
    note = []
    if any(row["elapsed_scope"] == "Continuation attempt" for row in rows):
        note = ["Continuation times cover the recorded attempt. Pre-interruption work is outside those timings."]
    return [
        "### Recorded stage elapsed time",
        "",
        "Stage times are observational wall-clock measurements. Run receipts do not record overlapping workloads "
        "or establish exclusive use of the host or GPU. Use controlled timing runs for runtime comparisons.",
        *note,
        "",
        "| Stage | Elapsed (s) | Timing scope | Saved cases before continuation |",
        "| --- | ---: | --- | ---: |",
        *[_timing_line(row) for row in rows],
        "",
    ]


def _timing_line(row):
    """Format the recorded scope without imputing a missing duration."""
    count = row["saved_case_bindings"]
    cells = (
        STAGE_LABELS[row["stage"]],
        f"{row['elapsed_seconds']:,.2f}",
        row["elapsed_scope"],
        "—" if count is None else count,
    )
    return "| " + " | ".join(map(_cell, cells)) + " |"


def _iov_lines(rows):
    """Show the current run's high-level IoV alarm and support diagnosis."""
    if not rows:
        return []
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["role"]), []).append(row)
    return [
        "### CIC-IoV-2024 working-point diagnosis",
        "",
        (
            "FAR and FNR maxima use seed-pooled row counts. The support column "
            "gives the range of benign feature groups per client and seed."
        ),
        "",
        "| Mode | Role | Benign groups | FAR max | Seeds above FAR cap | FNR max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *[_iov_line(mode, role, block) for (mode, role), block in groups.items()],
        "",
    ]


def _iov_line(mode, role, rows):
    """Aggregate row FAR and retain feature-group support per client."""
    seeds = {}
    for row in rows:
        counts = seeds.setdefault(row["seed"], [0, 0, 0, 0])
        for index, field in enumerate(
            ("false_alarm_rows", "benign_rows", "missed_attack_rows", "attack_rows")
        ):
            counts[index] += int(row[field])
    far = [alerts / benign for alerts, benign, _, _ in seeds.values() if benign]
    fnr = [missed / attacks for _, _, missed, attacks in seeds.values() if attacks]
    cap = float(rows[0]["target_far"])
    support = [int(row["benign_groups"]) for row in rows]
    far_max = f"{max(far):.3%}" if far else "Unavailable"
    fnr_max = f"{max(fnr):.3%}" if fnr else "Unavailable"
    return (
        f"| {mode} | {role} | {min(support):,}–{max(support):,} | {far_max} | "
        f"{sum(value > cap for value in far)}/{len(far)} | {fnr_max} |"
    )


def _iov_fixed_lines(status, classes):
    """Report fixed-base subtype evidence and the inherited FAR decision."""
    if not status:
        return []
    cap, maximum = status["val_far_cap"], status["val_max_far"]
    observed_far = f"maximum {maximum:.3%}" if maximum is not None else "unavailable FAR"
    return [
        "### IoV subtype assignment with the fixed base alarm mask",
        "",
        (
            "Both fixed-base branches retain the base threshold, ordered attack scores and binary alarm mask "
            "in CAL and VAL. The development comparison changes the subtype assignment on alerted rows."
        ),
        "",
        (
            f"Fixed-base VAL {observed_far} with"
            f" {status['val_seeds_over_far_cap']}/{status['val_seed_count']} seeds above the {cap:.2%} cap."
            " Deployment promotion is false."
        ),
        "",
        *_iov_class_lines(classes),
        *_iov_effect_lines(status["paired_effects"]),
    ]


def _iov_class_lines(rows):
    """Summarize observed class coverage without treating missing support as zero recall."""
    if not rows:
        return []
    labels = {row["class"] for row in rows}
    observed = {row["class"] for row in rows if int(row["support"]) > 0}
    missing = ", ".join(sorted(labels - observed)) or "none"
    return [
        (
            f"VAL class support covered {len(observed)}/{len(labels)} classes. Unobserved classes were"
            f" {missing}."
        ),
        "",
    ]


def _iov_effect_lines(effects):
    """Show the main subtype and binary trade-offs on their own evaluation roles."""
    labels = {
        "fixed_full_vs_base_val_macro_f1": "VAL macro-F1, fixed federated minus base",
        "fixed_full_vs_fixed_local_val_macro_f1": "VAL macro-F1, fixed federated minus fixed local",
        "fixed_full_vs_adapted_cal_fnr": "CAL FNR, fixed federated minus original federated",
        "fixed_full_vs_adapted_val_far": "VAL FAR, fixed federated minus original federated",
        "fixed_full_vs_adapted_val_fnr": "VAL FNR, fixed federated minus original federated",
    }
    return [
        "| Paired comparison | Difference [95% interval] |",
        "| --- | ---: |",
        *[f"| {label} | {_interval(effects[key])} |" for key, label in labels.items() if key in effects],
        "",
    ]


def _nb15_binary_lines(assessment):
    """Show the absolute FAR cap and paired FNR guard side by side."""
    scopes, criteria = assessment.get("scopes"), assessment.get("criteria")
    if not scopes or not criteria:
        return []
    far_cap = criteria["max_far_each_seed"]
    fnr_cap = criteria["max_paired_fnr_increase_ci95_upper"]
    return [
        "### NB15 binary acceptance on grouped VAL",
        "",
        (
            f"FAR cap {far_cap:.2%} in each seed. Maximum paired FNR increase at the 95% upper bound "
            f"{fnr_cap * 100:.2f} percentage points."
        ),
        "",
        (
            "| Population | Candidate max FAR | Seeds above cap, reference → candidate | ΔFNR 95% upper | FAR"
            " / FNR |"
        ),
        "| --- | ---: | ---: | ---: | --- |",
        *[_nb15_binary_line(scope, scopes[scope]) for scope in ("receiver", "overall") if scope in scopes],
        "",
        *_nb15_macro_lines(assessment.get("classification_guard", {})),
    ]


def _nb15_binary_line(scope, row):
    """Format the exact scope decision with a visible validation denominator."""
    delta = row["candidate_minus_reference_fnr"]
    reference, candidate = (row[f"{name}_far_seeds_over_limit"] for name in ("reference", "candidate"))
    seeds = delta["seeds_with_metric"]
    verdict = " / ".join("met" if row[f"{metric}_pass"] else "missed" for metric in ("far", "fnr"))
    return (
        f"| {scope} | {row['candidate_far_max']:.3%} | {reference}/{seeds} → {candidate}/{seeds} | "
        f"{delta['ci95_high'] * 100:+.3f} pp | {verdict} |"
    )


def _nb15_macro_lines(classification):
    """Keep the detector objective's classification guard visible."""
    paired = classification.get("candidate_minus_reference_macro_f1")
    if not paired:
        return []
    verdict = "met" if classification["pass"] else "missed"
    return [
        f"Overall macro-F1 candidate minus reference {_interval(paired)}. Classification guard {verdict}.",
        "",
    ]


def _nb15_donor_lines(rows):
    """Distinguish donor feasibility, receiver approval and observed VAL effects."""
    if not rows:
        return []
    seeds = len(rows)
    viable = sum(row["donor_status"] == "feasible" for row in rows)
    approved = sum(row["receiver_status"] == "enabled" for row in rows)
    deltas = [int(row["val_target_tp_delta"]) for row in rows if row.get("val_target_tp_delta")]
    lines = [
        "### NB15 donor CAL and receiver check",
        "",
        (
            f"The federated branch had a donor-feasible cutoff in {viable}/{seeds} seeds. "
            f"Receiver CAL approved it in {approved}/{seeds}."
        ),
    ]
    if len(deltas) == seeds:
        lines.append(
            f"On VAL, its net target true-positive change versus base TabM was {sum(deltas):+d} "
            "summed across seed evaluations of shared captures."
        )
    return [*lines, ""]


def _external_lines(record):
    """Show the acquisition and model-seed units for a completed confirmation."""
    confirmation = record.get("confirmation", {})
    captures, seeds = confirmation.get("capture_count"), confirmation.get("seed_count")
    if record.get("status") != "completed" or not captures or not seeds:
        return []
    uncertainty = (
        "A single capture provides no between-capture interval."
        if captures == 1
        else "See the capture-level intervals for uncertainty across acquisitions."
    )
    return [
        "### Hulk independent confirmation",
        "",
        f"Scored {captures} declared capture units with {seeds} frozen model seeds. {uncertainty}",
        "",
        *_external_capture_lines(confirmation.get("capture_evidence", {})),
    ]


def _external_capture_lines(evidence):
    """Summarize capture-level pairing and the development-overlap screen."""
    pairing = evidence.get("target_f1_pairing", {})
    screen = evidence.get("row_screen", {})
    lines = []
    if "captures_with_complete_seed_pairs" in pairing:
        lines.append(
            f"Target F1 had complete seed pairs in {pairing['captures_with_complete_seed_pairs']} captures."
        )
        if pairing.get("mean") is not None:
            if pairing.get("ci95_low") is None or pairing.get("ci95_high") is None:
                lines.append(f"Capture-level paired mean {pairing['mean']:+.4f}.")
            else:
                lines.append(f"Capture-level paired mean and 95% interval {_interval(pairing)}.")
    if "excluded_rows" in screen:
        lines.append(
            f"The development-input screen excluded {screen['excluded_rows']:,} of "
            f"{screen['before_rows']:,} declared rows."
        )
    return [*lines, ""] if lines else []


def _paired_lines(rows):
    """Show the prespecified contrasts most relevant to the review."""
    keys = ("stage", "contrast", "scope", "metric")
    selected = [row for row in rows if tuple(row[key] for key in keys) in HIGHLIGHTED_PAIRS]
    if not selected:
        return []
    lines = [
        "Paired 95% intervals describe variability across seeds sharing captures.",
        "",
        "| Experiment | Paired contrast | Metric | Seeds | Difference [95% interval] |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in selected:
        cells = (
            _paired_label(row),
            row["comparison"],
            f"{row['scope']} {row['metric']}",
            row["seeds_with_metric"],
            _interval(row),
        )
        lines.append("| " + " | ".join(map(_cell, cells)) + " |")
    return [*lines, ""]


def _paired_label(row):
    """Distinguish both NB15 branches even in older current-run receipts."""
    label = STAGE_LABELS[row["stage"]]
    if row["stage"] != "nb15_pair":
        return label
    details = ", ".join(part for part in (row["branch"], row["detector_objective"]) if part)
    return f"{label} ({details})"


def _interval(row):
    """Format the visible interval while preserving exact values in the CSV."""
    stats = (row["mean"], row["ci95_low"], row["ci95_high"])
    if any(value in ("", None) for value in stats):
        return "Unavailable"
    mean, lower, upper = map(float, stats)
    return f"{mean:+.4f} [{lower:+.4f}, {upper:+.4f}]"


def _branch_lines(rows):
    """Make absolute Hulk outcomes visible alongside the paired effects."""
    if not rows:
        return []
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in (*CONDITION_FIELDS, "branch"))
        groups.setdefault(key, []).append(row)
    return [
        "### Hulk branches on matched TabM alarms",
        "",
        (
            "VAL metrics are seed means except FAR, which is the observed seed maximum. "
            "Support gives the seed range. R and O mean receiver and overall. "
            "Payload is mean logical bytes per round and for training. "
            "The CSV records receiver delivery and setup separately where available. "
            "Wire overhead is outside these counts."
        ),
        "",
        (
            "| Branch | Target support | Target P / R / F1 | Macro-F1 | FAR R / O | FNR R / O | Bytes round /"
            " train |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_branch_line(group) for group in groups.values()],
        "",
    ]


def _branch_line(rows):
    """Show target metrics and matched binary operating rates for one branch."""

    def summary(metric, statistic=mean):
        return statistic([float(row[metric]) for row in rows])

    support = f"{summary('target_support', min):,.0f}–{summary('target_support', max):,.0f}"
    target = " / ".join(f"{summary(f'target_{metric}'):.4f}" for metric in ("precision", "recall", "f1"))
    far = " / ".join(f"{summary(f'{scope}_far', max):.2%}" for scope in ("receiver", "overall"))
    fnr = " / ".join(f"{summary(f'{scope}_fnr'):.2%}" for scope in ("receiver", "overall"))
    payload = " / ".join(f"{summary(f'{part}_payload_bytes'):,.0f}" for part in ("round", "training"))
    cells = (rows[0]["branch"], support, target, f"{summary('overall_macro_f1'):.4f}", far, fnr, payload)
    return "| " + " | ".join(cells) + " |"


def _report_evidence(run_root, working, sources, tables):
    """Bind compact summaries to tables already verified in this run."""
    decisions = list(_decisions(working))
    paired = list(_paired(working, decisions))
    audits = list(_audit_rows(working, tables))
    branches = list(_branch_rows(working, decisions))
    plan_path = _artifact_path(run_root, "final_plan.json")
    evidence = {"run_directory": str(run_root), "sources": sources}
    if plan_path.is_file():
        evidence["run_plan_sha256"] = _hash(plan_path)
    _write(working / "evidence_plan.json", evidence)
    _write_table(working / "operating_decisions.csv", decisions)
    _write_table(working / "paired_evidence.csv", paired)
    _write_table(working / "data_role_audit.csv", audits)
    _write_table(working / "matched_branch_metrics.csv", branches)
    return {
        "decisions": decisions,
        "paired": paired,
        "audits": audits,
        "branches": branches,
        "run_plan_sha256": evidence.get("run_plan_sha256"),
        "run_plan_link": "../final_plan.json",
        "refresh_note": [],
    }


def _collect(run_root, stages, working, *, refreshed=False):
    """Collect scoped evidence and write the report metadata."""
    sources, tables = {}, {}
    for stage, record in stages.items():
        verified, tables[stage] = _collect_stage(run_root, stage, record, working)
        sources.update(verified)
    evidence = _report_evidence(run_root, working, sources, tables)
    evidence["coverage"] = list(_coverage(stages))
    evidence["matrix_links"] = _matrix_links(tables.get("benchmark_matrix", []))
    evidence.update(_nb15_evidence(working))
    evidence.update(_matrix_evidence(working))
    evidence.update(_timing_evidence(run_root, stages, working, sources, evidence["run_plan_sha256"]))
    evidence.update(_iov_evidence(working, tables.get("iov_diagnosis", [])))
    evidence.update(_auxiliary_evidence(working, tables))
    if refreshed:
        evidence.update(
            run_plan_link="source_run_plan.json",
            refresh_note=[
                "Reporting-only export with zero cases trained.",
                "",
                "[Source verification](refresh_source.json)",
                "",
            ],
        )
    _write_table(working / "evidence_coverage.csv", evidence["coverage"])
    metadata = _status_metadata(stages, evidence["decisions"], working)
    _write(working / "stage_status.json", metadata)
    (working / "SUMMARY.md").write_text(_summary(run_root, metadata, tables, evidence), encoding="utf-8")
    return tables


def _auxiliary_evidence(working, tables):
    """Collect summaries only from the staged, hash-verified native artifacts."""
    links, rows = [], []
    for stage, labels in AUXILIARY_TABLES.items():
        for path in tables.get(stage, []):
            name = Path(path).name
            if name in labels:
                protocol = Path(path).parent.name if stage == "can_quantization" else ""
                suffix = f" ({protocol})" if protocol and protocol != stage else ""
                links.append(f"[{labels[name]}{suffix}]({path})")
            if stage == "loopback" and name == "loopback_summary.csv":
                rows.extend(_rows(working, path))
    return {"auxiliary_links": links, "loopback_rows": rows}


def _matrix_evidence(working):
    """Derive scope-aware comparisons from the already checked native tables."""
    source = working / "tables" / "benchmark_matrix"
    rows = benchmark_paper_rows(source)
    ablations = _ablation_rows(source)
    classes = _class_support_rows(source)
    _write_table(working / "benchmark_operating_points.csv", rows)
    _write_table(working / ABLATION_TABLE, ablations)
    _write_table(working / CLASS_SUPPORT_TABLE, classes)
    return {"matrix": rows, "ablation": ablations, "class_support": classes}


def _ablation_rows(source):
    """Export all requested paired effects with their design and estimate units."""
    rows = defaultdict(dict)
    design_fields = ("dataset", "protocol", "scenario", "clients", "alpha", "role", "contrast")
    for row in _rows(source, "paired_differences.csv"):
        if not _grouped_matrix_row(row):
            continue
        if row.get("contrast") in ABLATION_CONTRASTS and row.get("metric") in ABLATION_METRICS:
            design = tuple(row[field] for field in design_fields)
            if row["metric"] in rows[design]:
                raise ValueError(f"Duplicate benchmark ablation: {design} {row['metric']}")
            rows[design][row["metric"]] = row
    return [
        _ablation_record(dict(zip(design_fields, design)), metrics)
        for design, metrics in sorted(rows.items(), key=_ablation_order)
    ]


def _ablation_order(item):
    """Keep all datasets and ablation types in presentation order."""
    design, _ = item
    return (MATRIX_DATASETS.index(design[0]), tuple(ABLATION_CONTRASTS).index(design[-1]), design)


def _grouped_matrix_row(row):
    """Read the matched development population used in the benchmark."""
    return (row.get("protocol"), row.get("scenario"), row.get("role")) == ("grouped", "main", "val")


def _ablation_record(design, metrics):
    """Convert ratio differences to percentage points for each tracked metric."""
    record = {**design, "comparison": ABLATION_CONTRASTS[design["contrast"]]}
    for metric in ABLATION_METRICS:
        values = metrics.get(metric, {})
        record[f"{metric}_paired_seeds"] = values.get("n") or 0
        for field in ("mean", "ci95_low", "ci95_high"):
            value = values.get(field)
            record[f"{metric}_{field}_pp"] = float(value) * 100 if value not in ("", None) else None
    return record


def _class_support_rows(source):
    """Retain all defined classes, including classes unobserved in every seed."""
    groups = defaultdict(list)
    for row in _rows(source, "per_class.csv"):
        if _grouped_matrix_row(row) and row.get("mode") == "full":
            groups[row["dataset"], row["class"]].append(row)
    return [_class_support_record(key, rows) for key, rows in sorted(groups.items())]


def _class_support_record(key, rows):
    """Summarize seed-level support and means without filling absent labels with zero."""
    dataset, label = key
    support = [int(row["support"]) for row in rows]
    supported = [row for row in rows if int(row["support"]) > 0]
    recall = [float(row["recall"]) for row in supported if row.get("recall") not in ("", None)]
    result = {
        "dataset": dataset,
        "class": label,
        "total_seeds": len({row["seed"] for row in rows}),
        "supported_seeds": len(supported),
        "support_min": min(support),
        "support_max": max(support),
        "zero_recall_seeds": sum(value == 0 for value in recall),
    }
    for metric in ("precision", "recall", "f1"):
        values = [float(row[metric]) for row in supported if row.get(metric) not in ("", None)]
        result.update({f"{metric}_seeds": len(values), f"{metric}_mean": mean(values) if values else None})
    return result


def _timing_evidence(run_root, stages, working, sources, plan_hash):
    """Use the verified continuation evidence and stage receipts for timing scope."""
    saved = _read_timing_continuation(run_root, working, stages, sources, plan_hash)
    rows = [
        _timing_record(stage, record["elapsed_seconds"], saved)
        for stage, record in stages.items()
        if record["status"] == "completed" and type(record.get("elapsed_seconds")) in (int, float)
    ]
    _write_table(working / STAGE_TIMING_TABLE, rows)
    return {"timing": rows, "continuation": saved}


def _read_timing_continuation(run_root, working, stages, sources, plan_hash):
    """Verify the original plan and same-run artifact hashes before labeling continuation time."""
    path = _artifact_path(run_root, "continuation_plan.json")
    if not path.is_file():
        return {}
    digest = _hash(path)
    receipt = _read(path)
    preserved = receipt["preserved_stages"]
    if receipt["original_plan_sha256"] != plan_hash or not set(preserved) <= set(stages):
        raise ValueError("Timing continuation identifies a different source plan or stage set")
    counts = Counter()
    for name, expected in receipt["preserved_case_sha256"].items():
        source = _artifact_path(run_root, name)
        stage = source.relative_to(run_root).parts[0]
        if source.name != "result.json" or sources.get(str(source)) != expected or stage in preserved:
            raise ValueError(f"Timing continuation case is absent from verified stage evidence: {name}")
        counts[stage] += 1
    _copy_table(path, Path("source_continuation_plan.json"), digest, working)
    return {"preserved_stages": set(preserved), "saved_case_counts": counts}


def _timing_record(stage, seconds, saved):
    """Mark whether a valid continuation preceded the recorded attempt."""
    if not saved:
        scope, saved_cases = "Recorded stage execution", None
    elif stage in saved["preserved_stages"]:
        scope, saved_cases = "Recorded before continuation", None
    else:
        scope, saved_cases = "Continuation attempt", saved["saved_case_counts"][stage]
    return {
        "stage": stage,
        "elapsed_seconds": seconds,
        "elapsed_scope": scope,
        "saved_case_bindings": saved_cases,
        "concurrent_workload": "unrecorded",
    }


def _iov_evidence(working, paths):
    """Link the same-run score diagnosis and alarm parity tables."""
    labels = {
        "iov_operating_diagnostics.csv": "IoV CAL and VAL working-point diagnostics by client",
        "iov_alarm_overlap.csv": "IoV matched alarm overlap by seed and client",
        IOV_CLASS_FILE: "IoV CAL and VAL per-class support, precision, recall and F1",
        IOV_PAIRED_FILE: "IoV paired subtype and operating effects",
        IOV_STATUS_FILE: "IoV fixed-base scientific criteria",
    }
    links, rows, classes = [], [], []
    status = {}
    for path in paths:
        name = Path(path).name
        if name in labels:
            links.append(f"[{labels[name]}]({path})")
        if name == "iov_operating_diagnostics.csv":
            rows.extend(_rows(working, path))
        elif name == IOV_CLASS_FILE:
            classes.extend(
                row
                for row in _rows(working, path)
                if row["role"] == "val" and row["scope"] == "overall" and row["mode"] == "fixed_base_full"
            )
        elif name == IOV_STATUS_FILE:
            status = _read(working / path)
    return {"iov_links": links, "iov": rows, "iov_classes": classes, "iov_fixed_status": status}


def _nb15_evidence(working):
    """Generate links and report text from copied, verified NB15 evidence."""
    counts = write_nb15_review_tables(working, working / "tables")
    labels = {
        BINARY_TABLE: "NB15 absolute CAL and VAL working points by seed",
        CLASS_TABLE: "NB15 receiver per-class precision, recall, F1 and support",
        FRONTIER_TABLE: "NB15 donor feasibility, receiver check and VAL recovery",
    }
    donor = [row for row in _rows(working, FRONTIER_TABLE) if row["branch"] == "donor_checked_head_federated"]
    return {
        "nb15_links": [f"[{labels[name]}]({name})" for name in labels if counts.get(name)],
        "nb15_donor": donor,
    }


def _matrix_links(paths):
    """Label the original adapter matrix separately from the shared head tables."""
    labels = {
        "paper_baseline.csv": "Four-dataset K−1 benchmark metrics",
        "paper_paired.csv": "Four-dataset K−1 paired effects",
    }
    return [f"[{labels[Path(path).name]}]({path})" for path in paths if Path(path).name in labels]


def _status_metadata(stages, decisions, working):
    """Attach assessments from their verified, published source artifacts."""
    metadata = {
        stage: {key: value for key, value in record.items() if key not in {"artifacts", "binary_assessment"}}
        for stage, record in stages.items()
    }
    for row in decisions:
        metadata[row["stage"]].setdefault("native_development_decisions", []).append(row)
    binary_status = working / "tables/nb15_pair/binary_pair_status.json"
    if binary_status.is_file():
        metadata["nb15_pair"]["binary_assessment"] = _read(binary_status)
    return metadata


def _publish(working, output, tables, *, refresh=None):
    """Publish the complete report with hashes that verify every exported file."""
    bindings = {
        path.relative_to(working).as_posix(): _hash(path)
        for path in sorted(working.rglob("*"))
        if path.is_file()
    }
    result = {
        "status": "completed",
        "output": str(output),
        "report_path": str(output),
        "summary_path": str(output / "SUMMARY.md"),
        "table_counts": {key: len(value) for key, value in tables.items()},
        "execution_origin": "current_run_evidence_only",
        "reviewer_acceptance": "not_assessed",
        "artifacts": bindings,
    }
    if refresh is not None:
        result.update(
            execution_origin="verified_source_report_refresh", trained_cases=0, refresh_source=refresh
        )
    path = working / "complete.json"
    _write(path, result)
    digest = _hash(path)
    working.rename(output)
    return {**result, "artifacts": {**bindings, "complete.json": digest}}


def _check_stages(stages):
    """Accept the formal matrix, prior full runs and standalone IoV diagnostics."""
    if not isinstance(stages, Mapping):
        raise ValueError("The fresh report requires records for all planned experiment stages")
    names = set(stages)
    required = set(STAGE_LABELS) - OPTIONAL_PEERS
    if names != {"iov_diagnosis"} and not required <= names <= set(STAGE_LABELS):
        raise ValueError("The fresh report requires records for all planned experiment stages")


def run_fresh_report(output, stages):
    """Export the current experiment run to its dedicated report directory."""
    output = Path(output)
    if output.name != "report" or output.resolve() != output or output.exists():
        raise ValueError("Expected a new, non-redirected report directory inside the current run")
    _check_stages(stages)
    run_root = output.parent.resolve(strict=True)
    with TemporaryDirectory(prefix=".fresh-report-", dir=run_root) as temporary:
        working = Path(temporary) / "report"
        working.mkdir()
        tables = _collect(run_root, stages, working)
        return _publish(working, output, tables)


def _refresh_receipts(source):
    """Verify the source run's finalized control receipts and original report."""
    complete_path = _artifact_path(source, "final_complete.json")
    completion = _read(complete_path)
    if completion.get("status") not in {"completed", "completed_with_pending", "completed_with_failures"}:
        raise ValueError("Report refresh requires a finalized run")
    bindings = {
        name: completion[digest]
        for name, digest in (("final_plan.json", "plan_sha256"), ("final_state.json", "state_sha256"))
    }
    plan, state = [_read(path) for _, path, _ in _verified_files(source, bindings)]
    steps = {
        name: {key: val for key, val in record.items() if key != "artifacts"}
        for name, record in state.items()
    }
    if completion.get("steps") != steps or set(plan["stages"]) != set(state):
        raise ValueError("Source run stages differ from their finalized plan and receipt")
    if any(Path(receipt["run_directory"]) != source for receipt in (plan, completion)):
        raise ValueError("Source run directory differs from its finalized plan and receipt")
    original_report = state.pop("report")
    if original_report["status"] != "completed" or not original_report.get("artifacts"):
        raise ValueError("The source run lacks its finalized report receipt")
    root = _stage_root(source, "report", original_report)
    list(_verified_files(root, original_report["artifacts"]))
    _check_stages(state)
    return state, bindings


def _refresh_provenance(source, receipt_hashes):
    """Record reporting code and source receipts without starting an experiment."""
    module_names = (
        "revision_fresh_report.py",
        "revision_benchmark_matrix.py",
        "revision_nb15_review_tables.py",
    )
    modules = {name: _hash(Path(__file__).with_name(name)) for name in module_names}
    return {
        "source_run": source.name,
        "source_run_directory": str(source),
        "source_receipt_sha256": receipt_hashes,
        "report_module_sha256": modules,
        "trained_cases": 0,
    }


def regenerate_verified_report(source, output):
    """Write a fresh reporting-only export beside an immutable finalized run."""
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.resolve() != output or output.is_relative_to(source):
        raise ValueError("Report refresh requires a new output directory outside the source experiment")
    source_complete = _artifact_path(source, "final_complete.json")
    digest = _hash(source_complete)
    stages, bindings = _refresh_receipts(source)
    provenance = _refresh_provenance(source, {"final_complete.json": digest, **bindings})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".report-refresh-", dir=output.parent) as temporary:
        working = Path(temporary) / "report"
        _copy_table(
            source / "final_plan.json", Path("source_run_plan.json"), bindings["final_plan.json"], working
        )
        _write(working / "refresh_source.json", provenance)
        tables = _collect(source, stages, working, refreshed=True)
        list(_verified_files(source, provenance["source_receipt_sha256"]))
        return _publish(working, output, tables, refresh=provenance)
