"""Evaluate the exact donor CAL frontier on the five frozen NB15 splits."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .revision_data import _feature_keys
from .revision_donor_check_guard import EXACT_SEARCH, fit_donor_check_guard, predict_donor_check_guard
from .revision_external import _load_receipt, _model_states
from .revision_nb15_source import _bindings, _check_bindings, _inside, _source
from .revision_report import score, summarize, write_csv
from .revision_shared_head import DONOR, FEDERATED, LOCAL, head_probabilities
from .revision_subtype_guard import guarded_subtype_predictions
from .revision_targeted_subtypes import (
    SEEDS,
    _case_context,
    _execution_code,
    _load_context,
    _pair_for_output,
    _source_case_paths,
)
from .revision_v12 import _hash, _output_lock, _read, _write
from .utils import hierarchical_decisions


METHODS = (LOCAL, DONOR, FEDERATED)
OBJECTIVE = "multiclass"
MIN_F1_GAIN = 0.01
ABSOLUTE_FAR_MAX = 0.01
GUARD_SETTINGS = {
    "max_donor_error_rate": 0.05,
    "max_receiver_error_rate": 0.07,
    "max_class_harm_rate": 0.07,
    "min_donor_target_groups": 30,
    "min_donor_negative_groups": 30,
    "min_receiver_groups": 30,
    "min_correct_class_groups": 30,
}
CONTRASTS = {
    "guard_vs_legacy": ("donor_checked_" + FEDERATED, "legacy_" + FEDERATED),
    "aggregation_matched_donor_cal": ("donor_checked_" + FEDERATED, "donor_checked_" + LOCAL),
    "receiver_participation_matched_donor_cal": ("donor_checked_" + FEDERATED, "donor_checked_" + DONOR),
}


def _verified_comparison(root, stage):
    """Bind the existing guards and runtime state to the current-run receipt."""
    receipt_path = _inside(root, "complete.json")
    receipt = _read(receipt_path)
    expected = {**receipt["artifacts"], "complete.json": _hash(receipt_path)}
    if stage is not None and stage["artifacts"] != expected:
        raise ValueError("Current targeted stage receipt differs from its completion record")
    _check_bindings(root, expected)
    if receipt["status"] != "completed" or tuple(receipt["seeds"]) != SEEDS:
        raise ValueError("The donor-CAL replay requires five completed targeted seeds")
    if receipt["plan_sha256"] != expected["targeted_plan.json"]:
        raise ValueError("The targeted plan does not match the completion receipt")
    if receipt["state_sha256"] != expected["targeted_state.json"]:
        raise ValueError("The targeted state does not match the completion receipt")
    state = _read(root / "targeted_state.json")
    for seed in SEEDS:
        key = f"seed_{seed}/{OBJECTIVE}"
        directory = _inside(root, key)
        _check_bindings(directory, state[key])
        expected.update({f"{key}/{name}": digest for name, digest in state[key].items()})
    return expected


def _archived_model(case_path, record, names, features):
    """Require the receiver's exact archived head inventory and schema."""
    export = record["frozen_head_artifacts"]["reserved"]
    head = _inside(case_path.parent, export["file"])
    info, entry = _load_receipt(head)
    if any(entry.get(key) != export.get(key) for key in entry):
        raise ValueError("Frozen head bytes differ from their original case receipt")
    identity = record["identity"]
    if (
        (info["seed"], info["dataset"], info["target_class"], info["receiver"])
        != (identity["seed"], identity["dataset"], identity["target_class"], record["audit"]["receiver"])
        or info["class_order"] != names
        or info["feature_order"] != features
    ):
        raise ValueError("Frozen head identity differs from the matched experiment")
    return _model_states(entry, info), entry


def _cal_positions(route, rows):
    """Locate the selected CAL rows within the verified score array."""
    positions = np.searchsorted(route["cal"], rows)
    if np.any(positions >= len(route["cal"])) or not np.array_equal(route["cal"][positions], rows):
        raise ValueError("CAL rows do not align with their archived scores")
    return positions


def _calibration_rows(context):
    """Use donor CAL-fit and a receiver check which excludes alarm threshold-fit groups."""
    policy, receiver = context["policy"], context["receiver"]
    eligible = {}
    for client, binding in enumerate(context["union"]):
        split = policy["splits"][client][0]
        rows = policy["roster"][client]["cal"][split]
        if client == receiver:
            features = context["parts"]["cal"][0][rows]
            seed = context["identity"]["seed"] + 8123 + receiver
            rows = rows[_receiver_check_mask(features, seed)]
        eligible[client] = (rows, _cal_positions(binding, rows))
    return eligible


def _receiver_check_mask(features, seed):
    """Select one deterministic third of the receiver CAL-fit feature groups."""
    unique, groups = np.unique(_feature_keys(features), return_inverse=True)
    ordering = np.random.default_rng(seed).permutation(len(unique))
    return np.isin(groups.reshape(-1), ordering[: len(unique) // 3])


def _group_inventory(features, inventory):
    """Build one exact-input group ID space for donors and receiver."""
    keys = [_feature_keys(features[rows]) for rows, _ in inventory.values()]
    boundaries = np.cumsum([len(values) for values in keys])[:-1]
    _, groups = np.unique(np.concatenate(keys), return_inverse=True)
    return dict(zip(inventory, np.split(groups.reshape(-1), boundaries), strict=True))


def _cal_owners(groups, receiver):
    """Assign duplicate donor groups to the lowest client ID without inspecting outcomes."""
    keep, owned = {}, np.unique(groups[receiver])
    for client, identities in sorted(groups.items()):
        if client == receiver:
            continue
        keep[client] = ~np.isin(identities, owned)
        owned = np.union1d(owned, identities)
    keep[receiver] = slice(None)
    return keep


def _base_decisions(context):
    """Apply each client's already frozen binary threshold to its own scores."""
    return [
        hierarchical_decisions(values, threshold)
        for values, threshold in zip(
            context["probabilities"]["base_threshold"], context["thresholds"], strict=True
        )
    ]


def _cal_blocks(context, weights, method, base, inventory, groups):
    """Use receiver branch coefficients and each donor's own archived alert mask."""
    features, labels = context["parts"]["cal"]
    ownership = _cal_owners(groups, context["receiver"])
    blocks = {}
    for client, (rows, positions) in inventory.items():
        selected = ownership[client]
        rows, positions = rows[selected], positions[selected]
        if client == context["receiver"]:
            values = context["probabilities"][method][client][positions]
        else:
            values = head_probabilities(weights, features[rows], device="cpu")
        blocks[client] = {
            "base_predictions": base[client]["predicted"][positions],
            "scores": values,
            "labels": labels[rows],
            "group_ids": groups[client][selected],
        }
    receiver = blocks.pop(context["receiver"])
    return list(blocks.values()), receiver


def _val_populations(context, base):
    """Return aligned development VAL populations without exposing TEST."""
    predictions, probability, truth = [], [], []
    labels = context["parts"]["val"][1]
    for decision, route in zip(base, context["union"], strict=True):
        start = len(route["cal"])
        predictions.append(decision["predicted"][start:])
        probability.append(decision["attack_score"][start:])
        truth.append(labels[route["val"]])
    return predictions, probability, truth


def _metric_rows(seed, branch, scope, measures):
    return [
        {"seed": seed, "branch": branch, "scope": scope, "metric": name, "value": value}
        for name, value in measures.items()
    ]


def _measure(context, names, val, prediction, branch):
    """Measure the receiver target and the same all-client development population."""
    original, probability, truth = val
    receiver = context["receiver"]
    if not np.array_equal(prediction > 0, original[receiver] > 0):
        raise ValueError("The donor-CAL replay changed the frozen binary mask")
    all_predictions = original.copy()
    all_predictions[receiver] = prediction
    overall, _, _ = score(
        np.concatenate(truth), np.concatenate(all_predictions), np.concatenate(probability), names
    )
    detector, classes, _ = score(truth[receiver], prediction, probability[receiver], names)
    target = {**classes[context["target"]], "far": detector["far"], "fnr": detector["fnr"]}
    target.pop("class")
    seed = context["identity"]["seed"]
    rows = _metric_rows(seed, branch, "overall", overall) + _metric_rows(seed, branch, "receiver", target)
    spoiled = (prediction != truth[receiver]) & (original[receiver] == truth[receiver])
    for row in classes:
        class_label = row["class"]
        measures = {key: value for key, value in row.items() if key != "class"}
        measures["spoiled_rows"] = int(
            np.count_nonzero(spoiled & (original[receiver] == names.index(class_label)))
        )
        rows.extend(_metric_rows(seed, branch, "receiver_class:" + class_label, measures))
    return rows


def _guard_row(context, method, receipt):
    """Document the exact frozen policy, fit information and fallback."""
    return {
        "seed": context["identity"]["seed"],
        "branch": "donor_checked_" + method,
        "status": receipt["status"],
        "selected_threshold": receipt["selected_threshold"],
        "threshold": receipt.get("threshold"),
        "threshold_search": receipt["settings"]["threshold_search"],
        "donor_proposal_margin_rows": sum(row["proposal_margin_rows"] for row in receipt["donor_summaries"]),
        **receipt["frontier"],
        "communication_scope": receipt["communication"]["scope"],
        "logical_bytes": receipt["communication"]["logical_payload_bytes"],
        "partition_sha256": context["policy"]["partition_sha256"],
        "receipt": json.dumps(receipt, sort_keys=True, allow_nan=False),
    }


def _head_replay(context, names, method, weights, old, base, inventory, groups, val):
    """Fit on CAL, check once, and then evaluate both immutable decisions on VAL."""
    donor_blocks, receiver_block = _cal_blocks(context, weights, method, base, inventory, groups)
    receiver, target = context["receiver"], context["target"]
    receipt = fit_donor_check_guard(donor_blocks, receiver_block, target, thresholds=None, **GUARD_SETTINGS)
    rows = len(context["union"][receiver]["cal"])
    values = context["probabilities"][method][receiver][rows:]
    original = val[0][receiver]
    legacy = guarded_subtype_predictions(original, values, (target,), old["guards"][method]["legacy"])
    candidate = predict_donor_check_guard(original, values, target, receipt)
    metrics = _measure(context, names, val, legacy, "legacy_" + method)
    _legacy_parity(metrics, old, method)
    metrics.extend(_measure(context, names, val, candidate, "donor_checked_" + method))
    return metrics, _guard_row(context, method, receipt)


def _legacy_parity(metrics, old, method):
    """Fail if replaying the incumbent changes any bound reference metric."""
    reference = {}
    for scope, name in (("overall", "metrics"), ("receiver", "target_metrics")):
        rows = [row for row in old["tables"][name] if row["rule"] == "legacy" and row["method"] == method]
        if len(rows) != 1:
            raise ValueError("Incumbent metric not uniquely bound")
        reference[scope] = rows[0]
    aliases = {"far": "receiver_far", "fnr": "receiver_fnr"}
    for row in metrics:
        scope, metric = row["scope"], row["metric"]
        if scope not in reference:
            continue
        key = aliases.get(metric, metric) if scope == "receiver" else metric
        if key in reference[scope] and not np.isclose(
            row["value"], reference[scope][key], atol=1e-12, rtol=0
        ):
            raise ValueError(f"Replay changed the incumbent {method}/{scope}/{metric}")


def _seed_replay(data, cli, source_run, path, root, weights):
    """Recover the original split and score the three matched branches."""
    original = _read(path)
    context = _case_context(data, cli, original, path.parent, source_run)
    seed, receiver = context["identity"]["seed"], context["receiver"]
    old = _read(root / f"seed_{seed}/{OBJECTIVE}/result.json")
    if (
        old["source_case_sha256"] != _hash(path)
        or old["partition_sha256"] != context["policy"]["partition_sha256"]
    ):
        raise ValueError("Current targeted and archived reference partitions differ")
    base, inventory = _base_decisions(context), _calibration_rows(context)
    groups = _group_inventory(context["parts"]["cal"][0], inventory)
    val = _val_populations(context, base)
    metrics = _measure(context, data.names, val, val[0][receiver], "base_threshold")
    guard_rows = []
    for method in METHODS:
        rows, audit = _head_replay(
            context, data.names, method, weights[method], old, base, inventory, groups, val
        )
        metrics.extend(rows)
        guard_rows.append(audit)
    return metrics, guard_rows


def _summaries(metrics):
    """Pair each contrast on all five shared development populations."""
    indexed = {(row["seed"], row["branch"], row["scope"], row["metric"]): row["value"] for row in metrics}
    summary = []
    for contrast, (candidate, reference) in CONTRASTS.items():
        for scope, metric in (
            ("receiver", "f1"),
            ("receiver", "recall"),
            ("receiver", "precision"),
            ("overall", "macro_f1"),
            ("overall", "far"),
            ("overall", "fnr"),
            ("receiver", "far"),
            ("receiver", "fnr"),
        ):
            deltas = [
                indexed[seed, candidate, scope, metric] - indexed[seed, reference, scope, metric]
                for seed in SEEDS
            ]
            stats = summarize(deltas, bounds=(-1, 1))
            summary.append({
                "contrast": contrast,
                "candidate": candidate,
                "reference": reference,
                "scope": scope,
                "metric": metric,
                "seeds": len(SEEDS),
                **stats,
            })
    return summary


def _assessment(metrics, summaries, guards):
    """Keep the guard hypothesis, absolute FAR and inference promotion independent."""
    primary = {
        (row["scope"], row["metric"]): row for row in summaries if row["contrast"] == "guard_vs_legacy"
    }
    gain, macro = primary["receiver", "f1"], primary["overall", "macro_f1"]
    far = [
        row["value"]
        for row in metrics
        if row["metric"] == "far" and row["branch"] == "donor_checked_" + FEDERATED
    ]
    enabled = sum(
        row["status"] == "enabled" and row["branch"] == "donor_checked_" + FEDERATED for row in guards
    )
    checks = {
        "target_f1_mean_gain_at_least_0_01": gain["mean"] >= MIN_F1_GAIN,
        "target_f1_paired_ci95_low_positive": gain["ci95_low"] > 0,
        "overall_macro_f1_mean_nondecreasing": macro["mean"] >= 0,
        "candidate_enabled_in_at_least_one_seed": enabled > 0,
    }
    return {
        "scientific_status": (
            "development_guard_criteria_met" if all(checks.values()) else "development_guard_criteria_not_met"
        ),
        "checks": checks,
        "fed_enabled_seeds": enabled,
        "all_seed_binary_mask_parity": True,
        "absolute_far_met": len(far) == 2 * len(SEEDS) and all(value <= ABSOLUTE_FAR_MAX for value in far),
        "independent_confirmation": False,
        "promotion_allowed": False,
    }


def _prepare_replay(root, project, stage, pair_source=None):
    """Bind source checkpoints, target receipts and fixed replay protocol."""
    target_bindings = _verified_comparison(root, stage)
    source = _pair_for_output(root, project, pair_source)
    source_receipt, _, runs = _source(source)
    if pair_source is not None:
        targeted_plan = _read(root / "targeted_plan.json")
        if (
            targeted_plan.get("score_source_scope") != "current_run_only"
            or targeted_plan["source"] != source_receipt
            or Path(targeted_plan["selected_source"]) != source / "reference"
        ):
            raise ValueError("Donor replay and targeted fit require the same fresh NB15 source")
        if targeted_plan["current_reconstruction_and_fitting_sha256"] != _execution_code():
            raise ValueError("NB15 code changed after targeted fitting. Start a fresh run.")
    cases = _source_case_paths(source / "reference")
    frozen = {str(seed): _read(path)["frozen_head_artifacts"]["reserved"] for seed, path in cases.items()}
    plan = {
        "objective": OBJECTIVE,
        "seeds": list(SEEDS),
        "source": source_receipt,
        "targeted_output": str(root),
        "targeted_artifacts": target_bindings,
        "threshold_search": EXACT_SEARCH,
        "threshold_rule": (
            "Donor CAL only: maximize corrected groups within the donor budget, then minimize wrong groups"
            " and prefer the larger cutoff. Receiver CAL-check audits the selected cutoff once."
        ),
        "threshold_breakpoints": "Unique donor feature-group maxima for recoveries and wrong rewrites",
        "communication_scope": (
            "In-process CAL margin search. Network traffic and protection were not measured."
        ),
        "min_f1_gain": MIN_F1_GAIN,
        "guard_settings": GUARD_SETTINGS,
        "frozen_heads": frozen,
        "calibration_roles": "Donor CAL-fit, receiver one third of original CAL-fit feature groups",
        "receiver_check_scope": "Disjoint from donor selection and the original alarm threshold CAL holdout",
        "receiver_check_seed": "paired seed + 8123 + receiver client ID",
        "donor_overlap_policy": (
            "Exclude receiver-check groups, assign duplicate donor groups to smallest donor client ID"
        ),
        "development_criteria": (
            "Paired receiver F1 gain >= .01 and lower 95% > 0, overall macro-F1 nondecreasing, at least one"
            " enabled seed; absolute FAR separate"
        ),
        "absolute_far_max": ABSOLUTE_FAR_MAX,
        "scope": "CAL-only policy and shared-capture development VAL",
        "code": {
            name: _hash(Path(__file__).with_name(name))
            for name in (
                "revision_donor_check_replay.py",
                "revision_donor_check_guard.py",
                "revision_targeted_subtypes.py",
            )
        },
    }
    return source, runs["reference"], cases, plan


def _execute(root, source, source_run, cases, plan):
    """Verify all weights, then reuse each case without fitting a new head."""
    print("FedCAID | donor CAL frontier | verifying NB15 data and score alignment", flush=True)
    data, cli = _load_context(source / "reference", source_run, "cpu")
    metrics, guards = [], []
    for seed, path in sorted(cases.items()):
        print(f"FedCAID | donor CAL frontier | seed={seed} (no retraining in this stage)", flush=True)
        weights, entry = _archived_model(path, _read(path), data.names, data.features)
        rows, audits = _seed_replay(data, cli, source_run, path, root, weights)
        metrics.extend(rows)
        guards.extend(audits)
        if plan["frozen_heads"][str(seed)] != entry:
            raise ValueError("Frozen replay model changed after the plan was written")
    return metrics, guards


def _commit(output, plan, metrics, guards):
    """Publish the complete artifacts, keeping the receipt as the final commit marker."""
    summary = _summaries(metrics)
    with TemporaryDirectory(prefix=".replay-", dir=output) as staging:
        root = Path(staging)
        _write(root / "replay_plan.json", plan)
        write_csv(root / "summary.csv", summary)
        write_csv(root / "metrics.csv", metrics)
        write_csv(root / "guard_audit.csv", guards)
        names = ("replay_plan.json", "summary.csv", "metrics.csv", "guard_audit.csv")
        result = {
            "schema_version": 1,
            "status": "completed",
            "output": str(output),
            "execution_origin": "current_score_replay",
            "threshold_search": plan["threshold_search"],
            "executed_seeds": list(SEEDS),
            "head_training": False,
            "test_evaluated": False,
            **_assessment(metrics, summary, guards),
            "artifacts": _bindings(root, names),
        }
        _write(root / "complete.json", result)
        for name in (*names, "complete.json"):
            (root / name).replace(output / name)
    return {**result, "artifacts": {**result["artifacts"], "complete.json": _hash(output / "complete.json")}}


def _resume(output, plan):
    """Reuse the previous replay only if its full plan and evidence are identical."""
    marker = _inside(output, "complete.json")
    result = _read(marker)
    _check_bindings(output, result["artifacts"])
    previous = _read(output / "replay_plan.json")
    for entry in previous["frozen_heads"].values():
        for field in ("file", "receipt", "tabm_checkpoint"):
            sha_field = "sha256" if field == "file" else field + "_sha256"
            if _hash(Path(entry[field])) != entry[sha_field]:
                raise ValueError("Replay's frozen source checkpoint changed")
    if previous != plan:
        raise ValueError("Saved replay plan differs; preserve the output and use a separate run")
    return {
        **result,
        "execution_origin": "verified_replay_outputs",
        "artifacts": {**result["artifacts"], "complete.json": _hash(marker)},
    }


def run_donor_check_replay(output, project, targeted=None, *, pair_source=None):
    """Run or recover the prespecified five-seed CAL-check replay."""
    output, project = Path(output).resolve(), Path(project).resolve(strict=True)
    if pair_source is not None and not targeted:
        raise ValueError("Fresh donor replay requires this run's targeted stage receipt")
    root = (Path(targeted["output"]) if targeted else project / "outputs/fedcaid_review/targeted").resolve()
    if not (root / "complete.json").is_file() and not (targeted and targeted.get("status") == "completed"):
        return {"status": "pending", "reason": "Completed NB15 targeted comparison is required"}
    if output.parent != root.parent or output == root:
        raise ValueError("Donor replay needs a separate sibling output")
    source, source_run, cases, plan = _prepare_replay(root, project, targeted, pair_source)
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        if (output / "complete.json").exists():
            return _resume(output, plan)
        planned = output / "replay_plan.json"
        if planned.exists() and _read(planned) != plan:
            raise ValueError("Interrupted replay plan differs; use a separate run")
        _write(planned, plan)
        metrics, guards = _execute(root, source, source_run, cases, plan)
        return _commit(output, plan, metrics, guards)
