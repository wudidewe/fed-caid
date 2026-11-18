"""Run the reproducible FedCAID revision experiment matrix."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import torch

from . import utils as core
from ._iov_layout import SNAPSHOT_SUPPORT_FILES, project_root, source_path
from .experiment_config import DATASET_PARAMETERS, DATASETS
from .privacy_run import PrivateClient
from .revision_adaptation import (
    FIXED_BASE_MODE_DESCRIPTIONS,
    MATCHED_MODES,
    MODE_DESCRIPTIONS,
    THREAT_MODEL,
    AdapterClient,
    run_adapter_ablations,
)
from .revision_baseline import run_fedavg
from .revision_cal_split import cal_group_splits
from .revision_controlled import (
    PredictionCache,
    choose_controlled_receiver,
    development_cells,
    plan_targets,
    remove_receiver_class,
)
from .revision_data import load_revision_data
from .revision_iov_diagnostics import DATASET as IOV_DATASET
from .revision_iov_diagnostics import MODES as IOV_DIAGNOSTIC_MODES
from .revision_iov_diagnostics import IoVOperatingDiagnostics, write_iov_diagnostics
from .revision_measurement import ResidentInference, Resources, benchmark_tcp
from .revision_policies import policy_specs, repair_base
from .revision_report import score, write_report
from .revision_supplement import (
    MIN_DONOR_ROWS,
    MIN_DONOR_TRAIN_ROWS,
    PROFILES,
    REPAIR_FIXED,
    REPAIR_MODES,
    REPAIR_RESCUE,
    REPAIR_UNION,
    AttackSubtypeHead,
    TransferScoreDiagnostics,
    add_subtype_mode,
    missing_class_repair,
    rebalance_scale,
    subtype_diagnostics,
    supplement_cases,
    transfer_rows,
)
from .secure_aggregation import PRIME

ROOT = project_root(__file__)
SEEDS = (42, 43, 44, 45, 46)
SCALING = (
    ("scale_reference", 3, 0.8),
    ("scale_alpha_low", 3, 0.3),
    ("scale_alpha_high", 3, 2.0),
    ("scale_clients_5", 5, 0.8),
    ("scale_clients_10", 10, 0.8),
)
SCALE_DATASETS = ("Car-Hacking", "CIC-UNSW-NB15")
SCALE_TRAIN_ROWS = 100000
BALANCED_MODES = ("base", "base_threshold", "local_only", "full")
RUN_SOURCE = (
    "revision_run.py",
    "revision_data.py",
    "revision_adaptation.py",
    "revision_baseline.py",
    "revision_fixed_subtype.py",
    "revision_iov_diagnostics.py",
    "revision_measurement.py",
    "revision_report.py",
    "review_evidence.py",
    *SNAPSHOT_SUPPORT_FILES,
)
SUPPLEMENT_SOURCE = ("revision_supplement.py", "revision_temporal.py")


def experiment_cases(datasets, protocol, scenarios, *, smoke=False, check_data=False):
    """Build the reference, scale and paired training-objective scenarios."""
    protocols = ("frozen", "grouped") if protocol == "both" else (protocol,)
    cases = {}
    for dataset in datasets:
        for split in protocols:
            selected = []
            if scenarios in ("all", "main") or scenarios == "balance" and split == "grouped":
                selected.append(("main", 3, 0.8))
            if scenarios in ("all", "scale") and split == "grouped" and (dataset in SCALE_DATASETS or smoke):
                selected.extend(SCALING)
            if scenarios in ("all", "balance") and split == "grouped":
                selected.append(("balance", 3, 0.8))
            if selected or check_data:
                cases[dataset, split] = selected
    return cases


def allocate_clients(parts, classes, clients, alpha, seed):
    """Simulate label-skew development clients and route TEST without its labels."""
    rate_rng, *role_rngs = np.random.SeedSequence(seed).spawn(len(parts) + 1)
    rates = np.random.default_rng(rate_rng).dirichlet(np.full(clients, alpha), size=classes)
    result = [{} for _ in range(clients)]
    repairs = {}
    for (role, (_, labels)), role_seed in zip(parts.items(), role_rngs):
        rng = np.random.default_rng(role_seed)
        if role == "test":
            for client, indices in zip(result, np.array_split(rng.permutation(len(labels)), clients)):
                client[role] = indices
            repairs[role] = 0
            continue
        bags = [[] for _ in range(clients)]
        for label in range(classes):
            indices = rng.permutation(np.flatnonzero(labels == label))
            cuts = np.cumsum(rng.multinomial(len(indices), rates[label]))[:-1]
            for bag, indices in zip(bags, np.split(indices, cuts)):
                bag.append(indices)
        rows = [rng.permutation(np.concatenate(bag)) for bag in bags]
        minimum = 2 if role in ("train", "cal") else int(len(labels) >= clients)
        if len(labels) < minimum * clients:
            raise ValueError(f"{role.upper()} has too few rows for {clients} clients")
        moves = 0
        for i, indices in enumerate(rows):
            while len(indices) < minimum:
                donor = max(range(clients), key=lambda index: len(rows[index]))
                if len(rows[donor]) <= minimum:
                    raise RuntimeError(f"Cannot balance the minimum {role.upper()} support")
                indices = np.append(indices, rows[donor][-1])
                rows[donor] = rows[donor][:-1]
                moves += 1
            rows[i] = indices
        if sum(map(len, rows)) != len(labels):
            raise RuntimeError(f"Invalid {role.upper()} client allocation")
        for client, indices in zip(result, rows):
            client[role] = indices
        repairs[role] = moves
    return result, repairs


def _roster_hash(roster):
    """Fingerprint every client's ordered row allocation for matched comparisons."""
    digest = sha256()
    for client in roster:
        for role, indices in sorted(client.items()):
            digest.update(role.encode("ascii"))
            digest.update(len(indices).to_bytes(8, "little"))
            digest.update(np.ascontiguousarray(indices, dtype="<i8"))
    return digest.hexdigest()


def _cap_role(parts, role, limit, seed, *, exact=False):
    x, y = parts[role]
    if len(y) <= limit:
        if exact and len(y) < limit:
            raise ValueError(f"{role.upper()} has {len(y):,} rows but the profile requires {limit:,}")
        return parts
    rng = np.random.default_rng(seed)
    reserved = np.asarray([rng.choice(np.flatnonzero(y == label)) for label in np.unique(y)])
    available = np.ones(len(y), dtype=bool)
    available[reserved] = False
    sampled = rng.choice(np.flatnonzero(available), limit - len(reserved), replace=False)
    chosen = np.sort(np.concatenate((sampled, reserved)))
    return {**parts, role: (x[chosen], y[chosen])}


def _scale_parts(parts, seed):
    return _cap_role(parts, "train", SCALE_TRAIN_ROWS, seed + 905)


def _synthetic(name, protocol):
    from types import SimpleNamespace

    index = DATASETS.index(name)
    parts, names, features = core.synthetic_data(index)
    rng = np.random.default_rng(28800 + index)
    y = np.arange(240, dtype=np.int64) % len(names)
    x = (rng.normal(size=(len(y), len(features))) + y[:, None] * 0.6).astype(np.float32)
    parts["test"] = (x, y)
    return SimpleNamespace(
        parts=parts,
        names=names,
        features=features,
        test_unseen_mask=np.ones(len(y), dtype=bool),
        audit={"synthetic_only": True, "protocol": protocol},
    )


def _gather_eval(parts, client, roles):
    return np.concatenate([parts[role][0][client[role]] for role in roles], axis=0)


def _train_clients(
    parts,
    partitions,
    names,
    features,
    args,
    scratch,
    *,
    with_subtype=False,
    cal_splits=None,
    on_fitted_client=None,
):
    inputs, timings, subtype_guesses = [], [], []
    for i, client in enumerate(partitions):
        client_name = f"client_{i}"
        if not len(client["val"]):
            raise ValueError(f"{client_name} has no VAL rows")
        development = {role: parts[role] for role in ("train", "cal", "val")}
        rows = {role: client[role] for role in development}
        node = PrivateClient(client_name, development, rows, names, features, args, scratch / client_name)
        head = None
        if with_subtype:
            head_started = time.perf_counter()
            head = AttackSubtypeHead(*node.parts["train"], seed=args.seed + i)
            head_seconds = time.perf_counter() - head_started
            subtype_started = time.perf_counter()
            val_x = node.parts["val"][0]
            val_fallback = node.probabilities["val"][:, 1:].argmax(axis=1) + 1
            subtype_chunks = [head.predict(val_x, val_fallback)]
            subtype_seconds = time.perf_counter() - subtype_started
        test_indices = client["test"]
        test_probabilities = np.empty((0, len(names)))
        test_start = time.perf_counter()
        subtype_test_seconds = 0.0
        if len(test_indices):
            model = node.model.to(args.device)
            chunks = []
            for start in range(0, len(test_indices), 32 * args.batch_size):
                raw = parts["test"][0][test_indices[start : start + 32 * args.batch_size]]
                transformed = node.transform(raw)
                chunk_p = core.probabilities(
                    core.log_prediction(model, transformed, args.device, args.batch_size)
                )
                chunks.append(chunk_p)
                if head is not None:
                    head_inference_started = time.perf_counter()
                    subtype_chunks.append(head.predict(transformed, chunk_p[:, 1:].argmax(axis=1) + 1))
                    subtype_test_seconds += time.perf_counter() - head_inference_started
            test_probabilities = np.concatenate(chunks)
            model.cpu()
        stability_started = time.perf_counter()
        local_stability = dict(node.ple_stability)
        sample_positions = np.linspace(0, len(test_indices) - 1, min(len(test_indices), 4096), dtype=np.int64)
        sample = parts["test"][0][test_indices[sample_positions]]
        local_stability["test"] = core.ple_input_diagnostics(node.model, node.transform(sample))
        local_stability["test"]["input_rows"] = len(test_indices)
        local_stability["test"]["sampling"] = "deterministic evenly spaced routed TEST rows, at most 4096"
        stability_seconds = time.perf_counter() - stability_started
        probe = ResidentInference(
            node.model,
            node.transform,
            parts["val"][0][client["val"][:1]],
            args.device,
        )
        inputs.append(
            AdapterClient(
                node.probabilities["cal"],
                node.parts["cal"][1],
                np.concatenate((node.probabilities["val"], test_probabilities)),
                node.log_prior,
                probe,
                cal_split=cal_splits[i] if cal_splits is not None else None,
            )
        )
        timings.append({
            "stage": "local_detector",
            "mode": "shared_local_detectors",
            "client": client_name,
            **node.performance,
            "model_parameters": node.parameter_count,
            "ple_stability_private": local_stability,
            "test_ple_diagnostics_seconds": stability_seconds,
            "test_inference_with_transform_seconds": (
                time.perf_counter() - test_start - subtype_test_seconds - stability_seconds
            ),
            "test_inference_rows": len(test_indices),
        })
        if head is not None:
            timings.append({
                "stage": "local_subtype_head",
                "mode": "base_threshold_subtype",
                "client": client_name,
                "training_seconds": head_seconds,
                "inference_seconds": subtype_seconds + subtype_test_seconds,
                "training_rows": head.training_rows,
                "trained_attack_classes": len(head.classes),
                "fallback": head.model is None,
            })
            subtype_guesses.append(np.concatenate(subtype_chunks))
        if on_fitted_client is not None:
            on_fitted_client(i, node)
        del node
    return inputs, timings, subtype_guesses


def _aligned_modes(
    adapted,
    adapter_scores,
    parts,
    partitions,
    args,
    classes,
    with_fedavg,
    *,
    repair_clients=None,
    prediction_cache=None,
    cal_splits=None,
):
    if not with_fedavg:
        return None, [], []
    train = [(parts["train"][0][c["train"]], parts["train"][1][c["train"]]) for c in partitions]
    evaluation = [_gather_eval(parts, c, ("cal", "val", "test")) for c in partitions]
    result = run_fedavg(train, evaluation, classes, args)
    adapted["fedavg"], adapted["fedavg_threshold"] = [], []
    adapter_scores["fedavg"], adapter_scores["fedavg_threshold"] = [], []
    repair_elapsed = 0.0
    if repair_clients is not None:
        policies = policy_specs(args)
        for _, _, suffix in policies:
            for mode in REPAIR_MODES:
                adapted[mode + suffix], adapter_scores[mode + suffix] = [], []
        train_counts = [np.bincount(y, minlength=classes) for _, y in train]
        total_counts = np.sum(train_counts, axis=0)
    calibration = []
    for i, (client, prob) in enumerate(zip(partitions, result.pop("probabilities"), strict=True)):
        cal_labels = parts["cal"][1][client["cal"]]
        cal, evaluation_p = prob[: len(cal_labels)], prob[len(cal_labels) :]
        if prediction_cache is not None:
            prediction_cache.donor(i, cal, evaluation_p)
        if cal_splits is None:
            _, holdout = core.split_calibration(len(cal_labels), args.cal_fit_fraction, args.seed + 335 + i)
        else:
            _, holdout = cal_splits[i]
        threshold, status = core.empirical_alarm_threshold(
            cal[holdout],
            cal_labels[holdout],
            args.target_far,
            args.threshold_floor,
            getattr(args, "threshold_min_benign_rows", 0),
            getattr(args, "threshold_insufficient_policy", "floor"),
        )
        q = np.clip(evaluation_p[:, 1:].sum(axis=1), 0, 1)
        adapted["fedavg"].append(evaluation_p.argmax(axis=1))
        adapted["fedavg_threshold"].append(core.hierarchical_decisions(evaluation_p, threshold)["predicted"])
        adapter_scores["fedavg"].append(q)
        adapter_scores["fedavg_threshold"].append(q)
        calibration.append({"mode": "fedavg_threshold", "client": f"client_{i}", **status})
        if repair_clients is not None:
            started = time.perf_counter()
            local_counts = train_counts[i]
            eligible = (local_counts == 0) & (total_counts - local_counts >= MIN_DONOR_TRAIN_ROWS)
            eligible[0] = False
            for gate_fraction, residual_fraction, suffix in policies:
                policy_args = argparse.Namespace(**{
                    **vars(args),
                    "repair_gate_target_fraction": gate_fraction,
                    "repair_residual_budget_fraction": residual_fraction,
                })
                repaired, policy_scores, repair_status = missing_class_repair(
                    repair_clients[i],
                    holdout,
                    cal,
                    evaluation_p,
                    adapted["base_threshold"][i],
                    eligible,
                    policy_args,
                )
                for mode, values in repaired.items():
                    adapted[mode + suffix].append(values)
                    adapter_scores[mode + suffix].append(policy_scores[mode])
                calibration.extend(
                    {**record, "mode": record["mode"] + suffix, "client": f"client_{i}"}
                    for record in repair_status
                )
            repair_elapsed += time.perf_counter() - started
    timings = [{"stage": "fedavg", "mode": "fedavg", **result["performance"]}]
    if repair_clients is not None:
        timings.append({
            "stage": "missing_class_routing", "mode": "shared_repair", "elapsed_seconds": repair_elapsed
        })
    return result, timings, calibration


def _evaluate(parts, partitions, predictions, probabilities, names, unseen, *, roles=("val", "test")):
    results = []
    for role in roles:
        actual = parts[role][1]
        for mode, values in predictions.items():
            guessed, attack_score = np.empty(len(actual), dtype=np.int64), np.empty(len(actual), dtype=float)
            for client, pred, q in zip(partitions, values, probabilities[mode], strict=True):
                first = len(client["val"])
                section = slice(None, first) if role == "val" else slice(first, None)
                guessed[client[role]], attack_score[client[role]] = pred[section], q[section]
            views = [(role, slice(None))]
            if role == "test":
                views.append(("test_unique_unseen", unseen))
            for view, selection in views:
                metrics, classes, matrix = score(
                    actual[selection], guessed[selection], attack_score[selection], names
                )
                score_kind = (
                    "selected_branch_binary_probability" if repair_base(mode) else "binary_attack_probability"
                )
                results.append({
                    "role": view,
                    "mode": mode,
                    "metrics": metrics,
                    "per_class": classes,
                    "matrix": matrix,
                    "score_kind": score_kind,
                })
    return results


def _supports(parts, partitions, names):
    rows = []
    for role, (_, y) in parts.items():
        for client_name, labels in [
            ("pooled", y),
            *((f"client_{i}", y[c[role]]) for i, c in enumerate(partitions)),
        ]:
            rows.extend(
                {"role": role, "client": client_name, "class": name, "rows": int(count)}
                for name, count in zip(names, np.bincount(labels, minlength=len(names)), strict=True)
            )
    return rows


def _atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    core.save_json(temporary, value)
    temporary.replace(path)


def _source_hashes(supplement=False):
    files = (
        (*RUN_SOURCE, "revision_policies.py", *SUPPLEMENT_SOURCE)
        if supplement
        else (*RUN_SOURCE, "revision_policies.py", "revision_supplement.py")
    )
    files = (*files, "revision_controlled.py", "revision_conditional.py", "revision_cal_split.py")
    return {relative: core.sha(source_path(relative)) for relative in files}


def _supplement_scale_parts(parts, scenario, clients, seed, *, smoke):
    if smoke:
        return parts
    local = "_local_" in scenario
    train_budget = 33_000 * clients if local else SCALE_TRAIN_ROWS
    cal_budget = 3_300 * clients if local else 9_900
    parts = _cap_role(parts, "train", train_budget, seed + 905, exact=True)
    return _cap_role(parts, "cal", cal_budget, seed + 906, exact=True)


def _run_quantization(inputs, args, predictions, scores, diagnostics):
    fine = argparse.Namespace(**{**vars(args), "secure_scale": 10**9})
    fine_predictions, fine_scores, fine_diagnostics = run_adapter_ablations(
        inputs,
        fine,
        modes=("local_only", "full"),
    )
    predictions.update({f"{mode}_fine": values for mode, values in fine_predictions.items()})
    scores.update({f"{mode}_fine": values for mode, values in fine_scores.items()})
    for key in ("rounds", "private_rounds", "calibration", "postprocess", "resident_inference"):
        diagnostics[key].extend({**row, "mode": f"{row['mode']}_fine"} for row in fine_diagnostics[key])
    diagnostics["collector_startup_seconds"].update({
        f"{mode}_fine": seconds for mode, seconds in fine_diagnostics["collector_startup_seconds"].items()
    })


def _controlled_case(parts, roster, target, seed, fit_fraction, cli, *, planned=None):
    scope = getattr(cli, "controlled_absence_scope", "train_and_cal")
    grouped = getattr(cli, "group_cal", False)
    preferred = cli.controlled_receiver
    policy = "preferred_then_train_cal_support" if grouped and scope == "cal_only" else "fixed"
    receiver = preferred if planned is None else planned["receiver"]
    eligibility = planned.get("receiver_eligibility") if planned is not None else None
    if policy != "fixed" and planned is None:
        receiver, eligibility = choose_controlled_receiver(
            parts,
            roster,
            target,
            seed,
            fit_fraction,
            absence_scope=scope,
            preferred=preferred,
            grouped_cal=True,
        )
    roster, receipt = remove_receiver_class(
        parts, roster, target, receiver, seed, fit_fraction, scope, grouped_cal=grouped
    )
    receipt["receiver_policy"] = policy
    if eligibility is not None:
        receipt["receiver_eligibility"] = eligibility
    return roster, receipt


def _grouped_cal(parts, roster, args, enabled):
    if not enabled:
        return None, None
    cal_x, cal_y = parts["cal"]
    return cal_group_splits(cal_x, cal_y, roster, args.cal_fit_fraction, args.seed)


def _check_sampled_target(parts, roster, splits, target, receiver, args, classes):
    from .revision_conditional import select_fit_indices

    support = []
    for index, (client, (fit, _)) in enumerate(zip(roster, splits, strict=True)):
        fit_labels = parts["cal"][1][client["cal"][fit]]
        chosen = select_fit_indices(fit_labels, args, index, classes)
        support.append(int((fit_labels[chosen] == target).sum()))
    if support[receiver] or sum(support) < MIN_DONOR_ROWS:
        raise ValueError(f"Target {target}: receiver or donor actual CAL-fit support invalid: {support}")
    return support


def _run_case(
    data,
    name,
    protocol,
    scenario,
    seed,
    clients,
    alpha,
    cli,
    directory,
    *,
    controlled_target=None,
    controlled_preflight=None,
):
    profile = {
        **DATASET_PARAMETERS[name],
        "seed": seed,
        "clients": clients,
        "alpha": alpha,
        "device": cli.device,
        "detector_objective": "class_balanced" if scenario == "balance" else "prior_adjusted",
        "ple_boundary_policy": getattr(cli, "ple_boundary_policy", "clip_to_train_edges"),
    }
    if cli.smoke:
        profile.update(base_epochs=1, adapter_epochs=1, rounds=1, batch_size=64)
    conditional = getattr(cli, "conditional_adapter", False)
    development_only = getattr(cli, "development_only", False)
    profile.update(
        threshold_min_benign_rows=getattr(cli, "threshold_min_benign_rows", 0),
        threshold_insufficient_policy=getattr(cli, "threshold_insufficient_policy", "floor"),
    )
    if conditional:
        profile.update(
            conditional_rank=cli.conditional_rank,
            conditional_fit_limit=cli.conditional_fit_limit,
            conditional_sampling=getattr(cli, "conditional_sampling", "class_blind"),
            conditional_min_class_rows=getattr(cli, "conditional_min_class_rows", 0),
            conditional_class_balance_beta=getattr(cli, "conditional_class_balance_beta", 0.0),
            conditional_mask_absent_classes=getattr(cli, "conditional_mask_absent_classes", False),
            conditional_optimizer=getattr(cli, "conditional_optimizer", "sgd"),
            conditional_argmax_modes=getattr(cli, "conditional_argmax_modes", False),
        )
    args = argparse.Namespace(**profile)
    is_controlled = scenario.startswith("controlled_transfer_")
    is_transfer = scenario == "supplement_transfer" or is_controlled
    full_transfer_train = is_controlled or is_transfer and getattr(cli, "transfer_full_train", False)
    repair = is_transfer and getattr(cli, "repair_missing_classes", False)
    if repair:
        profile.update(
            repair_gate_target_fraction=getattr(cli, "repair_gate_target_fraction", 0.5),
            repair_residual_budget_fraction=getattr(cli, "repair_residual_budget_fraction", 1.0),
            repair_sensitivity=getattr(cli, "repair_sensitivity", False),
            repair_union_fallback=getattr(cli, "repair_union_fallback", "local"),
        )
        args = argparse.Namespace(**profile)
    score_diagnostics_enabled = is_transfer and getattr(cli, "transfer_score_diagnostics", False)
    is_temporal = scenario == "supplement_temporal"
    with_subtype = is_temporal and getattr(cli, "subtype_ablation", False)
    is_quantization = scenario == "supplement_quantization"
    is_scale = scenario.startswith("supplement_scale_")
    parts = data.parts
    if (is_transfer and not full_transfer_train) or scenario.startswith("scale_"):
        parts = _scale_parts(parts, seed)
    if is_scale:
        parts = _supplement_scale_parts(parts, scenario, clients, seed, smoke=cli.smoke)
    if development_only:
        parts = {**parts, "test": (parts["test"][0][:0], parts["test"][1][:0])}
    unseen = np.zeros(0, dtype=bool) if development_only else data.test_unseen_mask
    roster, repairs = allocate_clients(parts, len(data.names), clients, alpha, seed)
    controlled_receipt = None
    if is_controlled:
        if controlled_target is None:
            raise ValueError("Controlled case requires a frozen TRAIN/CAL-selected target")
        roster, controlled_receipt = _controlled_case(
            parts,
            roster,
            controlled_target["class_index"],
            seed,
            args.cal_fit_fraction,
            cli,
            planned=controlled_preflight,
        )
    quota_moves = rebalance_scale(roster, seed) if is_scale else {}
    cal_splits, cal_audit = _grouped_cal(parts, roster, args, getattr(cli, "group_cal", False))
    if cal_audit is not None:
        _atomic_json(directory / "cal_group_split.json", cal_audit)
    fixed_base_alarm = getattr(cli, "iov_fixed_base_alarm", False)
    iov_diagnostic = (
        IoVOperatingDiagnostics(
            parts, roster, cal_splits, class_names=data.names, fixed_base_alarm=fixed_base_alarm
        )
        if getattr(cli, "iov_diagnostics", False)
        else None
    )
    if controlled_receipt is not None:
        if cal_splits is not None and conditional:
            controlled_receipt["conditional_selected_target_support"] = _check_sampled_target(
                parts,
                roster,
                cal_splits,
                controlled_target["class_index"],
                controlled_receipt["receiver"],
                args,
                len(data.names),
            )
        if controlled_preflight is not None and cal_audit is not None:
            expected = controlled_preflight["cal_group_split"]["assignment_sha256"]
            if cal_audit["assignment_sha256"] != expected:
                raise RuntimeError("CAL feature group assignment differs from preflight")
        _atomic_json(directory / "controlled_partition.json", controlled_receipt)
    print(f"{name} | {protocol} | {scenario} | seed={seed} | clients={clients} | alpha={alpha}", flush=True)
    if cli.device == "cuda":
        torch.cuda.empty_cache()
    transfer, development = [], []
    score_diagnostics = None
    conditional_diagnostics = None
    controlled_development = []
    cache_receipt = None
    iov_evidence = {}
    prediction_cache = (
        PredictionCache(directory, data.names, parts, roster, development_only=development_only)
        if getattr(cli, "cache_predictions", False)
        else None
    )
    with Resources(cli.device) as resource:
        with TemporaryDirectory(prefix="local_", dir=directory) as scratch:
            adapter_inputs, local_times, subtype_guesses = _train_clients(
                parts,
                roster,
                data.names,
                data.features,
                args,
                Path(scratch),
                with_subtype=with_subtype,
                cal_splits=cal_splits,
            )
            _atomic_json(
                directory / "ple_stability_private.json",
                {
                    row["client"]: row.pop("ple_stability_private")
                    for row in local_times
                    if "ple_stability_private" in row
                },
            )
        if prediction_cache is not None:
            prediction_cache.local(adapter_inputs)
        if scenario == "balance":
            predictions, q, diagnostics = run_adapter_ablations(adapter_inputs, args, modes=BALANCED_MODES)
        elif is_transfer or is_temporal:
            if score_diagnostics_enabled:
                score_diagnostics = TransferScoreDiagnostics(parts, roster, data.names, unseen)
            matched_modes = (*MATCHED_MODES, "base_fixed") if cal_splits is not None else MATCHED_MODES
            predictions, q, diagnostics = run_adapter_ablations(
                adapter_inputs,
                args,
                modes=matched_modes,
                include_development=True,
                observe_scores=score_diagnostics.observe if score_diagnostics is not None else None,
            )
        elif is_quantization or is_scale:
            predictions, q, diagnostics = run_adapter_ablations(
                adapter_inputs,
                args,
                modes=("base_threshold", "local_only", "full"),
            )
        else:
            predictions, q, diagnostics = run_adapter_ablations(
                adapter_inputs,
                args,
                observe_operating=iov_diagnostic.observe if iov_diagnostic else None,
                fixed_base_alarm=fixed_base_alarm,
            )
        if is_quantization:
            _run_quantization(adapter_inputs, args, predictions, q, diagnostics)
        if conditional:
            from .revision_conditional import run_conditional_ablations

            cond_predictions, cond_q, conditional_diagnostics = run_conditional_ablations(
                adapter_inputs, args
            )
            predictions.update(cond_predictions)
            q.update(cond_q)
        if with_subtype:
            local_times.append(add_subtype_mode(predictions, q, diagnostics, subtype_guesses))
            for i, client in enumerate(roster):
                size = len(client["val"])
                original = adapter_inputs[i].eval_probabilities[:size, 1:].argmax(axis=1) + 1
                for row in subtype_diagnostics(
                    parts["val"][1][client["val"]],
                    predictions["base_threshold"][i][:size],
                    original,
                    subtype_guesses[i][:size],
                    data.names,
                ):
                    development.append({"client": f"client_{i}", "role": "val", **row})
        with_baseline = iov_diagnostic is None and (
            scenario in ("main", "scale_reference", "balance") or is_temporal or repair
        )
        repair_clients = adapter_inputs if repair else None
        del adapter_inputs
        baseline, baseline_times, baseline_calibration = _aligned_modes(
            predictions,
            q,
            parts,
            roster,
            args,
            len(data.names),
            with_baseline,
            repair_clients=repair_clients,
            prediction_cache=prediction_cache,
            cal_splits=cal_splits,
        )
        del repair_clients
        if is_transfer:
            transfer = transfer_rows(
                parts,
                roster,
                diagnostics["cal_fit_class_counts"],
                data.names,
                predictions,
                unseen,
            )
        if is_controlled:
            target_index = controlled_target["class_index"]
            receiver = controlled_receipt["receiver"]
            if diagnostics["cal_fit_class_counts"][receiver][target_index] != 0:
                raise RuntimeError("Controlled target unexpectedly entered adapter CAL-fit")
            controlled_development = development_cells(
                parts, roster, predictions, target_index, receiver, data.names
            )
        score_result = (
            {"transfer_scores": score_diagnostics.paired_rows(transfer)} if score_diagnostics else {}
        )
        del score_diagnostics
        evaluations = _evaluate(
            parts,
            roster,
            predictions,
            q,
            data.names,
            unseen,
            roles=("val",) if development_only else ("val", "test"),
        )
        if iov_diagnostic is not None:
            iov_evidence = {"iov_diagnostics": iov_diagnostic.finish(evaluations)}
        if prediction_cache is not None:
            cache_receipt = prediction_cache.save(predictions, q)
        predictions_repair_modes = [mode for mode in predictions if repair_base(mode)]
        del predictions, q
    settings = {
        **profile,
        **core.detector_objective_settings(args.detector_objective),
        "prime": PRIME,
        "probability_floor": 1e-12,
        "detector_tabm_ensemble": 4,
        "detector_tabm_blocks": 2,
        "detector_tabm_width": 64,
        "detector_ple_embedding": 8,
        "detector_ple_bins_max": 16,
        "detector_cross_layers": 2,
        "detector_cross_rank": 16,
        "detector_tabm_dropout": 0.1,
        "detector_gradient_clip": 5,
        "train_rows": len(parts["train"][1]),
        "repair_train": repairs["train"],
        "repair_cal": repairs["cal"],
        "repair_val": repairs["val"],
        "repair_test": repairs["test"],
        "test_client_assignment": "label-blind fixed permutation with equal-sized client blocks",
    }
    if development_only or conditional or is_controlled or prediction_cache is not None:
        settings.update(
            evaluation_stage="development_only" if development_only else "existing_test_exploratory",
            test_evaluated=not development_only,
            conditional_adapter=conditional,
        )
    if is_controlled:
        settings.update(
            controlled_target_class=controlled_target["class"],
            controlled_target_index=controlled_target["class_index"],
            controlled_receiver=controlled_receipt["receiver"],
            controlled_receiver_policy=controlled_receipt["receiver_policy"],
            controlled_absence_scope=getattr(cli, "controlled_absence_scope", "train_and_cal"),
            controlled_protocol=controlled_receipt,
        )
    if cal_audit is not None:
        settings.update(cal_split_protocol="exact_feature_group", cal_split_audit=cal_audit)
    if iov_diagnostic is not None:
        settings.update(iov_diagnostics=True, fedavg_evaluated=False, iov_fixed_base_alarm=fixed_base_alarm)
    if is_transfer:
        train_rows_before = len(data.parts["train"][1])
        settings.update(
            train_sampling=(
                "all_rows" if len(parts["train"][1]) == train_rows_before else "one_per_class_then_uniform"
            ),
            train_row_limit=None if full_transfer_train else SCALE_TRAIN_ROWS,
            train_rows_before_sampling=train_rows_before,
            train_sampling_seed=seed + 905 if len(parts["train"][1]) < train_rows_before else None,
            train_rows_by_client=[len(client["train"]) for client in roster],
            evaluation_partition_sha256=_roster_hash([
                {role: indices for role, indices in client.items() if role != "train"} for client in roster
            ]),
        )
    if is_scale:
        settings.update(
            quota_train_moves=quota_moves["train"],
            quota_cal_moves=quota_moves["cal"],
            cal_rows=len(parts["cal"][1]),
            scale_budget="fixed_local" if "_local_" in scenario else "fixed_total",
        )
    if is_quantization:
        settings["secure_scale_fine"] = 10**9
    if is_temporal:
        settings["subtype_head"] = (
            "local TRAIN attacks, averaged linear classifier, five passes, class weight cap eight"
            if with_subtype
            else None
        )
        settings["temporal_purge_exact"] = not getattr(cli, "temporal_raw", False)
    if repair:
        settings.update(
            repair_missing_classes=True,
            repair_min_donor_train_rows=MIN_DONOR_TRAIN_ROWS,
            repair_source_mode="base_threshold",
            repair_donor_kind="full_weight_fedavg",
        )
    communication = []
    if baseline:
        communication.append({"mode": "fedavg", **baseline["communication"]})
        if repair:
            communication.extend(
                {
                    "mode": mode,
                    **baseline["communication"],
                    "reuses_training_of": "fedavg",
                    "class_support_setup_payload_bytes": 2 * clients * len(data.names) * 8,
                }
                for mode in predictions_repair_modes
            )
        _atomic_json(
            directory / "fedavg.json",
            {key: baseline[key] for key in ("settings", "rounds", "performance", "communication")},
        )
    for mode in sorted({row["mode"] for row in diagnostics["rounds"]}):
        dimension = len(data.names) - 1
        communication.append({
            "mode": mode,
            "parameter_count": dimension,
            "round_uplink_payload_bytes": (2 * clients + 2) * dimension * 8,
            "round_downlink_payload_bytes": clients * dimension * 8,
            "round_payload_bytes": (3 * clients + 2) * dimension * 8,
            "training_payload_bytes": (3 * clients + 2) * dimension * 8 * args.rounds,
            "scope": (
                "logical bytes: both client shares, two collector sums, and coordinator vector to each client"
                " each round"
            ),
        })
    if conditional_diagnostics is not None:
        communication.extend(conditional_diagnostics.get("communication", []))
        baseline_calibration.extend(conditional_diagnostics.get("calibration", []))
        _atomic_json(directory / "conditional_diagnostics_private.json", conditional_diagnostics)
    if is_temporal and baseline:
        adapter_bytes = next(row["training_payload_bytes"] for row in communication if row["mode"] == "full")
        full_round_bytes = baseline["communication"]["round_payload_bytes"]
        settings["fedavg_full_rounds_at_adapter_logical_budget"] = adapter_bytes // full_round_bytes
    network = []
    if not cli.no_tcp:
        try:
            network = benchmark_tcp(
                clients,
                (len(data.names) - 1) * 8,
                baseline["communication"]["model_payload_bytes"] if baseline else None,
                repeats=3 if cli.smoke else 20,
            )
        except OSError as error:
            print(f"TCP microbenchmark unavailable: {error}", file=sys.stderr, flush=True)
            network = [
                {"stage": "loopback_tcp_payload_transfer_ack", "status": "unavailable", "error": str(error)}
            ]
    _atomic_json(directory / "adapter_diagnostics_private.json", diagnostics)
    return {
        "dataset": name,
        "protocol": protocol,
        "scenario": scenario,
        "seed": seed,
        "clients": clients,
        "alpha": alpha,
        "settings": settings,
        "partition_sha256": _roster_hash(roster),
        "resources": {**resource.result, "measurement_scope": "paired_case_all_modes"},
        "evaluations": evaluations,
        "supports": _supports(parts, roster, data.names),
        "rounds": diagnostics["rounds"],
        "calibration": diagnostics["calibration"] + baseline_calibration,
        "communication": communication,
        "transfer": transfer,
        "controlled_development": controlled_development,
        "prediction_cache": cache_receipt,
        "conditional_adapter": (
            {key: value for key, value in conditional_diagnostics.items() if not key.endswith("_private")}
            if conditional_diagnostics
            else None
        ),
        **score_result,
        **iov_evidence,
        "development_subtype": development,
        "development_rank": diagnostics.get("development_class_diagnostics", []),
        "private_round_summary": diagnostics["private_rounds"] if is_quantization else [],
        "measurement": (
            local_times
            + baseline_times
            + [
                {"stage": "conditional_postprocess", **row}
                for row in (conditional_diagnostics or {}).get("postprocess", [])
            ]
            + [
                {"stage": "conditional_federated_round", **row}
                for row in (conditional_diagnostics or {}).get("rounds", [])
            ]
            + [{"stage": "adapter_postprocess", **row} for row in diagnostics["postprocess"]]
            + [{"stage": "resident_single_inference", **row} for row in diagnostics["resident_inference"]]
            + network
        ),
    }


def _parser(*, supplement=False):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    if supplement:
        p.add_argument(
            "--profile",
            choices=PROFILES,
            default="core",
            help="Core runs grouped transfer and the NB15 time holdout",
        )
        p.add_argument(
            "--transfer-score-diagnostics",
            action="store_true",
            help="Export same-input class score contrasts with --profile transfer",
        )
        p.add_argument(
            "--transfer-full-train",
            action="store_true",
            help="Use all grouped TRAIN rows with --profile transfer",
        )
        p.add_argument(
            "--repair-missing-classes",
            action="store_true",
            help="Compare support-aware shared detection branches with --profile transfer",
        )
        p.add_argument(
            "--repair-gate-target-fraction",
            type=float,
            default=0.5,
            help="Donor fraction of FAR target; union local branch gets the complement",
        )
        p.add_argument(
            "--repair-residual-budget-fraction",
            type=float,
            default=1.0,
            help="Fraction of remaining benign CAL alert count available to rescue",
        )
        p.add_argument(
            "--repair-sensitivity",
            action="store_true",
            help="Evaluate all five declared donor/residual policies on the same detector predictions",
        )
        p.add_argument(
            "--temporal-raw",
            action="store_true",
            help="Retain historical temporal roles instead of forward purging exact inputs",
        )
        p.add_argument(
            "--subtype-ablation",
            action="store_true",
            help="Explicitly train and evaluate the additional temporal subtype head",
        )
        p.add_argument(
            "--transfer-alpha", type=float, help="One transfer Dirichlet alpha per output run (default: 0.3)"
        )
        p.add_argument(
            "--transfer-clients", type=int, help="One transfer client count per output run (default: 3)"
        )
        p.add_argument(
            "--controlled-count",
            type=int,
            default=3,
            help="Number of TRAIN/CAL-selected class tasks (synthetic default at most two)",
        )
        p.add_argument(
            "--controlled-classes",
            nargs="+",
            help="Explicit target names for a single selected dataset; never chosen from TEST",
        )
        p.add_argument("--controlled-receiver", type=int, default=0)
        p.add_argument(
            "--controlled-absence-scope",
            choices=("train_and_cal", "cal_only"),
            default="train_and_cal",
            help="cal_only removes the target from receiver CAL while TRAIN retains it (adapter-level test)",
        )
        p.add_argument("--conditional-adapter", action="store_true")
        p.add_argument("--conditional-rank", type=int, default=4)
        p.add_argument(
            "--conditional-fit-limit",
            type=int,
            default=128,
            help="CAL-fit budget shared by conditional local/federated modes; 0 means all",
        )
        p.add_argument(
            "--conditional-sampling",
            choices=("class_blind", "coverage_first"),
            default="class_blind",
            help="CAL-fit row selection; coverage_first reserves rows for every present class",
        )
        p.add_argument(
            "--conditional-min-class-rows",
            type=int,
            default=0,
            help="Reserved CAL-fit rows per present class under coverage_first",
        )
        p.add_argument(
            "--conditional-class-balance-beta",
            type=float,
            default=0.0,
            help="Inverse-frequency exponent for the weighted conditional fit; zero is unweighted",
        )
        p.add_argument(
            "--conditional-mask-absent-classes",
            action="store_true",
            help="Do not fit class coordinates that have no CAL-fit support",
        )
        p.add_argument("--conditional-optimizer", choices=("sgd", "adam"), default="sgd")
        p.add_argument(
            "--conditional-argmax-modes",
            action="store_true",
            help="Also evaluate matched argmax decisions without a global alarm threshold",
        )
        p.add_argument(
            "--threshold-min-benign-rows",
            type=int,
            default=0,
            help="Fall back to the declared floor when a CAL holdout has fewer benign rows",
        )
        p.add_argument(
            "--repair-union-fallback",
            choices=("local", "base"),
            default="local",
            help="Fallback decisions for repair union; base preserves every base alarm",
        )
    else:
        p.add_argument("--protocol", choices=("both", "frozen", "grouped"), default="both")
        p.add_argument(
            "--scenarios",
            choices=("all", "main", "scale", "balance"),
            default="all",
            help="Balance runs grouped reference and class-balanced objectives on matched splits",
        )
        p.add_argument(
            "--iov-diagnostics",
            action="store_true",
            help="Observe existing IoV CAL/VAL gates in a grouped development run",
        )
        p.add_argument(
            "--iov-fixed-base-alarm",
            action="store_true",
            help=(
                "Observe adapted attack subtypes while holding the original IoV TabM binary alarm and scores"
                " fixed"
            ),
        )
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument(
        "--group-cal",
        action="store_true",
        help="Assign identical model inputs federation-wide to one CAL side",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--ple-boundary-policy",
        choices=("clip_to_train_edges", "extrapolate"),
        default="clip_to_train_edges",
        help="Versioned PLE boundary rule shared by local and FedAvg",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run with synthetic data and one training epoch")
    p.add_argument("--threshold-insufficient-policy", choices=("floor", "disable"), default="floor")
    p.add_argument(
        "--check-data", action="store_true", help="Verify the selected data and splits without training"
    )
    p.add_argument("--no-tcp", action="store_true", help="Skip the standalone loopback TCP microbenchmark")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--development-only",
        action="store_true",
        help="Fit TRAIN/CAL and evaluate VAL only; TEST inference is disabled",
    )
    p.add_argument(
        "--cache-predictions",
        action="store_true",
        help="Save private role-indexed probabilities and predictions with SHA256",
    )
    return p


def _validate_iov_diagnostics(cli):
    """Restrict the opt-in observer to the matched normal IoV development case."""
    if getattr(cli, "iov_fixed_base_alarm", False) and not getattr(cli, "iov_diagnostics", False):
        raise ValueError("--iov-fixed-base-alarm requires --iov-diagnostics")
    if getattr(cli, "iov_diagnostics", False) and (
        hasattr(cli, "profile")
        or tuple(cli.dataset) != (IOV_DATASET,)
        or cli.protocol != "grouped"
        or cli.scenarios != "main"
        or not cli.group_cal
        or not cli.development_only
        or cli.cache_predictions
        or cli.check_data
    ):
        raise ValueError(
            "--iov-diagnostics requires only CIC-IoV-2024, grouped main, --group-cal and"
            " --development-only without --cache-predictions or --check-data"
        )


def _objective_comparison(cases, development_only):
    """Advertise the optional balance contrast only when this run planned it."""
    if not any(case[0] == "balance" for selected in cases.values() for case in selected):
        return {}
    population = "grouped VAL" if development_only else "grouped exact-input unique TEST"
    return {
        "objective_comparison": {
            "primary": f"{population} macro-F1 of balance/full minus main/full",
            "training_only": core.detector_objective_settings("class_balanced"),
            "selection": "The paired detector objectives, partitions and seeds are prespecified.",
            "scope": "Retrospective benchmark on previously inspected captures.",
        },
    }


def execute(cli):
    _validate_iov_diagnostics(cli)
    supplement = hasattr(cli, "profile")
    controlled = getattr(cli, "profile", None) == "controlled_transfer"
    transfer_profiles = ("transfer", "controlled_transfer")
    if controlled:
        if cli.controlled_count < 1 or cli.controlled_receiver < 0:
            raise ValueError("Controlled count must be positive and receiver nonnegative")
        if cli.controlled_classes is not None and len(cli.dataset) != 1:
            raise ValueError("Explicit controlled classes require a single dataset")
        if cli.controlled_classes is not None:
            cli.controlled_count = len(cli.controlled_classes)
        elif cli.smoke:
            cli.controlled_count = min(cli.controlled_count, 2)
        cli.transfer_full_train = True
        cli.repair_missing_classes = True
    elif (
        getattr(cli, "controlled_classes", None)
        or getattr(cli, "controlled_count", 3) != 3
        or getattr(cli, "controlled_receiver", 0) != 0
    ):
        raise ValueError("Controlled target options require controlled_transfer")
    if getattr(cli, "controlled_absence_scope", "train_and_cal") != "train_and_cal" and not controlled:
        raise ValueError("Controlled absence scope requires controlled_transfer")
    if getattr(cli, "conditional_adapter", False):
        if getattr(cli, "profile", None) not in (*transfer_profiles, "temporal"):
            raise ValueError("Conditional candidate requires transfer, controlled_transfer or temporal")
        if cli.conditional_rank < 1 or cli.conditional_fit_limit < 0:
            raise ValueError("Conditional rank must be positive and CAL-fit limit nonnegative")
        if cli.conditional_sampling not in ("class_blind", "coverage_first"):
            raise ValueError("Unsupported conditional CAL-fit sampling")
        if cli.conditional_min_class_rows < 0:
            raise ValueError("Conditional minimum per-class rows must be nonnegative")
        if not np.isfinite(cli.conditional_class_balance_beta) or cli.conditional_class_balance_beta < 0:
            raise ValueError("Conditional class-balance beta must be finite and nonnegative")
    if getattr(cli, "threshold_min_benign_rows", 0) < 0:
        raise ValueError("Threshold minimum benign rows must be nonnegative")
    if getattr(cli, "repair_union_fallback", "local") != "local" and not getattr(
        cli, "repair_missing_classes", False
    ):
        raise ValueError("Repair union fallback requires --repair-missing-classes")
    transfer_scores = getattr(cli, "transfer_score_diagnostics", False)
    full_transfer_train = getattr(cli, "transfer_full_train", False)
    repair = getattr(cli, "repair_missing_classes", False)
    if getattr(cli, "repair_sensitivity", False) and not repair:
        raise ValueError("--repair-sensitivity requires --repair-missing-classes")
    if (getattr(cli, "temporal_raw", False) or getattr(cli, "subtype_ablation", False)) and getattr(
        cli, "profile", None
    ) not in ("temporal", "core", "all"):
        raise ValueError("Temporal options require a profile containing temporal experiments")
    gate_fraction = getattr(cli, "repair_gate_target_fraction", 0.5)
    residual_fraction = getattr(cli, "repair_residual_budget_fraction", 1.0)
    if not np.isfinite(gate_fraction) or not 0 < gate_fraction <= 1:
        raise ValueError("Repair donor FAR fraction must be in (0, 1]")
    if not np.isfinite(residual_fraction) or not 0 <= residual_fraction <= 1:
        raise ValueError("Repair residual budget fraction must be in [0, 1]")
    if not repair and (gate_fraction != 0.5 or residual_fraction != 1.0):
        raise ValueError("Repair fractions require --repair-missing-classes")
    transfer_alpha = getattr(cli, "transfer_alpha", None)
    transfer_clients = getattr(cli, "transfer_clients", None)
    if transfer_alpha is not None and (not np.isfinite(transfer_alpha) or transfer_alpha <= 0):
        raise ValueError("Transfer alpha must be finite and positive")
    if transfer_clients is not None and transfer_clients < 3:
        raise ValueError("Secure transfer needs at least three clients")
    if (transfer_alpha is not None or transfer_clients is not None) and getattr(
        cli, "profile", None
    ) not in transfer_profiles:
        raise ValueError("Transfer alpha and clients require --profile transfer")
    if transfer_scores and cli.profile not in transfer_profiles:
        raise ValueError("--transfer-score-diagnostics requires --profile transfer")
    if full_transfer_train and cli.profile not in transfer_profiles:
        raise ValueError("--transfer-full-train requires --profile transfer")
    if repair and cli.profile not in transfer_profiles:
        raise ValueError("--repair-missing-classes requires --profile transfer")
    if cli.smoke and cli.check_data:
        raise ValueError("--check-data verifies real files and cannot be combined with --smoke")
    if cli.resume and (cli.out is None or cli.check_data):
        raise ValueError("--resume needs an existing explicit --out and a training run")
    if len(set(cli.dataset)) != len(cli.dataset):
        raise ValueError("Each dataset may be selected once")
    seeds = cli.seeds if cli.seeds is not None else ([42] if cli.smoke else list(SEEDS))
    if len(set(seeds)) != len(seeds) or any(not 0 <= value < 2**32 - 1000 for value in seeds):
        raise ValueError("Seeds must be distinct and within the supported unsigned range")
    if cli.device == "auto":
        cli.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    cases = (
        supplement_cases(
            cli.dataset,
            cli.profile,
            transfer_alpha=0.3 if transfer_alpha is None else transfer_alpha,
            transfer_clients=3 if transfer_clients is None else transfer_clients,
            controlled_count=getattr(cli, "controlled_count", 3),
        )
        if supplement
        else experiment_cases(
            cli.dataset, cli.protocol, cli.scenarios, smoke=cli.smoke, check_data=cli.check_data
        )
    )
    if not cases:
        if supplement:
            raise ValueError("The selected datasets are unsupported by the supplement profile")
        raise ValueError(
            "Balance uses the grouped protocol. Scale uses grouped Car-Hacking and CIC-UNSW-NB15."
        )
    prefix = "revision_supplement_v4" if supplement else "revision_v4"
    if repair:
        prefix += "_transfer_repair"
    elif transfer_scores:
        prefix += "_transfer_scores"
    if full_transfer_train:
        prefix += "_full_train"
    root = (
        cli.out or ROOT / "outputs" / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    ).resolve()
    data_root = cli.data_root.resolve()
    if root == data_root or root.is_relative_to(data_root):
        raise ValueError("Use an output directory outside the raw dataset tree")
    config = {
        "experiment": "fedcaid_review",
        "version": 4,
        "ple_boundary_policy": cli.ple_boundary_policy,
        "ple_boundary_policy_version": core.PLE_BOUNDARY_VERSION,
        "synthetic_only": cli.smoke,
        "data_only": cli.check_data,
        "seeds": seeds,
        "device": cli.device,
        "no_tcp": cli.no_tcp,
        "figures": not cli.no_figures,
        "data_root": str(data_root) if not cli.smoke else None,
        "matrix": [
            {"dataset": d, "protocol": p, "scenarios": [list(case) for case in values]}
            for (d, p), values in cases.items()
        ],
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if cli.device == "cuda" else None,
        },
        "source_sha256": _source_hashes(supplement),
        "mode_descriptions": {
            **MODE_DESCRIPTIONS,
            "fedavg": "Full detector-weight FedAvg with native multiclass argmax",
            "fedavg_threshold": "Same FedAvg detector with a client CAL holdout attack threshold",
        },
        "threat_model": THREAT_MODEL,
        **_objective_comparison(cases, cli.development_only),
    }
    if getattr(cli, "iov_diagnostics", False):
        config["iov_diagnostics"] = True
        config["iov_diagnostics_scope"] = {
            "modes": list(IOV_DIAGNOSTIC_MODES),
            "roles": ["cal_holdout", "val"],
            "test_evaluated": False,
            "fedavg_evaluated": False,
            "selection": "Frozen scalar threshold for each existing branch and client",
        }
        if getattr(cli, "iov_fixed_base_alarm", False):
            config["iov_fixed_base_alarm"] = True
            config["mode_descriptions"].update(FIXED_BASE_MODE_DESCRIPTIONS)
            config["iov_diagnostics_scope"]["modes"].extend(FIXED_BASE_MODE_DESCRIPTIONS)
            config["iov_diagnostics_scope"]["fixed_alarm_reference"] = "base_threshold"
    if (
        cli.development_only
        or cli.group_cal
        or cli.cache_predictions
        or controlled
        or getattr(cli, "conditional_adapter", False)
    ):
        config["revision_v6"] = {
            "evaluation_stage": "development_only" if cli.development_only else "existing_test_exploratory",
            "cache_predictions_private": cli.cache_predictions,
            "controlled": controlled,
            "controlled_count": getattr(cli, "controlled_count", None) if controlled else None,
            "controlled_classes": getattr(cli, "controlled_classes", None) if controlled else None,
            "controlled_receiver": getattr(cli, "controlled_receiver", None) if controlled else None,
            "controlled_absence_scope": (
                getattr(cli, "controlled_absence_scope", None) if controlled else None
            ),
            "conditional_adapter": getattr(cli, "conditional_adapter", False),
            "conditional_rank": getattr(cli, "conditional_rank", None),
            "conditional_fit_limit": getattr(cli, "conditional_fit_limit", None),
            "conditional_sampling": getattr(cli, "conditional_sampling", None),
            "conditional_min_class_rows": getattr(cli, "conditional_min_class_rows", None),
            "conditional_class_balance_beta": getattr(cli, "conditional_class_balance_beta", None),
            "conditional_mask_absent_classes": getattr(cli, "conditional_mask_absent_classes", None),
            "conditional_optimizer": getattr(cli, "conditional_optimizer", None),
            "conditional_argmax_modes": getattr(cli, "conditional_argmax_modes", None),
            "threshold_min_benign_rows": getattr(cli, "threshold_min_benign_rows", None),
            "threshold_insufficient_policy": cli.threshold_insufficient_policy,
            "cal_split_protocol": "exact_feature_group" if getattr(cli, "group_cal", False) else "row",
            "repair_union_fallback": getattr(cli, "repair_union_fallback", None),
        }
    if supplement:
        config["supplement"] = {
            "profile": cli.profile,
            "donor_cal_fit_min_rows": MIN_DONOR_ROWS,
            "transfer_endpoint": "unweighted mean recall of eligible receiver/class cells within each seed",
            "transfer_population": (
                "local CAL-fit class absent; separate local TRAIN-absent and TRAIN-present strata"
            ),
            "selection": (
                "Every declared mode and seed is evaluated. TEST does not select a method or threshold."
            ),
            "scope": (
                "Grouped results revisit the existing benchmark; NB15 time holdout remains within the same"
                " source."
            ),
            "temporal_purge_exact": not cli.temporal_raw,
            "subtype_ablation": cli.subtype_ablation,
        }
        if any(
            scenario == "supplement_transfer" or scenario.startswith("controlled_transfer_")
            for selected in cases.values()
            for scenario, _, _ in selected
        ):
            config["supplement"]["transfer_training"] = {
                "mode": "full" if full_transfer_train else "capped",
                "train_row_limit": None if full_transfer_train else SCALE_TRAIN_ROWS,
                "sampling": (
                    "all grouped TRAIN rows"
                    if full_transfer_train
                    else "reserve one per class and sample uniformly without replacement"
                ),
                "sampling_seed_offset": None if full_transfer_train else 905,
            }
        if transfer_scores:
            config["supplement"]["transfer_score_diagnostics"] = {
                "population": (
                    "same eligible receivers, attack labels and exact-input unique TEST as hard transfer"
                ),
                "direction": "federated minus local on each input; rank one is best",
                "output": "cell-level aggregates, then equal cell weight within seed and equal seed weight",
            }
        if repair:
            config["supplement"]["missing_class_repair"] = {
                "source_mode": "base_threshold",
                "donor_kind": "full_weight_fedavg",
                "gate_target_fraction": gate_fraction,
                "union_local_target_fraction": 1 - gate_fraction,
                "residual_budget_fraction": residual_fraction,
                "evaluated_policies": [
                    dict(donor_fraction=g, residual_fraction=r, suffix=suffix)
                    for g, r, suffix in policy_specs(cli)
                ],
                "probability_output": (
                    "selected-branch binary attack probability; ECE evaluated, calibration not assumed"
                ),
                "donor": (
                    "one shared TRAIN-only FedAvg detector with pooled TRAIN preprocessing in the local"
                    " simulator"
                ),
                "routing": (
                    "shared top attack class absent from local TRAIN and at least five remote TRAIN rows"
                ),
                "calibration": (
                    "fixed gate, split-budget union, and residual-budget rescue preserving all base alerts"
                ),
                "scope": (
                    "additional shared-model experiment; carries full FedAvg costs and no secret-sharing"
                    " claim"
                ),
            }
        config["mode_descriptions"].update({
            "base_threshold_subtype": (
                "Fixed base_threshold alarm, local attack-only linear subtype classifier"
            ),
            "local_only_fine": "Local-only control at fixed-point scale 1e9",
            "full_fine": "Full adapter at fixed-point scale 1e9",
        })
        if repair:
            config["mode_descriptions"].update({
                REPAIR_FIXED: "Shared missing subtype can replace an already issued base_threshold alert",
                REPAIR_UNION: "Local or shared missing-class branch alerts at its own CAL cutoff",
                REPAIR_RESCUE: "Keep local alarms and allow shared alerts within residual benign CAL budget",
            })
    config = json.loads(json.dumps(config))
    if cli.resume:
        if json.loads((root / "run.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume with a different experiment, source revision or runtime")
    else:
        root.mkdir(parents=True, exist_ok=False)
        _atomic_json(root / "run.json", config)
    print(f"REVISION OUTPUT: {root}", flush=True)
    results = []
    for (dataset, protocol), selected in cases.items():
        print(f"DATA AUDIT: {dataset} | {protocol}", flush=True)
        if cli.smoke:
            data = _synthetic(dataset, protocol)
        elif protocol == "temporal":
            from .revision_temporal import load_temporal_data

            data = load_temporal_data(data_root, purge_exact=not cli.temporal_raw)
        else:
            data = load_revision_data(dataset, data_root, split_protocol=protocol)
        parent = root / dataset / protocol
        parent.mkdir(parents=True, exist_ok=True)
        audit = parent / "data_audit.json"
        if audit.exists() and json.loads(audit.read_text(encoding="utf-8")) != json.loads(
            json.dumps(data.audit)
        ):
            raise ValueError(f"Cannot resume: source or split audit differs for {dataset} {protocol}")
        _atomic_json(audit, data.audit)
        controlled_plan = None
        controlled_checks = {}
        if controlled:
            controlled_plan = plan_targets(
                data.parts, data.names, cli.controlled_count, cli.controlled_classes
            )
            plan_path = parent / "controlled_plan.json"
            if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != controlled_plan:
                raise ValueError("Controlled class plan changed on resume")
            _atomic_json(plan_path, controlled_plan)
            partition_checks = []
            # Validate every predeclared partition before fitting any target.
            for _, clients, alpha in selected[:1]:
                if cli.controlled_receiver >= clients:
                    raise ValueError("Controlled receiver lies outside the client roster")
                for seed in seeds:
                    roster, _ = allocate_clients(data.parts, len(data.names), clients, alpha, seed)
                    for target_info in controlled_plan["targets"]:
                        fit_fraction = DATASET_PARAMETERS[dataset]["cal_fit_fraction"]
                        candidate_roster, partition_check = _controlled_case(
                            data.parts, roster, target_info["class_index"], seed, fit_fraction, cli
                        )
                        if getattr(cli, "group_cal", False) and cli.conditional_adapter:
                            fit_args = argparse.Namespace(
                                **vars(cli), cal_fit_fraction=fit_fraction, seed=seed
                            )
                            splits, _ = _grouped_cal(data.parts, candidate_roster, fit_args, True)
                            partition_check["conditional_selected_target_support"] = _check_sampled_target(
                                data.parts,
                                candidate_roster,
                                splits,
                                target_info["class_index"],
                                partition_check["receiver"],
                                fit_args,
                                len(data.names),
                            )
                        checked = {
                            "seed": seed,
                            "class": target_info["class"],
                            "receiver": partition_check["receiver"],
                            "receiver_policy": partition_check["receiver_policy"],
                            "receiver_eligibility": partition_check.get("receiver_eligibility"),
                            "receiver_train_target_rows": partition_check["roles"]["train"][
                                "receiver_target_rows_after"
                            ],
                            "conditional_selected_target_support": partition_check.get(
                                "conditional_selected_target_support"
                            ),
                            "cal_group_split": partition_check.get("cal_group_split"),
                            "donor_train_rows": partition_check["donor_train_rows"],
                            "donor_cal_fit_rows": partition_check["donor_cal_fit_rows"],
                            "receiver_val_target_support": partition_check["receiver_val_target_support"],
                            "val_routing": partition_check["val_routing"],
                            "test_routes_unchanged": partition_check["test_routes_unchanged"],
                            "selection": "VAL support reported after frozen TRAIN/CAL selection",
                        }
                        controlled_checks[seed, target_info["class_index"]] = checked
                        partition_checks.append(checked)
            preflight_path = parent / "controlled_support_preflight.json"
            if (
                preflight_path.exists()
                and json.loads(preflight_path.read_text(encoding="utf-8")) != partition_checks
            ):
                raise ValueError("Controlled partition preflight changed on resume")
            _atomic_json(preflight_path, partition_checks)
        if not cli.check_data:
            for scenario, clients, alpha in selected:
                for seed in seeds:
                    target = parent / scenario / f"seed_{seed}"
                    target.mkdir(parents=True, exist_ok=True)
                    saved = target / "result.json"
                    if saved.is_file():
                        result = json.loads(saved.read_text(encoding="utf-8"))
                        expected = (dataset, protocol, scenario, seed, clients, alpha)
                        if (
                            tuple(
                                result[k]
                                for k in ("dataset", "protocol", "scenario", "seed", "clients", "alpha")
                            )
                            != expected
                        ):
                            raise ValueError(f"Saved experiment identity differs at {saved}")
                    else:
                        controlled_target = (
                            controlled_plan["targets"][int(scenario.rsplit("_", 1)[1])]
                            if controlled_plan
                            else None
                        )
                        check = (
                            controlled_checks[seed, controlled_target["class_index"]]
                            if controlled_target
                            else None
                        )
                        result = _run_case(
                            data,
                            dataset,
                            protocol,
                            scenario,
                            seed,
                            clients,
                            alpha,
                            cli,
                            target,
                            controlled_target=controlled_target,
                            controlled_preflight=check,
                        )
                        _atomic_json(saved, result)
                    results.append(result)
        del data
        gc.collect()
    if results:
        write_report(root, results, figures=not cli.no_figures)
        if getattr(cli, "iov_diagnostics", False):
            write_iov_diagnostics(root, results)
    _atomic_json(root / "complete.json", {"replicates": len(results), "data_audits": len(cases)})
    if (
        not supplement
        and not cli.smoke
        and not cli.check_data
        and not cli.development_only
        and cli.scenarios == "all"
        and cli.protocol == "both"
        and set(cli.dataset) == set(DATASETS)
        and len(seeds) >= 5
    ):
        from .review_evidence import generate

        print(
            f"REVIEW EVIDENCE: {generate(root, root / 'review_evidence', figures=not cli.no_figures)}",
            flush=True,
        )
    print(f"REVISION COMPLETE: {len(results)} replicates; {len(cases)} data audits", flush=True)


def main(argv=None, *, supplement=False):
    cli = _parser(supplement=supplement).parse_args(argv)
    try:
        execute(cli)
    except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"REVISION STOP: {error}", file=sys.stderr, flush=True)
        return 2
    return 0
