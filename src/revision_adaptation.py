"""Matched class adaptation contrasts on fixed local detector predictions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .privacy_core import ClassScoreAdapter, digest, fit_class_adapter
from .revision_fixed_subtype import fixed_subtype_predictions
from .secure_aggregation import SecureAggregationSession, _clip_quantize
from .utils import empirical_alarm_threshold, hierarchical_decisions, split_calibration

MODES = (
    "base",
    "base_fixed",
    "base_threshold",
    "local_only",
    "local_fixed",
    "local_unclipped",
    "full",
    "without_prox",
    "without_threshold",
)

MATCHED_MODES = ("base", "base_threshold", "local_rounds", "local_only", "full_global", "full")
CAL_THRESHOLD_MODES = frozenset(MATCHED_MODES[1:]) | {"local_unclipped", "without_prox"}
FIXED_BASE_MODES = {"local_only": "fixed_base_local", "full": "fixed_base_full"}
FIXED_BASE_MODE_DESCRIPTIONS = {
    "fixed_base_local": "Base TabM binary alerts with locally adapted attack subtype labels",
    "fixed_base_full": "Base TabM binary alerts with federated and personalized attack subtype labels",
}

MODE_DESCRIPTIONS = {
    "base": "Detector alone with native multiclass argmax",
    "base_fixed": "Detector alone with a fixed attack threshold of 0.5",
    "base_threshold": "Detector alone with a CAL holdout attack threshold",
    "local_rounds": "Locally anchored, clipped and encoded round fits without final personalization",
    "local_only": "Locally anchored adapter with matched clipping, encoding and CAL threshold",
    "local_fixed": "Locally anchored adapter with a fixed attack threshold of 0.5",
    "local_unclipped": "Local adapter with a CAL threshold and no clipping or quantization between fits",
    "full_global": "Shared adapter with a CAL holdout attack threshold and no final personalization",
    "full": "Shared adapter, local personalization and a CAL holdout attack threshold",
    "without_prox": "Full pipeline with adapter proximal weight set to zero",
    "without_threshold": "Full pipeline with a fixed attack threshold of 0.5",
}

THREAT_MODEL = {
    "host": "Trusted local host; client code and coordinator share one Python process.",
    "clients": "Fixed roster; clients are trusted to clip and submit honest updates.",
    "collectors": (
        "Two local subprocesses, assumed honest but curious and noncolluding. "
        "Each receives one uniform field share per update and maintains a sum."
    ),
    "coordinator": "Trusted; reconstructs the complete roster aggregate, which is disclosed.",
    "missing_client_or_worker": (
        "An incomplete roster or worker error invalidates the session. "
        "No partial aggregate is accepted or used for personalization."
    ),
    "out_of_scope": [
        "malicious clients or coordinator",
        "colluding collectors",
        "host compromise",
        "remote deployment or transport security",
        "differential privacy",
    ],
}


@dataclass(frozen=True)
class AdapterClient:
    cal_probabilities: np.ndarray
    cal_labels: np.ndarray
    eval_probabilities: np.ndarray
    log_prior: np.ndarray
    measure_inference: Callable | None = None
    cal_split: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class _CalibrationSplit:
    fit_probabilities: np.ndarray
    fit_labels: np.ndarray
    holdout_probabilities: np.ndarray
    holdout_labels: np.ndarray
    fit_indices: np.ndarray


def _split_clients(clients, adapter, args):
    splits = []
    for index, client in enumerate(clients):
        cal = adapter._probabilities(client.cal_probabilities)
        labels = np.asarray(client.cal_labels)
        if (
            labels.shape != (len(cal),)
            or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any()
            or (labels >= adapter.classes).any()
        ):
            raise ValueError("Valid integer CAL labels in the public class space required")
        adapter._probabilities(client.eval_probabilities)
        if client.cal_split is None:
            fit, holdout = split_calibration(len(labels), args.cal_fit_fraction, args.seed + 335 + index)
        else:
            fit, holdout = map(np.asarray, client.cal_split)
            if any(
                indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices)
                for indices in (fit, holdout)
            ):
                raise ValueError("Explicit CAL split requires nonempty integer index arrays")
            all_rows = np.sort(np.concatenate((fit, holdout)))
            if not np.array_equal(all_rows, np.arange(len(cal))):
                raise ValueError("Explicit CAL split must partition the client CAL rows")
        splits.append(_CalibrationSplit(cal[fit], labels[fit], cal[holdout], labels[holdout], fit))
    return splits


def _fit(initial, client, split, args, prox, diagnostics=None):
    return fit_class_adapter(
        initial,
        split.fit_probabilities,
        split.fit_labels,
        client.log_prior,
        args.adapter_epochs,
        args.adapter_lr,
        args.prior_strength,
        prox,
        diagnostics=diagnostics,
    )


def _local_round_vectors(clients, splits, adapter, args, *, match_encoding):
    vectors = []
    for client, split in zip(clients, splits):
        vector = np.zeros(adapter.classes - 1)
        for _ in range(args.rounds):
            delta = _fit(vector, client, split, args, args.adapter_prox)
            if match_encoding:
                encoded = _clip_quantize(delta, len(vector), args.clip, args.secure_scale)
                delta = np.asarray(encoded, dtype=np.float64) / args.secure_scale
            vector = adapter.project(vector + delta)
        vectors.append(vector)
    return vectors


def _personalize_vectors(vectors, clients, splits, adapter, args, prox):
    return [
        adapter.project(vector + _fit(vector, client, split, args, prox))
        for vector, client, split in zip(vectors, clients, splits, strict=True)
    ]


def _record_client_update(delta, fit_details, split, args):
    norm = float(np.linalg.norm(delta))
    quantized = _clip_quantize(delta, len(delta), args.clip, args.secure_scale)
    clipped = np.asarray(delta) * min(1.0, args.clip / norm) if norm else np.asarray(delta)
    decoded = np.asarray(quantized, dtype=float) / args.secure_scale
    error = float(np.linalg.norm(decoded - clipped))
    clipped_norm = float(np.linalg.norm(clipped))
    rows = np.arange(len(split.fit_labels))
    saturated = split.fit_probabilities[rows, split.fit_labels] >= 1 - 1e-6
    return {
        "update_l2": norm,
        "encoding_all_zero": not any(quantized),
        "encoding_nonzero_coordinate_fraction": float(np.count_nonzero(quantized) / len(quantized)),
        "quantization_l2_error": error,
        "quantization_relative_l2_error": error / clipped_norm if clipped_norm else 0.0,
        "encoded_update_l2": float(np.linalg.norm(decoded)),
        "l2_clipped": norm > args.clip,
        "first_gradient_l2": fit_details["first_gradient_l2"],
        "projection_steps": fit_details["projection_steps"],
        "cal_fit_rows": len(rows),
        "base_true_class_probability_at_least_0_999999": int(saturated.sum()),
    }


def _explain_round(previous, mean, proposed, projected, updates):
    if not np.array_equal(projected, previous):
        return "global_adapter_changed"
    if not any(record["update_l2"] for record in updates):
        return "all_local_deltas_exactly_zero"
    if all(record["encoding_all_zero"] for record in updates):
        return "all_local_deltas_zero_after_clipping_and_encoding"
    if not np.any(mean):
        return "encoded_nonzero_updates_cancelled"
    if np.array_equal(proposed, previous):
        return "nonzero_mean_lost_in_float_addition"
    return "global_projection_removed_mean_update"


def _federated_round_vector(clients, splits, adapter, args, mode, diagnostics, prox):
    client_ids = [f"client_{i}" for i in range(len(clients))]
    current = np.zeros(adapter.classes - 1)
    started = time.perf_counter()
    with SecureAggregationSession(
        client_ids,
        len(current),
        clip=args.clip,
        scale=args.secure_scale,
        timeout=args.secure_timeout,
    ) as collector:
        diagnostics["collector_startup_seconds"][mode] = time.perf_counter() - started
        session = uuid4().hex
        for round_index in range(1, args.rounds + 1):
            round_started = time.perf_counter()
            contract = collector.start_round(session, round_index, digest(current.tolist()))
            summaries = []
            fit_seconds = 0.0
            submission_seconds = 0.0
            for client_id, client, split in zip(client_ids, clients, splits):
                fit_details = {}
                fit_started = time.perf_counter()
                delta = _fit(current, client, split, args, prox, fit_details)
                fit_seconds += time.perf_counter() - fit_started
                submission_started = time.perf_counter()
                collector.submit(client_id, delta, contract=contract)
                submission_seconds += time.perf_counter() - submission_started
                summaries.append(_record_client_update(delta, fit_details, split, args))
                del delta
            aggregation_started = time.perf_counter()
            mean, receipt = collector.aggregate()
            proposed = current + mean
            projected = adapter.project(proposed)
            aggregation_seconds = time.perf_counter() - aggregation_started
            elapsed = time.perf_counter() - round_started
            public = {
                "mode": mode,
                "round": round_index,
                "adapter_prox": prox,
                "adapter_fit_seconds": fit_seconds,
                "share_submission_seconds": submission_seconds,
                "aggregate_and_project_seconds": aggregation_seconds,
                "round_wall_seconds_including_diagnostics": elapsed,
                "mean_update_l2": float(np.linalg.norm(mean)),
                "global_adapter_change_l2": float(np.linalg.norm(projected - current)),
                "global_adapter_l2": float(np.linalg.norm(projected)),
                "global_projection_changed_proposal": not np.array_equal(projected, proposed),
                "accepted_update": not np.array_equal(projected, current),
                **receipt,
            }
            diagnostics["rounds"].append(public)
            fit_rows = sum(r["cal_fit_rows"] for r in summaries)
            diagnostics["private_rounds"].append({
                "mode": mode,
                "round": round_index,
                "outcome": _explain_round(current, mean, proposed, projected, summaries),
                "clients_with_exact_zero_delta": sum(r["update_l2"] == 0 for r in summaries),
                "clients_with_zero_encoded_delta": sum(r["encoding_all_zero"] for r in summaries),
                "clients_clipped": sum(r["l2_clipped"] for r in summaries),
                "secure_scale": args.secure_scale,
                "mean_nonzero_coordinate_fraction": float(
                    np.mean([r["encoding_nonzero_coordinate_fraction"] for r in summaries])
                ),
                "max_quantization_l2_error": max(r["quantization_l2_error"] for r in summaries),
                "max_quantization_relative_l2_error": max(
                    r["quantization_relative_l2_error"] for r in summaries
                ),
                "encoded_nonzero_updates_cancelled": bool(
                    not np.any(mean) and any(r["encoded_update_l2"] for r in summaries)
                ),
                "max_local_delta_l2": max(r["update_l2"] for r in summaries),
                "max_initial_gradient_l2": max(r["first_gradient_l2"] for r in summaries),
                "local_projection_steps": sum(r["projection_steps"] for r in summaries),
                "cal_fit_rows": fit_rows,
                "base_true_class_saturation_fraction": (
                    sum(r["base_true_class_probability_at_least_0_999999"] for r in summaries) / fit_rows
                ),
                "saturation_threshold": 1 - 1e-6,
            })
            current = projected
    return current


def _classify(mode, client, split, adapter, local_vector, args):
    optimized = mode in CAL_THRESHOLD_MODES
    source = "CAL holdout" if optimized else "native argmax" if mode == "base" else "fixed 0.5"
    calibration = {
        "threshold_source": source,
        "cal_fit_rows": len(split.fit_labels),
        "cal_holdout_rows": len(split.holdout_labels),
        "benign_holdout_rows": int((split.holdout_labels == 0).sum()),
        "threshold": None if mode == "base" else 0.5,
        "no_benign_fallback": False,
    }
    if local_vector is not None:
        adapter.set_vector(local_vector)
    holdout = None
    if optimized:
        holdout = split.holdout_probabilities
        if local_vector is not None:
            holdout = adapter.transform(holdout)
        _, details = empirical_alarm_threshold(
            holdout,
            split.holdout_labels,
            args.target_far,
            args.threshold_floor,
            getattr(args, "threshold_min_benign_rows", 0),
            getattr(args, "threshold_insufficient_policy", "floor"),
        )
        calibration.update(details)

    started = time.perf_counter()
    eval_p = client.eval_probabilities
    if local_vector is not None:
        eval_p = adapter.transform(eval_p)
    if mode == "base":
        predicted = eval_p.argmax(1)
        score = np.clip(eval_p[:, 1:].sum(1), 0, 1)
    else:
        decisions = hierarchical_decisions(eval_p, calibration["threshold"])
        predicted = decisions["predicted"]
        score = decisions["attack_score"]
    elapsed = time.perf_counter() - started
    return predicted, score, calibration, elapsed, eval_p, holdout


def _fixed_alarm_labels(base, adapted):
    """Relabel original alerts only and keep base labels for missing score rows."""
    result = base.copy()
    if adapted is None:
        return result
    adapted = np.asarray(adapted, dtype=np.float64)
    if adapted.ndim != 2 or adapted.shape[0] != len(base) or adapted.shape[1] < 2:
        raise ValueError("Adapted subtype scores must align with base alarm rows and classes")
    alerts = np.flatnonzero(base > 0)
    scores = adapted[alerts]
    valid = np.isfinite(scores).all(axis=1) & (scores >= 0).all(axis=1) & (scores[:, 1:] > 0).any(axis=1)
    rows = alerts[valid]
    result[rows] = fixed_subtype_predictions(base[rows], scores[valid], range(1, adapted.shape[1]))
    return result


def _fixed_base_specs(modes, enabled):
    """Require the same recorded base and adapted paths in scoring order."""
    if not enabled:
        return {}
    required = {"base_threshold", *FIXED_BASE_MODES}
    if not required <= set(modes) or any(
        modes.index(mode) < modes.index("base_threshold") for mode in FIXED_BASE_MODES
    ):
        raise ValueError("Fixed-base alarms require base_threshold before local_only and full")
    return FIXED_BASE_MODES


def _fixed_base_decisions(split, receipt, base_val, adapted_cal, adapted_val):
    """Use the original base threshold for CAL and the original base labels for VAL alarms."""
    cal = hierarchical_decisions(split.holdout_probabilities, receipt["threshold"])
    return _fixed_alarm_labels(cal["predicted"], adapted_cal), _fixed_alarm_labels(base_val, adapted_val)


def _development_class_diagnostics(split, args):
    """Count default-head alarm and subtype errors on CAL data only."""
    threshold, _ = empirical_alarm_threshold(
        split.holdout_probabilities,
        split.holdout_labels,
        args.target_far,
        args.threshold_floor,
        getattr(args, "threshold_min_benign_rows", 0),
        getattr(args, "threshold_insufficient_policy", "floor"),
    )
    roles = (
        ("cal_fit", split.fit_probabilities, split.fit_labels),
        ("cal_holdout", split.holdout_probabilities, split.holdout_labels),
    )
    # With an orthonormal class basis, any pairwise log odds can change by at most sqrt(2) * radius.
    max_pairwise_bias = np.sqrt(2.0) * ClassScoreAdapter.RADIUS
    for role, probabilities, labels in roles:
        for label in np.unique(labels[labels > 0]):
            values = probabilities[labels == label]
            decisions = hierarchical_decisions(values, threshold)
            subtype = values[:, 1:]
            true_score = values[:, label]
            # Stable class order breaks ties the same way as the deployed argmax.
            higher_rank = (subtype > true_score[:, None]) | (
                (subtype == true_score[:, None]) & (np.arange(1, values.shape[1]) < label)
            )
            top_gap = np.log(np.clip(subtype.max(axis=1), 1e-12, 1.0)) - np.log(
                np.clip(true_score, 1e-12, 1.0)
            )
            alarm = decisions["attack_alert"]
            correct_subtype = decisions["subtype"] == label
            yield {
                "role": role,
                "class_index": int(label),
                "support": len(values),
                "missed_alarm": int((~alarm).sum()),
                "alerted_wrong_subtype": int((alarm & ~correct_subtype).sum()),
                "attack_top1_correct": int(correct_subtype.sum()),
                "attack_top3_correct": int((higher_rank.sum(axis=1) < 3).sum()),
                "outside_bias_margin": int((top_gap > max_pairwise_bias).sum()),
                "base_cal_holdout_threshold": threshold,
            }


def run_adapter_ablations(
    clients,
    args,
    *,
    modes=MODES,
    include_development=False,
    observe_scores=None,
    observe_operating=None,
    fixed_base_alarm=False,
):
    """Return mode/client predictions, binary scores and diagnostics without accessing eval labels."""
    clients = tuple(clients)
    modes = tuple(modes)
    if not clients or not modes or set(modes) - MODE_DESCRIPTIONS.keys() or len(set(modes)) != len(modes):
        raise ValueError("Expected client inputs and distinct, supported ablation modes")
    fixed_modes = _fixed_base_specs(modes, fixed_base_alarm)
    first = np.asarray(clients[0].cal_probabilities)
    if first.ndim != 2:
        raise ValueError("A multiclass CAL probability matrix is required")
    adapter = ClassScoreAdapter(first.shape[1])
    splits = _split_clients(clients, adapter, args)
    diagnostics = {
        "rounds": [],
        "private_rounds": [],
        "collector_startup_seconds": {},
        "calibration": [],
        "postprocess": [],
        "resident_inference": [],
    }
    if include_development:
        diagnostics["cal_fit_class_counts"] = [
            np.bincount(split.fit_labels, minlength=adapter.classes).tolist() for split in splits
        ]
        diagnostics["development_class_diagnostics"] = [
            {"client": index, **row}
            for index, split in enumerate(splits)
            for row in _development_class_diagnostics(split, args)
        ]
    vectors = {}
    if {"local_rounds", "local_only", "local_fixed"}.intersection(modes):
        anchors = _local_round_vectors(clients, splits, adapter, args, match_encoding=True)
        vectors["local_rounds"] = anchors
        if {"local_only", "local_fixed"}.intersection(modes):
            vectors["local_only"] = _personalize_vectors(
                anchors,
                clients,
                splits,
                adapter,
                args,
                args.adapter_prox,
            )
            vectors["local_fixed"] = vectors["local_only"]
    if "local_unclipped" in modes:
        anchors = _local_round_vectors(clients, splits, adapter, args, match_encoding=False)
        vectors["local_unclipped"] = _personalize_vectors(
            anchors,
            clients,
            splits,
            adapter,
            args,
            args.adapter_prox,
        )
    full_personalized = {"full", "without_threshold"}.intersection(modes)
    if full_personalized or "full_global" in modes:
        round_mode = "full" if full_personalized else "full_global"
        anchor = _federated_round_vector(
            clients, splits, adapter, args, round_mode, diagnostics, args.adapter_prox
        )
        vectors["full_global"] = [anchor] * len(clients)
        if full_personalized:
            vectors["full"] = _personalize_vectors(
                vectors["full_global"],
                clients,
                splits,
                adapter,
                args,
                args.adapter_prox,
            )
            vectors["without_threshold"] = vectors["full"]
    if "without_prox" in modes:
        anchor = _federated_round_vector(clients, splits, adapter, args, "without_prox", diagnostics, 0.0)
        vectors["without_prox"] = _personalize_vectors(
            [anchor] * len(clients), clients, splits, adapter, args, 0.0
        )

    output_modes = (*modes, *fixed_modes.values())
    predictions, scores = {mode: [] for mode in output_modes}, {mode: [] for mode in output_modes}
    base_calibration = []
    shared_scores = {}
    score_groups = {
        "base_fixed": "base",
        "base_threshold": "base",
        "local_fixed": "local_only",
        "without_threshold": "full",
    }
    for mode in modes:
        local_vectors = vectors.get(mode, [None] * len(clients))
        for index, (client, split, local_vector) in enumerate(zip(clients, splits, local_vectors)):
            pred, score, calibration, elapsed, eval_p, holdout = _classify(
                mode,
                client,
                split,
                adapter,
                local_vector,
                args,
            )
            if observe_scores is not None:
                observe_scores(mode, index, eval_p, pred)
            if observe_operating is not None and holdout is not None:
                observe_operating(mode, index, holdout, calibration, score, pred)
            if fixed_modes and mode == "base_threshold":
                base_calibration.append(calibration)
            if mode in fixed_modes:
                fixed_mode = fixed_modes[mode]
                base_receipt = base_calibration[index]
                fixed_cal, fixed_val = _fixed_base_decisions(
                    split, base_receipt, predictions["base_threshold"][index], holdout, eval_p
                )
                base_score = scores["base_threshold"][index].copy()
                predictions[fixed_mode].append(fixed_val)
                scores[fixed_mode].append(base_score)
                fixed_receipt = {
                    **base_receipt,
                    "threshold_reference_mode": "base_threshold",
                    "subtype_source_mode": mode,
                }
                diagnostics["calibration"].append({"mode": fixed_mode, "client": index, **fixed_receipt})
                if observe_operating is not None:
                    observe_operating(
                        fixed_mode,
                        index,
                        split.holdout_probabilities,
                        fixed_receipt,
                        base_score,
                        fixed_val,
                        cal_predictions=fixed_cal,
                    )
            del eval_p, holdout
            predictions[mode].append(pred)
            scores[mode].append(shared_scores.setdefault((score_groups.get(mode, mode), index), score))
            diagnostics["calibration"].append({"mode": mode, "client": index, **calibration})
            diagnostics["postprocess"].append({
                "mode": mode,
                "client": index,
                "elapsed_seconds": elapsed,
                "rows": len(pred),
            })
            if mode == "full" and client.measure_inference is not None:

                def decide(p):
                    return hierarchical_decisions(adapter.transform(p), calibration["threshold"])

                diagnostics["resident_inference"].append({
                    "mode": mode, "client": index, **client.measure_inference(decide)
                })
    return predictions, scores, diagnostics
