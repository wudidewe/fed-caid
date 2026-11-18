"""Experimental probability-conditioned adapters with matched local controls.

The public feature map uses the ordered class log-probabilities, not private
detector hidden coordinates. A fixed Helmert projection avoids nonidentifiable
factor averaging. Conditional adapters are a new model family: their payload is
(K-1)*(rank+1). Rank zero is a matched intercept-only control.

Only CAL-fit labels optimize weights. CAL-holdout selects the same empirical
alarm rule used by the existing baselines; evaluation labels are not accepted.
No hyperparameters, classes or policies are chosen from evaluation results.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .privacy_core import ClassScoreAdapter, digest
from .revision_adaptation import _split_clients
from .secure_aggregation import SecureAggregationSession, _clip_quantize
from .utils import empirical_alarm_threshold, hierarchical_decisions

MODES = ("conditional_local_rounds", "conditional_local", "conditional_global", "conditional_full")
ARGMAX_MODES = tuple(f"{mode}_argmax" for mode in MODES)
SAMPLING_POLICIES = ("class_blind", "coverage_first")
OPTIMIZERS = ("sgd", "adam")
SCHEMA = "public_logprob_helmert_residual_v1"
FEATURE_LOG_LIMIT = 12.0
PROBABILITY_FLOOR = 1e-12
BATCH_ROWS = 16384


class ConditionalScoreAdapter:
    """Zero-sum residual logits, linear in a fixed bounded public feature map."""

    def __init__(self, classes, rank=4, radius=2.0):
        if type(classes) is not int or classes < 2:
            raise ValueError("At least two ordered public classes are required")
        if type(rank) is not int or rank < 0:
            raise ValueError("Conditional rank must be a nonnegative integer")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Conditional radius must be finite and positive")
        self.classes, self.rank, self.radius = classes, min(rank, classes - 1), float(radius)
        self.basis = ClassScoreAdapter(classes).basis.cpu().numpy().copy()
        # A truncated prefix of Helmert columns ignores later class positions.
        # Fixed public dense mixing gives every ordered class a feature loading.
        self.feature_seed = 20261112
        if self.rank == 0:
            mixing = np.empty((classes - 1, 0), dtype=np.float64)
        elif self.rank == classes - 1:
            mixing = np.eye(classes - 1)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.feature_seed, classes]))
            mixing, triangular = np.linalg.qr(rng.normal(size=(classes - 1, self.rank)), mode="reduced")
            mixing *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
        self.feature_basis = self.basis @ mixing
        self.dimension = (classes - 1) * (self.rank + 1)
        self.theta = np.zeros(self.dimension, dtype=np.float64)

    def _probabilities(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        if (
            p.ndim != 2
            or p.shape[1] != self.classes
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("Expected finite normalized probabilities in public class order")
        return p

    def project(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("Invalid conditional parameter vector")
        result = values.copy()
        largest = float(np.abs(result).max())
        if largest:
            unit = result / largest
            norm = float(np.linalg.norm(unit))
            if largest > self.radius / norm:
                result = unit * (self.radius / norm)
        return result

    def set_vector(self, values):
        self.theta = self.project(values)

    def features(self, probabilities):
        """Return [1, centered clipped log(p) @ public basis / sqrt(K)]."""
        p = self._probabilities(probabilities)
        logp = np.log(np.clip(p, PROBABILITY_FLOOR, 1.0))
        centered = logp - logp.mean(axis=1, keepdims=True)
        bounded = np.clip(centered, -FEATURE_LOG_LIMIT, FEATURE_LOG_LIMIT) / FEATURE_LOG_LIMIT
        z = bounded @ self.feature_basis / np.sqrt(self.classes)
        return logp, np.column_stack((np.ones(len(p)), z))

    def transform(self, probabilities):
        p = self._probabilities(probabilities)
        transformed = np.empty_like(p)
        class_weights = self.theta.reshape(self.classes - 1, self.rank + 1).T @ self.basis.T
        for start in range(0, len(p), BATCH_ROWS):
            block = slice(start, start + BATCH_ROWS)
            logp, features = self.features(p[block])
            logits = logp + features @ class_weights
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            transformed[block] = exp / exp.sum(axis=1, keepdims=True)
        return transformed

    def spec(self):
        return {
            "schema": SCHEMA,
            "classes": self.classes,
            "rank": self.rank,
            "parameter_count": self.dimension,
            "parameter_radius": self.radius,
            "feature_map": (
                "fixed dense orthonormal mixture of all Helmert contrasts in public class order"
                if self.rank
                else "intercept only"
            ),
            "feature_seed": self.feature_seed,
            "feature_basis_sha256": digest(self.feature_basis.tolist()),
            "feature_log_clip": FEATURE_LOG_LIMIT,
            "feature_normalization": "divide by sqrt(K)",
            "probability_floor": PROBABILITY_FLOOR,
            "public_feature_map_floats": int(self.feature_basis.size),
            "public_feature_map_transfer_bytes": 0,
            "public_feature_map_setup": "deterministically regenerated from ordered K/rank/schema",
            "fit_objective": (
                "CAL-fit multiclass cross entropy + ridge ||theta||^2/2 + prox ||theta-anchor||^2/2"
            ),
            "prior_offset": "none; identical deploy and fit logits use fixed detector probabilities",
            "comparison_scope": "conditional_full versus identical conditional_local architecture",
            "security_scope": "same trusted local simulation as legacy two-collector implementation",
        }


@dataclass(frozen=True)
class _FitData:
    logp: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def coverage_sample(labels, size, classes, min_per_class, rng):
    """Select CAL-fit rows with guaranteed coverage of every present class.

    A class-blind draw from a benign-dominated CAL partition can miss every
    rare attack class.  The first pass reserves ``min_per_class`` rows for each
    class that is present at all (or one row each when the budget is smaller
    than the per-class reservation would require). An impossible coverage
    budget is rejected. The remainder is drawn without
    replacement with probability proportional to ``1 / sqrt(class count)`` so
    that majorities still dominate the fit but cannot crowd out rare classes.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Coverage sampling requires nonempty one-dimensional labels")
    if type(size) is not int or size < 1 or size > len(labels):
        raise ValueError("Coverage sample size must be in [1, number of labels]")
    if type(min_per_class) is not int or min_per_class < 0:
        raise ValueError("Minimum per-class rows must be a nonnegative integer")
    counts = np.bincount(labels, minlength=classes)
    present = np.flatnonzero(counts)
    if not len(present):
        raise ValueError("Coverage sampling requires at least one present class")
    if min_per_class and size < len(present):
        raise ValueError("CAL-fit budget cannot cover every present class")
    reserved = min(min_per_class, size // len(present))
    chosen = []
    for label in present:
        pool = np.flatnonzero(labels == label)
        take = min(len(pool), reserved)
        if take:
            chosen.append(rng.choice(pool, take, replace=False))
    chosen = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    remaining = size - len(chosen)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(labels)), chosen, assume_unique=False)
        if len(available):
            weights = 1.0 / np.sqrt(np.maximum(counts[labels[available]], 1))
            weights = weights / weights.sum()
            extra = rng.choice(available, min(remaining, len(available)), replace=False, p=weights)
            chosen = np.concatenate((chosen, extra))
    if len(chosen) < size:
        available = np.setdiff1d(np.arange(len(labels)), chosen, assume_unique=False)
        if len(available):
            chosen = np.concatenate(
                (chosen, rng.choice(available, min(size - len(chosen), len(available)), replace=False))
            )
    return np.sort(chosen)


def class_weights_for(labels, classes, beta, max_ratio=20.0):
    """Inverse-frequency class weights with a cap, normalized over present classes."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Class-balance beta must be finite and nonnegative")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("Maximum class weight ratio must be finite and at least one")
    counts = np.bincount(np.asarray(labels), minlength=classes).astype(np.float64)
    weights = np.zeros(classes, dtype=np.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("Class weights require at least one present class")
    weights[present] = 1.0 / counts[present] ** beta
    weights[present] /= weights[present].mean()
    weights[present] = np.minimum(weights[present], max_ratio)
    return weights


def class_mask_for(labels, classes):
    """Return attack-class support in the public label order."""
    counts = np.bincount(np.asarray(labels), minlength=classes)
    return (counts[1:] > 0).astype(np.float64)


def _fit_data(adapter, probabilities, labels):
    logp, features = adapter.features(probabilities)
    labels = np.asarray(labels)
    if (
        labels.shape != (len(logp),)
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= adapter.classes).any()
    ):
        raise ValueError("Conditional fitting requires nonempty valid CAL-fit labels")
    return _FitData(logp, features, labels)


def fit_conditional(
    initial,
    data,
    adapter,
    *,
    epochs,
    lr,
    prox,
    ridge,
    class_weights=None,
    class_mask=None,
    optimizer="sgd",
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """Projected full-batch gradient (or Adam) steps on a weighted objective.

    ``class_weights`` are per-class multipliers applied to each row; they are
    renormalized by the actual row-weight sum so the gradient remains a proper
    weighted mean. With a class mask, the update preserves every unsupported
    class's residual log odds against BENIGN relative to the round anchor.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Conditional epochs must be a positive integer")
    if not math.isfinite(lr) or lr <= 0 or any(not math.isfinite(v) or v < 0 for v in (prox, ridge)):
        raise ValueError("Invalid conditional learning rate or penalty")
    if optimizer not in OPTIMIZERS:
        raise ValueError("Unsupported conditional optimizer")
    if class_weights is not None:
        class_weights = np.asarray(class_weights, dtype=np.float64)
        if (
            class_weights.shape != (adapter.classes,)
            or not np.isfinite(class_weights).all()
            or (class_weights < 0).any()
        ):
            raise ValueError("Invalid conditional class weights")
    if class_mask is not None:
        class_mask = np.asarray(class_mask, dtype=np.float64)
        if class_mask.shape != (adapter.classes - 1,) or not np.isfinite(class_mask).all():
            raise ValueError("Invalid conditional class mask")
        if not np.isin(class_mask, (0, 1)).all():
            raise ValueError("Conditional class mask entries must be binary")
    rows = len(data.labels)
    row_weights = class_weights[data.labels] if class_weights is not None else np.ones(rows, dtype=np.float64)
    weight_sum = float(row_weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Conditional fit requires positive total row weight")
    anchor = adapter.project(initial)
    theta = anchor.copy()
    tangent = None
    if class_mask is not None and (class_mask == 0).any():
        # Helmert rows of theta are contrasts. Construct the constraints in
        # class space and project back into the actual parameter coordinates.
        constrained = adapter.basis[1:][class_mask == 0] - adapter.basis[0]
        q, _ = np.linalg.qr(constrained.T, mode="complete")
        tangent = q[:, len(constrained) :]
        anchor_matrix = anchor.reshape(adapter.classes - 1, adapter.rank + 1)
        fixed = anchor_matrix - tangent @ (tangent.T @ anchor_matrix)
        free_radius = math.sqrt(max(0.0, adapter.radius**2 - float(np.sum(fixed**2))))
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    projection_steps, first_gradient = 0, None
    for step in range(epochs):
        class_theta = theta.reshape(adapter.classes - 1, adapter.rank + 1)
        class_logits = class_theta.T @ adapter.basis.T
        gradient = np.zeros((adapter.classes, adapter.rank + 1), dtype=np.float64)
        for start in range(0, rows, BATCH_ROWS):
            stop = min(start + BATCH_ROWS, rows)
            features, labels = data.features[start:stop], data.labels[start:stop]
            logits = data.logp[start:stop] + features @ class_logits
            logits -= logits.max(axis=1, keepdims=True)
            errors = np.exp(logits)
            errors /= errors.sum(axis=1, keepdims=True)
            errors[np.arange(len(labels)), labels] -= 1.0
            gradient += (errors * row_weights[start:stop, None]).T @ features
        grad_matrix = (adapter.basis.T @ gradient / weight_sum).reshape(adapter.classes - 1, adapter.rank + 1)
        grad = grad_matrix.reshape(-1) + ridge * theta + prox * (theta - anchor)
        if tangent is not None:
            grad = (tangent @ (tangent.T @ grad.reshape(adapter.classes - 1, -1))).ravel()
        if not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite conditional gradient")
        if first_gradient is None:
            first_gradient = float(np.linalg.norm(grad))
        if optimizer == "adam":
            first_moment = beta1 * first_moment + (1 - beta1) * grad
            second_moment = beta2 * second_moment + (1 - beta2) * grad * grad
            corrected_first = first_moment / (1 - beta1 ** (step + 1))
            corrected_second = second_moment / (1 - beta2 ** (step + 1))
            proposal = theta - lr * corrected_first / (np.sqrt(corrected_second) + epsilon)
        else:
            proposal = theta - lr * grad
        if tangent is None:
            theta = adapter.project(proposal)
        else:
            free = tangent.T @ proposal.reshape(adapter.classes - 1, -1)
            norm = float(np.linalg.norm(free))
            if norm > free_radius:
                free *= free_radius / norm
            theta = (fixed + tangent @ free).ravel()
        projection_steps += int(not np.allclose(theta, proposal, atol=1e-14, rtol=1e-14))
    return theta - anchor, {
        "first_gradient_l2": first_gradient,
        "projection_steps": projection_steps,
        "fit_rows": rows,
        "gradient_steps": epochs,
        "optimizer": optimizer,
        "weighted_fit": class_weights is not None,
        "masked_class_count": int(np.sum(class_mask == 0)) if class_mask is not None else 0,
    }


def _validate_args(args):
    for name in ("rounds", "adapter_epochs"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("adapter_lr", "clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("adapter_prox", args.adapter_prox),
        ("conditional_ridge", getattr(args, "conditional_ridge", 0.001)),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _log_loss(probabilities, labels):
    if not len(labels):
        return None
    return float(
        -np.log(np.clip(probabilities[np.arange(len(labels)), labels], PROBABILITY_FLOOR, 1.0)).mean()
    )


def select_fit_indices(labels, args, client_index, classes):
    limit = getattr(args, "conditional_fit_limit", 128)
    size = min(len(labels), limit) if limit else len(labels)
    rng = np.random.default_rng(args.seed + 1729 + client_index)
    if getattr(args, "conditional_sampling", "class_blind") == "coverage_first":
        return coverage_sample(labels, size, classes, getattr(args, "conditional_min_class_rows", 0), rng)
    return np.sort(rng.choice(len(labels), size, replace=False))


def run_conditional_ablations(clients, args, *, modes=MODES, observe_scores=None):
    """Return predictions, binary scores and diagnostics, without eval labels.

    local and federated branches both perform rounds*adapter_epochs fits plus
    one matching personalization fit. Each round update uses the identical L2
    clipping and fixed-point encoding. Personalization is projected and local,
    matching the original experiment's final-fit convention. The optional
    observer receives final scores only, after both branches have been fitted.
    """
    _validate_args(args)
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("Expected distinct supported conditional modes")
    clients = tuple(clients)
    if len(clients) < 3:
        raise ValueError("The conditional secure comparison requires at least three clients")
    first = np.asarray(clients[0].cal_probabilities)
    if first.ndim != 2:
        raise ValueError("A multiclass CAL probability matrix is required")
    adapter = ConditionalScoreAdapter(
        first.shape[1], getattr(args, "conditional_rank", 4), getattr(args, "conditional_radius", 2.0)
    )
    ridge = getattr(args, "conditional_ridge", 0.001)
    fit_limit = getattr(args, "conditional_fit_limit", 128)
    if type(fit_limit) is not int or fit_limit < 0:
        raise ValueError("Conditional fit limit must be a nonnegative integer (zero means all)")
    sampling = getattr(args, "conditional_sampling", "class_blind")
    min_class_rows = getattr(args, "conditional_min_class_rows", 0)
    balance_beta = getattr(args, "conditional_class_balance_beta", 0.0)
    mask_absent = getattr(args, "conditional_mask_absent_classes", False)
    optimizer = getattr(args, "conditional_optimizer", "sgd")
    argmax_modes = bool(getattr(args, "conditional_argmax_modes", False))
    threshold_min_benign_rows = getattr(args, "threshold_min_benign_rows", 0)
    if type(threshold_min_benign_rows) is not int or threshold_min_benign_rows < 0:
        raise ValueError("Threshold minimum benign rows must be a nonnegative integer")
    if sampling not in SAMPLING_POLICIES:
        raise ValueError("Unsupported conditional CAL-fit sampling policy")
    if type(min_class_rows) is not int or min_class_rows < 0:
        raise ValueError("Conditional minimum per-class rows must be a nonnegative integer")
    if not math.isfinite(balance_beta) or balance_beta < 0:
        raise ValueError("Conditional class-balance beta must be finite and nonnegative")
    if optimizer not in OPTIMIZERS:
        raise ValueError("Unsupported conditional optimizer")
    splits = _split_clients(clients, adapter, args)
    selected, selected_weights, selected_masks, selection_receipts = [], [], [], []
    for index, (client, split) in enumerate(zip(clients, splits, strict=True)):
        count = len(split.fit_labels)
        seed = args.seed + 1729 + index
        chosen = select_fit_indices(split.fit_labels, args, index, adapter.classes)
        original_fit = split.fit_indices
        support = np.bincount(split.fit_labels[chosen], minlength=adapter.classes)
        present = np.bincount(split.fit_labels, minlength=adapter.classes) > 0
        missing = np.flatnonzero(~present).tolist()
        selected.append(chosen)
        selected_weights.append(
            class_weights_for(split.fit_labels[chosen], adapter.classes, balance_beta)
            if balance_beta > 0
            else None
        )
        selected_masks.append(
            class_mask_for(split.fit_labels[chosen], adapter.classes) if mask_absent else None
        )
        selection_receipts.append({
            "client": index,
            "available_cal_fit_rows": count,
            "selected_cal_fit_rows": len(chosen),
            "sampling_seed": seed,
            "sampling": sampling,
            "selection": (
                "class-blind uniform without replacement within existing CAL-fit"
                if sampling == "class_blind"
                else "coverage-first reservation per present class, sqrt-inverse-frequency fill"
            ),
            "selected_cal_row_indices_sha256": digest(original_fit[chosen].tolist()),
            "present_class_support": np.bincount(split.fit_labels, minlength=adapter.classes).tolist(),
            "selected_class_support": support.tolist(),
            "covered_class_count": int((support > 0).sum()),
            "present_class_count": int(present.sum()),
            "missing_class_indices": missing,
            "min_class_rows": min_class_rows,
            "class_balance_beta": balance_beta,
            "mask_absent_classes": mask_absent,
        })
    fitted = [
        _fit_data(adapter, split.fit_probabilities[chosen], split.fit_labels[chosen])
        for split, chosen in zip(splits, selected, strict=True)
    ]
    diagnostics = {
        "model_spec": {
            **adapter.spec(),
            "ridge": ridge,
            "prox": args.adapter_prox,
            "epochs_per_fit": args.adapter_epochs,
            "learning_rate": args.adapter_lr,
            "rounds": args.rounds,
            "fits_per_client_per_branch": args.rounds + 1,
            "cal_fit_limit": fit_limit,
            "evaluated_modes": list(modes),
            "sampling": sampling,
            "min_class_rows": min_class_rows,
            "class_balance_beta": balance_beta,
            "mask_absent_classes": mask_absent,
            "optimizer": optimizer,
            "argmax_modes": argmax_modes,
            "threshold_min_benign_rows": threshold_min_benign_rows,
            "cal_fit_scope": (
                "same deterministic CAL-fit subset for conditional local and federated; legacy bias uses its"
                " original full fit set"
            ),
        },
        "rounds": [],
        "private_rounds": [],
        "calibration": [],
        "postprocess": [],
        "resident_inference": [],
        "collector_startup_seconds": {},
        "development_metrics": [],
        "communication": [],
        "fit_selection_private": selection_receipts,
        "fit_selection": [
            {name: value for name, value in receipt.items() if name != "selected_cal_row_indices_sha256"}
            for receipt in selection_receipts
        ],
    }

    def fit(vector, data, index):
        return fit_conditional(
            vector,
            data,
            adapter,
            epochs=args.adapter_epochs,
            lr=args.adapter_lr,
            prox=args.adapter_prox,
            ridge=ridge,
            class_weights=selected_weights[index],
            class_mask=selected_masks[index],
            optimizer=optimizer,
        )

    local = []
    if {"conditional_local_rounds", "conditional_local"}.intersection(modes):
        for index, data in enumerate(fitted):
            vector = np.zeros(adapter.dimension)
            for _ in range(args.rounds):
                delta, _ = fit(vector, data, index)
                encoded = _clip_quantize(delta, adapter.dimension, args.clip, args.secure_scale)
                vector = adapter.project(vector + np.asarray(encoded, dtype=float) / args.secure_scale)
            local.append(vector)

    current = np.zeros(adapter.dimension)
    ids = [f"client_{index}" for index in range(len(clients))]
    if {"conditional_global", "conditional_full"}.intersection(modes):
        start = time.perf_counter()
        with SecureAggregationSession(
            ids, adapter.dimension, clip=args.clip, scale=args.secure_scale, timeout=args.secure_timeout
        ) as collector:
            diagnostics["collector_startup_seconds"]["conditional_full"] = time.perf_counter() - start
            session = uuid4().hex
            for round_index in range(1, args.rounds + 1):
                start = time.perf_counter()
                contract = collector.start_round(session, round_index, digest(current.tolist()))
                details, fit_seconds, submission_seconds = [], 0.0, 0.0
                for index, (client_id, data) in enumerate(zip(ids, fitted, strict=True)):
                    started = time.perf_counter()
                    delta, detail = fit(current, data, index)
                    fit_seconds += time.perf_counter() - started
                    encoded = np.asarray(
                        _clip_quantize(delta, adapter.dimension, args.clip, args.secure_scale), dtype=float
                    )
                    detail.update(
                        update_l2=float(np.linalg.norm(delta)),
                        zero_encoded=not np.any(encoded),
                        clipped=bool(np.linalg.norm(delta) > args.clip),
                    )
                    details.append(detail)
                    started = time.perf_counter()
                    collector.submit(client_id, delta, contract=contract)
                    submission_seconds += time.perf_counter() - started
                started = time.perf_counter()
                mean, receipt = collector.aggregate()
                proposal = current + mean
                updated = adapter.project(proposal)
                diagnostics["rounds"].append({
                    "mode": "conditional_full",
                    "round": round_index,
                    "adapter_prox": args.adapter_prox,
                    "adapter_fit_seconds": fit_seconds,
                    "share_submission_seconds": submission_seconds,
                    "aggregate_and_project_seconds": time.perf_counter() - started,
                    "round_wall_seconds_including_diagnostics": time.perf_counter() - start,
                    "mean_update_l2": float(np.linalg.norm(mean)),
                    "global_adapter_change_l2": float(np.linalg.norm(updated - current)),
                    "global_adapter_l2": float(np.linalg.norm(updated)),
                    "global_projection_changed_proposal": not np.array_equal(updated, proposal),
                    "accepted_update": not np.array_equal(updated, current),
                    **receipt,
                })
                diagnostics["private_rounds"].append({
                    "mode": "conditional_full",
                    "round": round_index,
                    "secure_scale": args.secure_scale,
                    "clients_clipped": sum(row["clipped"] for row in details),
                    "clients_with_exact_zero_delta": sum(row["update_l2"] == 0 for row in details),
                    "clients_with_zero_encoded_delta": sum(row["zero_encoded"] for row in details),
                    "max_initial_gradient_l2": max(row["first_gradient_l2"] for row in details),
                    "max_local_delta_l2": max(row["update_l2"] for row in details),
                    "cal_fit_rows": sum(row["fit_rows"] for row in details),
                    "local_projection_steps": sum(row["projection_steps"] for row in details),
                })
                current = updated

    vectors = {"conditional_local_rounds": local, "conditional_global": [current] * len(clients)}
    if "conditional_local" in modes:
        vectors["conditional_local"] = [
            adapter.project(vector + fit(vector, data, index)[0])
            for index, (vector, data) in enumerate(zip(local, fitted, strict=True))
        ]
    if "conditional_full" in modes:
        vectors["conditional_full"] = [
            adapter.project(current + fit(current, data, index)[0]) for index, data in enumerate(fitted)
        ]
    predictions, scores = {mode: [] for mode in modes}, {mode: [] for mode in modes}
    for mode in modes:
        for index, (client, split, vector) in enumerate(zip(clients, splits, vectors[mode], strict=True)):
            adapter.set_vector(vector)
            holdout = adapter.transform(split.holdout_probabilities)
            threshold, calibration = empirical_alarm_threshold(
                holdout,
                split.holdout_labels,
                args.target_far,
                args.threshold_floor,
                threshold_min_benign_rows,
                getattr(args, "threshold_insufficient_policy", "floor"),
            )
            diagnostics["calibration"].append({
                "mode": mode,
                "client": index,
                "threshold_source": "CAL holdout",
                "cal_fit_rows": len(selected[index]),
                "cal_fit_available_rows": len(split.fit_labels),
                "cal_holdout_rows": len(split.holdout_labels),
                **calibration,
            })
            # Development diagnostic labels never cross into fitting or candidate selection.
            for role, p, labels in (
                ("cal_fit", split.fit_probabilities[selected[index]], split.fit_labels[selected[index]]),
                ("cal_holdout", split.holdout_probabilities, split.holdout_labels),
            ):
                transformed = adapter.transform(p)
                diagnostics["development_metrics"].append({
                    "mode": mode,
                    "client": index,
                    "role": role,
                    "rows": len(labels),
                    "base_log_loss": _log_loss(p, labels),
                    "candidate_log_loss": _log_loss(transformed, labels),
                })
            started = time.perf_counter()
            probabilities = adapter.transform(client.eval_probabilities)
            decision = hierarchical_decisions(probabilities, threshold)
            if observe_scores is not None:
                probabilities.setflags(write=False)
                observe_scores(mode, index, probabilities)
            predictions[mode].append(decision["predicted"])
            scores[mode].append(decision["attack_score"])
            diagnostics["postprocess"].append({
                "mode": mode,
                "client": index,
                "elapsed_seconds": time.perf_counter() - started,
                "rows": len(probabilities),
            })

    if argmax_modes:
        # A single global alarm threshold can suppress low-confidence rare
        # classes.  These matched argmax variants isolate model quality from
        # that decision rule; they are never used to fit or select the adapter.
        for source in modes:
            target = f"{source}_argmax"
            predictions[target], scores[target] = [], []
            for index, (client, split, vector) in enumerate(
                zip(clients, splits, vectors[source], strict=True)
            ):
                adapter.set_vector(vector)
                probabilities = adapter.transform(client.eval_probabilities)
                predictions[target].append(probabilities.argmax(1))
                scores[target].append(np.clip(1.0 - probabilities[:, 0], 0.0, 1.0))
                diagnostics["calibration"].append({
                    "mode": target,
                    "client": index,
                    "threshold_source": "argmax",
                    "threshold": None,
                    "cal_fit_rows": len(selected[index]),
                    "cal_fit_available_rows": len(split.fit_labels),
                    "cal_holdout_rows": len(split.holdout_labels),
                    "calibration_rule": "no alarm threshold; multiclass argmax",
                    "formal_conformal_guarantee": False,
                })
                diagnostics["postprocess"].append({
                    "mode": target,
                    "client": index,
                    "elapsed_seconds": 0.0,
                    "rows": len(probabilities),
                    "decision_rule": "argmax",
                })

    upload = 2 * len(clients) * adapter.dimension * 8
    aggregate = 2 * adapter.dimension * 8
    broadcast = len(clients) * adapter.dimension * 8
    for mode in modes:
        federated = mode in ("conditional_global", "conditional_full")
        diagnostics["communication"].append({
            "mode": mode,
            "parameter_count": adapter.dimension,
            "model_payload_bytes": adapter.dimension * 8,
            "round_uplink_payload_bytes": upload + aggregate if federated else 0,
            "round_downlink_payload_bytes": broadcast if federated else 0,
            "collector_sum_payload_bytes": aggregate if federated else 0,
            "round_payload_bytes": upload + aggregate + broadcast if federated else 0,
            "training_payload_bytes": args.rounds * (upload + aggregate + broadcast) if federated else 0,
            "reuses_training_of": (
                "conditional_full" if mode == "conditional_global" and "conditional_full" in modes else None
            ),
            "scope": (
                "Two uint64 field shares/client, two aggregate shares and float64 model broadcast; analytical"
                " numeric bytes; serialization/IPC separately in round receipts"
            ),
            "model_schema": SCHEMA,
            "feature_map_setup_payload_bytes": 0,
        })
    return predictions, scores, diagnostics
