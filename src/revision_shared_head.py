"""Matched TRAIN-only linear heads on one public, stateless feature map."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from hashlib import sha256
import math
import time
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from .revision_baseline import _epoch_schedule
from .secure_aggregation import PRIME, SecureAggregationSession, _clip_quantize


MODEL_SCHEMA = "public_bounded_signed_log_linear_v1"
LOCAL, FEDERATED, DONOR = "head_local", "head_federated", "head_donor"
SINGLE_DONOR = "head_donor_single"
MODES = (LOCAL, FEDERATED, DONOR)
MULTICLASS, ATTACK_ONLY, BALANCED_ATTACK = "multiclass", "attack_only", "balanced_attack"
HEAD_OBJECTIVES = (MULTICLASS, ATTACK_ONLY, BALANCED_ATTACK)


def _public_features(raw, device):
    """Use a declared transform with a common column order and no fitted state."""
    features = torch.as_tensor(raw, dtype=torch.float32, device=device)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.sign() * features.abs().log1p().div_(8.0).clamp_max_(1.0)


def _parameters(state, device):
    weights = torch.tensor(state, dtype=torch.float32, device=device)
    return weights[:, :-1], weights[:, -1]


def head_probabilities(state, features, *, device="cpu", batch_size=65_536):
    """Score ordered feature rows using one frozen head and no labels."""
    state, features = np.asarray(state), np.asarray(features)
    if (
        state.ndim != 2
        or min(state.shape) < 2
        or state.dtype.kind != "f"
        or not np.isfinite(state).all()
        or features.ndim != 2
        or features.shape[1] != state.shape[1] - 1
        or features.dtype.kind not in "iuf"
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("Expected a finite shared head and aligned feature rows")
    weights, bias = _parameters(state, device)
    result = np.empty((len(features), len(state)), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            rows = slice(start, start + batch_size)
            logits = F.linear(_public_features(features[rows], device), weights, bias)
            result[rows] = logits.to(torch.float64).softmax(dim=1).cpu().numpy()
    return result


def _local_delta(state, features, labels, settings, client, start_epoch, epochs, attack_rows, attack_weights):
    row_count = len(labels) if attack_rows is None else len(attack_rows)
    if not row_count:
        return np.zeros(state.size, dtype=np.float64)
    weights, bias = _parameters(state, settings["device"])
    weights.requires_grad_(True)
    bias.requires_grad_(True)
    class_weights = (
        None
        if attack_weights is None
        else torch.as_tensor(attack_weights, dtype=torch.float32, device=settings["device"])
    )
    for epoch in range(start_epoch, start_epoch + epochs):
        order = np.random.default_rng(np.random.SeedSequence([settings["seed"], client, epoch])).permutation(
            row_count
        )
        if attack_rows is not None:
            order = attack_rows[order]
        for start in range(0, len(order), settings["batch_size"]):
            chosen = order[start : start + settings["batch_size"]]
            x = _public_features(features[chosen], settings["device"])
            y = torch.as_tensor(labels[chosen], dtype=torch.int64, device=settings["device"])
            if attack_rows is None:
                loss = F.cross_entropy(F.linear(x, weights, bias), y)
            else:
                logits, target = F.linear(x, weights[1:], bias[1:]), y - 1
                if class_weights is None:
                    loss = F.cross_entropy(logits, target)
                else:
                    per_row = F.cross_entropy(logits, target, reduction="none")
                    loss = (per_row * class_weights[target]).mean()
            grad_weights, grad_bias = torch.autograd.grad(loss, (weights, bias))
            with torch.no_grad():
                weights.sub_(grad_weights, alpha=settings["learning_rate"])
                bias.sub_(grad_bias, alpha=settings["learning_rate"])
    updated = torch.cat((weights.detach(), bias.detach()[:, None]), dim=1).cpu().numpy()
    if not np.isfinite(updated).all():
        raise FloatingPointError("Nonfinite shared-head update")
    return (updated.astype(np.float64) - state).ravel()


def _communication(mode, state, client_count, rounds, *, donor_control=False):
    parameters = state.size
    if mode == FEDERATED:
        upload, aggregate = 2 * client_count * parameters * 8, 2 * parameters * 8
        broadcast = client_count * state.nbytes
        delivery = 0
        scope = (
            "Two uint64 shares per client, two aggregate shares, and float32 model broadcast in local"
            " simulation"
        )
    elif mode == DONOR or donor_control:
        upload, aggregate, broadcast = client_count * state.size * 8, 0, client_count * state.nbytes
        delivery = state.nbytes
        scope = (
            "Plaintext donor control: int64 fixed-point updates, float32 donor broadcasts, and one final"
            " receiver delivery"
        )
    else:
        upload, aggregate, broadcast, delivery = 0, 0, 0, 0
        scope = "Local training; no inter-client model exchange"
    per_round = upload + aggregate + broadcast
    return {
        "mode": mode,
        "parameter_count": parameters,
        "model_payload_bytes": state.nbytes,
        "round_uplink_payload_bytes": upload,
        "collector_sum_payload_bytes": aggregate,
        "round_downlink_payload_bytes": broadcast,
        "round_payload_bytes": per_round,
        "training_payload_bytes": rounds * per_round,
        "receiver_delivery_payload_bytes": delivery,
        "total_payload_bytes": rounds * per_round + delivery,
        "feature_map_setup_payload_bytes": 0,
        "scope": scope,
        "model_schema": MODEL_SCHEMA,
    }


def _validate_donor_subsets(donor_subsets, receiver, clients):
    if donor_subsets is None:
        return {}
    if not isinstance(donor_subsets, Mapping):
        raise ValueError("Donor subsets must map unique donor mode names to ordered client IDs")
    normalized = {}
    for mode, donors in donor_subsets.items():
        if (
            not isinstance(mode, str)
            or not mode.startswith(DONOR + "_")
            or mode == DONOR + "_"
            or not mode.isascii()
            or not mode.isidentifier()
            or not isinstance(donors, (tuple, list))
            or not donors
            or receiver is None
            or any(type(i) is not int or not 0 <= i < clients or i == receiver for i in donors)
            or len(set(donors)) != len(donors)
            or (mode == SINGLE_DONOR and len(donors) != 1)
        ):
            raise ValueError("A donor subset needs a valid mode and distinct non-receiver client IDs")
        normalized[mode] = tuple(donors)
    return normalized


def fit_shared_heads(train_parts, classes, args, *, receiver=None, modes=None, donor_subsets=None):
    """Fit local, federated, and optional plaintext donor heads using TRAIN only.

    The local and federated branches use the same per-client TRAIN visit order,
    number of SGD steps, minibatch size, initialization, and round clipping and
    quantization. Federation enforces equal-client mean aggregation. The donor
    branch omits the declared receiver and uses the same round update in plaintext.
    Explicit donor subsets use that same fit with only the declared contributors.
    All fitted states are returned in the original inference client order.

    Both attack-only objectives train on the same attack rows and logits. The
    balanced objective gives each locally observed attack class equal total
    loss weight per client epoch. A benign-only client sends a zero update
    in each round and counts toward the equal-client mean.
    """
    train = tuple((np.asarray(x), np.asarray(y)) for x, y in train_parts)
    subsets = _validate_donor_subsets(donor_subsets, receiver, len(train))
    defaults = MODES if receiver is not None else MODES[:2]
    modes = tuple(modes) if modes is not None else (*defaults, *subsets)
    if (
        not train
        or type(classes) is not int
        or classes < 2
        or not modes
        or len(set(modes)) != len(modes)
        or set(modes) - (set(MODES) | subsets.keys())
        or (receiver is not None and (type(receiver) is not int or not 0 <= receiver < len(train)))
        or (FEDERATED in modes and len(train) < 3)
        or (DONOR in modes and (receiver is None or len(train) < 2))
    ):
        raise ValueError("Invalid class space, client roster, receiver, or head modes")
    width = train[0][0].shape[1] if train[0][0].ndim == 2 else 0
    if not width or any(
        y.ndim != 1
        or x.ndim != 2
        or x.shape != (len(y), width)
        or x.dtype.kind not in "iuf"
        or y.dtype.kind not in "iu"
        or len(y) == 0
        or (y < 0).any()
        or (y >= classes).any()
        for x, y in train
    ):
        raise ValueError(
            "Expected nonempty TRAIN sets with the same numerical feature width and valid labels"
        )

    rounds = args.rounds
    epochs = getattr(args, "head_epochs", None)
    schedule = _epoch_schedule(rounds if epochs is None else epochs, rounds)
    clip, scale = getattr(args, "clip", 0.1), getattr(args, "secure_scale", 1_000_000)
    head_seed = getattr(args, "head_seed", None)
    head_batch_size = getattr(args, "head_batch_size", None)
    objective = getattr(args, "head_objective", MULTICLASS)
    settings = {
        "seed": args.seed if head_seed is None else head_seed,
        "device": str(args.device),
        "batch_size": args.batch_size if head_batch_size is None else head_batch_size,
        "learning_rate": getattr(args, "head_lr", 0.05),
        "objective": objective,
    }
    if (
        type(objective) is not str
        or objective not in HEAD_OBJECTIVES
        or type(settings["seed"]) is not int
        or settings["seed"] < 0
        or type(settings["batch_size"]) is not int
        or settings["batch_size"] < 1
        or isinstance(settings["learning_rate"], bool)
        or not math.isfinite(settings["learning_rate"])
        or settings["learning_rate"] <= 0
        or type(scale) is not int
        or scale < 1
        or isinstance(clip, bool)
        or not math.isfinite(clip)
        or clip <= 0
        or not math.isfinite(clip * scale)
        or clip * scale >= (1 << 52)
        or math.ceil(clip * scale) * len(train) >= PRIME // 2
    ):
        raise ValueError("Invalid shared-head training or fixed-point settings")
    support = [np.bincount(y.astype(np.int64, copy=False), minlength=classes) for _, y in train]
    attack_rows, attack_weights = [None] * len(train), [None] * len(train)
    row_counts = [len(y) for _, y in train]
    if objective != MULTICLASS:
        attack_rows = [np.flatnonzero(y != 0) for _, y in train]
        row_counts = [len(rows) for rows in attack_rows]
    if objective == BALANCED_ATTACK:
        for client, (count, support_by_class) in enumerate(zip(row_counts, support, strict=True)):
            attack_support = support_by_class[1:]
            observed = attack_support > 0
            weights = np.zeros(classes - 1, dtype=np.float32)
            if count:
                weights[observed] = count / (observed.sum() * attack_support[observed])
            attack_weights[client] = weights
    steps_per_epoch = [math.ceil(count / settings["batch_size"]) for count in row_counts]
    initial = np.zeros((classes, width + 1), dtype=np.float32)
    states, performance, communication, history = {}, [], [], []
    all_clients = tuple(range(len(train)))
    donate = {DONOR: tuple(i for i in all_clients if i != receiver), **subsets}
    for mode in modes:
        donor_control = mode in donate
        active = donate.get(mode, all_clients)
        donor_receipt = {"active_donor_clients": list(active)} if mode in subsets else {}
        rows_per_epoch = sum(row_counts[i] for i in active)
        mode_steps_per_epoch = sum(steps_per_epoch[i] for i in active)
        empty_clients = sum(row_counts[i] == 0 for i in active)
        ids = tuple(f"client_{i}" for i in active)
        local_states = [initial.copy() for _ in train] if mode == LOCAL else None
        shared_state = initial.copy()
        started = time.perf_counter()
        context = (
            SecureAggregationSession(
                ids, initial.size, clip=clip, scale=scale, timeout=getattr(args, "secure_timeout", 15.0)
            )
            if mode == FEDERATED
            else nullcontext()
        )
        with context as collector:
            collector_startup = time.perf_counter() - started if collector is not None else 0.0
            session = uuid4().hex
            start_epoch = 0
            for round_index, local_epochs in enumerate(schedule, start=1):
                round_start = time.perf_counter()
                contract = (
                    collector.start_round(session, round_index, sha256(shared_state.tobytes()).hexdigest())
                    if collector is not None
                    else None
                )
                donor_update = np.zeros(initial.size, dtype=np.int64) if donor_control else None
                update_norms, fit_seconds = [], 0.0
                for client, client_id in zip(active, ids, strict=True):
                    state = local_states[client] if mode == LOCAL else shared_state
                    fit_started = time.perf_counter()
                    delta = _local_delta(
                        state,
                        *train[client],
                        settings,
                        client,
                        start_epoch,
                        local_epochs,
                        attack_rows[client],
                        attack_weights[client],
                    )
                    fit_seconds += time.perf_counter() - fit_started
                    update_norms.append(float(np.linalg.norm(delta)))
                    if collector is not None:
                        collector.submit(client_id, delta, contract=contract)
                    else:
                        encoded = np.array(_clip_quantize(delta, initial.size, clip, scale), dtype=np.int64)
                        if mode == LOCAL:
                            local_states[client] = (state + encoded.reshape(initial.shape) / scale).astype(
                                np.float32
                            )
                        else:
                            donor_update += encoded

                receipt = {}
                if collector is not None:
                    mean, receipt = collector.aggregate()
                    shared_state = (shared_state + mean.reshape(initial.shape)).astype(np.float32)
                elif donor_control:
                    mean = donor_update / (scale * len(active))
                    shared_state = (shared_state + mean.reshape(initial.shape)).astype(np.float32)
                history.append({
                    "mode": mode,
                    "round": round_index,
                    "local_epochs": local_epochs,
                    "training_row_visits": local_epochs * rows_per_epoch,
                    "optimizer_steps": local_epochs * mode_steps_per_epoch,
                    "local_fit_seconds": fit_seconds,
                    "round_elapsed_seconds": time.perf_counter() - round_start,
                    "clipped_updates": sum(norm > clip for norm in update_norms),
                    "participating_clients": len(active),
                    "participating_clients_without_training_rows": empty_clients,
                    "raw_update_l2_min": min(update_norms),
                    "raw_update_l2_mean": sum(update_norms) / len(update_norms),
                    "raw_update_l2_max": max(update_norms),
                    **donor_receipt,
                    **receipt,
                })
                start_epoch += local_epochs
        selected = local_states if mode == LOCAL else [shared_state] * len(train)
        for state in selected:
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite fitted shared head")
            state.setflags(write=False)
        states[mode] = tuple(selected)
        by_client = [start_epoch * count if i in active else 0 for i, count in enumerate(row_counts)]
        performance.append({
            "mode": mode,
            "stage": "train_shared_head",
            "training_seconds": time.perf_counter() - started,
            "collector_startup_seconds": collector_startup,
            "training_row_visits": sum(by_client),
            "training_row_visits_by_client": by_client,
            "receiver_train_row_visits": None if receiver is None else by_client[receiver],
            "optimizer_steps": start_epoch * mode_steps_per_epoch,
            **donor_receipt,
        })
        communication.append({
            **_communication(mode, initial, len(active), rounds, donor_control=donor_control), **donor_receipt
        })

    objective_description, optimization_scope = {
        MULTICLASS: ("multiclass cross entropy", "All TRAIN rows and class logits."),
        ATTACK_ONLY: (
            "unweighted attack-only cross entropy",
            "Attack rows and attack logits only. Each row has unit weight.",
        ),
        BALANCED_ATTACK: (
            "locally class-balanced attack-only cross entropy",
            (
                "Attack rows and attack logits only. Local inverse class frequencies are normalized to a row"
                " mean of one."
            ),
        ),
    }[objective]
    subset_receipt = {mode: list(subsets[mode]) for mode in modes if mode in subsets}
    return {
        "states": states,
        "performance": performance,
        "communication": communication,
        "rounds": history,
        "model_spec": {
            "model_schema": MODEL_SCHEMA,
            "feature_map": "sign(x) * min(log1p(abs(x))/8, 1); nonfinite inputs mapped to zero",
            "width": width,
            "classes": classes,
            "parameter_count": initial.size,
            "epochs_by_round": schedule,
            "settings": settings,
            "clip": clip,
            "secure_scale": scale,
            "initialization": "zero weights and intercept",
            "objective": objective_description,
            "optimization_scope": optimization_scope,
            "local_train_class_support": [counts.tolist() for counts in support],
            "provenance_scope": (
                "Local class supports and raw update norms are single-host diagnostics outside the collector"
                " protocol."
            ),
            "aggregation": "equal-client mean of clipped fixed-point updates",
            "train_scope": (
                "TRAIN only. Donor control excludes the receiver from fitting and uses plaintext aggregation."
            ),
            **({"donor_subsets": subset_receipt} if subset_receipt else {}),
        },
    }
