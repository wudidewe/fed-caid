"""Predeclared development-only selection and controlled client class removal.

Selection reads pooled TRAIN/CAL support, never VAL/TEST outcomes. Target rows
are reassigned to donors; global training/calibration budgets are conserved.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from .utils import split_calibration


def _hash(values):
    return sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _file_hash(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def plan_targets(parts, names, count=3, requested=None, *, minimum_train=20, minimum_cal=20):
    if type(count) is not int or count < 1:
        raise ValueError("Controlled target count must be positive")
    train = np.bincount(parts["train"][1], minlength=len(names))
    cal = np.bincount(parts["cal"][1], minlength=len(names))
    eligible = [i for i in range(1, len(names)) if train[i] >= minimum_train and cal[i] >= minimum_cal]
    ordered = sorted(eligible, key=lambda i: (int(train[i]), i))
    if requested is not None:
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Controlled classes must be nonempty and distinct")
        if any(name not in names for name in requested):
            raise ValueError("Unknown controlled class for this dataset")
        selected = [names.index(name) for name in requested]
        if any(i not in eligible for i in selected):
            raise ValueError("Requested class lacks prespecified TRAIN/CAL support or is BENIGN")
    else:
        if len(ordered) < count:
            raise ValueError(f"Only {len(ordered)} eligible TRAIN/CAL classes for {count} targets")
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        selected = [ordered[i] for i in positions]
    return {
        "version": 1,
        "selection": (
            "TRAIN-support rank endpoints and evenly spaced ranks; CAL support eligibility only; no VAL/TEST"
            " selection"
        ),
        "minimum_train": minimum_train,
        "minimum_cal": minimum_cal,
        "requested_classes": requested,
        "class_order": list(names),
        "train_class_counts": train.tolist(),
        "cal_class_counts": cal.tolist(),
        "targets": [
            {
                "slot": slot,
                "class_index": int(i),
                "class": names[i],
                "train_rows": int(train[i]),
                "cal_rows": int(cal[i]),
            }
            for slot, i in enumerate(selected)
        ],
    }


def _remove_training_roles(
    parts,
    roster,
    target,
    receiver,
    seed,
    fit_fraction,
    absence_scope,
    grouped_cal,
    donor_fit_min,
    min_holdout_benign,
):
    """Qualify a controlled scenario using TRAIN/CAL alone, before VAL routing."""
    if not 0 <= receiver < len(roster) or target <= 0:
        raise ValueError("Invalid receiver or benign controlled target")
    if absence_scope not in ("train_and_cal", "cal_only"):
        raise ValueError("Controlled absence scope must be train_and_cal or cal_only")
    if type(donor_fit_min) is not int or donor_fit_min < 1:
        raise ValueError("Donor CAL-fit minimum must be a positive integer")
    result = [{role: indices.copy() for role, indices in client.items()} for client in roster]
    donors = [i for i in range(len(roster)) if i != receiver]
    if len(donors) < 2:
        raise ValueError("Controlled secure experiment needs at least three clients")
    receipt = {
        "receiver": receiver,
        "class_index": target,
        "absence_scope": (
            "receiver CAL only; TRAIN retains the target class"
            if absence_scope == "cal_only"
            else "receiver TRAIN and entire CAL (fit plus holdout); VAL is evaluation only"
        ),
        "reassignment": "removed rows redistributed round-robin to donor clients; no row duplication",
        "roles": {},
    }
    if absence_scope == "cal_only":
        train_labels = parts["train"][1]
        if not np.any(train_labels[result[receiver]["train"]] == target):
            raise ValueError("CAL-only receiver must have positive target TRAIN support before intervention")
        if not np.any(parts["cal"][1][result[receiver]["cal"]] == target):
            raise ValueError("CAL-only receiver must have positive target CAL support before intervention")
        receipt["roles"]["train"] = {
            "modified": False,
            "total_rows_before": int(sum(len(client["train"]) for client in result)),
            "total_rows_after": int(sum(len(client["train"]) for client in result)),
            "receiver_target_rows_before": int((train_labels[result[receiver]["train"]] == target).sum()),
            "receiver_target_rows_after": int((train_labels[result[receiver]["train"]] == target).sum()),
            "receiver_borrowed_non_target_rows": 0,
            "client_rows": [len(client["train"]) for client in result],
            "client_target_rows": [int((train_labels[client["train"]] == target).sum()) for client in result],
            "identity_sha256": [_hash(client["train"]) for client in result],
        }
    for role in ("cal",) if absence_scope == "cal_only" else ("train", "cal"):
        labels = parts[role][1]
        before = np.concatenate([client[role] for client in result])
        original = result[receiver][role]
        moved = original[labels[original] == target]
        kept = original[labels[original] != target]
        result[receiver][role] = kept
        for i, donor in enumerate(donors):
            result[donor][role] = np.concatenate((result[donor][role], moved[i :: len(donors)]))
        borrowed = []
        while len(result[receiver][role]) < 2:
            donor = next(
                (i for i in donors if len(result[i][role]) > 2 and np.any(labels[result[i][role]] != target)),
                None,
            )
            if donor is None:
                raise ValueError(f"Cannot retain two non-target receiver {role} rows")
            position = int(np.flatnonzero(labels[result[donor][role]] != target)[-1])
            index = result[donor][role][position]
            result[donor][role] = np.delete(result[donor][role], position)
            result[receiver][role] = np.append(result[receiver][role], index)
            borrowed.append(int(index))
        for i, client in enumerate(result):
            rng = np.random.default_rng(seed + 1711 + 31 * i + (0 if role == "train" else 1))
            client[role] = rng.permutation(client[role])
        after = np.concatenate([client[role] for client in result])
        if not np.array_equal(np.sort(before), np.sort(after)) or len(np.unique(after)) != len(after):
            raise RuntimeError("Controlled reassignment lost or duplicated row identities")
        if np.any(labels[result[receiver][role]] == target):
            raise RuntimeError("Target leaked into receiver TRAIN/CAL")
        receipt["roles"][role] = {
            "modified": True,
            "total_rows_before": len(before),
            "total_rows_after": len(after),
            "receiver_target_rows_before": len(moved),
            "receiver_target_rows_after": 0,
            "receiver_borrowed_non_target_rows": len(borrowed),
            "client_rows": [len(client[role]) for client in result],
            "client_target_rows": [int((labels[client[role]] == target).sum()) for client in result],
            "identity_sha256": [_hash(client[role]) for client in result],
        }
    if grouped_cal:
        from .revision_cal_split import cal_group_splits

        splits, audit = cal_group_splits(
            *parts["cal"], result, fit_fraction, seed, min_holdout_benign=min_holdout_benign
        )
        receipt["cal_group_split"] = audit
    else:
        splits = [
            split_calibration(len(client["cal"]), fit_fraction, seed + 335 + i)
            for i, client in enumerate(result)
        ]
    fit_counts = []
    for client, (fit, _) in zip(result, splits, strict=True):
        fit_counts.append(int((parts["cal"][1][client["cal"]][fit] == target).sum()))
    donor_fit = sum(fit_counts[i] for i in donors)
    if donor_fit < donor_fit_min:
        minimum = "five" if donor_fit_min == 5 else str(donor_fit_min)
        raise ValueError(
            f"Prespecified controlled target has fewer than {minimum} donor CAL-fit rows; no resampling"
            " allowed"
        )
    receipt.update(
        donor_cal_fit_rows=donor_fit,
        client_cal_fit_target_rows=fit_counts,
        donor_cal_fit_minimum=donor_fit_min,
        donor_train_rows=sum(receipt["roles"]["train"]["client_target_rows"][i] for i in donors),
    )
    return result, receipt


def choose_controlled_receiver(
    parts,
    roster,
    target,
    seed,
    fit_fraction,
    *,
    absence_scope="cal_only",
    preferred=0,
    grouped_cal=True,
    donor_fit_min=5,
    min_holdout_benign=0,
):
    """Prefer the declared receiver, then ascending client index, using TRAIN/CAL.

    Eligibility includes the actual deterministic post-reassignment CAL split.
    No validation/test rows or outcomes participate, and seeds are never retried.
    """
    if not 0 <= preferred < len(roster):
        raise ValueError("Invalid preferred controlled receiver")
    allowed_parts = {role: parts[role] for role in ("train", "cal")}
    allowed_roster = [{role: client[role] for role in ("train", "cal")} for client in roster]
    attempts = []
    for receiver in [preferred, *[i for i in range(len(roster)) if i != preferred]]:
        attempt = {
            "client": receiver,
            "train_target_rows_before": int((parts["train"][1][roster[receiver]["train"]] == target).sum()),
            "cal_target_rows_before": int((parts["cal"][1][roster[receiver]["cal"]] == target).sum()),
        }
        try:
            _, check = _remove_training_roles(
                allowed_parts,
                allowed_roster,
                target,
                receiver,
                seed,
                fit_fraction,
                absence_scope,
                grouped_cal,
                donor_fit_min,
                min_holdout_benign,
            )
        except ValueError as error:
            attempt.update(eligible=False, reason=str(error))
            attempts.append(attempt)
            continue
        attempt.update(eligible=True, donor_cal_fit_rows=check["donor_cal_fit_rows"])
        attempts.append(attempt)
        return receiver, {
            "method": (
                "preferred client then ascending index; TRAIN/CAL support and post-reassignment CAL"
                " feasibility only"
            ),
            "preferred_receiver": preferred,
            "selected_receiver": receiver,
            "absence_scope": absence_scope,
            "class_index": target,
            "val_test_used": False,
            "seed_retries": 0,
            "candidates_checked": attempts,
            "cal_group_assignment_sha256": check.get("cal_group_split", {}).get("assignment_sha256"),
        }
    raise ValueError("No eligible controlled receiver using TRAIN/CAL only: " + json.dumps(attempts))


def remove_receiver_class(
    parts,
    roster,
    target,
    receiver,
    seed,
    fit_fraction,
    absence_scope="train_and_cal",
    *,
    grouped_cal=False,
    donor_fit_min=5,
    min_holdout_benign=0,
):
    result, receipt = _remove_training_roles(
        parts,
        roster,
        target,
        receiver,
        seed,
        fit_fraction,
        absence_scope,
        grouped_cal,
        donor_fit_min,
        min_holdout_benign,
    )
    # Routing is predeclared and independent of the controlled class and labels.
    val_before = [_hash(client["val"]) for client in result]
    val_order = np.random.default_rng(seed + 2251).permutation(len(parts["val"][1]))
    for client, indices in zip(result, np.array_split(val_order, len(result)), strict=True):
        client["val"] = indices
    receipt["val_routing"] = {
        "method": (
            "label-blind equal blocks of seeded permutation; independent of target and all model outcomes"
        ),
        "seed": seed + 2251,
        "identity_sha256_before": val_before,
        "identity_sha256_after": [_hash(client["val"]) for client in result],
    }
    receipt.update(
        receiver_val_target_support=int((parts["val"][1][result[receiver]["val"]] == target).sum()),
        test_routes_unchanged=all(
            np.array_equal(result[i]["test"], roster[i]["test"]) for i in range(len(roster))
        ),
    )
    return result, receipt


def development_cells(parts, roster, predictions, target, receiver, names):
    y = parts["val"][1][roster[receiver]["val"]]
    positive = y == target
    support = int(positive.sum())
    rows = []
    for mode, values in predictions.items():
        predicted = np.asarray(values[receiver])[: len(y)]
        selected = predicted == target
        tp, count = int((positive & selected).sum()), int(selected.sum())
        benign = y == 0
        benign_support = int(benign.sum())
        false_alerts = int((benign & (predicted > 0)).sum())
        rows.append({
            "role": "val",
            "client": f"client_{receiver}",
            "class": names[target],
            "class_index": target,
            "mode": mode,
            "support": support,
            "true_positive": tp,
            "predicted": count,
            "receiver_benign_support": benign_support,
            "receiver_false_alerts": false_alerts,
            "receiver_far": false_alerts / benign_support if benign_support else None,
            "recall": tp / support if support else None,
            "precision": tp / count if count else (0.0 if support else None),
            "f1": 2 * tp / (support + count) if support else None,
            "status": "available" if support else "no_val_target_support",
        })
    return rows


class PredictionCache:
    """Per-role private numerical receipt; never publish raw predictions by default."""

    def __init__(self, directory, names, parts, roster, *, development_only):
        self.directory = Path(directory)
        self.arrays = {}
        self.metadata = {
            "version": 1,
            "private": True,
            "class_order": list(names),
            "development_only": development_only,
            "identity_scope": (
                "indices within effective per-case role arrays after any budget sampling, not provider row"
                " ids; source protocol audit and result sampling settings define reconstruction"
            ),
            "effective_role_rows": {role: len(pair[1]) for role, pair in parts.items()},
            "evaluation_order": ["val", "test"],
            "clients": len(roster),
        }
        audit = self.directory.parent.parent / "data_audit.json"
        if audit.is_file():
            self.metadata["data_audit_sha256"] = _file_hash(audit)
        for i, client in enumerate(roster):
            for role, indices in client.items():
                self.arrays[f"client_{i}_{role}_indices"] = np.asarray(indices)
                self.arrays[f"client_{i}_{role}_labels"] = parts[role][1][indices]

    def local(self, inputs):
        for i, client in enumerate(inputs):
            self.arrays[f"client_{i}_local_cal_probabilities"] = client.cal_probabilities
            self.arrays[f"client_{i}_local_eval_probabilities"] = client.eval_probabilities
            self.arrays[f"client_{i}_local_log_prior"] = client.log_prior
            split = getattr(client, "cal_split", None)
            if split is not None:
                fit, holdout = split
                self.arrays[f"client_{i}_cal_fit_positions"] = fit
                self.arrays[f"client_{i}_cal_holdout_positions"] = holdout

    def donor(self, index, cal, evaluation):
        self.arrays[f"client_{index}_donor_cal_probabilities"] = cal
        self.arrays[f"client_{index}_donor_eval_probabilities"] = evaluation

    def save(self, predictions, scores):
        self.metadata["modes"] = list(predictions)
        for mode, values in predictions.items():
            for i, value in enumerate(values):
                self.arrays[f"client_{i}_{mode}_prediction"] = value
                self.arrays[f"client_{i}_{mode}_attack_probability"] = scores[mode][i]
        path = self.directory / "predictions_private.npz"
        temporary = self.directory / "predictions_private.partial"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **self.arrays)
        temporary.replace(path)
        self.metadata.update(
            file=path.name,
            sha256=_file_hash(path),
            arrays={
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in self.arrays.items()
            },
        )
        meta = self.directory / "predictions_private.json"
        temporary = meta.with_suffix(".partial")
        temporary.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        temporary.replace(meta)
        return {
            "file": path.name,
            "sha256": self.metadata["sha256"],
            "manifest": meta.name,
            "manifest_sha256": _file_hash(meta),
            "private": True,
            "development_only": self.metadata["development_only"],
        }


def load_prediction_cache(directory):
    """Read only integrity-checked numeric arrays; pickle payloads are forbidden."""
    directory = Path(directory)
    meta = json.loads((directory / "predictions_private.json").read_text(encoding="utf-8"))
    if meta.get("version") != 1 or meta.get("file") != "predictions_private.npz":
        raise ValueError("Unsupported prediction cache manifest")
    path = directory / meta["file"]
    if _file_hash(path) != meta["sha256"]:
        raise ValueError("Prediction cache SHA256 mismatch")
    audit = directory.parent.parent / "data_audit.json"
    if "data_audit_sha256" in meta and (
        not audit.is_file() or _file_hash(audit) != meta["data_audit_sha256"]
    ):
        raise ValueError("Prediction cache source audit differs")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != set(meta["arrays"]):
            raise ValueError("Prediction cache array schema differs")
        arrays = {name: source[name] for name in source.files}
    for name, value in arrays.items():
        if (
            list(value.shape) != meta["arrays"][name]["shape"]
            or str(value.dtype) != meta["arrays"][name]["dtype"]
        ):
            raise ValueError("Prediction cache shape or dtype differs")
    return meta, arrays
