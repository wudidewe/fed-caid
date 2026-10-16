"""Shared model, preprocessing, calibration and evaluation methods."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional

# Preserve historical checkpoint lookups after the module rename.
sys.modules.setdefault("Utils", sys.modules[__name__])
core = sys.modules[__name__]

NEURAL_KINDS = ("ple_cross",)


class LowRankCross(nn.Module):
    """DCN-V2 form: z_next = z + z0 * (U V^T z + bias)."""

    def __init__(self, width, rank=16):
        super().__init__()
        if width < 1 or rank < 1 or rank > width:
            raise ValueError("Cross rank must be positive and no larger than input width")
        self.project_in = nn.Linear(width, rank, bias=False)
        self.project_out = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        # Identity start preserves the base model's initial predictions.
        nn.init.zeros_(self.project_out.weight)

    def forward(self, original, current):
        return current + original * (self.project_out(self.project_in(current)) + self.bias)


class CrossTabM(nn.Module):
    """Shared numerical embedding -> two rank16 crosses -> unchanged TabM."""

    def __init__(self, base):
        super().__init__()
        if base.num_module is None or base.cat_module is not None:
            raise ValueError("CrossTabM requires numeric PLE input only")
        self.base = base
        self.input_features, embedding_width = base.num_module.get_output_shape()
        self.width = self.input_features * embedding_width
        # Additional parameters must not change the base/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.cross_layers = nn.ModuleList([LowRankCross(self.width, 16) for _ in range(2)])

    def apply_cross(self, original):
        current = original
        for layer in self.cross_layers:
            current = layer(original, current)
        return current

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError("CrossTabM requires a 2D numerical feature matrix")
        embedded = self.base.num_module(x).flatten(1)
        hidden = self.base.backbone(self.base.ensemble_view(self.apply_cross(embedded)))
        return self.base.output(hidden)

    @torch.no_grad()
    def diagnostics(self, x):
        embedded = self.base.num_module(x).flatten(1)
        crossed = self.apply_cross(embedded)
        return {
            "layers": 2,
            "rank": 16,
            "embedded_width": self.width,
            "parameter_count": sum(p.numel() for p in self.cross_layers.parameters()),
            "out_projection_norms": [
                float(layer.project_out.weight.norm()) for layer in self.cross_layers
            ],
            "bias_norms": [float(layer.bias.norm()) for layer in self.cross_layers],
            "mean_relative_residual_norm": float(
                ((crossed - embedded).norm(dim=1) / (embedded.norm(dim=1) + 1e-8)).mean()
            ),
        }


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def probabilities(logp, temperature=1.0):
    z = np.asarray(logp, dtype=float) / temperature
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def nll(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def prediction_metrics(y, p):
    pred, labels = p.argmax(1), np.arange(p.shape[1])
    return {
        "accuracy": float((pred == y).mean()),
        "weighted_precision": float(
            precision_score(y, pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "nll": nll(y, p),
    }


def selection_key(metrics):
    prefix = "val_" if "val_weighted_f1" in metrics else ""
    return tuple(
        round(metrics[prefix + k], 8)
        for k in ("weighted_f1", "macro_f1", "accuracy", "weighted_precision")
    ) + (-metrics[prefix + "nll"],)


def log_prediction(model, x, device, batch=1024):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(torch.as_tensor(x[start : start + batch], device=device))
            logp = (
                logits.softmax(-1).mean(1).clamp_min(1e-12).log()
                if logits.ndim == 3
                else logits.log_softmax(-1)
            )
            result.append(logp.cpu().numpy())
    return np.concatenate(result)


class Transform:
    """TRAIN median; neural inputs additionally use TRAIN z-score and clip12."""

    def fit(self, x, neural=True):
        if x.ndim != 2 or not len(x) or not x.shape[1]:
            raise ValueError("Nonempty TRAIN matrix required")
        finite = np.where(np.isfinite(x), x, np.nan)
        self.median = np.nan_to_num(pd.DataFrame(finite).median().to_numpy(), nan=0).astype(
            np.float32
        )
        filled = np.where(np.isfinite(x), x, self.median).astype(np.float64)
        self.mean, self.scale = filled.mean(0), filled.std(0)
        self.scale[self.scale < 1e-6] = 1
        self.neural = neural
        return self

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.median):
            raise ValueError("Transform width differs")
        filled = np.where(np.isfinite(x), x, self.median)
        return (
            np.clip((filled - self.mean) / self.scale, -12, 12) if self.neural else filled
        ).astype(np.float32)

    def state(self):
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "neural": self.neural,
        }

    @classmethod
    def restore(cls, state):
        obj = cls()
        obj.median, obj.mean, obj.scale = (
            np.asarray(state[k]) for k in ("median", "mean", "scale")
        )
        obj.neural = state["neural"]
        return obj


def make_model(d, classes, kind, x=None, bins=None):
    if kind not in NEURAL_KINDS:
        raise ValueError("Unknown neural model")
    vendor = str(Path(__file__).resolve().parent / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from tabm import TabM
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings

    if bins is None:
        if x is None or x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("TRAIN-only finite PLE input required")
        q = torch.quantile(torch.as_tensor(x), torch.linspace(0, 1, 17), dim=0).T
        bins = []
        for column in q:
            edges = column.unique().contiguous()
            if len(edges) == 1:
                edges = torch.tensor(
                    [float(edges[0]) - 0.5, float(edges[0]) + 0.5], dtype=torch.float32
                )
            bins.append(edges)
    if len(bins) != d or any(
        len(b) < 2 or not torch.isfinite(b).all() or not (b.diff() > 0).all() for b in bins
    ):
        raise ValueError("Invalid saved PLE bins")
    model = TabM.make(
        n_num_features=d,
        d_out=classes,
        n_blocks=2,
        d_block=64,
        k=4,
        dropout=0.1,
        num_embeddings=PiecewiseLinearEmbeddings(
            bins, d_embedding=8, activation=False, version="B"
        ),
    )
    return CrossTabM(model), bins


def class_weights(y):
    y = np.asarray(y)
    if (
        y.ndim != 1
        or not len(y)
        or not np.issubdtype(y.dtype, np.integer)
        or not np.array_equal(np.unique(y), np.arange(y.max() + 1))
    ):
        raise ValueError("TRAIN labels must cover contiguous classes")
    counts = np.bincount(y)
    weights = counts.astype(float) ** -0.25
    return weights / np.sum(counts / len(y) * weights)


def classification_losses(logits, labels, weights=None):
    if logits.ndim == 2:
        logits = logits[:, None, :]
    b, k, c = logits.shape
    loss = (
        functional.cross_entropy(
            logits.reshape(b * k, c), labels[:, None].expand(-1, k).reshape(-1), reduction="none"
        )
        .reshape(b, k)
        .mean(1)
    )
    return loss if weights is None else loss * weights[labels]


def diagnostic_rows(y, cap=10000):
    rng = np.random.default_rng(123)
    if len(y) <= cap:
        return np.arange(len(y))
    counts = np.bincount(y)
    if cap < len(counts):
        raise ValueError("Diagnostic cap must cover all classes")
    minimum = np.minimum(counts, 256)
    if minimum.sum() > cap:
        minimum = np.minimum(counts, cap // len(counts))
    remaining = cap - int(minimum.sum())
    quotas = np.minimum(counts, minimum + np.floor(remaining * counts / len(y)).astype(int))
    while quotas.sum() > cap:
        quotas[np.argmax(quotas)] -= 1
    selected = []
    for c, n in enumerate(counts):
        selected.extend(rng.choice(np.flatnonzero(y == c), int(quotas[c]), replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def calibrate(logp, y):
    grid = np.unique(np.r_[1.0, np.geomspace(0.25, 4, 81)])
    temperature = float(min(grid, key=lambda t: nll(y, probabilities(logp, t))))
    p = probabilities(logp, temperature)
    benign = 1 - p[y == 0, 0]
    if not len(benign):
        raise ValueError("CAL benign support required")
    threshold = float(np.nextafter(np.quantile(benign, 0.99, method="higher"), np.inf))
    return {
        "temperature": temperature,
        "attack_threshold": threshold,
        "fit_role": "CAL",
        "cal_far": float((benign >= threshold).mean()),
    }


def client_partitions(parts, classes, clients, alpha, seed_value, cap):
    """Create fixed label-skew clients for simulation, not real vehicle identities."""
    rng = np.random.default_rng(seed_value)
    rates = rng.dirichlet(np.full(clients, alpha), size=classes)
    result = [{r: [] for r in parts} for _ in range(clients)]
    for role, (_, y) in parts.items():
        for c in range(classes):
            rows = rng.permutation(np.flatnonzero(y == c))
            cuts = np.cumsum(rng.multinomial(len(rows), rates[c]))[:-1]
            for i, indices in enumerate(np.split(rows, cuts)):
                result[i][role].append(indices)
    for client in result:
        for role, rows in client.items():
            rows = rng.permutation(np.concatenate(rows))
            limit = cap if role == "train" else max(64, cap // 3) if cap else 0
            if limit:
                rows = rows[:limit]
            if not len(rows):
                raise ValueError(
                    "Empty simulated client role; use more rows or larger Dirichlet alpha"
                )
            client[role] = rows
    return result


def synthetic_data(index):
    """Four tiny, different-feature-space fixtures; never research evidence."""
    rng = np.random.default_rng(1700 + index)
    classes, width = 3 + index % 2, 4 + index * 2
    parts = {}
    for role, n in (("train", 480), ("cal", 240), ("val", 240)):
        y = np.arange(n, dtype=np.int64) % classes
        x = rng.normal(size=(n, width)) + y[:, None] * 0.6
        parts[role] = (x.astype(np.float32), y)
    return (
        parts,
        ["BENIGN"] + [f"ATTACK_{c}" for c in range(1, classes)],
        [f"f{i}" for i in range(width)],
    )


def load_development(path):
    """Load only the pre-existing development roles, never TEST/confirmation."""
    from data_io import load_data, read_json

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        expected = {"class_names", "feature_names"} | {
            f"{xy}_{r}" for xy in ("X", "y") for r in ("train", "val", "cal")
        }
        if set(archive.files) != expected:
            raise ValueError(
                "Privacy input must contain exactly TRAIN/VAL/CAL; TEST/confirmation forbidden"
            )
    preparation = path.parent / "preparation.json"
    if preparation.exists():
        audit = read_json(preparation)
        if audit["data_sha256"] != sha(path):
            raise ValueError("Frozen development cache hash differs")
        identities = path.parent / "selected_train_identities.npz"
        if not identities.exists() or audit["identity_sha256"] != sha(identities):
            raise ValueError("Frozen TRAIN identity cache differs")
    return load_data(path)


def cached_train_log_prior(y, classes):
    """Local TRAIN only; absent classes receive no artificial prior boost."""
    y = np.asarray(y)
    if (
        type(classes) is not int
        or classes < 2
        or y.ndim != 1
        or not np.issubdtype(y.dtype, np.integer)
        or not len(y)
        or (y < 0).any()
        or (y >= classes).any()
    ):
        raise ValueError("Nonempty integer TRAIN labels in the public class space required")
    counts = np.bincount(y, minlength=classes)
    if len(counts) != classes or counts.sum() == 0:
        raise ValueError("Nonempty TRAIN labels in the public class space required")
    prior = (counts + 0.5) / (counts.sum() + 0.5 * classes)
    return np.where(counts > 0, np.log(prior), 0.0).astype(np.float64)


def split_calibration(rows, fraction=0.5, seed_value=42):
    if rows < 1 or not 0 < fraction < 1:
        raise ValueError("Nonempty CAL and a split fraction between zero and one required")
    order = np.random.default_rng(seed_value).permutation(rows)
    count = min(rows - 1, max(1, int(rows * fraction))) if rows > 1 else 1
    return order[:count], order[count:]


def empirical_alarm_threshold(p, y, target_far=0.01, floor=0.5):
    """Benign CAL holdout quantile; empirical, not a conformal guarantee."""
    if not 0 < target_far < 1 or not 0 <= floor <= 1:
        raise ValueError("Invalid target FAR or threshold floor")
    y = np.asarray(y)
    q = np.clip(np.asarray(p)[:, 1:].sum(1), 0.0, 1.0)
    if len(q) != len(y):
        raise ValueError("Calibration rows differ")
    benign = q[y == 0]
    threshold = (
        max(floor, float(np.quantile(benign, 1 - target_far, method="higher")))
        if len(benign)
        else 1.0
    )
    return threshold, {
        "threshold": threshold,
        "target_far": target_far,
        "benign_holdout_rows": len(benign),
        "threshold_floor": floor,
        "calibration_far": float((benign > threshold).mean()) if len(benign) else None,
        "calibration_rule": "empirical_benign_holdout_quantile_higher",
        "formal_conformal_guarantee": False,
        "no_benign_fallback": not bool(len(benign)),
    }


def subtype_joint_probabilities(p, attack_subtype_scores=None):
    """Joint probabilities for NLL/reporting; binary decisions retain original q."""
    if attack_subtype_scores is None:
        return p
    scores = np.asarray(attack_subtype_scores, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if (
        scores.shape != (len(p), p.shape[1] - 1)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
    ):
        raise ValueError("Finite nonnegative attack subtype scores with matching classes required")
    total = scores.sum(1, keepdims=True)
    conditional = np.divide(
        scores, total, out=np.full_like(scores, 1 / scores.shape[1]), where=total > 0
    )
    joint = p.copy()
    joint[:, 1:] = p[:, 1:].sum(1, keepdims=True) * conditional
    return joint


def hierarchical_decisions(p, threshold=0.5, attack_subtype_scores=None):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("Finite multiclass probabilities required")
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), (len(p),))
    if not np.isfinite(threshold).all() or ((threshold < 0) | (threshold > 1)).any():
        raise ValueError("Thresholds must be finite probabilities")
    q = np.clip(p[:, 1:].sum(1), 0.0, 1.0)
    alert = q > threshold
    scores = (
        p[:, 1:]
        if attack_subtype_scores is None
        else np.asarray(attack_subtype_scores, dtype=np.float64)
    )
    if scores.shape != p[:, 1:].shape or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Matching finite nonnegative subtype scores required")
    subtype = 1 + scores.argmax(1)
    return {
        "attack_score": q,
        "attack_alert": alert,
        "subtype": subtype,
        "predicted": np.where(alert, subtype, 0),
        "threshold": threshold,
    }


def risk_metrics(y, p, names=None, threshold=0.5, attack_subtype_scores=None):
    y = np.asarray(y)
    q, attack = p[:, 1:].sum(1), y > 0
    joint = subtype_joint_probabilities(p, attack_subtype_scores)
    argmax_metrics = core.prediction_metrics(y, joint)
    result = {
        "nll": argmax_metrics["nll"],
        **{"argmax_" + k: argmax_metrics[k] for k in ("accuracy", "weighted_f1", "macro_f1")},
    }
    result.update(
        binary_brier=float(np.mean((q - attack) ** 2)),
        binary_nll=float(
            -np.mean(
                np.where(attack, np.log(np.clip(q, 1e-12, 1)), np.log(np.clip(1 - q, 1e-12, 1)))
            )
        ),
        far_at_half=float((q[~attack] > 0.5).mean()) if (~attack).any() else None,
        fnr_at_half=float((q[attack] <= 0.5).mean()) if attack.any() else None,
        binary_auc=float(roc_auc_score(attack, q)) if len(np.unique(attack)) == 2 else None,
    )
    groups = np.minimum((q * 10).astype(int), 9)
    result["binary_ece_10"] = float(
        sum(
            np.abs(q[groups == i].mean() - attack[groups == i].mean()) * (groups == i).mean()
            for i in range(10)
            if (groups == i).any()
        )
    )
    predicted = hierarchical_decisions(p, threshold, attack_subtype_scores)["predicted"]
    argmax = joint.argmax(1)
    result.update(
        rows=len(y),
        far=float((predicted[~attack] > 0).mean()) if (~attack).any() else None,
        fnr=float((predicted[attack] == 0).mean()) if attack.any() else None,
        far_argmax=float((argmax[~attack] > 0).mean()) if (~attack).any() else None,
        fnr_argmax=float((argmax[attack] == 0).mean()) if attack.any() else None,
        false_positive_count=int(((~attack) & (predicted > 0)).sum()),
        false_negative_count=int((attack & (predicted == 0)).sum()),
        wrong_attack_subtype_count=int((attack & (predicted > 0) & (predicted != y)).sum()),
    )
    names = names or [str(i) for i in range(p.shape[1])]
    report = classification_report(
        y,
        predicted,
        labels=np.arange(p.shape[1]),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    result.update(
        accuracy=float((predicted == y).mean()),
        weighted_precision=float(report["weighted avg"]["precision"]),
        macro_precision=float(report["macro avg"]["precision"]),
        weighted_f1=float(report["weighted avg"]["f1-score"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        decision_rule="total_attack_probability_then_attack_subtype",
    )
    result["per_class"] = {name: report[name] for name in names}
    result["confusion_matrix"] = confusion_matrix(
        y, predicted, labels=np.arange(p.shape[1])
    ).tolist()
    result["class_names"] = names
    return result
