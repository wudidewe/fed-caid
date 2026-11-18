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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, roc_auc_score
from torch import nn
from torch.nn import functional

from .vendor.rtdl_num_embeddings import PiecewiseLinearEmbeddings
from .vendor.tabm import TabM

# Preserve historical checkpoint lookups after the module rename.
sys.modules.setdefault('Utils', sys.modules[__name__])
core = sys.modules[__name__]

NEURAL_KINDS = ('ple_cross',)
DETECTOR_OBJECTIVES = ('prior_adjusted', 'class_balanced')
CLASS_BALANCE_POWER = 0.25
CLASS_BALANCE_MAX_RATIO = 8.0
PLE_BOUNDARY_POLICIES = ('clip_to_train_edges', 'extrapolate')
PLE_BOUNDARY_VERSION = 'train_edges_v1'


class LowRankCross(nn.Module):
    """DCN-V2 form: z_next = z + z0 * (U V^T z + bias)."""

    def __init__(self, width, rank=16):
        super().__init__()
        if width < 1 or rank < 1 or rank > width:
            raise ValueError('Cross rank must be positive and no larger than input width')
        self.project_in = nn.Linear(width, rank, bias=False)
        self.project_out = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        # Identity start preserves the base model's initial predictions.
        nn.init.zeros_(self.project_out.weight)

    def forward(self, original, current):
        """Apply one low rank cross update to the current features using the original feature tensor."""
        return current + original * (self.project_out(self.project_in(current)) + self.bias)


class CrossTabM(nn.Module):
    """Shared numerical embedding -> two rank16 crosses -> unchanged TabM."""

    def __init__(self, base, bins=None, *, ple_boundary_policy='clip_to_train_edges'):
        super().__init__()
        if base.num_module is None or base.cat_module is not None:
            raise ValueError('CrossTabM requires numeric PLE input only')
        self.base = base
        self.input_features, embedding_width = base.num_module.get_output_shape()
        self.width = self.input_features * embedding_width
        if ple_boundary_policy not in PLE_BOUNDARY_POLICIES:
            raise ValueError('Unknown PLE boundary policy')
        if bins is None:
            raise ValueError('TRAIN-fitted PLE bins required for boundary handling')
        self.ple_boundary_policy = ple_boundary_policy
        # Bounds are derived from the separately saved TRAIN bins. Nonpersistent
        # buffers preserve strict loading of historical checkpoints without new keys.
        self.register_buffer('ple_train_lower', torch.stack([b[0] for b in bins]), persistent=False)
        self.register_buffer('ple_train_upper', torch.stack([b[-1] for b in bins]), persistent=False)
        self.register_buffer(
            'ple_min_bin_width',
            torch.stack([b.diff().min() for b in bins]),
            persistent=False,
        )
        # Additional parameters must not change the base/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.cross_layers = nn.ModuleList([LowRankCross(self.width, 16) for _ in range(2)])

    def apply_cross(self, original):
        """Apply the cross layers to the original embedded features and return the resulting tensor."""
        current = original
        for layer in self.cross_layers:
            current = layer(original, current)
        return current

    def prepare_ple_input(self, x):
        """Return numerical input after applying the configured boundary policy and TRAIN bin limits."""
        if self.ple_boundary_policy == 'clip_to_train_edges':
            return torch.maximum(torch.minimum(x, self.ple_train_upper), self.ple_train_lower)
        return x

    def forward(self, x):
        """Return per member class logits for a matrix of numerical feature rows."""
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError('CrossTabM requires a 2D numerical feature matrix')
        embedded = self.base.num_module(self.prepare_ple_input(x)).flatten(1)
        hidden = self.base.backbone(self.base.ensemble_view(self.apply_cross(embedded)))
        return self.base.output(hidden)

    @torch.no_grad()
    def diagnostics(self, x):
        """Return cross layer size, parameter norms and mean relative residual for the supplied input without
        gradients.
        """
        embedded = self.base.num_module(self.prepare_ple_input(x)).flatten(1)
        crossed = self.apply_cross(embedded)
        return {
            'layers': 2,
            'rank': 16,
            'embedded_width': self.width,
            'parameter_count': sum(p.numel() for p in self.cross_layers.parameters()),
            'out_projection_norms': [float(layer.project_out.weight.norm()) for layer in self.cross_layers],
            'bias_norms': [float(layer.bias.norm()) for layer in self.cross_layers],
            'mean_relative_residual_norm': float(
                ((crossed - embedded).norm(dim=1) / (embedded.norm(dim=1) + 1e-8)).mean()
            ),
        }


class TrainingFiniteMonitor:
    """Accumulate loss/gradient checks on-device; synchronize at progress boundaries."""

    def __init__(self, device, context):
        self.finite = torch.ones((), dtype=torch.bool, device=device)
        self.max_gradient_norm = torch.zeros((), dtype=torch.float64, device=device)
        self.context = context

    def observe(self, loss, gradient_norm):
        """Update the finiteness flag and maximum gradient norm from the supplied loss and norm tensors."""
        self.finite.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(gradient_norm.detach()))
        self.max_gradient_norm = torch.maximum(self.max_gradient_norm, gradient_norm.detach().double())

    def check(self):
        """Raise FloatingPointError if any observed loss or gradient norm was nonfinite."""
        if not bool(self.finite):
            raise FloatingPointError(f'Nonfinite {self.context} loss or gradient')


@torch.no_grad()
def ple_input_diagnostics(model, x, *, max_rows=4096, batch_size=512):
    """Audit a deterministic feature-only sample; never fit bounds on this input.

    The sampled maximum is a diagnostic, not a guaranteed maximum over all rows.
    Selecting spaced row positions consumes no RNG and cannot change training.
    """
    if not isinstance(model, CrossTabM):
        return {'available': False, 'reason': 'model does not expose CrossTabM PLE'}
    if max_rows < 1 or batch_size < 1:
        raise ValueError('Positive PLE diagnostic sample and batch sizes required')
    indices = np.linspace(0, len(x) - 1, min(len(x), max_rows), dtype=np.int64)
    device = model.ple_train_lower.device
    below = above = rows_outside = 0
    maximum = 0.0
    all_finite = True
    for start in range(0, len(indices), batch_size):
        values = torch.as_tensor(np.asarray(x)[indices[start : start + batch_size]], device=device)
        low, high = values < model.ple_train_lower, values > model.ple_train_upper
        below += int(low.sum())
        above += int(high.sum())
        rows_outside += int((low | high).any(dim=1).sum())
        encoded = model.base.num_module.impl(model.prepare_ple_input(values))
        all_finite = all_finite and bool(torch.isfinite(encoded).all())
        maximum = max(maximum, float(encoded.abs().max()))
    count = len(indices)
    return {
        'available': True,
        'ple_boundary_policy': model.ple_boundary_policy,
        'ple_boundary_version': PLE_BOUNDARY_VERSION,
        'bounds_source': 'TRAIN-fitted PLE bin edges; no evaluation refitting',
        'train_lower': model.ple_train_lower.cpu().tolist(),
        'train_upper': model.ple_train_upper.cpu().tolist(),
        'minimum_bin_width_by_feature': model.ple_min_bin_width.cpu().tolist(),
        'minimum_bin_width': float(model.ple_min_bin_width.min()),
        'input_rows': len(x),
        'sample_rows': count,
        'sampling': 'deterministic evenly spaced row positions, at most 4096 by default',
        'sample_values_below_train_edges': below,
        'sample_values_above_train_edges': above,
        'sample_rows_outside_train_edges': rows_outside,
        'sample_outside_row_fraction': rows_outside / count if count else None,
        'sample_encoding_abs_max': maximum if count and all_finite else None,
        'sample_encoding_all_finite': all_finite,
    }


def sha(path):
    """Return the hexadecimal SHA256 digest of the file at the supplied path."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    """Write the value to the given path as UTF-8 JSON and reject nonfinite numbers."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seed(seed_value=42):
    """Set the Python, NumPy and PyTorch random states and select deterministic cuDNN settings and CPU thread count.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def probabilities(logp, temperature=1.0):
    """Return rowwise softmax probabilities from log scores scaled by the supplied temperature."""
    z = np.asarray(logp, dtype=float) / temperature
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def nll(y, p):
    """Return the mean negative log likelihood of the integer labels under the supplied class probabilities."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def prediction_metrics(y, p):
    """Return accuracy, weighted and macro precision and F1, and negative log likelihood for labels and probabilities.
    """
    pred, labels = p.argmax(1), np.arange(p.shape[1])
    return {
        'accuracy': float((pred == y).mean()),
        'weighted_precision': float(precision_score(y, pred, labels=labels, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, labels=labels, average='weighted', zero_division=0)),
        'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
        'nll': nll(y, p),
    }


def selection_key(metrics):
    """Return the model selection tuple from validation metrics when present, or from unprefixed metrics otherwise.
    """
    prefix = 'val_' if 'val_weighted_f1' in metrics else ''
    return tuple(
        round(metrics[prefix + k], 8) for k in ('weighted_f1', 'macro_f1', 'accuracy', 'weighted_precision')
    ) + (-metrics[prefix + 'nll'],)


def log_prediction(model, x, device, batch=1024):
    """Set the model to evaluation mode and return log class probabilities for input rows scored in device batches.
    """
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(torch.as_tensor(x[start : start + batch], device=device))
            logp = logits.softmax(-1).mean(1).clamp_min(1e-12).log() if logits.ndim == 3 else logits.log_softmax(-1)
            result.append(logp.cpu().numpy())
    return np.concatenate(result)


class Transform:
    """TRAIN median; neural inputs additionally use TRAIN z-score and clip12."""

    def fit(self, x, neural=True):
        """Fit imputation and scaling statistics from the TRAIN feature matrix and return this transform.

        Enable neural preprocessing to apply standardization and clipping during transformation.
        """
        if x.ndim != 2 or not len(x) or not x.shape[1]:
            raise ValueError('Nonempty TRAIN matrix required')
        finite = np.where(np.isfinite(x), x, np.nan)
        self.median = np.nan_to_num(pd.DataFrame(finite).median().to_numpy(), nan=0).astype(np.float32)
        filled = np.where(np.isfinite(x), x, self.median).astype(np.float64)
        self.mean, self.scale = filled.mean(0), filled.std(0)
        self.scale[self.scale < 1e-6] = 1
        self.neural = neural
        return self

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.median):
            raise ValueError('Transform width differs')
        filled = np.where(np.isfinite(x), x, self.median)
        return (np.clip((filled - self.mean) / self.scale, -12, 12) if self.neural else filled).astype(np.float32)

    def state(self):
        """Return the fitted medians, means, scales and preprocessing mode as a serializable state."""
        return {
            'median': self.median.tolist(),
            'mean': self.mean.tolist(),
            'scale': self.scale.tolist(),
            'neural': self.neural,
        }

    @classmethod
    def restore(cls, state):
        """Return a transform reconstructed from the saved state dictionary."""
        obj = cls()
        obj.median, obj.mean, obj.scale = (np.asarray(state[k]) for k in ('median', 'mean', 'scale'))
        obj.neural = state['neural']
        return obj


def make_model(d, classes, kind, x=None, bins=None, *, ple_boundary_policy='clip_to_train_edges'):
    """Build a detector with the declared feature and class counts, and return the model and its embedding bins.

    Reuse supplied bins or fit them on numerical TRAIN inputs. The boundary policy governs later scoring.
    """
    if kind not in NEURAL_KINDS:
        raise ValueError('Unknown neural model')

    if bins is None:
        if x is None or x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('TRAIN-only finite PLE input required')
        q = torch.quantile(torch.as_tensor(x), torch.linspace(0, 1, 17), dim=0).T
        bins = []
        for column in q:
            edges = column.unique().contiguous()
            if len(edges) == 1:
                edges = torch.tensor([float(edges[0]) - 0.5, float(edges[0]) + 0.5], dtype=torch.float32)
            bins.append(edges)
    if len(bins) != d or any(len(b) < 2 or not torch.isfinite(b).all() or not (b.diff() > 0).all() for b in bins):
        raise ValueError('Invalid saved PLE bins')
    model = TabM.make(
        n_num_features=d,
        d_out=classes,
        n_blocks=2,
        d_block=64,
        k=4,
        dropout=0.1,
        num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=8, activation=False, version='B'),
    )
    return CrossTabM(model, bins, ple_boundary_policy=ple_boundary_policy), bins


def class_weights(y, classes=None, *, max_ratio=None):
    """Calculate inverse-frequency weights with a unit mean over TRAIN rows."""
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('TRAIN labels must be a nonempty vector of nonnegative integers')
    if classes is None:
        if not np.array_equal(np.unique(y), np.arange(y.max() + 1)):
            raise ValueError('TRAIN labels must cover contiguous classes')
        classes = int(y.max()) + 1
    if type(classes) is not int or classes < 1 or int(y.max()) >= classes:
        raise ValueError('TRAIN labels must be in the declared class space')
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=classes)
    observed = counts > 0
    weights = np.zeros(classes)
    weights[observed] = counts[observed].astype(float) ** -CLASS_BALANCE_POWER
    if max_ratio is not None:
        if not np.isfinite(max_ratio) or max_ratio < 1:
            raise ValueError('Class weight ratio must be finite and at least one')
        weights[observed] = np.minimum(weights[observed], weights[observed].min() * max_ratio)
    return weights / np.sum(counts / len(y) * weights)


def detector_objective_settings(objective):
    """Return the settings associated with the selected detector training objective."""
    if objective not in DETECTOR_OBJECTIVES:
        raise ValueError(f'Unknown detector objective: {objective}')
    weighted = objective == 'class_balanced'
    return {
        'detector_objective': objective,
        'class_weight_power': CLASS_BALANCE_POWER if weighted else None,
        'class_weight_max_ratio': CLASS_BALANCE_MAX_RATIO if weighted else None,
        'class_weight_scope': ('local TRAIN, unit sample-weighted mean, absent classes zero' if weighted else None),
    }


def detector_class_weights(y, classes, objective):
    """Return local TRAIN class weights for the selected objective, or None when it uses no weights."""
    settings = detector_objective_settings(objective)
    ratio = settings['class_weight_max_ratio']
    return None if ratio is None else class_weights(y, classes, max_ratio=ratio)


def classification_losses(logits, labels, weights=None):
    """Return per row cross entropy averaged over ensemble members, with optional weights indexed by label."""
    if logits.ndim == 2:
        logits = logits[:, None, :]
    b, k, c = logits.shape
    loss = (
        functional.cross_entropy(logits.reshape(b * k, c), labels[:, None].expand(-1, k).reshape(-1), reduction='none')
        .reshape(b, k)
        .mean(1)
    )
    return loss if weights is None else loss * weights[labels]


def diagnostic_rows(y, cap=10000):
    """Return sorted row indices for a deterministic class aware sample of the labels within the supplied cap."""
    rng = np.random.default_rng(123)
    if len(y) <= cap:
        return np.arange(len(y))
    counts = np.bincount(y)
    if cap < len(counts):
        raise ValueError('Diagnostic cap must cover all classes')
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
    """Select a temperature and a benign alarm cutoff from CAL log scores and labels and return fit diagnostics."""
    grid = np.unique(np.r_[1.0, np.geomspace(0.25, 4, 81)])
    temperature = float(min(grid, key=lambda t: nll(y, probabilities(logp, t))))
    p = probabilities(logp, temperature)
    benign = 1 - p[y == 0, 0]
    if not len(benign):
        raise ValueError('CAL benign support required')
    threshold = float(np.nextafter(np.quantile(benign, 0.99, method='higher'), np.inf))
    return {
        'temperature': temperature,
        'attack_threshold': threshold,
        'fit_role': 'CAL',
        'cal_far': float((benign >= threshold).mean()),
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
            limit = cap if role == 'train' else max(64, cap // 3) if cap else 0
            if limit:
                rows = rows[:limit]
            if not len(rows):
                raise ValueError('Empty simulated client role; use more rows or larger Dirichlet alpha')
            client[role] = rows
    return result


def synthetic_data(index):
    """Four tiny, different-feature-space fixtures; never research evidence."""
    rng = np.random.default_rng(1700 + index)
    classes, width = 3 + index % 2, 4 + index * 2
    parts = {}
    for role, n in (('train', 480), ('cal', 240), ('val', 240)):
        y = np.arange(n, dtype=np.int64) % classes
        x = rng.normal(size=(n, width)) + y[:, None] * 0.6
        parts[role] = (x.astype(np.float32), y)
    return (
        parts,
        ['BENIGN'] + [f'ATTACK_{c}' for c in range(1, classes)],
        [f'f{i}' for i in range(width)],
    )


def load_development(path):
    """Load only the pre-existing development roles, never TEST/confirmation."""
    from .data_io import load_data, read_json

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        expected = {'class_names', 'feature_names'} | {
            f'{xy}_{r}' for xy in ('X', 'y') for r in ('train', 'val', 'cal')
        }
        if set(archive.files) != expected:
            raise ValueError('Privacy input must contain exactly TRAIN/VAL/CAL; TEST/confirmation forbidden')
    preparation = path.parent / 'preparation.json'
    if preparation.exists():
        audit = read_json(preparation)
        if audit['data_sha256'] != sha(path):
            raise ValueError('Frozen development cache hash differs')
        identities = path.parent / 'selected_train_identities.npz'
        if not identities.exists() or audit['identity_sha256'] != sha(identities):
            raise ValueError('Frozen TRAIN identity cache differs')
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
        raise ValueError('Nonempty integer TRAIN labels in the public class space required')
    counts = np.bincount(y, minlength=classes)
    if len(counts) != classes or counts.sum() == 0:
        raise ValueError('Nonempty TRAIN labels in the public class space required')
    prior = (counts + 0.5) / (counts.sum() + 0.5 * classes)
    return np.where(counts > 0, np.log(prior), 0.0).astype(np.float64)


def split_calibration(rows, fraction=0.5, seed_value=42):
    """Return seeded fit and holdout position arrays for the supplied CAL row count and fit fraction."""
    if rows < 1 or not 0 < fraction < 1:
        raise ValueError('Nonempty CAL and a split fraction between zero and one required')
    order = np.random.default_rng(seed_value).permutation(rows)
    count = min(rows - 1, max(1, int(rows * fraction))) if rows > 1 else 1
    return order[:count], order[count:]


def empirical_alarm_threshold(p, y, target_far=0.01, floor=0.5, min_benign_rows=0, insufficient_policy='floor'):
    """Benign CAL holdout quantile with a small-sample support guard.

    A 1% quantile estimated from a few hundred benign rows is unstable and can
    land near 1.0, suppressing almost every attack.  When the benign holdout is
    below ``min_benign_rows``, use the declared floor for fixed-threshold
    diagnosis or disable the alert gate for the conservative operating point.
    This controls an empirical CAL operating point, not the future FAR.
    """
    if not 0 < target_far < 1 or not 0 <= floor <= 1:
        raise ValueError('Invalid target FAR or threshold floor')
    if type(min_benign_rows) is not int or min_benign_rows < 0:
        raise ValueError('Minimum benign rows must be a nonnegative integer')
    if insufficient_policy not in ('floor', 'disable'):
        raise ValueError('Invalid insufficient benign support policy')
    y = np.asarray(y)
    q = np.clip(np.asarray(p)[:, 1:].sum(1), 0.0, 1.0)
    if len(q) != len(y):
        raise ValueError('Calibration rows differ')
    benign = q[y == 0]
    if len(benign) and len(benign) >= min_benign_rows:
        threshold = max(floor, float(np.quantile(benign, 1 - target_far, method='higher')))
        fallback = False
    elif len(benign):
        threshold, fallback = (1.0 if insufficient_policy == 'disable' else float(floor)), True
    else:
        threshold, fallback = 1.0, True
    if not len(benign):
        rule = 'no_benign_support_disabled'
    elif fallback:
        rule = (
            'insufficient_benign_support_floor_fallback'
            if insufficient_policy == 'floor'
            else 'insufficient_benign_support_disabled'
        )
    else:
        rule = 'empirical_benign_holdout_quantile_higher'
    return threshold, {
        'threshold': threshold,
        'target_far': target_far,
        'benign_holdout_rows': len(benign),
        'threshold_floor': floor,
        'min_benign_rows': min_benign_rows,
        'calibration_far': float((benign > threshold).mean()) if len(benign) else None,
        'calibration_rule': rule,
        'insufficient_benign_policy': insufficient_policy,
        'alert_gate_disabled_for_support': bool(fallback and threshold == 1.0),
        'insufficient_benign_support': fallback and bool(len(benign)),
        'formal_conformal_guarantee': False,
        'no_benign_fallback': not bool(len(benign)),
    }


def subtype_joint_probabilities(p, attack_subtype_scores=None):
    """Joint probabilities for NLL/reporting; binary decisions retain original q."""
    if attack_subtype_scores is None:
        return p
    scores = np.asarray(attack_subtype_scores, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if scores.shape != (len(p), p.shape[1] - 1) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Finite nonnegative attack subtype scores with matching classes required')
    total = scores.sum(1, keepdims=True)
    conditional = np.divide(scores, total, out=np.full_like(scores, 1 / scores.shape[1]), where=total > 0)
    joint = p.copy()
    joint[:, 1:] = p[:, 1:].sum(1, keepdims=True) * conditional
    return joint


def hierarchical_decisions(p, threshold=0.5, attack_subtype_scores=None):
    """Return binary attack scores and alerts, attack subtype labels and final multiclass predictions.

    Apply a scalar or per row threshold to the attack score. Optional subtype scores select labels only after an
    alert.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError('Finite multiclass probabilities required')
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), (len(p),))
    if not np.isfinite(threshold).all() or ((threshold < 0) | (threshold > 1)).any():
        raise ValueError('Thresholds must be finite probabilities')
    q = np.clip(p[:, 1:].sum(1), 0.0, 1.0)
    alert = q > threshold
    scores = p[:, 1:] if attack_subtype_scores is None else np.asarray(attack_subtype_scores, dtype=np.float64)
    if scores.shape != p[:, 1:].shape or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Matching finite nonnegative subtype scores required')
    subtype = 1 + scores.argmax(1)
    return {
        'attack_score': q,
        'attack_alert': alert,
        'subtype': subtype,
        'predicted': np.where(alert, subtype, 0),
        'threshold': threshold,
    }


def risk_metrics(y, p, names=None, threshold=0.5, attack_subtype_scores=None):
    """Return binary and multiclass metrics for the supplied labels and class probabilities.

    Use the given class order, alarm thresholds and optional subtype scores for hierarchical decisions.
    """
    y = np.asarray(y)
    q, attack = p[:, 1:].sum(1), y > 0
    joint = subtype_joint_probabilities(p, attack_subtype_scores)
    argmax_metrics = core.prediction_metrics(y, joint)
    result = {
        'nll': argmax_metrics['nll'],
        **{'argmax_' + k: argmax_metrics[k] for k in ('accuracy', 'weighted_f1', 'macro_f1')},
    }
    result.update(
        binary_brier=float(np.mean((q - attack) ** 2)),
        binary_nll=float(-np.mean(np.where(attack, np.log(np.clip(q, 1e-12, 1)), np.log(np.clip(1 - q, 1e-12, 1))))),
        far_at_half=float((q[~attack] > 0.5).mean()) if (~attack).any() else None,
        fnr_at_half=float((q[attack] <= 0.5).mean()) if attack.any() else None,
        binary_auc=float(roc_auc_score(attack, q)) if len(np.unique(attack)) == 2 else None,
    )
    groups = np.minimum((q * 10).astype(int), 9)
    result['binary_ece_10'] = float(
        sum(
            np.abs(q[groups == i].mean() - attack[groups == i].mean()) * (groups == i).mean()
            for i in range(10)
            if (groups == i).any()
        )
    )
    predicted = hierarchical_decisions(p, threshold, attack_subtype_scores)['predicted']
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
        weighted_precision=float(report['weighted avg']['precision']),
        macro_precision=float(report['macro avg']['precision']),
        weighted_f1=float(report['weighted avg']['f1-score']),
        macro_f1=float(report['macro avg']['f1-score']),
        decision_rule='total_attack_probability_then_attack_subtype',
    )
    result['per_class'] = {name: report[name] for name in names}
    result['confusion_matrix'] = confusion_matrix(y, predicted, labels=np.arange(p.shape[1])).tolist()
    result['class_names'] = names
    return result
