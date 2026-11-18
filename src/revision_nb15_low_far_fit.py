"""Train matched binary TabM candidates on the upper benign TRAIN score tail."""

from copy import deepcopy
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import utils as core


TAIL_FRACTION = 0.01
EPOCHS = 2
LEARNING_RATE = 5e-4
BCE_WEIGHT = 0.25
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0


class BinaryAttackScorer(nn.Module):
    """Add a zero-initialized binary readout to the shared TabM score."""

    def __init__(self, reference_model, mode, device):
        super().__init__()
        self.detector = deepcopy(reference_model).to(device)
        self.detector.requires_grad_(mode == "full")
        width = self.detector.base.output.in_features
        # Construct parameters directly so adding the branch does not change RNG.
        self.weight = nn.Parameter(torch.zeros(width, device=device))
        self.bias = nn.Parameter(torch.zeros((), device=device))
        # Disable dropout in both arms so only trainable parameter scope differs.
        self.eval()

    def forward(self, x):
        """Return attack log odds of the mean ensemble attack probability."""
        detector = self.detector
        base = detector.base
        features = base.num_module(detector.prepare_ple_input(x)).flatten(1)
        crossed = detector.apply_cross(features)
        hidden = base.backbone(base.ensemble_view(crossed))
        logits = base.output(hidden)
        attack = logits[..., 1:].logsumexp(dim=-1)
        member_odds = attack - logits[..., 0]
        log_attack = F.logsigmoid(member_odds).logsumexp(dim=1)
        log_benign = F.logsigmoid(-member_odds).logsumexp(dim=1)
        correction = hidden.mean(dim=1) @ self.weight + self.bias
        return log_attack - log_benign + correction


def _feature_matrix(x, width, batch_size):
    """Check feature shape and the batch size before taking TRAIN or score slices."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != width or x.dtype.kind not in "fiu":
        raise ValueError("Expected numeric rows in the reference TabM feature order")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Low-FAR batch size must be a positive integer")
    return x


def _batch_tensor(x, transform, device):
    """Apply only the supplied TRAIN-fitted transform, if one is needed."""
    if transform is not None:
        x = transform(x)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class LowFARFit:
    """Trained binary scoring arm and its budget and objective receipt."""

    scorer: BinaryAttackScorer
    transform: object
    device: torch.device
    batch_size: int
    receipt: dict

    @torch.no_grad()
    def score(self, x):
        """Score features without consulting or fitting with evaluation labels."""
        x = _feature_matrix(x, self.scorer.detector.input_features, self.batch_size)
        output = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self.batch_size):
            rows = slice(start, start + self.batch_size)
            batch = _batch_tensor(x[rows], self.transform, self.device)
            output[rows] = self.scorer(batch).double().sigmoid().cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Low-FAR TabM produced nonfinite scores")
        return output


def _tail_loss(logits, attack, benign_count):
    """Rank attacks above the largest 1% of benign TRAIN scores in the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, attack.to(logits.dtype))
    if benign_count == 0 or benign_count == len(attack):
        return BCE_WEIGHT * bce, 0
    tail_size = ceil(TAIL_FRACTION * benign_count)
    tail = logits[~attack].topk(tail_size).values
    attack_scores = logits[attack]
    ranking = F.softplus(tail[:, None] - attack_scores[None, :]).mean()
    return ranking + BCE_WEIGHT * bce, tail_size * len(attack_scores)


def _train_epoch(model, parameters, optimizer, features, attack, order, config):
    """Fit one common TRAIN permutation and count scored pairs and steps."""
    transform, device, batch_size = config
    monitor = core.TrainingFiniteMonitor(device, "NB15 low-FAR detector")
    steps, pairs = 0, 0
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        known_attack = attack[rows]
        benign_count = len(rows) - int(known_attack.sum())
        x = _batch_tensor(features[rows], transform, device)
        y = torch.as_tensor(known_attack, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, pair_count = _tail_loss(model(x), y, benign_count)
        loss.backward()
        monitor.observe(loss, nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP))
        optimizer.step()
        pairs += pair_count
        steps += 1
    monitor.check()
    return steps, pairs


def _parameter_audit(reference, scorer):
    """Verify which original TabM tensors and readout parameters changed."""
    current = dict(scorer.detector.named_parameters())
    changed = [
        name
        for name, before in reference.named_parameters()
        if not torch.equal(before.detach().to(current[name].device), current[name].detach())
    ]
    changed_readout = bool(torch.count_nonzero(scorer.weight)) or bool(scorer.bias.detach() != 0)
    return changed, changed_readout


def _training_inputs(reference, x, y, mode, batch_size):
    """Require one real binary training population and the reference class order."""
    if not isinstance(reference, core.CrossTabM) or mode not in {"frozen", "full"}:
        raise ValueError("Low-FAR fitting requires frozen or full CrossTabM")
    x = _feature_matrix(x, reference.input_features, batch_size)
    y = np.asarray(y)
    if (
        y.shape != (len(x),)
        or y.dtype.kind not in "iu"
        or (y < 0).any()
        or (y >= reference.base.output.out_features).any()
        or not (y == 0).any()
        or not (y > 0).any()
    ):
        raise ValueError("Low-FAR fitting needs BENIGN and attack labels from TRAIN")
    return x, y > 0


def _fit_receipt(mode, seed, x, batch_size, steps, pairs, updated, readout, parameters):
    """Record the fixed fit budget, scoring geometry and effective updates."""
    return {
        "mode": mode,
        "seed": seed,
        "training_role": "TRAIN",
        "train_rows": len(x),
        "train_rows_visited": EPOCHS * len(x),
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "optimizer_steps": steps,
        "tail_pairs": pairs,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "base_parameters_updated": updated,
        "binary_readout_updated": readout,
        "base_score": "log odds of mean ensemble attack probability plus zero-start linear readout",
        "loss": "pairwise softplus over top benign TRAIN scores plus weighted BCE",
        "benign_tail_fraction": TAIL_FRACTION,
        "bce_weight": BCE_WEIGHT,
        "adamw_learning_rate": LEARNING_RATE,
        "adamw_weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP,
        "dropout_during_fine_tuning": False,
    }


def fit_low_far(reference_model, train_x, train_y, *, mode, seed, device, transform=None, batch_size=2048):
    """Clone a same-run reference and fit an isolated frozen or full binary arm."""
    x, attack = _training_inputs(reference_model, train_x, train_y, mode, batch_size)
    device = torch.device(device)
    scorer = BinaryAttackScorer(reference_model, mode, device)
    parameters = [p for p in scorer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    total_steps, total_pairs = 0, 0
    config = transform, device, batch_size
    for _ in range(EPOCHS):
        steps, pairs = _train_epoch(scorer, parameters, optimizer, x, attack, rng.permutation(len(x)), config)
        total_steps += steps
        total_pairs += pairs
    updated, readout = _parameter_audit(reference_model, scorer)
    if (mode == "frozen" and updated) or (mode == "full" and not updated) or not readout or not total_pairs:
        raise RuntimeError("The low-FAR update did not match the declared training scope")
    receipt = _fit_receipt(mode, seed, x, batch_size, total_steps, total_pairs, updated, readout, parameters)
    return LowFARFit(scorer, transform, device, batch_size, receipt)
