"""Matched TabM training objectives for developer-side detector experiments."""

import torch

from . import utils as core


HARD_BENIGN = "hard_benign"
DETECTOR_OBJECTIVES = (*core.DETECTOR_OBJECTIVES, HARD_BENIGN)
HARD_BENIGN_STRENGTH = 3.0
LOSS_NAMES = {
    "prior_adjusted": "tempered_prior_adjusted_ce",
    "class_balanced": "class_balanced_tempered_prior_adjusted_ce",
    HARD_BENIGN: "hard_benign_tempered_prior_adjusted_ce",
}


def objective_settings(objective):
    """Describe the chosen weighting rule in private and run-level receipts."""
    if objective != HARD_BENIGN:
        return core.detector_objective_settings(objective)
    return {
        **core.detector_objective_settings("prior_adjusted"),
        "detector_objective": HARD_BENIGN,
        "hard_benign_weight_strength": HARD_BENIGN_STRENGTH,
        "hard_benign_weight_rule": "1 + strength * attack_probability ** 2",
        "hard_benign_weight_scope": "current TRAIN minibatch, normalized among benign rows",
        "hard_benign_score": "sum of attack probabilities from detached, unadjusted TabM logits",
        "hard_benign_attack_weight": 1.0,
        "extra_optimizer_steps": 0,
    }


def objective_class_weights(labels, classes, objective):
    """Return static weights for historical objectives that use them."""
    return None if objective == HARD_BENIGN else core.detector_class_weights(labels, classes, objective)


@torch.no_grad()
def hard_benign_weights(logits, labels):
    """Upweight hard benign rows and preserve the minibatch benign mass."""
    benign = labels.eq(0)
    attack_probability = logits.float().softmax(dim=-1)[..., 1:].sum(dim=-1)
    if attack_probability.ndim == 2:
        attack_probability = attack_probability.mean(dim=1)
    raw = 1.0 + HARD_BENIGN_STRENGTH * attack_probability.square()
    benign_count = benign.sum()
    normalizer = (raw * benign).sum() / benign_count.clamp_min(1)
    return torch.where(benign, raw / normalizer.clamp_min(1), 1.0)


def training_losses(logits, labels, prior_offset, class_weights, objective):
    """Share the selected TRAIN loss between local TabM and FedAvg clients."""
    losses = core.classification_losses(logits + prior_offset, labels, class_weights)
    if objective == HARD_BENIGN:
        return losses * hard_benign_weights(logits, labels)
    return losses
