"""Matched TabM training objectives for developer-side detector experiments."""

import numpy as np
import torch

from . import utils as core
from .revision_data import _feature_keys

HARD_BENIGN = 'hard_benign'
GROUP_BALANCED = 'group_balanced'
TREE_DISTILLATION = 'tree_distillation'
DETECTOR_OBJECTIVES = (*core.DETECTOR_OBJECTIVES, HARD_BENIGN, GROUP_BALANCED, TREE_DISTILLATION)
HARD_BENIGN_STRENGTH = 3.0
GROUP_WEIGHT_MAX_RATIO = 16
LOSS_NAMES = {
    'prior_adjusted': 'tempered_prior_adjusted_ce',
    'class_balanced': 'class_balanced_tempered_prior_adjusted_ce',
    HARD_BENIGN: 'hard_benign_tempered_prior_adjusted_ce',
    GROUP_BALANCED: 'group_balanced_tempered_prior_adjusted_ce',
    TREE_DISTILLATION: 'tempered_prior_adjusted_ce_plus_tree_distillation_kl',
}


def objective_settings(objective):
    """Describe the chosen weighting rule in private and run-level receipts."""
    if objective == TREE_DISTILLATION:
        from .revision_distillation import (
            DISTILLATION_FOLDS,
            DISTILLATION_TEMPERATURE,
            DISTILLATION_WEIGHT,
            TEACHER_TREES,
        )

        return {
            **core.detector_objective_settings('prior_adjusted'),
            'detector_objective': TREE_DISTILLATION,
            'distillation_temperature': DISTILLATION_TEMPERATURE,
            'distillation_weight': DISTILLATION_WEIGHT,
            'distillation_teacher': 'Extra Trees with folds grouped by exact raw input',
            'distillation_teacher_trees': TEACHER_TREES,
            'distillation_group_folds': DISTILLATION_FOLDS,
            'distillation_scope': 'local TRAIN only; out-of-fold teacher predictions; inference uses TabM',
            'distillation_unsupported_row_policy': 'hard cross entropy only',
            'extra_optimizer_steps': 0,
        }
    if objective == GROUP_BALANCED:
        return {
            **core.detector_objective_settings('prior_adjusted'),
            'detector_objective': GROUP_BALANCED,
            'group_weight_max_ratio': GROUP_WEIGHT_MAX_RATIO,
            'group_weight_rule': '1 / sqrt(min(exact_input_row_count, max_ratio**2))',
            'group_weight_scope': 'canonical raw local TRAIN inputs; normalized to unit mean within each class',
            'extra_optimizer_steps': 0,
        }
    if objective != HARD_BENIGN:
        return core.detector_objective_settings(objective)
    return {
        **core.detector_objective_settings('prior_adjusted'),
        'detector_objective': HARD_BENIGN,
        'hard_benign_weight_strength': HARD_BENIGN_STRENGTH,
        'hard_benign_weight_rule': '1 + strength * attack_probability ** 2',
        'hard_benign_weight_scope': 'current TRAIN minibatch, normalized among benign rows',
        'hard_benign_score': 'sum of attack probabilities from detached, unadjusted TabM logits',
        'hard_benign_attack_weight': 1.0,
        'extra_optimizer_steps': 0,
    }


def objective_class_weights(labels, classes, objective):
    """Return static weights for historical objectives that use them."""
    if objective in (HARD_BENIGN, GROUP_BALANCED, TREE_DISTILLATION):
        return None
    return core.detector_class_weights(labels, classes, objective)


def objective_row_weights(raw_train, labels, classes, objective):
    """Return per-row weights for this client's TRAIN without dropping observations."""
    if objective != GROUP_BALANCED:
        return None
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or len(labels) != len(raw_train)
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= classes).any()
    ):
        raise ValueError('Group weights require aligned TRAIN features and labels in the class space')
    labels = labels.astype(np.int64, copy=False)
    distinct, group, group_counts = np.unique(_feature_keys(raw_train), return_inverse=True, return_counts=True)
    del distinct
    weights = 1.0 / np.sqrt(np.minimum(group_counts[group], GROUP_WEIGHT_MAX_RATIO**2))
    class_count = np.bincount(labels, minlength=classes)
    class_mass = np.bincount(labels, weights=weights, minlength=classes)
    scale = np.divide(class_count, class_mass, out=np.zeros(classes), where=class_mass > 0)
    return (weights * scale[labels]).astype(np.float32)


def objective_distillation_targets(raw_train, labels, classes, objective, seed):
    """Fit a teacher only for a distillation run and return TRAIN-aligned targets and coverage."""
    if objective != TREE_DISTILLATION:
        return None, {}
    from .revision_distillation import fit_tree_distillation

    probabilities, supported, metadata = fit_tree_distillation(raw_train, labels, classes, seed=seed)
    return (probabilities, supported), metadata


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


def training_losses(logits, labels, prior_offset, class_weights, objective, *, row_weights=None, teacher=None):
    """Share the selected TRAIN loss between local TabM and FedAvg clients."""
    losses = core.classification_losses(logits + prior_offset, labels, class_weights)
    if objective == HARD_BENIGN:
        return losses * hard_benign_weights(logits, labels)
    if objective == GROUP_BALANCED:
        if row_weights is None or row_weights.shape != labels.shape:
            raise ValueError('Group-balanced TRAIN minibatches require aligned row weights')
        return losses * row_weights
    if objective == TREE_DISTILLATION:
        from .revision_distillation import DISTILLATION_WEIGHT, temperature_kl_losses

        if teacher is None:
            raise ValueError('Tree distillation minibatches require aligned teacher targets and a coverage mask')
        return losses + DISTILLATION_WEIGHT * temperature_kl_losses(logits, *teacher)
    return losses
