"""Train attention TabM with separate alarm and conditional attack outputs."""

import math

import numpy as np
import torch
from torch.nn import functional

from . import utils as core
from .revision_data import _feature_keys
from .revision_iov_id_gate import ID_GATE_MODE


BINARY_GUARD_MODES = ('binary_swa_guard', 'binary_swa_recall', 'binary_swa_rank', ID_GATE_MODE)
PORTABLE_BINARY_MODES = ('binary_swa', 'binary_swa_subtype')
BINARY_SWA_MODES = (*PORTABLE_BINARY_MODES, 'binary_support_bins', *BINARY_GUARD_MODES)
BINARY_HEAD_MODES = ('binary', 'binary_tail', *BINARY_SWA_MODES)
BINARY_BCE_WEIGHT = 1.0
BINARY_SUBTYPE_WEIGHT = 1.0
BENIGN_TAIL_FRACTION = 0.10
BENIGN_TAIL_MARGIN = 1.0
BENIGN_TAIL_WEIGHT = 1.0
GROUP_WEIGHT_MAX_RATIO = 16
BINARY_SWA_EPOCHS = 3
BINARY_GUARD_FLOOR = 0.70
BINARY_RECALL_MARGIN = math.log(BINARY_GUARD_FLOOR / (1 - BINARY_GUARD_FLOOR))
BINARY_RANK_MARGIN = 1.0
BINARY_RANK_WEIGHT = 0.25
BINARY_RANK_GROUP_LIMIT = 32
SUPPORT_BIN_COUNT = 32


class BinaryAlarmTabM(core.CrossTabM):
    """Use one output projection for alarm and its other rows for attack subtype."""

    def forward(self, x):
        raw = super().forward(x)
        alarm, subtype = raw[..., :1], raw[..., 1:]
        return torch.cat((
            functional.logsigmoid(-alarm),
            functional.logsigmoid(alarm) + functional.log_softmax(subtype, dim=-1),
        ), dim=-1)


def make_binary_alarm_model(model, bins):
    """Keep the initialized attention backbone and partition its independent output rows."""
    if not isinstance(model, core.CrossTabM) or model.feature_attention is None:
        raise ValueError('The binary alarm model requires an attention CrossTabM backbone')
    binary = BinaryAlarmTabM(
        model.base,
        bins,
        ple_boundary_policy=model.ple_boundary_policy,
        feature_attention=True,
        can_id_training=model.can_id_values,
    )
    binary.load_state_dict(model.state_dict(), strict=True)
    return binary


def binary_alarm_logits(joint_logits):
    """Recover the raw alarm log odds from joint class log probabilities."""
    if joint_logits.ndim not in (2, 3) or joint_logits.shape[-1] < 2:
        raise ValueError('Joint logits must have batch/class or batch/member/class shape')
    return torch.logsumexp(joint_logits[..., 1:], dim=-1) - joint_logits[..., 0]


def conditional_attack_loss(joint_logits, labels):
    """Average conditional subtype cross entropy over attack rows and ensemble members."""
    conditional = functional.log_softmax(joint_logits[..., 1:], dim=-1)
    if conditional.ndim == 2:
        conditional = conditional.unsqueeze(1)
    targets = (labels - 1).clamp_min(0)[:, None, None].expand(-1, conditional.shape[1], 1)
    losses = -conditional.gather(-1, targets).squeeze(-1).mean(dim=1)
    attacks = labels > 0
    return BINARY_SUBTYPE_WEIGHT * losses.masked_fill(~attacks, 0).sum() / attacks.sum().clamp_min(1)


def support_quantile_bins(x):
    """Fit each embedding on its distinct standardized TRAIN feature values."""
    quantiles = torch.linspace(0, 1, SUPPORT_BIN_COUNT + 1)
    bins = []
    for column in x.T:
        values = torch.as_tensor(np.unique(column), dtype=torch.float32)
        if len(values) == 1:
            edges = values + torch.tensor([-0.5, 0.5], dtype=torch.float32)
        else:
            edges = torch.quantile(values, quantiles).unique()
        bins.append(edges.contiguous())
    return bins


def binary_group_row_weights(raw_train, labels):
    """Weight TRAIN benign groups by capped inverse square root frequency."""
    labels = np.asarray(labels)
    if labels.shape != (len(raw_train),) or not len(labels) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Group weights require aligned local TRAIN integer labels')
    _, groups, counts = np.unique(_feature_keys(raw_train), return_inverse=True, return_counts=True)
    weights = 1.0 / np.sqrt(np.minimum(counts[groups.reshape(-1)], GROUP_WEIGHT_MAX_RATIO**2))
    benign = labels == 0
    if benign.any():
        weights[benign] /= weights[benign].mean()
    weights[~benign] = 1.0
    return weights.astype(np.float32)


def binary_train_groups(raw_train):
    """Return exact input group identifiers using local TRAIN features only."""
    _, groups = np.unique(_feature_keys(raw_train), return_inverse=True)
    return groups.reshape(-1)


def binary_group_rank_loss(joint_logits, labels, groups):
    """Penalize overlap between high-scoring benign and low-scoring attack groups."""
    normal = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels > 0)
    normal_groups, normal_first = np.unique(groups[normal], return_index=True)
    attack_groups, attack_first = np.unique(groups[attack], return_index=True)
    # A single feature vector cannot be ranked above itself when its labels disagree.
    conflicting = np.intersect1d(normal_groups, attack_groups, assume_unique=True)
    normal = normal[normal_first[~np.isin(normal_groups, conflicting)]]
    attack = attack[attack_first[~np.isin(attack_groups, conflicting)]]
    if not len(normal) or not len(attack):
        return joint_logits.new_zeros(())
    alarm = binary_alarm_logits(joint_logits)
    if alarm.ndim == 2:
        alarm = alarm.mean(dim=1)
    normal_scores, attack_scores = alarm[normal], alarm[attack]
    high = normal_scores.detach().topk(min(len(normal), BINARY_RANK_GROUP_LIMIT)).indices
    low = (-attack_scores.detach()).topk(min(len(attack), BINARY_RANK_GROUP_LIMIT)).indices
    gap = attack_scores[low][None, :] - normal_scores[high][:, None]
    return BINARY_RANK_WEIGHT * functional.softplus(BINARY_RANK_MARGIN - gap).mean()


def binary_alarm_losses(joint_logits, labels, *, tail=False, row_weights=None, positive_margin=0.0):
    """Return per-row auxiliary BCE and an optional upper benign tail margin loss."""
    if labels.shape != (len(joint_logits),):
        raise ValueError('Binary alarm labels differ from the student batch')
    alarm = binary_alarm_logits(joint_logits)
    target = (labels > 0).to(dtype=alarm.dtype)
    if alarm.ndim == 2:
        target = target[:, None].expand_as(alarm)
    training_alarm = alarm - positive_margin * target if positive_margin else alarm
    losses = functional.binary_cross_entropy_with_logits(training_alarm, target, reduction='none')
    if alarm.ndim == 2:
        losses = losses.mean(dim=1)
    losses = BINARY_BCE_WEIGHT * losses
    if not tail:
        return losses
    if row_weights is None or row_weights.shape != labels.shape:
        raise ValueError('The benign tail objective requires aligned local TRAIN group weights')
    benign = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if not len(benign):
        return losses
    mean_alarm = alarm.mean(dim=1) if alarm.ndim == 2 else alarm
    count = max(1, math.ceil(BENIGN_TAIL_FRACTION * len(benign)))
    hardest = torch.topk(mean_alarm.detach()[benign], count, sorted=False).indices
    selected = benign[hardest]
    margin = functional.softplus(alarm[selected] + BENIGN_TAIL_MARGIN)
    if alarm.ndim == 2:
        margin = margin.mean(dim=1)
    penalty = torch.zeros_like(losses)
    penalty[selected] = row_weights[selected] * margin * len(benign) / count
    return losses + BENIGN_TAIL_WEIGHT * penalty


def binary_objective_settings(mode):
    """Describe the fixed auxiliary objective used by this optional TRAIN experiment."""
    if mode not in BINARY_HEAD_MODES:
        raise ValueError('Unknown attention TabM binary objective')
    return {
        'binary_alarm_head': True,
        'binary_output_rule': 'log sigmoid(-a); log sigmoid(a) + log softmax(attack subtype logits)',
        'binary_head_layout': 'one independent alarm output and K-1 independent subtype outputs on the shared backbone',
        'binary_bce_weight': BINARY_BCE_WEIGHT,
        'binary_bce_scope': 'all local TRAIN rows; mean over TabM members before batch reduction',
        **({
            'conditional_subtype_weight': BINARY_SUBTYPE_WEIGHT,
            'conditional_subtype_scope': 'mean over members and attack TRAIN rows in each batch; zero for benign-only batches',
        } if mode == 'binary_swa_subtype' else {}),
        'binary_prior_scope': 'existing class CE only; auxiliary BCE uses raw inference alarm log odds',
        'binary_tail_enabled': mode == 'binary_tail',
        'benign_tail_fraction': BENIGN_TAIL_FRACTION if mode == 'binary_tail' else None,
        'benign_tail_margin': BENIGN_TAIL_MARGIN if mode == 'binary_tail' else None,
        'benign_tail_weight': BENIGN_TAIL_WEIGHT if mode == 'binary_tail' else None,
        'benign_tail_selection': 'highest member-mean benign alarm log odds in each TRAIN minibatch',
        'benign_tail_group_weights': 'capped inverse square root raw TRAIN group frequency, unit mean among benign rows',
        'benign_tail_group_max_ratio': GROUP_WEIGHT_MAX_RATIO if mode == 'binary_tail' else None,
        'binary_teacher_models': 0,
        'binary_extra_optimizer_steps': 0,
        'binary_swa_last_epochs': BINARY_SWA_EPOCHS if mode in BINARY_SWA_MODES else None,
        'binary_support_bins': SUPPORT_BIN_COUNT if mode == 'binary_support_bins' else None,
        'binary_support_binning_scope': 'distinct standardized values in local TRAIN' if mode == 'binary_support_bins' else None,
        'binary_guard_floor': BINARY_GUARD_FLOOR if mode in BINARY_GUARD_MODES else None,
        'binary_positive_training_logit_margin': BINARY_RECALL_MARGIN if mode == 'binary_swa_recall' else None,
        'binary_train_rank_weight': BINARY_RANK_WEIGHT if mode == 'binary_swa_rank' else None,
        'binary_train_rank_margin': BINARY_RANK_MARGIN if mode == 'binary_swa_rank' else None,
        'binary_train_rank_group_limit': BINARY_RANK_GROUP_LIMIT if mode == 'binary_swa_rank' else None,
    }
