"""Make local TRAIN teacher scores with whole-input-group cross fitting."""

from hashlib import sha256
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.nn import functional

from .revision_data import _feature_keys
from .revision_tree_alarm import TREE_SETTINGS
from .utils import Transform


DISTILLATION_FOLDS = 3
TEACHER_TREES = 128
DISTILLATION_TEMPERATURE = 4.0
DISTILLATION_WEIGHT = 1.0
PROBABILITY_FLOOR = 1e-12


def _grouped_folds(inputs, labels, seed):
    keys, groups = np.unique(_feature_keys(inputs), return_inverse=True)
    groups = groups.reshape(-1)
    if len(keys) < DISTILLATION_FOLDS:
        raise ValueError('Grouped teacher fitting requires at least three distinct local TRAIN inputs')
    splitter = StratifiedGroupKFold(n_splits=DISTILLATION_FOLDS, shuffle=True, random_state=seed)
    assignment = np.full(len(keys), -1, dtype=np.int8)
    for fold, (fit, heldout) in enumerate(splitter.split(inputs, labels, groups)):
        heldout_groups = np.unique(groups[heldout])
        if not len(fit) or not len(heldout) or (assignment[heldout_groups] != -1).any():
            raise ValueError('Grouped teacher split has empty or repeated held-out groups')
        if np.isin(groups[fit], heldout_groups).any():
            raise RuntimeError('A teacher training fold overlaps its held-out input groups')
        assignment[heldout_groups] = fold
    if (assignment < 0).any():
        raise RuntimeError('Grouped teacher folds do not cover all local TRAIN input groups')
    return keys, groups, assignment


def fit_tree_distillation(raw_x, train_y, n_classes, *, seed=42):
    """Return raw OOF probabilities, a supported-target mask, and timing metadata."""
    started = time.perf_counter()
    inputs, labels = np.asarray(raw_x), np.asarray(train_y)
    if (
        inputs.ndim != 2 or not inputs.shape[1] or not len(inputs)
        or labels.shape != (len(inputs),) or not np.issubdtype(labels.dtype, np.integer)
        or type(n_classes) is not int or n_classes < 2
        or (labels < 0).any() or (labels >= n_classes).any()
        or type(seed) is not int or seed < 0
    ):
        raise ValueError('Teacher fitting needs local TRAIN inputs and aligned integer class labels')
    keys, groups, group_assignment = _grouped_folds(inputs, labels, seed)
    assignment = group_assignment[groups]
    partition_seconds = time.perf_counter() - started
    probabilities = np.zeros((len(labels), n_classes), dtype=np.float32)
    supported = np.zeros(len(labels), dtype=bool)
    fold_records = []
    settings = {**TREE_SETTINGS, 'n_estimators': TEACHER_TREES}
    for fold in range(DISTILLATION_FOLDS):
        fit, heldout = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        fit_labels = labels[fit]
        preprocess_started = time.perf_counter()
        transform = Transform().fit(inputs[fit], neural=False)
        fit_x = transform(inputs[fit])
        preprocessing_seconds = time.perf_counter() - preprocess_started
        model = ExtraTreesClassifier(random_state=seed + fold, **settings)
        fit_started = time.perf_counter()
        model.fit(fit_x, fit_labels)
        fit_seconds = time.perf_counter() - fit_started
        del fit_x
        query_started = time.perf_counter()
        prediction = np.zeros((len(heldout), n_classes), dtype=np.float32)
        prediction[:, model.classes_.astype(np.int64)] = model.predict_proba(transform(inputs[heldout]))
        query_seconds = time.perf_counter() - query_started
        if not np.isfinite(prediction).all() or (prediction < 0).any() or not np.allclose(prediction.sum(1), 1):
            raise FloatingPointError('OOF teacher probabilities are not finite normalized distributions')
        probabilities[heldout] = prediction
        supported[heldout] = np.isin(labels[heldout], model.classes_)
        fold_records.append({
            'fold': fold,
            'teacher_seed': seed + fold,
            'fit_rows': len(fit),
            'heldout_rows': len(heldout),
            'fit_input_groups': int(np.count_nonzero(group_assignment != fold)),
            'heldout_input_groups': int(np.count_nonzero(group_assignment == fold)),
            'fit_class_support': np.bincount(fit_labels, minlength=n_classes).tolist(),
            'heldout_class_support': np.bincount(labels[heldout], minlength=n_classes).tolist(),
            'hard_only_class_support': np.bincount(labels[heldout][~supported[heldout]], minlength=n_classes).tolist(),
            'preprocessing_seconds': preprocessing_seconds,
            'fit_seconds': fit_seconds,
            'oof_prediction_with_transform_seconds': query_seconds,
        })
        del model, prediction, transform
    counts = np.bincount(labels, minlength=n_classes)
    group_support = [int(len(np.unique(groups[labels == c]))) for c in range(n_classes)]
    return probabilities, supported, {
        'teacher_method': 'local TRAIN three-fold stratified exact-input-group Extra Trees',
        'teacher_seed': seed,
        'teacher_folds': DISTILLATION_FOLDS,
        'teacher_trees_per_fold': TEACHER_TREES,
        'teacher_tree_settings': settings,
        'teacher_train_rows': len(labels),
        'teacher_input_groups': len(keys),
        'teacher_train_class_support': counts.tolist(),
        'teacher_class_input_groups': group_support,
        'teacher_supported_rows': int(supported.sum()),
        'teacher_hard_only_rows': int((~supported).sum()),
        'teacher_hard_only_class_support': np.bincount(labels[~supported], minlength=n_classes).tolist(),
        'teacher_supported_fraction': float(supported.mean()),
        'teacher_partition_seconds': partition_seconds,
        'teacher_preprocessing_seconds': sum(row['preprocessing_seconds'] for row in fold_records),
        'teacher_fit_seconds': sum(row['fit_seconds'] for row in fold_records),
        'teacher_oof_prediction_seconds': sum(row['oof_prediction_with_transform_seconds'] for row in fold_records),
        'teacher_total_seconds': time.perf_counter() - started,
        'teacher_group_assignment_sha256': sha256(keys.tobytes() + group_assignment.tobytes()).hexdigest(),
        'teacher_oof_probabilities_sha256': sha256(probabilities.tobytes()).hexdigest(),
        'teacher_fold_details': fold_records,
        'teacher_target_type': 'raw class-aligned probabilities, temperature applied in student loss',
        'teacher_unsupported_policy': 'Retain every TRAIN row; use hard CE only when its class is absent in fold fit',
        'teacher_group_key': 'canonical raw float32 model inputs, labels excluded',
        'teacher_preprocessing_scope': 'fold-fit TRAIN only',
        'teacher_evaluation_roles_used': [],
        'teacher_models_retained_for_inference': False,
        'teacher_network_messages': 0,
    }


def temperature_kl_losses(logits, teacher_probabilities, mask=None):
    """Return per-row T-squared KL, averaging TabM members before batch reduction."""
    if logits.ndim not in (2, 3) or not logits.is_floating_point():
        raise ValueError('Student logits must have batch/class or batch/member/class shape')
    targets = torch.as_tensor(teacher_probabilities, dtype=logits.dtype, device=logits.device).detach()
    if targets.shape != (len(logits), logits.shape[-1]):
        raise ValueError('Teacher probability width or row count differs from student logits')
    target_log = targets.clamp_min(PROBABILITY_FLOOR).log() / DISTILLATION_TEMPERATURE
    softened = target_log.softmax(dim=-1)
    if logits.ndim == 3:
        softened = softened[:, None, :].expand(-1, logits.shape[1], -1)
    student_log = functional.log_softmax(logits / DISTILLATION_TEMPERATURE, dim=-1)
    losses = functional.kl_div(student_log, softened, reduction='none').sum(dim=-1)
    if logits.ndim == 3:
        losses = losses.mean(dim=1)
    losses = DISTILLATION_TEMPERATURE**2 * losses
    if mask is not None:
        selected = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if selected.shape != (len(logits),):
            raise ValueError('Distillation support mask differs from the student batch')
        losses = losses * selected
    return losses
