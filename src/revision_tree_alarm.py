"""Shared local tree alarm for the optional TabM subtype comparison."""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .revision_v12 import _write
from .utils import Transform, empirical_alarm_threshold, hierarchical_decisions


TREE_SETTINGS = {
    'n_estimators': 32,
    'max_depth': 12,
    'min_samples_leaf': 8,
    'max_features': None,
    'bootstrap': False,
    'n_jobs': 4,
}
TREE_ALARM_COUNT = 128
TREE_ALARM_SUBTYPE_WEIGHT = 0.5
TREE_FUSION_WEIGHT = 0.8


def tree_probabilities(model, transform, features, classes):
    probabilities = np.zeros((len(features), classes))
    probabilities[:, model.classes_] = model.predict_proba(transform(features))
    return probabilities


class LocalTreeScoreFusion:
    """Blend TRAIN-only tree and TabM probabilities before calibration and adaptation."""

    def __init__(self, train_x, train_y, classes, seed):
        self.classes = classes
        self.transform = Transform().fit(train_x, neural=False)
        self.model = ExtraTreesClassifier(
            random_state=seed, **{**TREE_SETTINGS, 'n_estimators': TREE_ALARM_COUNT},
        ).fit(self.transform(train_x), train_y)

    def __call__(self, raw, _transformed, tabm_probabilities):
        if not len(raw):
            return tabm_probabilities
        tree = tree_probabilities(self.model, self.transform, raw, self.classes)
        return TREE_FUSION_WEIGHT * tree + (1 - TREE_FUSION_WEIGHT) * tabm_probabilities


class SharedTreeAlarm:
    """Fit the same TRAIN-only alarm model for every branch at each client."""

    def __init__(self, parts, roster, cal_splits, args, classes, receipt):
        models, calibration = [], []
        for index, (client, (_, holdout)) in enumerate(zip(roster, cal_splits, strict=True)):
            train_rows = client['train']
            train_x, train_y = (values[train_rows] for values in parts['train'])
            transform = Transform().fit(train_x, neural=False)
            settings = {**TREE_SETTINGS, 'n_estimators': TREE_ALARM_COUNT}
            model = ExtraTreesClassifier(random_state=args.seed + index, **settings)
            model.fit(transform(train_x), train_y)
            cal_x, cal_y = (values[client['cal'][holdout]] for values in parts['cal'])
            p = tree_probabilities(model, transform, cal_x, classes)
            threshold, evidence = empirical_alarm_threshold(
                p, cal_y, args.target_far, args.threshold_floor, args.threshold_min_benign_rows,
                args.threshold_insufficient_policy,
            )
            models.append((model, transform, threshold))
            calibration.append({'client': index, 'train_rows': len(train_rows), **evidence})
        _write(receipt, {'trees': TREE_ALARM_COUNT, 'tree_subtype_weight': TREE_ALARM_SUBTYPE_WEIGHT,
                         'clients': calibration})

        self.predictions = {name: [] for name in ('tree_alarm', 'tree_alarm_local', 'tree_alarm_full')}
        self.scores = {name: [] for name in self.predictions}
        self.subtypes = []
        for client, (model, transform, threshold) in zip(roster, models, strict=True):
            p = tree_probabilities(model, transform, parts['val'][0][client['val']], classes)
            result = hierarchical_decisions(p, threshold)
            self.predictions['tree_alarm'].append(result['predicted'])
            self.scores['tree_alarm'].append(result['attack_score'])
            self.subtypes.append(p[result['attack_alert'], 1:])

    def observe(self, mode, client, probabilities, _predicted):
        """Blend conditional subtype scores only for already alerted records."""
        if mode not in ('local_only', 'full'):
            return
        name = 'tree_alarm_local' if mode == 'local_only' else 'tree_alarm_full'
        prediction = self.predictions['tree_alarm'][client].copy()
        alerts = prediction > 0
        tree = self.subtypes[client]
        tabm = probabilities[alerts, 1:]
        combined = TREE_ALARM_SUBTYPE_WEIGHT * tree / tree.sum(axis=1, keepdims=True)
        combined += (1 - TREE_ALARM_SUBTYPE_WEIGHT) * tabm / tabm.sum(axis=1, keepdims=True)
        prediction[alerts] = combined.argmax(axis=1) + 1
        self.predictions[name].append(prediction)
        self.scores[name].append(self.scores['tree_alarm'][client])
