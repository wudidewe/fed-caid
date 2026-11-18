"""Aggregate paired VAL target scores without retaining multivariate predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOCAL = 'conditional_local'
FULL = 'conditional_full'


@dataclass(frozen=True)
class _TargetScores:
    log_probability: np.ndarray
    attack_log_margin: np.ndarray | None
    attack_rank: np.ndarray


class TargetScoreObserver:
    """Collect one receiver's targeted scores, then emit only aggregate VAL rows.

    The hook receives normalized multiclass probabilities in VAL row order.
    Ranking uses raw probabilities and the first-class tie rule of NumPy
    argmax. The log probability and log margin use a finite probability floor.
    Working arrays use at most `max_block_rows` rows at a time and the paired
    state retains at most three one-dimensional arrays per observed mode.
    """

    def __init__(self, target_index, receiver_index, val_labels, *, max_block_rows=16384, probability_floor=1e-12):
        if (
            isinstance(target_index, (bool, np.bool_))
            or not isinstance(target_index, (int, np.integer))
            or target_index < 1
        ):
            raise ValueError('Target must be a positive attack class index')
        if (
            isinstance(receiver_index, (bool, np.bool_))
            or not isinstance(receiver_index, (int, np.integer))
            or receiver_index < 0
        ):
            raise ValueError('Receiver must be a nonnegative client index')
        if type(max_block_rows) is not int or max_block_rows < 1:
            raise ValueError('Target score blocks must have a positive integer row limit')
        if not np.isfinite(probability_floor) or not 0 < probability_floor <= 1:
            raise ValueError('Target probability floor must be in (0, 1]')
        labels = np.asarray(val_labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
            raise ValueError('Target score observer needs aligned integer VAL labels')
        self.target_index = int(target_index)
        self.receiver_index = int(receiver_index)
        self.val_labels = labels.copy()
        self.max_block_rows = max_block_rows
        self.probability_floor = float(probability_floor)
        self._scores = {}
        self._class_count = None

    def __call__(self, mode, index, probabilities):
        if index != self.receiver_index or mode not in (LOCAL, FULL):
            return
        if mode in self._scores:
            raise ValueError(f'Target scores already observed for {mode}')
        p = np.asarray(probabilities)
        if (
            p.ndim != 2
            or p.shape[0] != len(self.val_labels)
            or p.shape[1] <= self.target_index
            or not np.issubdtype(p.dtype, np.floating)
            or (self.val_labels >= p.shape[1]).any()
            or (self._class_count is not None and p.shape[1] != self._class_count)
        ):
            raise ValueError('Target score probabilities do not match receiver VAL labels or class space')
        class_count = p.shape[1]
        n = len(p)
        logp = np.empty(n, dtype=np.float64)
        margin = np.empty(n, dtype=np.float64) if class_count > 2 else None
        rank = np.empty(n, dtype=np.int32)
        c = self.target_index
        for start in range(0, n, self.max_block_rows):
            block = slice(start, start + self.max_block_rows)
            values = p[block]
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
            ):
                raise ValueError('Target scores require finite normalized probabilities')
            target = values[:, c]
            earlier, later = values[:, 1:c], values[:, c + 1 :]
            rank[block] = 1 + (earlier >= target[:, None]).sum(axis=1) + (later > target[:, None]).sum(axis=1)
            logp[block] = np.log(np.maximum(target.astype(np.float64), self.probability_floor))
            if margin is not None:
                competitor = np.maximum(earlier.max(axis=1, initial=0.0), later.max(axis=1, initial=0.0))
                margin[block] = logp[block] - np.log(np.maximum(competitor.astype(np.float64), self.probability_floor))
        self._class_count = class_count
        self._scores[mode] = _TargetScores(logp, margin, rank)

    @staticmethod
    def _mean(values, selected):
        return float(values[selected].mean()) if values is not None and selected.any() else None

    @staticmethod
    def _delta(left, right, selected):
        return float((right[selected] - left[selected]).mean()) if left is not None and selected.any() else None

    def _gated_predictions(self, predictions, mode):
        pred = np.asarray(predictions)
        if (
            pred.shape != self.val_labels.shape
            or not np.issubdtype(pred.dtype, np.integer)
            or (pred < 0).any()
            or (pred >= self._class_count).any()
        ):
            raise ValueError('Gated decisions must be aligned integer receiver VAL predictions')
        if not np.array_equal(pred == self.target_index, (pred > 0) & (self._scores[mode].attack_rank == 1)):
            raise ValueError('Gated target decisions differ from the observed attack ranking')
        return pred

    def summarize(self, local_predictions=None, full_predictions=None):
        """Return aggregate strata with full-minus-local score and hit changes.

        `non_target` includes both `benign` and `other_attack`. Gated predictions
        must be the corresponding hierarchical decision outputs for the scores.
        Empty strata and unavailable competing-attack margins have null means.
        """
        if self._scores.keys() != {LOCAL, FULL}:
            raise ValueError('Both conditional branches must be observed before summarizing')
        if (local_predictions is None) != (full_predictions is None):
            raise ValueError('Provide both gated prediction vectors or neither')
        local, full = self._scores[LOCAL], self._scores[FULL]
        true_target = self.val_labels == self.target_index
        benign = self.val_labels == 0
        top_local, top_full = local.attack_rank == 1, full.attack_rank == 1
        gained_top, lost_top = ~top_local & top_full, top_local & ~top_full
        have_gate = local_predictions is not None
        if have_gate:
            lp = self._gated_predictions(local_predictions, LOCAL)
            fp = self._gated_predictions(full_predictions, FULL)
            lt, ft = lp == self.target_index, fp == self.target_index
            gained_gate, lost_gate = ~lt & ft, lt & ~ft
        results = []
        for stratum, selected in (
            ('true_target', true_target),
            ('non_target', ~true_target),
            ('benign', benign),
            ('other_attack', ~true_target & ~benign),
        ):
            rows = int(selected.sum())
            row = {
                'receiver': self.receiver_index,
                'target_class_index': self.target_index,
                'stratum': stratum,
                'rows': rows,
                'probability_floor': self.probability_floor,
                'margin_rows': rows if local.attack_log_margin is not None else 0,
                'local_target_logp_mean': self._mean(local.log_probability, selected),
                'full_target_logp_mean': self._mean(full.log_probability, selected),
                'delta_target_logp_mean': self._delta(local.log_probability, full.log_probability, selected),
                'local_attack_log_margin_mean': self._mean(local.attack_log_margin, selected),
                'full_attack_log_margin_mean': self._mean(full.attack_log_margin, selected),
                'delta_attack_log_margin_mean': self._delta(local.attack_log_margin, full.attack_log_margin, selected),
                'local_attack_rank_mean': self._mean(local.attack_rank, selected),
                'full_attack_rank_mean': self._mean(full.attack_rank, selected),
                'delta_attack_rank_mean': self._delta(local.attack_rank, full.attack_rank, selected),
                'attack_rank_improved': int((selected & (full.attack_rank < local.attack_rank)).sum()),
                'attack_rank_worsened': int((selected & (full.attack_rank > local.attack_rank)).sum()),
                'local_attack_top1': int((selected & top_local).sum()),
                'full_attack_top1': int((selected & top_full).sum()),
                'local_attack_top3': int((selected & (local.attack_rank <= 3)).sum()),
                'full_attack_top3': int((selected & (full.attack_rank <= 3)).sum()),
                'new_attack_top1_target': int((selected & gained_top).sum()),
                'lost_attack_top1_target': int((selected & lost_top).sum()),
                'gated_decisions_available': have_gate,
            }
            if have_gate:
                row.update(
                    {
                        'local_gated_target': int((selected & lt).sum()),
                        'full_gated_target': int((selected & ft).sum()),
                        'new_gated_true_positive': int((selected & true_target & gained_gate).sum()),
                        'lost_gated_true_positive': int((selected & true_target & lost_gate).sum()),
                        'new_gated_false_positive': int((selected & ~true_target & gained_gate).sum()),
                        'removed_gated_false_positive': int((selected & ~true_target & lost_gate).sum()),
                        'local_attack_top1_blocked_by_gate': int((selected & top_local & (lp == 0)).sum()),
                        'full_attack_top1_blocked_by_gate': int((selected & top_full & (fp == 0)).sum()),
                        'new_gated_target_from_closed_gate': int((selected & gained_gate & (lp == 0)).sum()),
                        'new_gated_target_from_other_alert': int((selected & gained_gate & (lp > 0)).sum()),
                    }
                )
            results.append(row)
        return results
