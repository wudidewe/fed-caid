"""Apply fixed alarm thresholds using a client's raw TRAIN CAN IDs."""

import numpy as np


ID_GATE_MODE = 'binary_swa_id_gate'
ID_GATE_RULE = 'client_train_can_id_fixed_thresholds'
SEEN_ID_THRESHOLD = 0.50
UNSEEN_ID_THRESHOLD = 0.70
MAX_CAN_ID = (1 << 29) - 1


def _parse_id(value):
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _valid_ids(raw_ids):
    raw_ids = np.asarray(raw_ids)
    if raw_ids.ndim != 1:
        raise ValueError('CAN IDs must be a one-dimensional raw feature column')
    if raw_ids.dtype.kind in 'iuf':
        values = raw_ids.astype(np.float64, copy=False)
    else:
        values = np.fromiter((_parse_id(value) for value in raw_ids), dtype=np.float64, count=len(raw_ids))
    valid = np.isfinite(values) & (values >= 0) & (values <= MAX_CAN_ID) & (values == np.trunc(values))
    return values[valid].astype(np.int64), valid


def id_thresholds(seen):
    """Return fixed thresholds from a one-dimensional Boolean TRAIN membership mask."""
    seen = np.asarray(seen)
    if seen.ndim != 1 or seen.dtype.kind != 'b':
        raise ValueError('CAN ID membership must be a one-dimensional Boolean mask')
    return np.where(seen, SEEN_ID_THRESHOLD, UNSEEN_ID_THRESHOLD)


class TrainIdGate:
    """Freeze a TRAIN vocabulary and use the prespecified seen and unseen thresholds."""

    def __init__(self, train_ids):
        ids, _ = _valid_ids(train_ids)
        self._train_ids = np.unique(ids)

    @property
    def training_id_count(self):
        """Return the number of distinct valid IDs in local TRAIN."""
        return len(self._train_ids)

    def seen(self, raw_ids):
        """Return one membership flag per row. Invalid or missing IDs are unseen."""
        ids, valid = _valid_ids(raw_ids)
        seen = np.zeros(valid.shape, dtype=bool)
        seen[valid] = np.isin(ids, self._train_ids)
        return seen

