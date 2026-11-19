"""Compare mapped flow features without choosing features or thresholds.

Feature quantiles are computed before the frozen model transformation. The
rate checks are internal identities, not evidence that a feature extractor used
the same definition for every feature.
"""

import csv
from pathlib import Path

import numpy as np


CLASSES = ('BENIGN', 'DoS Hulk', 'DoS Slowhttptest')
RATE_CHECKS = {
    'packet_rate_duration_ratio': (
        'Flow Packets/s',
        'Total Fwd Packets',
        'Total Backward Packets',
    ),
    'byte_rate_duration_ratio': (
        'Flow Bytes/s',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
    ),
}
FIELDS = (
    'dataset', 'role', 'class', 'measurement', 'name', 'units', 'rows',
    'valid_rows', 'missing_fraction', 'ineligible_fraction', 'p05', 'p50',
    'p95', 'fraction_within_one_percent_of_expected_unit_ratio',
)


def _quantiles(values):
    """Summarize finite values and leave empty strata without quantiles."""
    if not len(values):
        return {}
    return dict(zip(('p05', 'p50', 'p95'), np.quantile(values, (0.05, 0.50, 0.95)), strict=True))


def _feature_rows(values, feature_names):
    """Describe missing values and the distribution of each canonical column."""
    for index, name in enumerate(feature_names):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        yield {
            'measurement': 'feature_value',
            'name': name,
            'units': 'canonical input units',
            'valid_rows': len(finite),
            'missing_fraction': 1 - len(finite) / len(column) if len(column) else None,
            **_quantiles(finite),
        }


def _rate_rows(values, indices):
    """Check rate times duration divided by packet or byte totals."""
    duration = values[:, indices['Flow Duration']].astype(np.float64)
    for name, (rate_name, forward_name, backward_name) in RATE_CHECKS.items():
        rate = values[:, indices[rate_name]].astype(np.float64)
        forward = values[:, indices[forward_name]].astype(np.float64)
        backward = values[:, indices[backward_name]].astype(np.float64)
        total = forward + backward
        eligible = (
            np.isfinite(duration) & np.isfinite(rate) & np.isfinite(total)
            & (duration > 0) & (rate >= 0) & (forward >= 0) & (backward >= 0) & (total > 0)
        )
        ratio = rate[eligible] * duration[eligible] / (1e6 * total[eligible])
        yield {
            'measurement': 'unit_consistency',
            'name': name,
            'units': 'dimensionless, expected 1 for microseconds and per-second rate',
            'valid_rows': len(ratio),
            'ineligible_fraction': 1 - len(ratio) / len(values) if len(values) else None,
            'fraction_within_one_percent_of_expected_unit_ratio': (
                np.mean(np.abs(ratio - 1) <= 0.01) if len(ratio) else None
            ),
            **_quantiles(ratio),
        }


def write_feature_audit(
    output, external_x, external_y, development_x, development_y,
    feature_names, class_names, *, development_role='development_reference',
):
    """Write a descriptive three-class audit for the 2017 and 2018 inputs."""
    feature_names, class_names = tuple(feature_names), tuple(class_names)
    indices = {name: index for index, name in enumerate(feature_names)}
    required = {'Flow Duration', *(field for names in RATE_CHECKS.values() for field in names)}
    if len(indices) != len(feature_names) or not required <= indices.keys():
        raise ValueError('Feature audit requires unique canonical duration, rate, and count columns.')
    class_ids = {name: class_names.index(name) for name in CLASSES}
    populations = (
        ('CIC-IDS-2017', development_role, np.asarray(development_x), np.asarray(development_y)),
        ('CSE-CIC-IDS2018', '2018_02_16_capture', np.asarray(external_x), np.asarray(external_y)),
    )
    for dataset, _, features, labels in populations:
        if (
            features.ndim != 2 or features.shape[1] != len(feature_names)
            or features.dtype.kind not in 'iuf'
            or labels.shape != (len(features),) or labels.dtype.kind not in 'iu'
            or (labels < 0).any() or (labels >= len(class_names)).any()
        ):
            raise ValueError(f'Feature audit population does not match the frozen schema: {dataset}.')

    path = Path(output) / 'feature_audit.csv'
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dataset, role, features, labels in populations:
            for name, class_id in class_ids.items():
                values = features[labels == class_id]
                common = {'dataset': dataset, 'role': role, 'class': name, 'rows': len(values)}
                writer.writerows(
                    {**common, **row} for row in _feature_rows(values, feature_names)
                )
                writer.writerows({**common, **row} for row in _rate_rows(values, indices))
    return path
