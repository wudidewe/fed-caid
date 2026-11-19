"""Recover the exact 2017 calibration inputs for the frozen Hulk receivers."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .revision_data import load_revision_data
from .revision_followup_hulk import _frozen_hulk_heads, _source_artifacts
from .revision_next import _design, _scope_data


def _read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def load_original_calibration(project, frozen_run, infos):
    """Return the verified CAL rows by seed and the grouped 2017 VAL population."""
    source = Path(frozen_run).resolve(strict=True)
    _source_artifacts(source)
    heads = _frozen_hulk_heads(source)
    seed_infos = {info['seed']: info for info in infos}
    if len(seed_infos) != len(infos) or set(seed_infos) != {42, 43, 44, 45, 46}:
        raise ValueError('Expected the five archived Hulk cases for seeds 42 through 46.')

    dataset = 'CIC-IDS-2017'
    target = 'DoS Hulk'
    stage = source / 'attribution'
    settings = _read_json(stage / 'run.json')['settings']
    if settings['primary_cal_policy'] != 'reserved' or settings['operating_policy'] != 'cal-select':
        raise ValueError('The archived run has a different operating calibration policy.')
    cli = SimpleNamespace(**settings)
    data = load_revision_data(dataset, Path(project) / 'dataset', split_protocol='grouped')
    if data.audit != _read_json(stage / dataset / 'data_audit.json'):
        raise ValueError('The reconstructed grouped 2017 data audit differs from the archived run.')

    partitions = _scope_data(data)
    target_index = data.names.index(target)
    calibration = {}
    for head in heads:
        archived = _read_json(head.with_name('result.json'))
        identity = archived['identity']
        seed = identity['seed']
        info = seed_infos[seed]
        if (
            identity['dataset'] != dataset
            or identity['target_class'] != target
            or info['seed'] != seed
            or info['dataset'] != dataset
            or info['target_class'] != target
            or list(data.features) != info['feature_order']
            or list(data.names) != info['class_order']
        ):
            raise ValueError(f'The frozen 2017 schema or identity differs for seed {seed}.')
        parts, _, policies, audit = _design(partitions, data.names, identity, target_index, None, cli)
        if audit != archived['audit'] or audit['receiver'] != info['receiver']:
            raise ValueError(f'The reconstructed calibration route differs for seed {seed}.')
        policy = policies['reserved']
        receiver = info['receiver']
        frozen = info['fixed_gate']['operating_selection']
        if (
            policy['status'] != 'eligible'
            or policy['operating_audit']['assignment_sha256'] != frozen['cal_group_assignment_sha256']
            or policy['partition_sha256'] != info['settings']['cal_partition_sha256']
            or audit['route_sha256'] != info['settings']['route_sha256']
        ):
            raise ValueError(f'The archived calibration assignment differs for seed {seed}.')
        holdout = policy['roster'][receiver]['cal'][policy['splits'][receiver][1]]
        rows = holdout[policy['operating_splits'][receiver]['threshold_fit']]
        features, labels = parts['cal']
        labels = labels[rows]
        frozen_support = frozen['support']
        benign = int(np.count_nonzero(labels == 0))
        if (
            benign != frozen_support['threshold_fit_benign_rows'][receiver]
            or len(labels) - benign != frozen_support['threshold_fit_attack_rows'][receiver]
        ):
            raise ValueError(f'The reconstructed threshold-fit support differs for seed {seed}.')
        calibration[seed] = features[rows], labels
    return calibration, data.parts['val']
