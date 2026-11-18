"""Build row-held-out and exact-feature-grouped evaluation datasets."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import stream_source
from .dataset_preparation import _identity_codes, _verify_raw, check_raw_sources

ROLES = ('train', 'cal', 'val', 'test')
GROUP_SPLIT_SEED = 42
GROUP_SPLIT_PROPORTIONS = (0.70, 0.10, 0.10, 0.10)


@dataclass(frozen=True)
class RevisionDataset:
    """Hold feature and label arrays by role, ordered class and feature names, the split audit and the unseen TEST
    mask.
    """
    parts: dict
    names: list
    features: list
    audit: dict
    test_unseen_mask: np.ndarray


def _class_counts(labels, names):
    return dict(zip(names, np.bincount(labels, minlength=len(names)).tolist()))


def _matching_positions(sorted_values, values):
    positions = np.searchsorted(sorted_values, values)
    rows = np.flatnonzero(positions < len(sorted_values))
    rows = rows[sorted_values[positions[rows]] == values[rows]]
    return rows, positions[rows]


def _feature_keys(x):
    """Use exact bytes of model features, with one representation for missing values."""
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError('Feature fingerprints need a nonempty feature dimension')
    canonical = np.array(x, dtype='<f4', copy=True, order='C')
    canonical[~np.isfinite(canonical)] = np.nan
    canonical[canonical == 0] = 0.0
    return canonical.view(np.dtype((np.void, canonical.shape[1] * canonical.itemsize))).ravel()


def _audit_exact_inputs(parts, names, chunk_size):
    evaluation_roles = ('cal', 'val', 'test')
    lengths = [len(parts[role][1]) for role in evaluation_roles]
    keys, inverse = np.unique(
        np.concatenate([_feature_keys(parts[role][0]) for role in evaluation_roles]),
        return_inverse=True,
    )
    group_ids = dict(zip(evaluation_roles, np.split(inverse, np.cumsum(lengths)[:-1])))
    group_counts = {role: np.bincount(group_ids[role], minlength=len(keys)) for role in evaluation_roles}
    group_counts['train'] = np.zeros(len(keys), dtype=np.int64)
    train = parts['train'][0]
    for start in range(0, len(train), chunk_size):
        _, matches = _matching_positions(keys, _feature_keys(train[start : start + chunk_size]))
        group_counts['train'] += np.bincount(matches, minlength=len(keys))

    overlaps = {}
    for index, role in enumerate(ROLES):
        for other in ROLES[index + 1 :]:
            left, right = group_counts[role], group_counts[other]
            shared = (left > 0) & (right > 0)
            overlaps[f'{role}:{other}'] = {
                'unique_shared_inputs': int(shared.sum()),
                f'{role}_rows': int(left[shared].sum()),
                f'{other}_rows': int(right[shared].sum()),
            }

    test_groups, test_y = group_ids['test'], parts['test'][1]
    seen = np.logical_or.reduce([group_counts[role] > 0 for role in ROLES[:-1]])
    novel = ~seen[test_groups]
    low = np.full(len(keys), len(names), dtype=np.int64)
    high = np.full(len(keys), -1, dtype=np.int64)
    np.minimum.at(low, test_groups, test_y)
    np.maximum.at(high, test_groups, test_y)
    unique, representatives = np.unique(test_groups, return_index=True)
    keep = np.zeros(len(test_y), dtype=bool)
    clean = (~seen[unique]) & (low[unique] == high[unique])
    keep[representatives[clean]] = True

    return {
        'comparison': (
            'Exact equality of ordered pre-normalization float32 model features, '
            'with signed zero and missing/nonfinite values canonicalized and labels excluded'
        ),
        'cross_role_exact_overlap': overlaps,
        'within_role_repeats': {
            role: {
                'distinct_inputs': int(np.count_nonzero(group_counts[role])),
                'rows_beyond_one_per_distinct_input': lengths[i] - int(np.count_nonzero(group_counts[role])),
            }
            for i, role in enumerate(evaluation_roles)
        },
        'test_rows_matching_development_class_support': _class_counts(test_y[~novel], names),
        'test_exact_input_unseen_before_deduplication': _class_counts(test_y[novel], names),
        'test_exact_input_unseen_deduplicated': _class_counts(test_y[keep], names),
        'unseen_test_groups_with_conflicting_labels_excluded': int(
            ((~seen[unique]) & (low[unique] != high[unique])).sum()
        ),
        'unseen_test_subset_rule': (
            'Exclude every TEST input seen in TRAIN/CAL/VAL, discard conflicting-label '
            'TEST groups, and keep the first source row of each remaining input'
        ),
        'scope_limit': 'Does not check near-duplicates, session overlap or post-imputation equality',
    }, keep


def _allocate_feature_groups(groups, y, num_classes):
    """Assign whole groups using fixed randomization, never model outcomes."""
    group_count = int(groups.max()) + 1
    if not 1 <= num_classes <= 64:
        raise ValueError('Feature-group coverage supports one to 64 classes')
    group_labels = np.zeros(group_count, dtype=np.uint64)
    np.bitwise_or.at(group_labels, groups, np.uint64(1) << y.astype(np.uint64))
    present = [np.flatnonzero(group_labels & (np.uint64(1) << np.uint64(c))) for c in range(num_classes)]
    assigned = np.full(group_count, -1, dtype=np.int8)
    rng = np.random.default_rng(GROUP_SPLIT_SEED)
    # Reserve groups for the rarest classes first, prioritizing TRAIN, then TEST.
    for role in (0, 3, 1, 2):
        for candidates in sorted(present, key=len):
            if (assigned[candidates] == role).any():
                continue
            choices = candidates[assigned[candidates] == -1]
            if len(choices):
                assigned[rng.choice(choices)] = role
    unassigned = assigned == -1
    assigned[unassigned] = rng.choice(4, size=int(unassigned.sum()), p=GROUP_SPLIT_PROPORTIONS)
    return assigned, group_labels, present


def _grouped_resplit(parts, source_ids, names):
    combined_x = np.concatenate([parts[role][0] for role in ROLES])
    combined_y = np.concatenate([parts[role][1] for role in ROLES])
    combined_ids = np.concatenate([source_ids[role] for role in ROLES])
    _, groups = np.unique(_feature_keys(combined_x), return_inverse=True)
    assigned, group_labels, present = _allocate_feature_groups(groups, combined_y, len(names))
    role_assignment = assigned[groups]
    original_train_ids = source_ids['train']
    grouped_parts, grouped_ids = {}, {}
    for index, role in enumerate(ROLES):
        rows = np.flatnonzero(role_assignment == index)
        if not len(rows):
            raise ValueError(f'Exact-feature grouped split has no {role.upper()} rows')
        rows = rows[np.argsort(combined_ids[rows])]
        grouped_parts[role] = (combined_x[rows], combined_y[rows])
        grouped_ids[role] = combined_ids[rows]
    if len(np.unique(grouped_parts['train'][1])) != len(names):
        raise ValueError('Exact-feature grouped TRAIN split does not cover the full class space')
    ordered_train = np.sort(original_train_ids)
    historical_train_matches, _ = _matching_positions(ordered_train, grouped_ids['test'])
    conflicts = (group_labels & (group_labels - np.uint64(1))) != 0
    return (
        grouped_parts,
        grouped_ids,
        {
            'split_seed': GROUP_SPLIT_SEED,
            'unreserved_group_assignment_probabilities': dict(zip(ROLES, GROUP_SPLIT_PROPORTIONS)),
            'coverage_reservation_order': ['train', 'test', 'cal', 'val'],
            'source_feature_groups': len(assigned),
            'groups_with_multiple_labels': int(conflicts.sum()),
            'groups_supporting_each_class': dict(zip(names, map(len, present))),
            'assigned_groups': dict(zip(ROLES, np.bincount(assigned, minlength=4).tolist())),
            'test_rows_formerly_in_original_train': len(historical_train_matches),
        },
    )


def load_revision_data(dataset, dataset_root, *, chunk_size=250000, split_protocol='frozen'):
    """Replay the frozen reserve or re-split whole exact feature groups."""
    if chunk_size < 1 or split_protocol not in {'frozen', 'grouped'}:
        raise ValueError('Use a positive chunk size and the frozen or grouped split protocol')
    spec = check_raw_sources(dataset, Path(dataset_root))
    names, features = spec['class_names'], spec['feature_names']
    excluded = pd.read_csv(spec['excluded_identities_path'])
    heldout = pd.read_csv(spec['heldout_identities_path'])
    files = spec['source_files']
    excluded_codes = _identity_codes(excluded, files)
    heldout_codes = _identity_codes(heldout, files)
    if set(heldout.role) != {'cal', 'val'}:
        raise ValueError('Frozen heldout identities must specify CAL and VAL')
    if not np.isin(heldout_codes, excluded_codes).all():
        raise ValueError('CAL and VAL identities must be excluded from TRAIN')
    parts, source_ids = {}, {}
    for role in ('cal', 'val'):
        frame = heldout.loc[heldout.role == role].sort_values('row_in_role')
        if not np.array_equal(frame.row_in_role, np.arange(len(frame))):
            raise ValueError('Frozen heldout positions must be unique and consecutive')
        source_ids[role] = _identity_codes(frame, files)
        y = np.asarray([names.index(label) for label in frame['class']], dtype=np.int64)
        if np.bincount(y, minlength=len(names)).tolist() != spec['heldout_class_counts'][role]:
            raise ValueError(f'Frozen {role.upper()} supports differ from the recipe')
        parts[role] = (np.empty((len(y), len(features)), np.float32), y)

    reserve = excluded.loc[~np.isin(excluded_codes, heldout_codes)].sort_values(
        ['source_file_id', 'zero_based_data_row']
    )
    source_ids['test'] = _identity_codes(reserve, files)
    test_y = np.asarray([names.index(label) for label in reserve['class']], dtype=np.int64)
    if not np.array_equal(np.unique(test_y), np.arange(len(names))):
        raise ValueError('Frozen reserve is empty or does not cover the declared class space')
    parts['test'] = (np.empty((len(test_y), len(features)), np.float32), test_y)
    count = spec['train_rows']
    parts['train'] = (np.empty((count, len(features)), np.float32), np.empty(count, dtype=np.int64))
    source_ids['train'] = np.empty(count, dtype=np.uint64)
    train_order = np.random.default_rng(spec['seed']).permutation(count)
    train_positions = np.empty_like(train_order)
    train_positions[train_order] = np.arange(count)
    del train_order
    orders = {role: np.argsort(source_ids[role]) for role in ROLES[1:]}
    sorted_ids = {role: source_ids[role][orders[role]] for role in ROLES[1:]}
    excluded_codes = np.sort(excluded_codes)
    seen_counts = dict.fromkeys(ROLES, 0)
    previous_code = None
    for file_id, rows, labels, values in stream_source(spec, True, chunk_size):
        if not len(rows):
            continue
        codes = (np.uint64(file_id) << np.uint64(48)) | rows.astype(np.uint64)
        if (previous_code is not None and previous_code >= codes[0]) or (codes[1:] <= codes[:-1]).any():
            raise ValueError('Raw source identities must be unique and in physical order')
        previous_code = codes[-1]
        excluded_rows, _ = _matching_positions(excluded_codes, codes)
        train = np.ones(len(rows), dtype=bool)
        train[excluded_rows] = False
        start = seen_counts['train']
        end = start + int(train.sum())
        if end > count:
            raise ValueError('Raw TRAIN rows exceed the frozen count')
        destination = train_positions[start:end]
        parts['train'][0][destination] = values[train]
        parts['train'][1][destination] = labels[train]
        source_ids['train'][destination] = codes[train]
        seen_counts['train'] = end
        for role in ROLES[1:]:
            selected, positions = _matching_positions(sorted_ids[role], codes)
            target = orders[role][positions]
            if not np.array_equal(labels[selected], parts[role][1][target]):
                raise ValueError(f'Raw labels differ from the {role.upper()} identity manifest')
            parts[role][0][target] = values[selected]
            seen_counts[role] += len(target)

    if any(seen_counts[role] != len(parts[role][1]) for role in ROLES):
        raise ValueError('Raw source replay did not find all frozen role identities')
    if np.bincount(parts['train'][1], minlength=len(names)).tolist() != spec['train_class_counts']:
        raise ValueError('Raw TRAIN label counts differ from the frozen recipe')
    source_counts = sum(np.bincount(parts[role][1], minlength=len(names)) for role in ROLES)
    if source_counts.tolist() != spec['source_class_counts']:
        raise ValueError('Raw source label counts differ from the frozen recipe')
    for source in files:
        _verify_raw(source)
    del train_positions
    parts = {role: parts[role] for role in ROLES}

    grouping = None
    if split_protocol == 'grouped':
        parts, source_ids, grouping = _grouped_resplit(parts, source_ids, names)
    exact, novel = _audit_exact_inputs(parts, names, chunk_size)
    if split_protocol == 'grouped' and any(
        pair['unique_shared_inputs'] for pair in exact['cross_role_exact_overlap'].values()
    ):
        raise ValueError('The grouped split has unexpected cross-role feature overlap')
    return RevisionDataset(
        parts,
        names,
        features,
        {
            'split_protocol': split_protocol,
            'evaluation_scope': (
                'held-out source rows from the pinned captures excluded from original TRAIN/CAL/VAL'
                if split_protocol == 'frozen'
                else 'retrospective exact-feature grouped re-split of the same pinned captures'
            ),
            'external_captures': False,
            'test_frozen_reserve_rule': 'excluded_identities minus original CAL and VAL identities',
            'source_recipe_sha256': spec['recipe_sha256'],
            'source_files_sha256': {source['filename']: source['sha256'] for source in files},
            'source_identity_disjoint_by_construction': True,
            'source_identity_encoding': (
                'Little-endian uint64, (source_file_id << 48) | zero_based_data_row, in role row order'
            ),
            'source_identity_sha256_by_role': {
                role: hashlib.sha256(np.ascontiguousarray(ids, dtype='<u8')).hexdigest()
                for role, ids in source_ids.items()
            },
            'role_rows': {role: len(parts[role][1]) for role in ROLES},
            'role_class_support': {role: _class_counts(parts[role][1], names) for role in ROLES},
            'nb15_data_csv_label_csv_used': False,
            'exact_feature_audit': exact,
            'grouped_resplit': grouping,
        },
        novel,
    )
