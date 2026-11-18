"""Reserve existing, all-benign CAL groups without touching TRAIN or evaluation."""

from hashlib import sha256

import numpy as np

from .revision_cal_split import InfeasibleDesign
from .revision_data import _feature_keys


def _digest(values):
    return sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def _group_codes(x, owned, group_ids):
    _, feature_groups = np.unique(_feature_keys(x[owned]), return_inverse=True)
    if group_ids is None:
        return feature_groups
    keys = np.asarray(group_ids)
    if keys.shape != (len(x),) or keys.dtype.kind not in 'iuUS':
        raise ValueError('CAL reservation group IDs must be one integer or string per CAL row')
    _, groups = np.unique(keys[owned], return_inverse=True)
    lowest = np.full(int(feature_groups.max()) + 1, len(owned), dtype=np.int64)
    highest = np.full(len(lowest), -1, dtype=np.int64)
    np.minimum.at(lowest, feature_groups, groups)
    np.maximum.at(highest, feature_groups, groups)
    if not np.array_equal(lowest, highest):
        raise ValueError('CAL reservation group IDs split an exact feature group')
    return groups


def _support(labels, groups, owners, client):
    selected = owners == client
    benign = selected & (labels == 0)
    return {
        'rows': int(selected.sum()),
        'benign_rows': int(benign.sum()),
        'feature_groups': int(len(np.unique(groups[selected]))),
        'benign_feature_groups': int(len(np.unique(groups[benign]))),
    }


def reserve_benign_cal(cal_x, cal_y, roster, floor, seed, *, group_ids=None):
    """Reallocate already-owned CAL groups for a controlled calibration-budget study.

    A globally excluded fit anchor per client provides a witness for a nonempty
    fit side. The remaining groups must supply `floor` benign holdout rows at
    each client. Only all-benign groups can move, and a moved group goes to one
    client in full. Donors retain the floor and their fit anchor. Untouched
    routes and the within-client order of retained CAL rows are preserved.

    This deterministic transfer policy can stop when no admissible group can
    fill a shortage. Its failure does not classify all possible assignments.
    Explicit groups may join exact feature groups. This function does not
    change the subsequent CAL-fit/holdout splitter.
    """
    x, y = np.asarray(cal_x), np.asarray(cal_y)
    if x.ndim != 2 or not x.shape[1] or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('CAL reservation requires aligned features and nonnegative integer labels')
    if type(floor) is not int or floor < 1 or type(seed) is not int or seed < 0:
        raise ValueError('CAL reservation requires a positive benign floor and a nonnegative integer seed')
    if not roster:
        raise ValueError('CAL reservation requires at least one client')
    indices = [np.asarray(client['cal']) for client in roster]
    if any(
        rows.ndim != 1
        or not len(rows)
        or not np.issubdtype(rows.dtype, np.integer)
        or (rows < 0).any()
        or (rows >= len(y)).any()
        for rows in indices
    ):
        raise ValueError('CAL reservation requires nonempty valid CAL row indices at each client')
    owned = np.concatenate(indices).astype(np.int64, copy=False)
    if len(np.unique(owned)) != len(owned):
        raise ValueError('CAL reservation requires one owner per CAL row')
    n_clients = len(indices)
    owners = np.repeat(np.arange(n_clients), list(map(len, indices)))
    labels = y[owned]
    benign = labels == 0
    if int(benign.sum()) < n_clients * floor:
        raise InfeasibleDesign('CAL reservation has insufficient total benign rows for the declared floor')

    groups = _group_codes(x, owned, group_ids)
    group_count = int(groups.max()) + 1
    rng = np.random.default_rng(seed + 4421)
    group_priority = rng.permutation(group_count)
    benign_count = np.bincount(groups[benign], minlength=group_count)
    group_size = np.bincount(groups, minlength=group_count)

    # Anchor the group with the lightest benign footprint among each client's
    # groups. All anchors count as fit globally, including shared groups.
    anchors = []
    for client in range(n_clients):
        choices = np.unique(groups[owners == client])
        index = np.lexsort((group_priority[choices], benign_count[choices]))[0]
        anchors.append(int(choices[index]))
    anchor = np.zeros(group_count, dtype=bool)
    anchor[anchors] = True
    working_benign = benign & ~anchor[groups]
    allowance = np.bincount(owners[working_benign], minlength=n_clients)
    if int(allowance.sum()) < n_clients * floor:
        raise InfeasibleDesign('CAL reservation cannot supply the floor after preserving fit groups')

    # Store only occupied group/client pairs to avoid a dense
    # (number of groups by number of clients) ownership table.
    owner_keys, owner_sizes = np.unique(groups * n_clients + owners, return_counts=True)
    offsets = np.searchsorted(owner_keys, np.arange(group_count + 1, dtype=np.int64) * n_clients)
    pair_owners = owner_keys % n_clients
    movable = np.flatnonzero(~anchor & (benign_count == group_size))
    ordered = movable[np.lexsort((group_priority[movable], group_size[movable]))]
    assigned = np.full(group_count, -1, dtype=np.int64)
    for client in np.argsort(allowance, kind='stable'):
        shortage = floor - allowance[client]
        if shortage <= 0:
            continue
        # Try the smallest groups first, which limits budget overshoot and
        # prefers reassignment of fewer rows when multiple groups are feasible.
        for group in ordered:
            if assigned[group] >= 0:
                continue
            positions = slice(offsets[group], offsets[group + 1])
            donors, moved = pair_owners[positions], owner_sizes[positions]
            others = donors != client
            donors, moved = donors[others], moved[others]
            if not len(donors) or (allowance[donors] - moved < floor).any():
                continue
            allowance[donors] -= moved
            allowance[client] += moved.sum()
            assigned[group] = client
            if allowance[client] >= floor:
                break
        if allowance[client] < floor:
            raise InfeasibleDesign(
                f'CAL reservation could not meet client {client} benign floor '
                f'({int(allowance[client])}/{floor}) using transferable all-benign groups'
            )

    replacements = assigned[groups]
    new_owners = np.where(replacements < 0, owners, replacements)
    moved = new_owners != owners
    if (moved & ~benign).any() or not np.array_equal(
        allowance, np.bincount(new_owners[working_benign], minlength=n_clients)
    ):
        raise RuntimeError('Invalid CAL reservation result')
    new_roster, receipts = [], []
    for client, original in enumerate(roster):
        stays = owned[(owners == client) & ~moved]
        incoming = np.sort(owned[(new_owners == client) & moved])
        rows = np.concatenate((stays, incoming))
        new_roster.append({**original, 'cal': rows})
        receipts.append(
            {
                'client': client,
                'before': _support(labels, groups, owners, client),
                'after': _support(labels, groups, new_owners, client),
                'moved_in_benign_rows': len(incoming),
                'moved_out_benign_rows': int(((owners == client) & moved).sum()),
                'non_anchor_benign_rows': int(allowance[client]),
                'cal_source_indices_sha256': _digest(rows),
            }
        )
    if not np.array_equal(np.sort(np.concatenate([v['cal'] for v in new_roster])), np.sort(owned)):
        raise RuntimeError('CAL reservation lost or duplicated row identities')
    return new_roster, {
        'method': 'deterministic whole all-benign CAL group reassignment',
        'grouping': (
            'exact model-input feature groups'
            if group_ids is None
            else 'explicit CAL groups preserving exact model-input feature groups'
        ),
        'seed': seed,
        'min_holdout_benign_rows': floor,
        'owned_cal_rows': len(owned),
        'reservation_groups': group_count,
        'witness_fit_groups': int(anchor.sum()),
        'moved_benign_rows': int(moved.sum()),
        'moved_attack_rows': 0,
        'moved_groups': int((assigned >= 0).sum()),
        'eligible_roles': ['cal'],
        'clients': receipts,
        'scope': 'Controlled CAL resource policy; no independent-capture, privacy or FAR guarantee',
    }
