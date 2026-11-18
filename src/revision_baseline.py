"""Full-detector FedAvg control for the local review experiment."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import utils as core
from .revision_detector_objective import objective_class_weights, objective_settings, training_losses


def _pooled_train_sample(train_parts, cap, seed):
    """Sample shared preprocessing rows by their positions in the pooled TRAIN."""
    offsets = np.cumsum([0, *(len(y) for _, y in train_parts)], dtype=np.int64)
    selected = np.random.default_rng(seed).choice(int(offsets[-1]), size=min(cap, int(offsets[-1])), replace=False)
    selected.sort()
    boundaries = np.searchsorted(selected, offsets)
    return np.concatenate(
        [x[selected[boundaries[i] : boundaries[i + 1]] - offsets[i]] for i, (x, _) in enumerate(train_parts)]
    )


def _epoch_schedule(epochs, rounds):
    if type(epochs) is not int or type(rounds) is not int or not 1 <= rounds <= epochs:
        raise ValueError('FedAvg requires 1 <= rounds <= detector epochs')
    epochs_per_round, remainder = divmod(epochs, rounds)
    return [epochs_per_round + int(i < remainder) for i in range(rounds)]


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _train_client(model, transform, train, loss_inputs, epochs, rng, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.detector_lr, weight_decay=args.detector_weight_decay)
    x, y = train
    prior, weights = loss_inputs
    objective = getattr(args, 'detector_objective', 'prior_adjusted')
    prior_offset = torch.as_tensor(prior, dtype=torch.float32, device=device)
    loss_weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    model.train()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    for _ in range(epochs):
        finite_monitor = core.TrainingFiniteMonitor(device, 'FedAvg detector')
        order = rng.permutation(len(y))
        for start in range(0, len(y), args.batch_size):
            rows = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(transform(x[rows]), device=device))
            labels = torch.as_tensor(y[rows], dtype=torch.long, device=device)
            loss = training_losses(logits, labels, prior_offset, loss_weights, objective).mean()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            finite_monitor.observe(loss, gradient_norm)
            optimizer.step()
            total_loss += loss.detach().double() * len(rows)
            if (start // args.batch_size + 1) % 500 == 0:
                finite_monitor.check()
        finite_monitor.check()
    mean_loss = float(total_loss) / (epochs * len(y))
    if not math.isfinite(mean_loss):
        raise FloatingPointError('Nonfinite FedAvg detector objective')
    return mean_loss


def _predict(model, transform, x, classes, device, batch_size):
    if not len(x):
        return np.empty((0, classes), dtype=np.float64)
    result = np.empty((len(x), classes), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        rows = slice(start, start + batch_size)
        log_p = core.log_prediction(model, transform(x[rows]), device, batch_size)
        result[rows] = core.probabilities(log_p)
    return result


def run_fedavg(train_parts, eval_features, classes, args, *, preprocessing_rows=65_536):
    """Fit shared CrossPLE TabM weights using only the matched clients' TRAIN rows.

    The local simulation fits a common TRAIN-only preprocessing artifact so that
    averaging PLE weights has a consistent meaning. It receives no evaluation
    labels. Each client sees the same total detector epochs as the local method.
    """
    train_parts = [(np.asarray(x), np.asarray(y)) for x, y in train_parts]
    eval_features = [np.asarray(x) for x in eval_features]
    if not train_parts or len(train_parts) != len(eval_features):
        raise ValueError('Matching nonempty FedAvg TRAIN and evaluation client lists required')
    if type(classes) is not int or classes < 2:
        raise ValueError('FedAvg requires a common class space with at least two classes')
    if type(preprocessing_rows) is not int or preprocessing_rows < 2:
        raise ValueError('FedAvg requires a preprocessing sample limit of at least two rows')
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError('FedAvg batch size must be a positive integer')
    for x, y in train_parts:
        if (
            x.ndim != 2
            or not x.shape[1]
            or y.ndim != 1
            or not len(y)
            or len(x) != len(y)
            or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any()
            or (y >= classes).any()
        ):
            raise ValueError('FedAvg requires nonempty TRAIN features and integer labels in the shared class space')
    width = train_parts[0][0].shape[1]
    if any(x.ndim != 2 or x.shape[1] != width for x in [*(x for x, _ in train_parts), *eval_features]):
        raise ValueError('FedAvg feature widths must match across clients and evaluation')
    schedule = _epoch_schedule(args.base_epochs, args.rounds)
    training_seed = getattr(args, 'training_seed', None)
    training_seed = args.seed if training_seed is None else training_seed
    device = torch.device(args.device)
    counts = [len(y) for _, y in train_parts]
    total_count = sum(counts)
    if total_count < 2:
        raise ValueError('FedAvg requires at least two pooled TRAIN rows')
    objective = getattr(args, 'detector_objective', 'prior_adjusted')
    loss_settings = objective_settings(objective)

    core.seed(training_seed)
    started = time.perf_counter()
    preprocessing_seed = training_seed + 104_729
    sample = _pooled_train_sample(train_parts, preprocessing_rows, preprocessing_seed)
    transform = core.Transform().fit(sample, neural=True)
    boundary_policy = getattr(args, 'ple_boundary_policy', 'clip_to_train_edges')
    model, bins = core.make_model(
        width,
        classes,
        'ple_cross',
        x=transform(sample),
        ple_boundary_policy=boundary_policy,
    )
    model.to(device)
    _sync(device)
    setup_seconds = time.perf_counter() - started
    del sample
    shared_setup_bytes = sum(a.nbytes for a in (transform.median, transform.mean, transform.scale))
    shared_setup_bytes += sum(b.numel() * b.element_size() for b in bins)
    loss_inputs = [
        (
            args.prior_strength * core.cached_train_log_prior(y, classes),
            objective_class_weights(y, classes, objective),
        )
        for _, y in train_parts
    ]
    shuffles = [np.random.default_rng(training_seed + 79) for _ in train_parts]

    trainable = dict(model.named_parameters())
    global_weights = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    parameter_count = sum(p.numel() for p in trainable.values())
    weight_bytes = sum(p.numel() * p.element_size() for p in trainable.values())
    client_payload = len(counts) * weight_bytes
    round_history = []
    training_started = time.perf_counter()
    for round_index, local_epochs in enumerate(schedule, start=1):
        round_started = time.perf_counter()
        weighted_sum = {name: torch.zeros_like(p, dtype=torch.float64) for name, p in global_weights.items()}
        mean_loss = 0.0
        for i, (train, local_loss, rng, count) in enumerate(
            zip(train_parts, loss_inputs, shuffles, counts, strict=True)
        ):
            with torch.no_grad():
                for name, p in trainable.items():
                    p.copy_(global_weights[name])
            torch.manual_seed(training_seed + 10_003 * round_index + i)
            client_weight = count / total_count
            loss = _train_client(model, transform, train, local_loss, local_epochs, rng, args, device)
            mean_loss += client_weight * loss
            for name, p in trainable.items():
                weighted_sum[name].add_(p.detach().cpu(), alpha=client_weight)

        round_delta_squared = 0.0
        for name, old in global_weights.items():
            merged = weighted_sum[name].to(old.dtype)
            if not torch.isfinite(merged).all():
                raise FloatingPointError('Nonfinite FedAvg detector weights')
            round_delta_squared += float((merged.double() - old.double()).square().sum())
            global_weights[name] = merged
        elapsed = time.perf_counter() - round_started
        round_history.append(
            {
                'round': round_index,
                'local_epochs': local_epochs,
                'training_row_visits': local_epochs * total_count,
                'mean_training_loss': mean_loss,
                'global_update_l2': math.sqrt(round_delta_squared),
                'uplink_payload_bytes': client_payload,
                'downlink_payload_bytes': client_payload,
                'seconds': elapsed,
            }
        )
        print(
            f'FEDAVG round {round_index}/{args.rounds}: local epochs={local_epochs}, '
            f'TRAIN visits={local_epochs * total_count:,}, loss={mean_loss:.6f}, seconds={elapsed:.1f}',
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started

    with torch.no_grad():
        for name, p in trainable.items():
            p.copy_(global_weights[name])
    _sync(device)
    inference_started = time.perf_counter()
    predictions = [_predict(model, transform, x, classes, device, args.batch_size) for x in eval_features]
    _sync(device)
    inference_seconds = time.perf_counter() - inference_started
    diagnostics_started = time.perf_counter()

    # Small, feature-only samples avoid retaining extra full transformed TRAIN
    # matrices and do not take part in fitting or model selection.
    def stability_sample(x):
        indices = np.linspace(0, len(x) - 1, min(len(x), 4096), dtype=np.int64)
        result = core.ple_input_diagnostics(model, transform(x[indices]))
        result['input_rows_before_sampling'] = len(x)
        return result

    stability = {
        'train_by_client': [stability_sample(x) for x, _ in train_parts],
        'evaluation_by_client': [stability_sample(x) for x in eval_features],
    }
    return {
        'probabilities': predictions,
        'performance': {
            'setup_seconds': setup_seconds,
            'training_seconds': training_seconds,
            'inference_seconds': inference_seconds,
            'inference_rows': sum(len(x) for x in eval_features),
            'training_row_visits': args.base_epochs * total_count,
            'optimizer_steps': args.base_epochs * sum(math.ceil(n / args.batch_size) for n in counts),
            'ple_diagnostics_seconds': time.perf_counter() - diagnostics_started,
        },
        'communication': {
            'parameter_count': parameter_count,
            'model_payload_bytes': weight_bytes,
            'round_uplink_payload_bytes': client_payload,
            'round_downlink_payload_bytes': client_payload,
            'round_payload_bytes': 2 * client_payload,
            'training_payload_bytes': 2 * args.rounds * client_payload,
            'shared_preprocessing_payload_bytes_per_client': shared_setup_bytes,
            'shared_preprocessing_setup_payload_bytes': len(counts) * shared_setup_bytes,
            'scope': (
                'Numeric trainable tensor bytes for one upload and one download per client per round. '
                'Fixed PLE grid and transform numeric bytes are separate one-time setup costs. '
                'Excludes headers, serialization, transport and preprocessing coordination.'
            ),
        },
        'settings': {
            'algorithm': 'FedAvg',
            'architecture': 'CrossPLE TabM',
            'epochs_by_round': schedule,
            'training_rows_by_client': counts,
            'aggregation': 'sample-count-weighted trainable tensors',
            'optimizer': 'fresh AdamW on each client at each round',
            'detector_lr': args.detector_lr,
            'detector_weight_decay': args.detector_weight_decay,
            'batch_size': args.batch_size,
            'prior_strength': args.prior_strength,
            'ple_boundary_policy': boundary_policy,
            'ple_boundary_version': core.PLE_BOUNDARY_VERSION,
            'ple_stability': stability,
            **loss_settings,
            'class_weights_by_client': [None if weights is None else weights.tolist() for _, weights in loss_inputs],
            'preprocessing_seed': preprocessing_seed,
            'preprocessing_rows': min(preprocessing_rows, total_count),
            'preprocessing': (
                'One common scaler and PLE grid fitted from a pooled TRAIN-only uniform sample '
                'in the local simulator. This baseline assumes access to pooled TRAIN features.'
            ),
            'final_model': 'last round, fixed in advance',
        },
        'rounds': round_history,
    }
