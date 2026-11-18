"""Train local detectors and private class adapters for independent federations."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import utils as core
from ._iov_layout import project_root
from .privacy_core import ClassScoreAdapter, fit_class_adapter
from .revision_detector_objective import LOSS_NAMES, objective_class_weights, objective_settings, training_losses
from .utils import (
    cached_train_log_prior,
    empirical_alarm_threshold,
    hierarchical_decisions,
    risk_metrics,
    split_calibration,
    subtype_joint_probabilities,
)

ROOT = project_root(__file__)
DATASETS = ('CIC-IoV-2024', 'Car-Hacking', 'CIC-IDS-2017', 'CIC-UNSW-NB15')
DEFAULT_DATA = ROOT / 'dataset'


class PrivateClient:
    """Simulate one client in a shared process, without physical runtime isolation."""

    def __init__(self, client_id, parts, rows, names, features, args, private_dir):
        self.client_id, self.names, self.features = client_id, names, features
        self.directory = Path(private_dir)
        self.directory.mkdir(parents=True)
        self.rows = rows
        preprocessing_started = time.perf_counter()
        training_seed = getattr(args, 'training_seed', None)
        training_seed = args.seed if training_seed is None else training_seed
        core.seed(training_seed + int(client_id.rsplit('_', 1)[1]))
        # Both normalization and PLE quantiles are fitted on LOCAL TRAIN only.
        raw = parts['train'][0][rows['train']]
        self.transform = core.Transform().fit(raw, neural=True)
        self.parts = {r: (self.transform(parts[r][0][indices]), parts[r][1][indices]) for r, indices in rows.items()}
        self.performance = {
            'preprocessing_seconds': time.perf_counter() - preprocessing_started,
            'adapter_training_seconds': 0.0,
            'clipping_noise_seconds': 0.0,
            'upload_total_seconds': 0.0,
        }
        self.private_upload_diagnostics = []
        training_started = time.perf_counter()
        model, self.bins = core.make_model(
            len(features),
            len(names),
            'ple_cross',
            x=self.parts['train'][0],
            ple_boundary_policy=getattr(args, 'ple_boundary_policy', 'clip_to_train_edges'),
        )
        self.model = model.to(args.device)
        self.parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(args, 'detector_lr', 0.002),
            weight_decay=getattr(args, 'detector_weight_decay', 1e-4),
        )
        x, y = self.parts['train']
        self.log_prior = cached_train_log_prior(y, len(names))
        self.detector_objective = getattr(args, 'detector_objective', 'prior_adjusted')
        self.train_class_weights = objective_class_weights(y, len(names), self.detector_objective)
        loss_weights = (
            None
            if self.train_class_weights is None
            else torch.as_tensor(self.train_class_weights, dtype=torch.float32, device=args.device)
        )
        self.prior_strength = args.prior_strength
        prior_offset = args.prior_strength * torch.as_tensor(self.log_prior, dtype=torch.float32, device=args.device)
        rng = np.random.default_rng(training_seed + 79)
        self.training_history = []
        for epoch in range(args.base_epochs):
            epoch_started = time.perf_counter()
            total_loss = torch.zeros((), dtype=torch.float64, device=args.device)
            finite_monitor = core.TrainingFiniteMonitor(args.device, 'private detector')
            model.train()
            order = rng.permutation(len(y))
            for start in range(0, len(y), args.batch_size):
                at = order[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(torch.as_tensor(x[at], device=args.device))
                labels = torch.as_tensor(y[at], device=args.device)
                loss = training_losses(logits, labels, prior_offset, loss_weights, self.detector_objective).mean()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)
                finite_monitor.observe(loss, gradient_norm)
                optimizer.step()
                # Check finite loss at progress intervals and epoch boundaries.
                total_loss += loss.detach().double() * len(at)
                if (start // args.batch_size + 1) % 500 == 0:
                    finite_monitor.check()
                    if not math.isfinite(float(total_loss)):
                        raise FloatingPointError('Nonfinite private detector objective')
                    print(
                        f'LOCAL {self.client_id}: epoch {epoch + 1}/{args.base_epochs}, '
                        f'processed {min(start + args.batch_size, len(y)):,}/{len(y):,} TRAIN rows',
                        flush=True,
                    )
            finite_monitor.check()
            epoch_loss = float(total_loss) / len(y)
            if not math.isfinite(epoch_loss):
                raise FloatingPointError('Nonfinite private detector objective')
            history = {
                'epoch': epoch + 1,
                'training_loss': epoch_loss,
                'seconds': time.perf_counter() - epoch_started,
                'observed_train_rows': len(y),
                'maximum_preclip_gradient_norm': float(finite_monitor.max_gradient_norm),
                'loss_and_gradient_finite': True,
            }
            self.training_history.append(history)
            print(
                f'LOCAL {self.client_id}: epoch {epoch + 1}/{args.base_epochs}, '
                f"TRAIN={len(y):,}, loss={history['training_loss']:.6f}, seconds={history['seconds']:.1f}",
                flush=True,
            )
        self.performance['detector_training_seconds'] = time.perf_counter() - training_started
        self.performance['optimizer_steps'] = args.base_epochs * math.ceil(len(y) / args.batch_size)
        self.performance['training_row_visits'] = args.base_epochs * len(y)
        self.performance['training_rows_per_second'] = self.performance['training_row_visits'] / max(
            self.performance['detector_training_seconds'], 1e-12
        )
        del optimizer
        inference_started = time.perf_counter()
        self.probabilities = {
            r: core.probabilities(core.log_prediction(model, data[0], args.device, args.batch_size))
            for r, data in self.parts.items()
            if r in ('cal', 'val')
        }
        self.performance['inference_seconds'] = time.perf_counter() - inference_started
        self.performance['inference_rows'] = sum(len(values) for values in self.probabilities.values())
        self.performance['inference_ms_per_sample'] = (
            1000 * self.performance['inference_seconds'] / self.performance['inference_rows']
        )
        self.model = model.cpu()
        diagnostics_started = time.perf_counter()
        self.ple_stability = {
            role: core.ple_input_diagnostics(self.model, values) for role, (values, _) in self.parts.items()
        }
        core.save_json(self.directory / 'ple_stability_private.json', self.ple_stability)
        train_stability = self.ple_stability['train']
        self.performance.update(
            {
                'ple_diagnostics_seconds': time.perf_counter() - diagnostics_started,
                'ple_boundary_policy': getattr(self.model, 'ple_boundary_policy', 'unavailable'),
                'ple_boundary_version': core.PLE_BOUNDARY_VERSION,
                'ple_minimum_bin_width': train_stability.get('minimum_bin_width'),
                'ple_train_sample_rows': train_stability.get('sample_rows'),
                'ple_train_sample_outside_row_fraction': train_stability.get('sample_outside_row_fraction'),
                'ple_train_sample_encoding_abs_max': train_stability.get('sample_encoding_abs_max'),
            }
        )
        self.adapter_epochs, self.adapter_lr = args.adapter_epochs, args.adapter_lr
        self.adapter_prox = args.adapter_prox
        self.target_far, self.threshold_floor = args.target_far, args.threshold_floor
        self.fit_cal_rows, self.threshold_cal_rows = split_calibration(
            len(self.parts['cal'][1]),
            args.cal_fit_fraction,
            args.seed + 335 + int(client_id.rsplit('_', 1)[1]),
        )
        self.last_fit_initial, self.last_fit_vector = None, None

    def fit_update(self, initial):
        """Fit an adapter starting from the supplied coordinates on local CAL fit rows. Return the fitted displacement
        and record the fit and elapsed time.
        """
        started = time.perf_counter()
        fit = self.fit_cal_rows
        delta = fit_class_adapter(
            initial,
            self.probabilities['cal'][fit],
            self.parts['cal'][1][fit],
            self.log_prior,
            self.adapter_epochs,
            self.adapter_lr,
            self.prior_strength,
            self.adapter_prox,
        )
        self.last_fit_initial = np.asarray(initial).copy()
        self.last_fit_vector = np.asarray(initial) + delta
        self.performance['adapter_training_seconds'] += time.perf_counter() - started
        return delta

    def submit_secure(self, channel, initial, contract):
        """Client-side share submission; no unmasked vector returned to server."""
        delta = self.fit_update(initial)
        started = time.perf_counter()
        channel.submit(self.client_id, delta, contract=contract)
        self.performance['secure_submission_seconds'] = (
            self.performance.get('secure_submission_seconds', 0.0) + time.perf_counter() - started
        )

    def save_private(self, global_vector):
        """Personalize the supplied global vector, calibrate the alarm threshold and write the local checkpoint and
        diagnostics. Return the VAL metrics.
        """
        adapter = ClassScoreAdapter(len(self.names))
        if self.last_fit_initial is not None and np.array_equal(global_vector, self.last_fit_initial):
            local_vector = self.last_fit_vector
        else:
            fit = self.fit_cal_rows
            fit_started = time.perf_counter()
            delta = fit_class_adapter(
                global_vector,
                self.probabilities['cal'][fit],
                self.parts['cal'][1][fit],
                self.log_prior,
                self.adapter_epochs,
                self.adapter_lr,
                self.prior_strength,
                self.adapter_prox,
            )
            self.performance['adapter_training_seconds'] += time.perf_counter() - fit_started
            local_vector = global_vector + delta
        adapter.set_vector(local_vector)
        self.final_local_vector = adapter.get_vector()
        calibration_probabilities = adapter.transform(self.probabilities['cal'][self.threshold_cal_rows])
        self.alarm_threshold, calibration = empirical_alarm_threshold(
            calibration_probabilities,
            self.parts['cal'][1][self.threshold_cal_rows],
            self.target_far,
            self.threshold_floor,
        )
        self.final_probabilities = adapter.transform(self.probabilities['val'])
        metrics = risk_metrics(self.parts['val'][1], self.final_probabilities, self.names, self.alarm_threshold)
        # Keep client diagnostics private; do not upload this directory.
        core.save_json(self.directory / 'validation_private.json', metrics)
        core.save_json(self.directory / 'training_history_private.json', self.training_history)
        core.save_json(self.directory / 'performance_private.json', self.performance)
        core.save_json(self.directory / 'clipping_private.json', self.private_upload_diagnostics)
        core.save_json(self.directory / 'calibration_private.json', calibration)
        core.save_json(
            self.directory / 'training_prior_private.json',
            {
                'log_prior': self.log_prior.tolist(),
                'prior_strength': self.prior_strength,
                'loss': LOSS_NAMES[self.detector_objective],
                **objective_settings(self.detector_objective),
            },
        )
        np.savez(
            self.directory / 'calibration_split_private.npz',
            fit=self.fit_cal_rows,
            threshold=self.threshold_cal_rows,
        )
        np.savez_compressed(self.directory / 'partition_private.npz', **self.rows)
        torch.save(
            {
                'state': self.model.state_dict(),
                'bins': self.bins,
                'ple_boundary_policy': self.model.ple_boundary_policy,
                'ple_boundary_version': core.PLE_BOUNDARY_VERSION,
                'transform': self.transform.state(),
                'class_names': self.names,
                'feature_names': self.features,
                'framework_version': getattr(self, 'framework_version', 2),
                'adapter_schema': 'class_logit_bias_v2',
                'mechanism': getattr(self, 'mechanism', 'dp'),
                'class_score_adapter': self.final_local_vector.tolist(),
                'global_class_score_adapter': global_vector.tolist(),
                'alarm_threshold': self.alarm_threshold,
            },
            self.directory / 'detector_private.pt',
        )
        return metrics


def _predict_private_saved(checkpoint, x, device='cpu', return_subtypes=False):
    """Read a trusted local v3 checkpoint and apply its fitted final adapter."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (
        saved.get('framework_version') != 3
        or saved.get('mechanism') != 'mpc'
        or saved.get('adapter_schema') != 'class_logit_bias_v2'
        or 'attack_subtype_calibration' in saved
    ):
        raise ValueError('A fresh FedCAID MPC v3 checkpoint is required')
    transform = core.Transform.restore(saved['transform'])
    model, _ = core.make_model(
        len(saved['feature_names']),
        len(saved['class_names']),
        'ple_cross',
        bins=saved['bins'],
        # Historical checkpoints used unbounded PLE extrapolation. Preserve their
        # predictions instead of silently applying the new training policy.
        ple_boundary_policy=saved.get('ple_boundary_policy', 'extrapolate'),
    )
    model.load_state_dict(saved['state'], strict=True)
    p = core.probabilities(core.log_prediction(model.to(device), transform(x), device))
    adapter = ClassScoreAdapter(len(saved['class_names']))
    adapter.set_vector(saved['class_score_adapter'])
    p = adapter.transform(p)
    if return_subtypes:
        return p, saved.get('alarm_threshold', 0.5), None
    return subtype_joint_probabilities(p), saved.get('alarm_threshold', 0.5)


def predict_private(checkpoint, x, device='cpu'):
    """Return class probabilities for feature rows x using a trusted local MPC checkpoint on the selected device.
    """
    return _predict_private_saved(checkpoint, x, device)[0]


def predict_private_decisions(checkpoint, x, device='cpu'):
    """Return probabilities and hierarchical predictions for feature rows x using a trusted local MPC checkpoint and
    its saved threshold.
    """
    p, threshold, _ = _predict_private_saved(checkpoint, x, device, return_subtypes=True)
    return {'probabilities': subtype_joint_probabilities(p), **hierarchical_decisions(p, threshold)}


def execute(args):
    """Normalize the original RNG arguments, then run fresh secure federation."""
    args.training_seed = args.seed if args.training_seed is None else args.training_seed
    args.partition_seed = args.seed if args.partition_seed is None else args.partition_seed
    args.seed = args.partition_seed
    for field in ('clients', 'rounds', 'base_epochs', 'adapter_epochs', 'batch_size'):
        if getattr(args, field) < 1:
            raise ValueError(f'{field} must be positive')
    if args.clients < 3 or args.rows_per_client < 0 or args.seed < 0 or args.training_seed < 0:
        raise ValueError('MPC requires at least three clients and a nonnegative row cap/seed')
    if not math.isfinite(args.alpha) or args.alpha <= 0 or not math.isfinite(args.adapter_lr) or args.adapter_lr <= 0:
        raise ValueError('Positive finite Dirichlet alpha and adapter learning rate required')
    if (
        not 0 <= args.prior_strength <= 1
        or not math.isfinite(args.adapter_prox)
        or args.adapter_prox <= 0
        or not 0 < args.target_far < 1
        or not 0 <= args.threshold_floor <= 1
        or not 0 < args.cal_fit_fraction < 1
    ):
        raise ValueError('Invalid detector or calibration settings')
    if getattr(args, 'mechanism', 'mpc') != 'mpc' or getattr(args, 'improve_from', None) is not None:
        raise ValueError('This release supports the fresh MPC experiment only')
    from .secure_run import execute_secure

    return execute_secure(args)


def parser():
    """Return the command-line argument parser for the local MPC workflow."""
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--out', default=None, help='NEW output directory; default creates a timestamped directory')
    p.add_argument(
        '--smoke',
        action='store_true',
        help='Small synthetic functional check, not research evidence',
    )
    p.add_argument('--scope', choices=('all', 'can', 'network'), default='all')
    p.add_argument('--data-root', default=str(DEFAULT_DATA))
    p.add_argument('--clients', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--base-epochs', type=int, default=5)
    p.add_argument('--adapter-epochs', type=int, default=10)
    p.add_argument('--adapter-lr', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument(
        '--rows-per-client',
        type=int,
        default=0,
        help='0 means all assigned rows; positive cap is a pilot',
    )
    p.add_argument('--alpha', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42, help='Detector training and client partition seed')
    p.add_argument(
        '--partition-seed',
        type=int,
        default=None,
        help='Fixed client membership seed; defaults to --seed',
    )
    p.add_argument('--training-seed', type=int, default=None, help='Training RNG only; defaults to --seed')
    p.add_argument('--secure-scale', type=int, default=1000000, help='Fixed-point secure sum scale')
    p.add_argument('--secure-timeout', type=float, default=20.0, help='Fail-closed worker timeout seconds')
    p.add_argument('--prior-strength', type=float, default=0.5)
    p.add_argument('--adapter-prox', type=float, default=1.0)
    p.add_argument('--target-far', type=float, default=0.01)
    p.add_argument('--threshold-floor', type=float, default=0.5)
    p.add_argument('--cal-fit-fraction', type=float, default=0.5)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument(
        '--ple-boundary-policy',
        choices=core.PLE_BOUNDARY_POLICIES,
        default='clip_to_train_edges',
        help='Versioned TRAIN-only PLE boundary handling',
    )
    p.set_defaults(mechanism='mpc', improve_from=None, ledger_dir=None)
    return p


def main(argv=None):
    """Parse the supplied arguments or the current command line. Return the run exit code or 2 after a handled
    execution failure.
    """
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'FEDCAID STOP: {error}', flush=True)
        return 2
