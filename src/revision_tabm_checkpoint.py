"""Freeze the raw local CrossTabM detector used by the two-stage IDS.

Only plain metadata and tensors are serialized. No historical class adapter is
part of this checkpoint. The TRAIN prior is saved for provenance and remains
a training-loss term, consistent with the original raw detector inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import torch

from . import utils as core
from .revision_iov_binary import BinaryAlarmTabM, make_binary_alarm_model

SCHEMA = 'fedcaid_raw_cross_tabm_v1'
MODEL_KIND = 'ple_cross'
BINARY_MODEL_KIND = 'ple_cross_binary_alarm'
INFERENCE_RULE = 'mean_ensemble_probability_without_prior_offset'


def _names(values):
    return (
        isinstance(values, (list, tuple))
        and bool(values)
        and all(isinstance(value, str) and value for value in values)
        and len(set(values)) == len(values)
    )


def _finite_vector(tensor, width, dtype):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.shape == (width,)
        and tensor.dtype == dtype
        and bool(torch.isfinite(tensor).all())
    )


def save_tabm_checkpoint(path, node, *, batch_size=1024, inference_device='cpu'):
    """Export a fitted local detector before the experiment releases it."""
    path = Path(path).resolve()
    payload = {
        'schema': SCHEMA,
        'model_kind': BINARY_MODEL_KIND if isinstance(node.model, BinaryAlarmTabM) else MODEL_KIND,
        'inference_rule': INFERENCE_RULE,
        'client_id': node.client_id,
        'class_names': list(node.names),
        'feature_names': list(node.features),
        'ple_boundary_policy': node.model.ple_boundary_policy,
        'ple_boundary_version': core.PLE_BOUNDARY_VERSION,
        **({'feature_attention': True} if node.model.feature_attention is not None else {}),
        **(
            {'can_id_values': node.model.can_id_values.detach().cpu().clone()}
            if node.model.can_id_values is not None else {}
        ),
        'bins': [value.detach().cpu().clone() for value in node.bins],
        'transform': {
            'median': torch.as_tensor(node.transform.median, dtype=torch.float32).clone(),
            'mean': torch.as_tensor(node.transform.mean, dtype=torch.float64).clone(),
            'scale': torch.as_tensor(node.transform.scale, dtype=torch.float64).clone(),
            'neural': bool(node.transform.neural),
        },
        'state': {key: value.detach().cpu().clone() for key, value in node.model.state_dict().items()},
        'train_log_prior': torch.as_tensor(node.log_prior, dtype=torch.float64).clone(),
        'prior_strength': float(node.prior_strength),
        'batch_size': batch_size,
        'inference_device': str(inference_device),
    }
    _validate(payload, node.names, node.features)
    with path.open('xb') as stream:
        try:
            torch.save(payload, stream)
        except BaseException:
            stream.close()
            path.unlink(missing_ok=True)
            raise
    return {
        'file': str(path),
        'sha256': core.sha(path),
        'schema': SCHEMA,
        'client_id': payload['client_id'],
        'batch_size': batch_size,
        'inference_device': payload['inference_device'],
    }


def _validate(payload, names, features):
    if (
        not isinstance(payload, dict)
        or payload.get('schema') != SCHEMA
        or payload.get('model_kind') not in (MODEL_KIND, BINARY_MODEL_KIND)
        or (payload.get('model_kind') == BINARY_MODEL_KIND and payload.get('feature_attention') is not True)
        or payload.get('inference_rule') != INFERENCE_RULE
        or payload.get('ple_boundary_version') != core.PLE_BOUNDARY_VERSION
        or payload.get('ple_boundary_policy') not in core.PLE_BOUNDARY_POLICIES
        or type(payload.get('feature_attention', False)) is not bool
    ):
        raise ValueError('Unrecognized frozen raw TabM model schema or boundary policy')
    classes, inputs = payload.get('class_names'), payload.get('feature_names')
    if not _names(classes) or len(classes) < 2 or classes[0] != 'BENIGN' or not _names(inputs):
        raise ValueError('Frozen TabM requires unique feature names and a BENIGN-first class order')
    if (names is not None and tuple(classes) != tuple(names)) or (
        features is not None and tuple(inputs) != tuple(features)
    ):
        raise ValueError('Frozen TabM class or feature order differs')
    width = len(inputs)
    if 'can_id_values' in payload:
        values = payload['can_id_values']
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or not len(values)
            or not _finite_vector(values, len(values), torch.float32)
            or not bool((values.diff() > 0).all())
            or not bool((values.abs() <= 12).all())
        ):
            raise ValueError('Invalid frozen TabM TRAIN CAN ID values')
    transform = payload.get('transform')
    if (
        not isinstance(transform, dict)
        or transform.get('neural') is not True
        or not _finite_vector(transform.get('median'), width, torch.float32)
        or not _finite_vector(transform.get('mean'), width, torch.float64)
        or not _finite_vector(transform.get('scale'), width, torch.float64)
        or not bool((transform['scale'] > 0).all())
    ):
        raise ValueError('Invalid frozen TabM TRAIN preprocessing')
    bins, state = payload.get('bins'), payload.get('state')
    if (
        not isinstance(bins, list)
        or len(bins) != width
        or not all(
            isinstance(edges, torch.Tensor)
            and edges.ndim == 1
            and len(edges) >= 2
            and edges.dtype == torch.float32
            and bool(torch.isfinite(edges).all())
            and bool((edges.diff() > 0).all())
            for edges in bins
        )
    ):
        raise ValueError('Invalid frozen TabM TRAIN bins')
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and value.layout == torch.strided
            and bool(torch.isfinite(value).all())
            for key, value in state.items()
        )
    ):
        raise ValueError('Invalid frozen TabM tensor state')
    prior, strength = payload.get('train_log_prior'), payload.get('prior_strength')
    client_id = payload.get('client_id')
    if (
        not isinstance(client_id, str)
        or not client_id
        or not _finite_vector(prior, len(classes), torch.float64)
        or type(strength) not in (int, float)
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
        or type(payload.get('batch_size')) is not int
        or payload['batch_size'] < 1
        or payload.get('inference_device') not in ('cpu', 'cuda')
    ):
        raise ValueError('Invalid frozen TabM provenance or inference settings')


@dataclass(frozen=True)
class FrozenTabMPredictor:
    """Callable raw detector with fixed transforms and ordered feature metadata."""

    model: torch.nn.Module
    transform: core.Transform
    names: tuple[str, ...]
    features: tuple[str, ...]
    client_id: str
    inference_device: str
    batch_size: int

    def __call__(self, x, *, batch_size=None):
        x = np.asarray(x)
        batch = self.batch_size if batch_size is None else batch_size
        if x.ndim != 2 or x.shape[1] != len(self.features) or x.dtype.kind not in 'fiu':
            raise ValueError('Expected a numeric feature matrix in the frozen TabM feature order')
        if type(batch) is not int or batch < 1:
            raise ValueError('Frozen TabM inference batch size must be a positive integer')
        output = np.empty((len(x), len(self.names)), dtype=np.float64)
        chunk = 32 * batch
        for start in range(0, len(x), chunk):
            at = slice(start, start + chunk)
            logp = core.log_prediction(self.model, self.transform(x[at]), self.inference_device, batch)
            output[at] = core.probabilities(logp)
        if not np.isfinite(output).all():
            raise FloatingPointError('Frozen raw TabM produced nonfinite probabilities')
        return output


def tabm_node_predictor(node, *, batch_size=1024):
    """Reuse the trained CPU node with the same scoring path as a frozen detector."""
    if any(value.device.type != 'cpu' for value in node.model.parameters()):
        raise ValueError('The raw TabM node must be on CPU for canonical inference')
    return FrozenTabMPredictor(
        node.model.eval(),
        node.transform,
        tuple(node.names),
        tuple(node.features),
        node.client_id,
        'cpu',
        batch_size,
    )


def load_tabm_checkpoint(path, *, expected_sha256=None, names=None, features=None, device=None):
    """Load and verify one frozen detector without unpickling custom objects."""
    data = Path(path).read_bytes()
    if expected_sha256 is not None and sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Frozen TabM checkpoint SHA256 differs')
    try:
        payload = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
    except (RuntimeError, UnpicklingError, EOFError) as error:
        raise ValueError('Invalid frozen TabM checkpoint data') from error
    _validate(payload, names, features)
    device = payload['inference_device'] if device is None else str(device)
    if device not in ('cpu', 'cuda') or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError(f'Unsupported or unavailable frozen TabM inference device: {device}')
    classes, inputs = tuple(payload['class_names']), tuple(payload['feature_names'])
    with torch.random.fork_rng(devices=[]):
        model, _ = core.make_model(
            len(inputs),
            len(classes),
            MODEL_KIND,
            bins=payload['bins'],
            ple_boundary_policy=payload['ple_boundary_policy'],
            feature_attention=payload.get('feature_attention', False),
            can_id_training=payload.get('can_id_values'),
        )
        if payload['model_kind'] == BINARY_MODEL_KIND:
            model = make_binary_alarm_model(model, payload['bins'])
    state = payload['state']
    expected = model.state_dict()
    if expected.keys() != state.keys() or any(
        expected[key].shape != state[key].shape or expected[key].dtype != state[key].dtype for key in expected
    ):
        raise ValueError('Frozen TabM tensor schema differs from the reconstructed detector')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    saved = payload['transform']
    transform = core.Transform.restore(
        {
            'median': saved['median'].numpy(),
            'mean': saved['mean'].numpy(),
            'scale': saved['scale'].numpy(),
            'neural': True,
        }
    )
    return FrozenTabMPredictor(model, transform, classes, inputs, payload['client_id'], device, payload['batch_size'])


def predict_tabm_checkpoint(path, x, *, batch_size=None, device=None):
    """Load the frozen detector at path and return class probabilities for feature rows x. Optional batch size and
    device override the saved defaults.
    """
    return load_tabm_checkpoint(path, device=device)(x, batch_size=batch_size)
