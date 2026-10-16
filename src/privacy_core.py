"""Class score adapters for the fresh two-collector secure-sum experiment.

Transport and clipping are implemented in secure_aggregation.py. This module
does not add Gaussian noise or claim differential privacy.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def vector(value, dimension=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("A nonempty finite vector is required")
    if dimension is not None and len(result) != dimension:
        raise ValueError("Adapter dimension differs from the public contract")
    return result

class ClassScoreAdapter(nn.Module):
    """Class-logit correction in K-1 public zero-sum coordinates.

    The fixed Helmert basis removes the redundant common softmax offset.
    Every dataset uses the same construction; only its public class count
    determines the transmitted dimension. No scores or local priors upload.
    """
    schema = "class_logit_bias_v2"
    radius = 2.

    def __init__(self, classes):
        super().__init__()
        if type(classes) is not int or classes < 2:
            raise ValueError("At least two public classes required")
        self.classes = classes
        basis = torch.zeros((classes, classes - 1), dtype=torch.float64)
        for j in range(classes - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[:j + 1, j] = 1. / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis)
        self.theta = nn.Parameter(torch.zeros(classes - 1, dtype=torch.float64))

    def project(self, values):
        values = vector(values, self.classes - 1).copy()
        # Scale first so a finite but extreme input cannot overflow its norm.
        scale = float(np.abs(values).max())
        if scale:
            unit = values / scale
            unit_norm = float(np.linalg.norm(unit))
            if scale > self.radius / unit_norm:
                values = unit * (self.radius / unit_norm)
        return values

    def set_vector(self, values):
        with torch.no_grad():
            self.theta.copy_(torch.as_tensor(self.project(values), device=self.theta.device))

    def get_vector(self):
        return self.theta.detach().cpu().numpy().copy()

    def _probabilities(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        if (p.ndim != 2 or p.shape[1] != self.classes or not np.isfinite(p).all() or
                (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-6)):
            raise ValueError("Normalized finite probabilities with the public class count required")
        return p

    def transform(self, probabilities):
        p = self._probabilities(probabilities)
        with torch.no_grad():
            correction = (self.basis @ self.theta).cpu().numpy()
        logits = np.log(np.clip(p, 1e-12, 1.)) + correction
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
        return output / output.sum(axis=1, keepdims=True)

def fit_class_adapter(initial, probabilities, labels, log_prior, epochs=10, lr=.1,
                      prior_strength=.5, prox=1.):
    """Fit the same projected class correction locally for every domain.

    The prior-adjusted objective helps expose low-frequency class errors.
    Local probabilities, labels, and training priors remain client-private.
    The caller clips the WHOLE vector, then applies the chosen DP mechanism or
    secure-sharing protocol before release; the two guarantees are distinct.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("A local multiclass probability matrix is required")
    adapter = ClassScoreAdapter(p.shape[1])
    initial = vector(initial, p.shape[1] - 1)
    adapter.set_vector(initial)
    adapter._probabilities(p)
    labels = np.asarray(labels)
    prior = vector(log_prior, p.shape[1])
    if (labels.shape != (len(p),) or not len(labels) or
            not np.issubdtype(labels.dtype, np.integer) or
            (labels < 0).any() or (labels >= p.shape[1]).any()):
        raise ValueError("Valid nonempty local calibration labels required")
    if (type(epochs) is not int or epochs < 1 or not math.isfinite(lr) or lr <= 0 or
            not math.isfinite(prior_strength) or prior_strength < 0 or
            not math.isfinite(prox) or prox < 0):
        raise ValueError("Invalid local class-adapter training parameters")
    scores = torch.as_tensor(np.log(np.clip(p, 1e-12, 1.)))
    target = torch.as_tensor(labels, dtype=torch.long)
    prior_tensor = torch.as_tensor(prior)
    anchor = torch.as_tensor(initial)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = scores + adapter.basis @ adapter.theta + prior_strength * prior_tensor
        loss = F.cross_entropy(logits, target) + prox / 2 * (adapter.theta - anchor).square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite local class-calibration objective")
        loss.backward()
        optimizer.step()
        adapter.set_vector(adapter.get_vector())
    return adapter.get_vector() - initial
