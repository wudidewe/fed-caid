"""Class score adapters for secure aggregation without differential privacy."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional


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
    """Correct class logits in K-1 zero-sum coordinates without uploading scores."""

    SCHEMA = "class_logit_bias_v2"
    RADIUS = 2.0

    def __init__(self, classes):
        super().__init__()
        if type(classes) is not int or classes < 2:
            raise ValueError("At least two public classes required")
        self.classes = classes
        basis = torch.zeros((classes, classes - 1), dtype=torch.float64)
        for j in range(classes - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
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
            if scale > self.RADIUS / unit_norm:
                values = unit * (self.RADIUS / unit_norm)
        return values

    def set_vector(self, values):
        with torch.no_grad():
            self.theta.copy_(torch.as_tensor(self.project(values), device=self.theta.device))

    def get_vector(self):
        return self.theta.detach().cpu().numpy().copy()

    def _probabilities(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        if (
            p.ndim != 2
            or p.shape[1] != self.classes
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(1), 1.0, atol=1e-6)
        ):
            raise ValueError("Normalized finite probabilities with the public class count required")
        return p

    def transform(self, probabilities):
        p = self._probabilities(probabilities)
        with torch.no_grad():
            correction = (self.basis @ self.theta).cpu().numpy()
        logits = np.log(np.clip(p, 1e-12, 1.0)) + correction
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
        return output / output.sum(axis=1, keepdims=True)


def fit_class_adapter(
    initial,
    probabilities,
    labels,
    log_prior,
    epochs=10,
    lr=0.1,
    prior_strength=0.5,
    prox=1.0,
    *,
    diagnostics=None,
):
    """Fit a local class correction; clip and securely share the update before release."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("A local multiclass probability matrix is required")
    adapter = ClassScoreAdapter(p.shape[1])
    initial = vector(initial, p.shape[1] - 1)
    adapter.set_vector(initial)
    adapter._probabilities(p)
    labels = np.asarray(labels)
    prior = vector(log_prior, p.shape[1])
    if (
        labels.shape != (len(p),)
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= p.shape[1]).any()
    ):
        raise ValueError("Valid nonempty local calibration labels required")
    if (
        type(epochs) is not int
        or epochs < 1
        or not math.isfinite(lr)
        or lr <= 0
        or not math.isfinite(prior_strength)
        or prior_strength < 0
        or not math.isfinite(prox)
        or prox < 0
    ):
        raise ValueError("Invalid local class-adapter training parameters")
    scores = torch.as_tensor(np.log(np.clip(p, 1e-12, 1.0)))
    target = torch.as_tensor(labels, dtype=torch.long)
    prior_tensor = torch.as_tensor(prior)
    anchor = torch.as_tensor(initial)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)
    projection_steps = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = scores + adapter.basis @ adapter.theta + prior_strength * prior_tensor
        loss = functional.cross_entropy(logits, target) + prox / 2 * (adapter.theta - anchor).square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite local class-calibration objective")
        loss.backward()
        if diagnostics is not None and epoch == 0:
            diagnostics["first_gradient_l2"] = float(torch.linalg.vector_norm(adapter.theta.grad))
        optimizer.step()
        proposal = adapter.get_vector()
        if diagnostics is not None:
            projection_steps += int(np.linalg.norm(proposal) > adapter.RADIUS)
        adapter.set_vector(proposal)
    if diagnostics is not None:
        diagnostics["projection_steps"] = projection_steps
    return adapter.get_vector() - initial
