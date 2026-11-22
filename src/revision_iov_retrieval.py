"""Supplement local TabM scores using distinct inputs from that client's TRAIN."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


NEIGHBORS = 5
RETRIEVAL_WEIGHT = 0.5


class LocalTrainRetrieval:
    """Index TRAIN inputs and blend distance-weighted label votes with TabM."""

    def __init__(self, x, y, classes):
        inputs, groups = np.unique(x, axis=0, return_inverse=True)
        counts = np.zeros((len(inputs), classes), dtype=np.int64)
        np.add.at(counts, (groups.reshape(-1), y), 1)
        self.class_votes = counts / counts.sum(axis=1, keepdims=True)
        self.neighbors = NearestNeighbors(
            n_neighbors=min(NEIGHBORS, len(inputs)), algorithm='kd_tree', n_jobs=1,
        ).fit(inputs)

    def __call__(self, _raw, x, tabm_probabilities):
        if not len(x):
            return tabm_probabilities
        # Query identical inputs once and map scores back to every original row.
        inputs, rows = np.unique(x, axis=0, return_inverse=True)
        distances, neighbors = self.neighbors.kneighbors(inputs)
        weights = 1 / np.maximum(distances, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        probabilities = (weights[:, :, None] * self.class_votes[neighbors]).sum(axis=1)
        return (1 - RETRIEVAL_WEIGHT) * tabm_probabilities + RETRIEVAL_WEIGHT * probabilities[rows.reshape(-1)]
