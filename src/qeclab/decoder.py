"""Minimum-weight perfect matching decoder for the toric code.

Defect plaquettes (syndrome = 1) always come in pairs on a torus. The
decoder pairs them up with minimum total toric Manhattan distance —
NetworkX's blossom-algorithm ``min_weight_matching`` on the complete defect
graph — then flips a shortest path between each matched pair. MWPM is the
classic decoder whose bit-flip threshold on the toric code is ~10.3%
(Dennis, Kitaev, Landahl & Preskill 2002).
"""

from __future__ import annotations

import numpy as np

from .toric import apply_path, logical_failure, sample_errors, syndrome, toric_distance

__all__ = ["mwpm_correction", "run_trial", "logical_error_rate"]


def mwpm_correction(defect_map: np.ndarray) -> np.ndarray:
    """Correction array from a plaquette defect map via MWPM."""
    import networkx as nx

    size = defect_map.shape[0]
    defects = [tuple(idx) for idx in np.argwhere(defect_map)]
    if len(defects) % 2:
        raise ValueError("odd number of defects is impossible on a torus")
    correction = np.zeros((2, size, size), dtype=bool)
    if not defects:
        return correction

    graph = nx.Graph()
    for a in range(len(defects)):
        for b in range(a + 1, len(defects)):
            graph.add_edge(a, b, weight=toric_distance(defects[a], defects[b], size))
    matching = nx.min_weight_matching(graph)
    for a, b in matching:
        apply_path(correction, defects[a], defects[b], size)
    return correction


def run_trial(size: int, p: float, rng: np.random.Generator) -> bool:
    """One decode cycle; True means the decoder caused a logical error."""
    errors = sample_errors(size, p, rng)
    correction = mwpm_correction(syndrome(errors))
    residual = errors ^ correction
    return logical_failure(residual)


def logical_error_rate(size: int, p: float, n_trials: int, seed: int = 0) -> float:
    """Monte Carlo logical error rate of distance-``size`` decoding at ``p``."""
    rng = np.random.default_rng(seed)
    failures = sum(run_trial(size, p, rng) for _ in range(n_trials))
    return failures / n_trials
