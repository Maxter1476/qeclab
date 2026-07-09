"""Toric code under independent bit-flip noise, with exact syndrome extraction.

Layout: an L x L torus with one qubit on every edge — 2 L^2 qubits stored as
a boolean array of shape (2, L, L): channel 0 holds horizontal edges
h[i, j] (connecting vertex (i, j) to (i, j+1)), channel 1 vertical edges
v[i, j] (connecting (i, j) to (i+1, j)).

Plaquette (i, j) touches h[i, j] (top), h[i+1, j] (bottom), v[i, j] (left),
v[i, j+1] (right); its Z-type stabilizer flags X errors on those four edges.
By CSS duality, phase-flip errors are the identical problem on the dual
lattice, so studying bit-flips alone loses nothing.

A residual error that commutes with every plaquette is a cycle; it is a
*logical* error iff it winds the torus — detected as odd parity across a
fixed cut in either direction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_errors",
    "syndrome",
    "logical_failure",
    "apply_path",
    "toric_distance",
]


def sample_errors(size: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """IID bit-flip errors on every edge with probability p."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    return rng.random((2, size, size)) < p


def syndrome(errors: np.ndarray) -> np.ndarray:
    """Plaquette syndrome: XOR of the four boundary edges of each plaquette."""
    h, v = errors[0], errors[1]
    return h ^ np.roll(h, -1, axis=0) ^ v ^ np.roll(v, -1, axis=1)


def toric_distance(a: tuple[int, int], b: tuple[int, int], size: int) -> int:
    """Manhattan distance on the torus."""
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return min(dy, size - dy) + min(dx, size - dx)


def apply_path(
    correction: np.ndarray, a: tuple[int, int], b: tuple[int, int], size: int
) -> None:
    """Flip the edges of a shortest defect-pairing path from plaquette a to b.

    Moving between vertically adjacent plaquettes (i, j) -> (i+1, j) crosses
    horizontal edge h[i+1, j]; moving horizontally (i, j) -> (i, j+1) crosses
    vertical edge v[i, j+1]. Wraparound direction is chosen per-axis to
    minimize distance.
    """
    i, j = a
    ti, tj = b
    dy = (ti - i) % size
    step_i = 1 if dy <= size - dy else -1
    while i != ti:
        nxt = (i + step_i) % size
        crossed = nxt if step_i == 1 else i
        correction[0, crossed, j] ^= True
        i = nxt
    dx = (tj - j) % size
    step_j = 1 if dx <= size - dx else -1
    while j != tj:
        nxt = (j + step_j) % size
        crossed = nxt if step_j == 1 else j
        correction[1, i, crossed] ^= True
        j = nxt


def logical_failure(residual: np.ndarray) -> bool:
    """True if a syndrome-free residual error winds the torus (logical flip).

    An X-residual is a logical error iff it anticommutes with a logical Z
    operator. The two logical Zs live on noncontractible primal loops: all
    horizontal edges of one row, and all vertical edges of one column.
    Odd overlap with either support = failure. For syndrome-free residuals
    (dual-lattice cycles) these parities are independent of which row or
    column is chosen — vertex stars overlap every such loop an even number
    of times.
    """
    if syndrome(residual).any():
        raise ValueError("residual has nonzero syndrome; not a cycle")
    anticommutes_z1 = bool(residual[0, 0, :].sum() % 2)  # row of horizontal edges
    anticommutes_z2 = bool(residual[1, :, 0].sum() % 2)  # column of vertical edges
    return anticommutes_z1 or anticommutes_z2
