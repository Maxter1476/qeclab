from itertools import combinations

import numpy as np
import pytest

from qeclab import (
    apply_path,
    logical_error_rate,
    logical_failure,
    mwpm_correction,
    run_trial,
    sample_errors,
    syndrome,
    toric_distance,
)


def test_no_errors_no_syndrome_no_failure():
    errors = np.zeros((2, 5, 5), dtype=bool)
    assert not syndrome(errors).any()
    assert not logical_failure(errors)


def test_single_error_flags_two_plaquettes():
    """Every single edge error must light exactly its two adjacent plaquettes."""
    size = 5
    for channel in (0, 1):
        for i in range(size):
            for j in range(size):
                errors = np.zeros((2, size, size), dtype=bool)
                errors[channel, i, j] = True
                assert syndrome(errors).sum() == 2


def test_vertex_star_is_invisible():
    """The four edges incident to a vertex form the X-type stabilizer A_v:
    applying X on them must produce no plaquette syndrome and no logical
    failure — including at vertex (0, 0), where the star straddles the rows
    and columns used by the logical-Z parity checks."""
    size = 5
    for i, j in [(2, 3), (0, 0), (4, 0), (0, 4)]:
        errors = np.zeros((2, size, size), dtype=bool)
        errors[0, i, j] ^= True  # h[i, j]
        errors[0, i, (j - 1) % size] ^= True  # h[i, j-1]
        errors[1, i, j] ^= True  # v[i, j]
        errors[1, (i - 1) % size, j] ^= True  # v[i-1, j]
        assert not syndrome(errors).any(), (i, j)
        assert not logical_failure(errors), (i, j)


def test_noncontractible_loop_is_logical():
    """A straight loop around the torus has no syndrome but flips a logical."""
    size = 5
    errors = np.zeros((2, size, size), dtype=bool)
    errors[0, :, 2] = True  # column of horizontal edges: winds vertically
    assert not syndrome(errors).any()
    assert logical_failure(errors)


def test_path_endpoints_match_syndrome():
    """apply_path must create defects exactly at its two endpoints."""
    size = 7
    for a, b in [((0, 0), (3, 4)), ((6, 6), (1, 2)), ((2, 5), (2, 5))]:
        correction = np.zeros((2, size, size), dtype=bool)
        apply_path(correction, a, b, size)
        defects = set(map(tuple, np.argwhere(syndrome(correction))))
        assert defects == (set() if a == b else {a, b})


def test_toric_distance_wraps():
    assert toric_distance((0, 0), (4, 4), 5) == 2
    assert toric_distance((0, 0), (2, 2), 5) == 4
    assert toric_distance((1, 1), (1, 1), 5) == 0


@pytest.mark.parametrize("size", [3, 5])
def test_every_single_error_is_corrected(size):
    """Distance d corrects any weight-1 error — exhaustively."""
    for channel in (0, 1):
        for i in range(size):
            for j in range(size):
                errors = np.zeros((2, size, size), dtype=bool)
                errors[channel, i, j] = True
                correction = mwpm_correction(syndrome(errors))
                assert not logical_failure(errors ^ correction)


def test_every_weight2_error_corrected_at_d5():
    """Distance 5 corrects any weight-2 error — all C(50, 2) of them."""
    size = 5
    edges = [(c, i, j) for c in (0, 1) for i in range(size) for j in range(size)]
    for e1, e2 in combinations(edges, 2):
        errors = np.zeros((2, size, size), dtype=bool)
        errors[e1] = errors[e2] = True
        correction = mwpm_correction(syndrome(errors))
        assert not logical_failure(errors ^ correction), (e1, e2)


def test_below_threshold_bigger_code_is_better():
    """At p = 0.05 (well below the ~10.3% MWPM threshold), increasing the
    distance must suppress the logical error rate decisively."""
    p, trials = 0.05, 400
    rate3 = logical_error_rate(3, p, trials, seed=11)
    rate7 = logical_error_rate(7, p, trials, seed=13)
    assert rate7 < rate3
    assert rate3 > 0.01  # d=3 visibly fails sometimes at this p
    assert rate7 < 0.06


def test_above_threshold_code_does_not_help():
    """At p = 0.20 (above threshold) a bigger code no longer suppresses
    errors: d = 7 fails at least as often as ~30%."""
    rate7 = logical_error_rate(7, 0.20, 200, seed=17)
    assert rate7 > 0.3


def test_sample_errors_rate():
    rng = np.random.default_rng(0)
    errors = sample_errors(20, 0.1, rng)
    assert errors.mean() == pytest.approx(0.1, abs=0.02)
    with pytest.raises(ValueError):
        sample_errors(5, 1.5, rng)


def test_run_trial_zero_noise_never_fails():
    rng = np.random.default_rng(0)
    assert not any(run_trial(5, 0.0, rng) for _ in range(20))
