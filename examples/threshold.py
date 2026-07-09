"""README figure: logical error rate vs physical error rate — the threshold.

Curves for different code distances cross near the MWPM toric-code threshold
(~10.3% for bit-flip noise with perfect measurements). Below it, bigger codes
win; above it, they lose. Takes a few minutes at these trial counts.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qeclab import logical_error_rate

SIZES = (3, 5, 7)
PHYSICAL = np.linspace(0.04, 0.16, 9)
TRIALS = 1200


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for size in SIZES:
        rates = [
            logical_error_rate(size, float(p), TRIALS, seed=1000 + size)
            for p in PHYSICAL
        ]
        err = [np.sqrt(max(r * (1 - r), 1e-9) / TRIALS) for r in rates]
        ax.errorbar(PHYSICAL, rates, yerr=err, fmt="o-", ms=4, capsize=2, label=f"d = {size}")
    ax.axvline(0.103, color="gray", ls="--", lw=1, label="MWPM threshold ≈ 10.3%")
    ax.set_xlabel("physical bit-flip probability p")
    ax.set_ylabel("logical error rate")
    ax.set_title("Toric code + MWPM decoder: the error-correction threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/threshold.png", dpi=150)
    print("wrote docs/figures/threshold.png")


if __name__ == "__main__":
    main()
