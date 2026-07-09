"""qeclab — toric-code error correction with an MWPM decoder."""

from .decoder import logical_error_rate, mwpm_correction, run_trial
from .toric import apply_path, logical_failure, sample_errors, syndrome, toric_distance

__all__ = [
    "apply_path",
    "logical_error_rate",
    "logical_failure",
    "mwpm_correction",
    "run_trial",
    "sample_errors",
    "syndrome",
    "toric_distance",
]

__version__ = "0.1.0"
