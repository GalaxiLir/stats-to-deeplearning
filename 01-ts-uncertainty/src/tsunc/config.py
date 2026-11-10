"""Project-wide constants and reproducibility helpers."""

import random

import numpy as np

SEED: int = 42
HORIZON: int = 24  # forecast horizon, in hours
ALPHA: float = 0.10  # miscoverage level: target coverage is 1 - ALPHA = 90 %

# Quantile levels predicted by the model. The two outer levels are derived from ALPHA, so
# they always match the interval targeted by conformalized quantile regression (CQR);
# the median (0.5) gives a point forecast, used for the MASE.
QUANTILES: tuple[float, ...] = (ALPHA / 2, 0.5, 1 - ALPHA / 2)


def set_seed(seed: int = SEED) -> None:
    """Seed the global random generators of Python and NumPy (like set.seed() in R).

    PyTorch seeds will be added here when the LSTM is introduced.
    """
    random.seed(seed)
    np.random.seed(seed)
