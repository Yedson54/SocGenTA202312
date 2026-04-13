"""Backward-compatible facade for data wrangling utilities.

This module now re-exports focused helpers from smaller modules to keep
responsibilities explicit and the codebase easier to navigate.
"""

from .data_io import load_data
from .normalization import normalize_columns, wrangle
from .statistics_utils import (
    compute_entropies,
    entropy,
    test_distribution_difference,
    test_distribution_difference_all,
    test_distribution_difference_categorical,
)
from .visualization import corr_matrix_threshold, plot_kde_grid

__all__ = [
    "load_data",
    "normalize_columns",
    "wrangle",
    "entropy",
    "compute_entropies",
    "plot_kde_grid",
    "test_distribution_difference",
    "test_distribution_difference_categorical",
    "test_distribution_difference_all",
    "corr_matrix_threshold",
]
