"""Visualization helpers."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_kde_grid(data, columns, grid_cols=3, hue=None, target=None):
    """Plot KDE plots for selected columns in a grid."""
    if isinstance(columns, (pd.Index, list)):
        columns = list(columns)
    if target and target in columns:
        columns.remove(target)

    grid_rows, reminder = divmod(len(columns) - (1 - (not hue)), grid_cols)
    grid_rows += reminder > 0

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        ax = axes[i]
        if not hue:
            sns.kdeplot(data[column], ax=ax, fill=True)
        else:
            for hue_level in data[hue].unique():
                subset_data = data[data[hue] == hue_level]
                sns.kdeplot(
                    subset_data[column],
                    ax=ax,
                    fill=True,
                    label=f"{hue_level}",
                    common_norm=False,
                )
        ax.set_title(column)
        ax.set_xlabel("")
        ax.legend()

    plt.tight_layout()
    plt.show()


def corr_matrix_threshold(
    df: pd.DataFrame,
    cols: List[str],
    method: str = "pearson",
    threshold: float = 0.7,
    cmap: str = "coolwarm",
):
    """Return styled correlation matrix for highly-correlated columns."""
    corr_matrix = df[cols].corr(method=method)
    np.fill_diagonal(corr_matrix.values, np.nan)
    selected_cols = (corr_matrix.abs() > threshold).any()
    high_corr_matrix = corr_matrix.loc[selected_cols, selected_cols]
    return high_corr_matrix.style.background_gradient(cmap=cmap)
