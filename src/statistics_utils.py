"""Statistical helpers for exploratory analysis."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats


def entropy(column_series: pd.Series, normalize: bool = False) -> float:
    """Compute entropy for a Pandas series."""
    n_classes = column_series.nunique()
    if n_classes <= 1:
        return 0.0

    value_counts = column_series.value_counts(normalize=True)
    value = -np.sum(value_counts * np.log2(value_counts))
    return float(value / np.log2(n_classes) if normalize else value)


def compute_entropies(df: pd.DataFrame, normalize: bool = False) -> Dict[str, float]:
    """Compute entropy for categorical and numeric binary columns."""
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    numeric_columns = df.select_dtypes("number")
    binary_numeric_columns = numeric_columns.columns[numeric_columns.nunique() == 2]
    selected_columns = categorical_columns.append(binary_numeric_columns)
    return {column: entropy(df[column], normalize=normalize) for column in selected_columns}


def test_distribution_difference(
    data: pd.DataFrame,
    target_variable: str,
    feature_variable: str,
    equal_var: bool = False,
):
    """Use KS test to compare class distributions for a numeric feature."""
    del equal_var  # kept for backward compatibility in signature
    class_0 = data[data[target_variable] == 0][feature_variable]
    class_1 = data[data[target_variable] == 1][feature_variable]
    return stats.ks_2samp(class_0, class_1)


def test_distribution_difference_categorical(
    data: pd.DataFrame,
    target_variable: str,
    categorical_variable: str,
):
    """Use Chi-squared test for categorical feature/target relationship."""
    contingency_table = pd.crosstab(data[target_variable], data[categorical_variable])
    chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2_statistic, p_value


def test_distribution_difference_all(
    data: pd.DataFrame,
    target_variable: str,
    feature_variables: List[str] | pd.Index,
):
    """Run distribution tests across all provided features."""
    results = {}
    for feature in feature_variables:
        if feature == target_variable:
            continue
        if pd.api.types.is_numeric_dtype(data[feature]):
            results[feature] = test_distribution_difference(data, target_variable, feature)
        else:
            results[feature] = test_distribution_difference_categorical(data, target_variable, feature)
    return results

# Prevent pytest from collecting helper functions imported into test modules.
test_distribution_difference.__test__ = False
test_distribution_difference_categorical.__test__ = False
test_distribution_difference_all.__test__ = False
