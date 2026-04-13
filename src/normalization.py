"""Column normalization and wrangling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_io import load_data


DATE_COLUMN_PATTERN = r"^S[3-7]"


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical and date-like columns."""
    normalized = data.copy().fillna(np.nan).replace({True: "True", False: "False"})

    catcols = normalized.select_dtypes(include=["object", "string"]).columns
    normalized[catcols] = normalized[catcols].apply(
        lambda col: col.str.upper().str.strip().replace({"TRUE": "True", "FALSE": "False"})
    )

    date_cols = normalized.filter(regex=DATE_COLUMN_PATTERN).columns
    normalized[date_cols] = normalized[date_cols].apply(
        lambda col: pd.to_datetime(col, errors="coerce", format="%Y-%m-%d")
    )

    return normalized.convert_dtypes(convert_string=False).replace(pd.NA, np.nan)


def wrangle(filepath: str) -> pd.DataFrame:
    """Load and normalize data from a folder path."""
    return normalize_columns(load_data(filepath))
