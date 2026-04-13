"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_READ_KWARGS = {"sep": ";", "decimal": "."}


def load_data(data_rootpath: str) -> pd.DataFrame:
    """Load all CSV files from a folder and merge them on ``ID``."""
    files = sorted(Path(data_rootpath).glob("*.csv"))
    if len(files) < 2:
        raise ValueError("At least two CSV files are required to merge on 'ID'.")

    frames = [pd.read_csv(file, **CSV_READ_KWARGS) for file in files]
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, how="inner", on="ID")
    return merged
