import pandas as pd
import numpy as np
import pytest

from src.data_wrangling import (
    load_data,
    normalize_columns,
    entropy,
    compute_entropies,
    test_distribution_difference_all,
)


def test_load_data(tmp_path):
    df1 = pd.DataFrame({'ID': [1, 2], 'A': [3, 4]})
    df2 = pd.DataFrame({'ID': [1, 2], 'B': [5, 6]})
    df1.to_csv(tmp_path / "a.csv", index=False, sep=";", decimal=".")
    df2.to_csv(tmp_path / "b.csv", index=False, sep=";", decimal=".")

    result = load_data(str(tmp_path))
    expected = pd.merge(df1, df2, on="ID")
    pd.testing.assert_frame_equal(result, expected)


def test_normalize_columns():
    df = pd.DataFrame({'cat': [' true ', 'FALSE'], 'S3': ['2020-01-01', '2020-01-02']})
    result = normalize_columns(df)
    assert result['cat'].tolist() == ['True', 'False']
    assert pd.api.types.is_datetime64_any_dtype(result['S3'])


def test_entropy():
    sr = pd.Series([0, 1, 0, 1])
    assert entropy(sr) == pytest.approx(1.0)


def test_compute_entropies():
    df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 1, 0, 0]})
    ents = compute_entropies(df, normalize=True)
    assert set(ents.keys()) == {'A', 'B'}
    assert all(isinstance(v, float) for v in ents.values())


def test_test_distribution_difference_all():
    df = pd.DataFrame({
        'target': [0, 0, 1, 1],
        'num': [1.0, 2.0, 1.5, 1.8],
        'cat': ['a', 'a', 'b', 'b']
    })
    result = test_distribution_difference_all(df, 'target', df.columns)
    assert set(result.keys()) == {'num', 'cat'}
    for stat, pvalue in result.values():
        assert isinstance(stat, float)
        assert isinstance(pvalue, float)
