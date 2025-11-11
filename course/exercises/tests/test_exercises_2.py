import pytest
import pandas as pd
from ..exercises_2 import (
  column_mean, select_row, frequencies_by_group,
  filter_rows, add_ratio_column, rename_columns,
  drop_missing, fill_missing, sort_by_column,
  unique_values)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': ['cat', 'dog', 'cat', 'dog', 'cat'],
        'Group': ['X', 'X', 'Y', 'Y', 'Y']
    })


@pytest.mark.parametrize("column,expected", [("A", 3.0), ("B", 3.0)])
def test_column_mean(sample_df, column, expected):
    assert column_mean(sample_df, column) == expected


@pytest.mark.parametrize("x,expected", [(0, 1), (4, 5)])
def test_select_row(sample_df, x, expected):
    assert select_row(sample_df, x)['A'] == expected


def test_frequencies_by_group(sample_df):
    result = frequencies_by_group(sample_df, 'C')
    print(result)
    print(type(result))
    assert result.loc['cat'] == 3
    assert result.loc['dog'] == 2


def test_filter_rows(sample_df):
    result = filter_rows(sample_df, 'A', 3)
    assert len(result) == 2


def test_add_ratio_column(sample_df):
    df = add_ratio_column(sample_df.copy(), 'A', 'B', 'Ratio')
    assert 'Ratio' in df.columns
    assert df.loc[0, 'Ratio'] == 1 / 5


def test_rename_columns(sample_df):
    df = rename_columns(sample_df.copy(), {'A': 'Alpha'})
    assert 'Alpha' in df.columns


def test_drop_missing():
    df = pd.DataFrame({'A': [1, None, 3]})
    result = drop_missing(df)
    assert len(result) == 2


def test_fill_missing():
    df = pd.DataFrame({'A': [1, None, 3]})
    result = fill_missing(df, 0)
    assert result.loc[1, 'A'] == 0


def test_sort_by_column(sample_df):
    result = sort_by_column(sample_df, 'B')
    assert result.iloc[0]['B'] == 1


def test_unique_values(sample_df):
    result = unique_values(sample_df, 'C')
    assert set(result) == {'cat', 'dog'}
