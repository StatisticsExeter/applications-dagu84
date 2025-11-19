import pytest
import pandas as pd
from numpy import nan
from ..python_exercises import (
  sum_list, max_value, reverse_string, filter_even, get_fifth_row, column_mean,
  lookup_key, count_occurrences, drop_missing, value_counts_df)


def test_sum_list_basic():
    # Basic test for sum_list
    assert callable(sum_list)


@pytest.mark.parametrize("input_list, expected", [
    ([1, 2, 3], 6),
    ([], 0),
    ([-1, -2, -3], -6),
])
def test_sum_list(input_list, expected):
    assert sum_list(input_list) == expected


@pytest.mark.parametrize("input_list, expected", [
    ([1, 5, 3], 5),
    ([-1, -5, -3], -1),
    ([-1, -2, -3], -1),
])
def test_max_value(input_list, expected):
    # Case 1 for max_value
    assert max_value(input_list) == expected


def test_reverse_string_basic():
    # Basic test for reverse_string
    assert callable(reverse_string)


@pytest.mark.parametrize("input, expected", [
    ("hello", "olleh"),
    ("", "")

])
def test_reverse_string(input, expected):
    # Case 1 for reverse_string
    assert reverse_string(input) == expected


def test_filter_even_basic():
    # Basic test for filter_even
    assert callable(filter_even)


@pytest.mark.parametrize("input_list, expected", [
    ([1, 2, 3, 4], [2, 4]),
    ([1, 3, 5], [])
])
def test_filter_even_case1(input_list, expected):
    # Case 1 for filter_even
    assert filter_even(input_list) == expected


def test_get_fifth_row_basic():
    # Basic test for get_fifth_row
    assert callable(get_fifth_row)


def test_get_fifth_row_case1():
    # Case 1 for get_fifth_row
    import pandas as pd
    df = pd.DataFrame({'A': range(10)})
    assert get_fifth_row(df).equals(df.iloc[4])


def test_get_fifth_row_case2():
    # Case 2 for get_fifth_row
    import pandas as pd
    df = pd.DataFrame({'A': range(3)})
    try:
        get_fifth_row(df)
        assert False
    except IndexError:
        assert True


def test_column_mean_basic():
    # Basic test for column_mean
    assert callable(column_mean)


@pytest.mark.parametrize("df, column, expected", [
    (pd.DataFrame({'A': [1, 2, 3]}), 'A', 2.0),
    (pd.DataFrame({'B': []}), 'B', nan),
    (pd.DataFrame({'C': [10, 20, 30, 40]}), 'C', 25.0),
])
def test_column_mean_valid(df, column, expected):
    result = column_mean(df, column)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected


def test_column_mean_missing_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(KeyError):
        column_mean(df, 'Z')


def test_lookup_key_basic():
    # Basic test for lookup_key
    assert callable(lookup_key)


def test_lookup_key_case1():
    # Case 1 for lookup_key
    assert lookup_key({'a': 1, 'b': 2}, 'b') == 2


def test_lookup_key_case2():
    # Case 2 for lookup_key
    assert lookup_key({'a': 1}, 'z') is None


def test_count_occurrences_basic():
    # Basic test for count_occurrences
    assert callable(count_occurrences)


def test_count_occurrences_case1():
    # Case 1 for count_occurrences
    assert count_occurrences(['a', 'b', 'a']) == {'a': 2, 'b': 1}


def test_count_occurrences_case2():
    # Case 2 for count_occurrences
    assert count_occurrences([]) == {}


def test_drop_missing_removes_rows():
    df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
    result = drop_missing(df)
    assert result.shape[0] == 1
    assert result.isnull().sum().sum() == 0


def test_value_counts_df_returns_correct_counts():
    df = pd.DataFrame({'type': ['car', 'bike', 'car', 'bus', 'bike']})
    result = value_counts_df(df, 'type')
    assert set(result['type']) == {'car', 'bike', 'bus'}
    assert result[result['type'] == 'car']['count'].values[0] == 2
