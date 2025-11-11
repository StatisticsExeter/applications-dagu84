def column_mean(df, column):
    """Given a data frame 'df' and a column name 'column'
    return the mean of the specified column."""
    output = df[column].mean()
    return output


def select_row(df, x):
    """Given a data frame 'df' and an integer 'x'
    return the xth row of the DataFrame."""
    output = df.iloc[x]
    return output


def frequencies_by_group(df, cat_col):
    """Given a dataframe 'df' and the name of a categorical
    variable column 'cat_col'
    return frequency counts of that categorical column."""
    output = df[cat_col].value_counts()
    return output


def filter_rows(df, column, threshold):
    """Given a dataframe 'df', the name of a column 'column'
    and a float indicating a threshold 'threshold'
    return rows where the column value is greater than the threshold."""
    output = df[df[column] > threshold]
    return output


def add_ratio_column(df, numerator, denominator, new_col):
    """Given a dataframe 'df' and two names of columns
    'numerator' and 'denominator', the name of a new column 'new_col'
    return a dataframe with this named new column that is the
    ratio of two existing columns."""
    df[new_col] = df[numerator] / df[denominator]
    return df


def rename_columns(df, columns_dict):
    """Given a dataframe 'df# and a dictionary that maps
    existing column names to new names, return a dataframe
    with the new names."""
    df = df.rename(columns=columns_dict)
    return df


def drop_missing(df):
    """Given a dataframe 'df'
    return a dataframe having dropped rows with any
    missing values."""
    df = df.dropna()
    return df


def fill_missing(df, value):
    """Given a dataframe 'df' and a marker for missing values 'value'
    (which could be NA)
    return a data frame where the missing values with this specified value."""
    df = df.fillna(value)
    return df


def sort_by_column(df, column, ascending=True):
    """Given the dataframe 'df' and the name of a column 'column'
    return a DataFrame sorted by that specified column."""
    df = df.sort_values(by=column, ascending=ascending)
    return df


def unique_values(df, column):
    """Given a dataframe 'df' and a named column 'column'
    return unique values from that specified column."""
    df = df[column].unique()
    return df
