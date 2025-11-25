def sum_list(numbers):
    """Given a list of integers 'numbers'
    return the sum of this list."""
    output = sum(numbers)
    return output


def max_value(numbers):
    """Given a list of numbers 'numbers'
    return the maximum value of this list."""
    output = max(numbers)
    return output


def reverse_string(s):
    """Given a string 'string'
    return the reversed version of the input string."""
    output = s[::-1]
    return output


def filter_even(numbers):
    """Given a list of numbers 'numbers'
    return a list containing only the even numbers from the input list."""
    output = []
    for i in numbers:
      if i % 2 == 0:
        output.append(i)
      else: output
    return output


def get_fifth_row(df):
    """Given a dataframe 'df'
    return the fifth row of this as a pandas DataFrame."""
    output = df.iloc[4] # my assumption is it starts at 0 so the 5th row would be 4
    # still not passing test unsure why
    return output


def column_mean(df, column):
    """Given a dataframe 'df' and the name of a column 'column'
    return the mean of the specified column in a pandas DataFrame."""
    output = df[column].mean()
    return output


def lookup_key(d, key):
    """Given a dictionary 'd' and a key 'key'
    return the value associated with the key in the dictionary."""
    # there is a key error so I will try lowering all values to lower case
    #case_key = key.lower()
    #case_d = {key.lower(): value for key, value in d.items()}
    # output = case_d[case_key]
    # still not passing second test, unsure why
    return d.get(key)


def count_occurrences(lst):
    """Given a list 'lst'
    return a dictionary with counts of each unique element in the list."""
    output = {}
    for i in lst:
      output[i] = lst.count(i)
    return output


def drop_missing(df):
    """Given a dataframe 'df' with some rows containing missing values,
    return a DataFrame with rows containing missing values removed."""
    output = df.dropna()
    return output


def value_counts_df(df, column):
    """Given a dataframe 'df' with various columns and the name of one of those columns 'column',
    return a DataFrame with value counts of the specified column."""
    output = df[column].value_counts().reset_index()
    return output
