import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import plotly.express as px


def plot_scatter(df, x_name, y_name):
    """Given a dataframe containing numeric columns specified by x_name and y_name
    return a plotly express scatterplot"""
    chart = px.scatter(x=df[x_name], y=df[y_name])
    return chart


def calculate_correlation(df, x1, x2):
    """Given a dataframe containing numeric columns specified by x_name and y_name
    return two objects (numbers), the first is the pearson correlation coefficient, the second 
    the significance of this estimate"""
    output = pearsonr(df[x1], df[x2])
    return output


def fit_regression(df, x_name, y_name):
    """Given a dataframe containing numeric columns specified by x_name and y_name
    return the stats models OLS fit of a regression model of y on x"""
    model = sm.OLS(exog=df[x_name], endog=df[y_name])
    results = model.fit()
    return results


def filter_data(df, year):
    """Given a dataframe of various rows including a column 'Year' and an integer year
    return a dataframe containing only those rows where the value in this column is less
    than the value of the supplied year"""
    output = df[df['Year'] < year]
    return output


def tyler_viglen():
    """Create the data used at
    https://www.tylervigen.com/
    spurious/correlation/19524_kerosene-used-in-india_correlates-with_the-divorce-rate-in-maine"""
    array_1 = np.array([
      227, 238.806, 220.93, 220.358, 216.652, 198.424, 198.587, 201.298,
      198.333, 196.481, 197.041, 189.078, 183, 166, 157, 154, 149, 128,
      88, 76.9508, 59.2101, 40.4265, 34.1946
      ])
    array_2 = np.array([
      5.1, 5, 4.7, 4.6, 4.4, 4.3, 4.1, 4.2, 4.2, 4.2, 4.1, 4.2, 4.2, 3.9,
      3.96973, 3.58172, 3.42805, 3.42852, 3.22627, 3.19709, 3.033, 2.40567, 2.72837
      ])
    years = np.arange(1999, 2022)
    return pd.DataFrame({'Year': years, 'Kerosene': array_1, 'DivorceRate': array_2})
