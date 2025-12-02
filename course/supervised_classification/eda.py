import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'supervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Energy variables showing different built_age type"
    fig = scatter_onecat(df, 'built_age', title)
    fig.write_html(outpath)


def scatter_onecat(df, cat_column, title):
    """Return a plotly express figure which is a scatterplot of all numeric columns in df
    with markers/colours given by the text in column cat_column
    and overall title specfied by title"""
    numeric_columns= df.select_dtypes('number').columns
    scatterplot = px.scatter(df, x=cat_column, y=numeric_columns, title=title)
    return scatterplot


def get_frequencies(df, cat_column):
    return df[cat_column].value_counts()


def get_grouped_stats(df, cat_column):
    numeric_cols = df.select_dtypes(include='number').columns
    grouped_stats = df.groupby(cat_column)[numeric_cols].describe()
    grouped_stats.columns = ['{}_{}'.format(var, stat) for var, stat in grouped_stats.columns]
    return grouped_stats.transpose()


def get_summary_stats():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    cat_column = 'built_age'
    frequencies = get_frequencies(df, cat_column)
    outpath_f = base_dir / VIGNETTE_DIR / 'frequencies.csv'
    frequencies.to_csv(outpath_f)
    summary_stats = get_grouped_stats(df, cat_column)
    outpath_s = base_dir / VIGNETTE_DIR / 'grouped_stats.csv'
    summary_stats.to_csv(outpath_s)
