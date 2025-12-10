import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
    boxplots = _boxplots() # my code
    outpath_box = base_dir / VIGNETTE_DIR / 'boxplots.png' # my code
    boxplots.savefig(outpath_box) # my code
    violinplots = _violin_plots() # my code
    outpath_violin = base_dir / VIGNETTE_DIR / 'violinplots.png' # my code
    violinplots.savefig(outpath_violin) # my code
    histplots = _histograms() # my code
    outpath_box = base_dir / VIGNETTE_DIR / 'histplots.png' # my code
    histplots.savefig(outpath_box) # my code
    qqplots_pre = _qqplots(df[df['built_age'] == 'Pre-30s']) # my code
    outpath_qq_pre = base_dir / VIGNETTE_DIR / 'qqplots_pre.png' # my code
    qqplots_pre.savefig(outpath_qq_pre) # my code
    qqplots_post = _qqplots(df[df['built_age'] == 'Post-30s']) # my code
    outpath_qq_post = base_dir / VIGNETTE_DIR / 'qqplots_post.png' # my code
    qqplots_post.savefig(outpath_qq_post) # my code


def scatter_onecat(df, cat_column, title):
    """Return a plotly express figure which is a scatterplot of all numeric columns in df
    with markers/colours given by the text in column cat_column
    and overall title specfied by title"""
    # numeric_columns= df.select_dtypes('number').columns
    scatterplot = px.scatter_matrix(df, color=cat_column, title=title)
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


def _boxplots():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    feature_columns = [i for i in df.columns if i != 'built_age']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        sns.boxplot(data=df, x='built_age', y=col, ax=ax)
        ax.set_title(col)
    return fig


def _violin_plots():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    feature_columns = [i for i in df.columns if i != 'built_age']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        sns.violinplot(data=df, x='built_age', y=col, ax=ax)
        ax.set_title(col)
    return fig


def _histograms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    feature_columns = [i for i in df.columns if i != 'built_age']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        sns.histplot(data=df, x=col, ax=ax, hue='built_age')
        ax.set_title(col)
    return fig


def _qqplots(df):
    feature_columns = [i for i in df.columns if i != 'built_age']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        stats.probplot(df[col], dist='norm', plot=ax)
        ax.set_title(col)
    return fig
