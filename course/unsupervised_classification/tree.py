from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(df_scaled)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)
    my_dendogram = ff.create_dendrogram(df_scaled, linkagefun=lambda x: linked)
    my_dendogram.write_html(base_dir / VIGNETTE_DIR / 'mydendo2.html')
    clusters = _cutree(linked, 14)  # adjust this value based on dendrogram scale
    number_of_comp(df)
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters['cluster'].astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    df_plot.to_csv(base_dir / VIGNETTE_DIR / 'hclustered_data.csv', index=False)
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _fit_dendrogram(df):
    """Given a dataframe containing only suitable values
    Return a scipy.cluster.hierarchy hierarchical clustering solution to these data"""
    output = linkage(df, method='ward', metric='euclidean')
    return output


def _plot_dendrogram(df):
    """Given a dataframe df containing only suitable variables
    Use plotly.figure_factory to plot a dendrogram of these data"""
    output = ff.create_dendrogram(df)
    output.update_layout(title='Interactive Hierarchical Clustering Dendrogram')
    return output


def _cutree(tree, height):
    """Given a scipy.cluster.hierarchy hierarchical clustering solution and a float of the height
    Cut the tree at that hight and return the solution (cluster group membership) as a
    data frame with one column called 'cluster'"""
    cluster = fcluster(tree, height, criterion='distance')
    output = pd.DataFrame(cluster, columns=['cluster'])
    return output


def _pca(df):
    """Given a dataframe of only suitable variables
    return a dataframe of the first two pca predictions (z values) with columns 'PC1' and 'PC2'"""
    model = PCA(n_components=2)
    output = model.fit_transform(df)
    new_df = pd.DataFrame(output, columns=['PC1', 'PC2'])
    return new_df


def _scatter_clusters(df):
    """Given a data frame containing columns 'PC1' and 'PC2' and 'cluster'
      (the first two principal component projections and the cluster groups)
    return a plotly express scatterplot of PC1 versus PC2
    with marks to denote cluster group membership"""
    output = px.scatter(df, x='PC1', y='PC2', color='cluster',
                        title='PCA Scatter Plot Colored by Cluster Labels')
    return output


def number_of_comp(df):
    model = PCA()
    output = model.fit(df)
    components = np.arange(1, len(output.explained_variance_) + 1)
    variance = output.explained_variance_ratio_
    sns.scatterplot(x=components, y=variance)
    sns.lineplot(x=components, y=variance)
    base_dir = find_project_root()
    plt.savefig(base_dir / VIGNETTE_DIR / 'screeplot.png')
