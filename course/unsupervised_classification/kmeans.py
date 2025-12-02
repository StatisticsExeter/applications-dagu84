import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from course.utils import find_project_root
from course.unsupervised_classification.tree import _scatter_clusters, _pca

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def _kmeans(df, k):
    """Given dataframe df containing only suitable variables and integer k
    Return a sci-kit learn KMeans solution fitted to these data"""
    model = KMeans(n_clusters=k)
    return model.fit(df)


def kmeans(k):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = _kmeans(df_scaled, k)
    clusters = kmeans.labels_
    scaled_centers = kmeans.cluster_centers_
    fig1, fig2 = _plot_centroids(scaled_centers, scaler, df.columns, k)
    outpath1 = base_dir / VIGNETTE_DIR / 'kcentroids1.html'
    outpath2 = base_dir / VIGNETTE_DIR / 'kcentroids2.html'
    fig1.write_html(outpath1)
    fig2.write_html(outpath2)
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'kscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _plot_centroids(scaled_centers, scaler, colnames, k):    # Melt for grouped bar plot
    original_centers = scaler.inverse_transform(scaled_centers)
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, [0]]
    centers_df['cluster'] = [f'Cluster {i}' for i in range(k)]
    centers_melted = centers_df.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig1 = px.bar(
        centers_melted,
        x='Feature',
        y='Value',
        color='cluster',
        barmode='group',
        title='Cluster Centers by Feature (Original Scale)'
    )
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, 1:]
    centers_df['cluster'] = [f'Cluster {i}' for i in range(k)]
    centers_melted = centers_df.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig2 = px.bar(
        centers_melted,
        x='Feature',
        y='Value',
        color='cluster',
        barmode='group',
        title='Cluster Centers by Feature (Original Scale)'
    )
    return fig1, fig2
