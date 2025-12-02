import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Crash types in each Local Authority"
    fig = _scatter(df, title)
    fig.write_html(outpath)


def _scatter(df, title):
    """When called with dataframe 'df' and a string 'title'
    Return a plotlty express object which is a scatterplot of all numeric variables
    in the dataframe. The title should be as provided in the function call"""
    numeric = df.select_dtypes('number')
    scatterplot = px.scatter_matrix(numeric, title=title)
    return scatterplot
