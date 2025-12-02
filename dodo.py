import pandas as pd
from pathlib import Path
from course.unsupervised_classification.visual_eda import (
  summary_stats, generate_raw_boxplot, generate_scaled_boxplot, generate_scatterplot)


def task_check_cache():
    def check_cache():
        """Check cache folder exists"""
        models_path = Path("data_cache/models")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache]
    }


def task_check_vignettes():
    def check_vignettes():
        """Check cache folder exists"""
        vignettes_path = Path("vignettes/unsupervised/cache")
        vignettes_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_vignettes]
    }


def task_load_data():
    def load_data():
        df = pd.read_csv('data/olive_oil.csv')
        df_reduced = df.iloc[:, 3:]
        df_reduced.to_csv('data_cache/unsupervised.csv', index=False)
    return {
        'actions': [load_data],
        'file_dep': ['data/olive_oil.csv'],
        'targets': ['data_cache/unsupervised.csv'],
    }


def task_summary_stats():
    return {
        'actions': [summary_stats],
        'file_dep': ['data_cache/unsupervised.csv',
                     'course/unsupervised_classification/visual_eda.py'],
        'targets': ['vignettes/unsupervised/cache/olive_oil_summary.html'],
    }


def task_plot_raw_boxplot():
    return {
        'file_dep': ['data_cache/unsupervised.csv',
                     'course/unsupervised_classification/visual_eda.py'],
        'targets': ['vignettes/unsupervised/cache/raw_boxplot.html'],
        'actions': [generate_raw_boxplot],
        'clean': True,
    }


def task_plot_scaled_boxplot():
    return {
        'file_dep': ['data_cache/unsupervised.csv',
                     'course/unsupervised_classification/visual_eda.py'],
        'targets': ['vignettes/unsupervised/cache/scaled_boxplot.html'],
        'actions': [generate_scaled_boxplot],
        'clean': True,
    }


def task_plot_scatterplot():
    return {
        'file_dep': ['data_cache/unsupervised.csv',
                     'course/unsupervised_classification/visual_eda.py'],
        'targets': ['vignettes/unsupervised/cache/scatterplot.html'],
        'actions': [generate_scatterplot],
        'clean': True,
    }


def task_render_quarto():
    return {
        'file_dep': ['vignettes/unsupervised/ict_unsupervised.qmd'],
        'targets': ['vignettes/unsupervised/ict_unsupervised.html'],
        'actions': ['quarto render vignettes/unsupervised/ict_unsupervised.qmd'],
        'clean': True,
    }
