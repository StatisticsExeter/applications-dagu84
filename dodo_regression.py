from pathlib import Path
from doit.tools import config_changed
from course.utils import load_pg_data
from course.regression.eda import boxplot_age, boxplot_rooms, boxplot_rva
from course.regression.fit_model import fit_model
from course.regression.caterpillar_reffs import plot_caterpillar


def task_check_cache_data():
    def check_cache_data():
        """Check cache folder exists"""
        models_path = Path("data_cache/models")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache_data]
    }


def task_check_cache_results():
    def check_cache_results():
        """Check cache folder exists"""
        models_path = Path("data_cache/vignettes/regression")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache_results]
    }


def task_energy_metrics_la():
    def energy_metrics_la():
        with open('sql/la_energy.sql', 'r') as file:
            QUERY = file.read()
        df = load_pg_data(QUERY).dropna()
        df.to_csv('data_cache/la_energy.csv', index=False)
    return {
        'actions': [energy_metrics_la],
        'file_dep': ['sql/la_energy.sql'],
        'targets': ['data_cache/la_energy.csv'],
        'uptodate': [config_changed({'sql_file': open('sql/la_energy.sql').read()})],
    }


def task_eda():
    return {
        'actions': [boxplot_age, boxplot_rooms, boxplot_rva],
        'file_dep': ['data_cache/la_energy.csv',
                     'course/regression/eda.py'],
        'targets': ['data_cache/vignettes/boxplot_age.html',
                    'data_cache/vignettes/boxplot_rooms.html',
                    'data_cache/vignettes/boxplot_rva.html'],
    }


def task_fit_model():
    return {
        'actions': [fit_model],
        'file_dep': ['data_cache/la_energy.csv',
                     'course/regression/fit_model.py'],
        'targets': ['data_cache/vignettes/model_fit.txt',
                    'data_cache/models/reffs.csv'],
    }


def task_caterpillar_plot():
    return {
        'actions': [plot_caterpillar],
        'file_dep': ['data_cache/models/reffs.csv',
                     'course/regression/caterpillar_reffs.py'],
        'targets': ['data_cache/vignettes/regression/model_fit.txt',
                    'data_cache/vignettes/regression/caterpillar.html'],
    }
