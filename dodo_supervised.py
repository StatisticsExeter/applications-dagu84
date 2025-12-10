from pathlib import Path
from doit.tools import config_changed
from course.utils import load_pg_data
from course.supervised_classification.classify import fit_qda, fit_lda, fit_random_forest
from course.supervised_classification.predict import pred_lda, pred_qda, pred_random_forest
from course.supervised_classification.metrics import metric_report_lda, metric_report_qda, confusion_matrix_rf
from course.supervised_classification.split_test_train import test_and_train
from course.supervised_classification.eda import plot_scatter, get_summary_stats
from course.supervised_classification.roc_curve import plot_roc_curve


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
        models_path = Path("data_cache/vignettes/supervised_classification")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache_results]
    }


def task_energy_metrics_new():
    def energy_metrics_new():
        with open('sql/energy.sql', 'r') as file:
            QUERY = file.read()
        df = load_pg_data(QUERY)
        df.to_csv('data_cache/energy.csv', index=False)
    return {
        'actions': [energy_metrics_new],
        'file_dep': ['sql/energy.sql'],
        'targets': ['data_cache/energy.csv'],
        'uptodate': [config_changed({'sql_file': open('sql/energy.sql').read()})],
    }


def task_eda():
    return {
      'actions': [plot_scatter],
      'file_dep': ['data_cache/energy.csv'],
      'targets': ['data_cache/vignettes/supervised_classification/scatterplot.html'],
    }


def task_tabulate_stats():
    return {
      'actions': [get_summary_stats],
      'file_dep': ['data_cache/energy.csv',
                   'course/supervised_classification/eda.py'],
      'targets': ['data_cache/vignettes/supervised_classification/frequencies.csv',
                  'data_cache/vignettes/supervised_classification/grouped_stats.csv']
    }


def task_test_and_train():
    return {
        'actions': [test_and_train],
        'file_dep': ['data_cache/energy.csv',
                     'course/supervised_classification/split_test_train.py'],
        'targets': ['data_cache/energy_X_train.csv', 'data_cache/energy_y_train.csv',
                    'data_cache/energy_X_test.csv', 'data_cache/energy_y_test.csv'],
    }


def task_fit_lda():
    return {
      'actions': [fit_lda],
      'file_dep': ['data_cache/energy_X_train.csv', 'data_cache/energy_y_train.csv',
                   'course/supervised_classification/classify.py'],
      'targets': ['data_cache/models/lda_model.joblib']
    }


def task_fit_qda():
    return {
      'actions': [fit_qda],
      'file_dep': ['data_cache/energy_X_train.csv', 'data_cache/energy_y_train.csv',
                   'course/supervised_classification/classify.py'],
      'targets': ['data_cache/models/qda_model.joblib']
    }


def task_fit_random_forest(): # my code
    return {
      'actions': [fit_random_forest],
      'file_dep': ['data_cache/energy_X_train_raw.csv', 'data_cache/energy_y_train_raw.csv',
                   'course/supervised_classification/classify.py'],
      'targets': ['data_cache/models/random_forest_model.joblib']
    }


def task_predict_lda():
    return {
      'actions': [pred_lda],
      'file_dep': ['data_cache/models/lda_model.joblib', 'data_cache/energy_X_test.csv',
                   'course/supervised_classification/predict.py'],
      'targets': ['data_cache/models/lda_y_pred.csv',
                  'data_cache/models/lda_y_pred_prob.csv',]
    }


def task_predict_qda():
    return {
      'actions': [pred_qda],
      'file_dep': ['data_cache/models/qda_model.joblib', 'data_cache/energy_X_test.csv',
                   'course/supervised_classification/predict.py'],
      'targets': ['data_cache/models/qda_y_pred.csv',
                  'data_cache/models/qda_y_pred_prob.csv',]
    }


def task_predict_rf(): # my code
    return {
      'actions': [pred_random_forest],
      'file_dep': ['data_cache/models/random_forest_model.joblib', 'data_cache/energy_X_test_raw.csv',
                   'course/supervised_classification/predict.py'],
      'targets': ['data_cache/models/rf_y_pred.csv',
                  'data_cache/models/rf_y_pred_prob.csv',]
    }


def task_metrics_lda():
    return {
      'actions': [metric_report_lda],
      'file_dep': ['data_cache/models/lda_y_pred.csv', 'data_cache/energy_y_test.csv',
                   'course/supervised_classification/metrics.py'],
      'targets': ['data_cache/vignettes/supervised_classification/lda.csv']
    }


def task_metrics_qda():
    return {
      'actions': [metric_report_qda],
      'file_dep': ['data_cache/models/qda_y_pred.csv', 'data_cache/energy_y_test.csv',
                   'course/supervised_classification/metrics.py'],
      'targets': ['data_cache/vignettes/supervised_classification/qda.csv']
    }


def task_confusion():
    return {
      'actions': [confusion_matrix_rf],
      'file_dep': ['data_cache/models/rf_y_pred.csv', 'data_cache/energy_y_test.csv',
                   'course/supervised_classification/metrics.py'],
      'targets': ['data_cache/vignettes/supervised_classification/rf.csv']
    }


def task_roc_curve():
    return {
      'actions': [plot_roc_curve],
      'file_dep': ['data_cache/energy_y_test.csv',
                   'data_cache/models/qda_y_pred_prob.csv'],
      'targets': ['vignettes/supervised_classication/roc.html']
    }
