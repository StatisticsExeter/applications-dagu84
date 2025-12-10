import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from course.utils import find_project_root


def metric_report(y_test_path, y_pred_path, report_path):
    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)
    """Create a pandas data frame called report which contains your classifier results"""
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report_dict)
    report.transpose().to_csv(report_path, index=True)


def metric_report_lda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'lda.csv'
    metric_report(y_test_path, y_pred_path, report_path)


def metric_report_qda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'qda.csv'
    metric_report(y_test_path, y_pred_path, report_path)


def confusion_matrix_rf():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test_raw.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'rf_y_pred.csv'
    path = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'confusion_matrix.png'

    y_test = pd.read_csv(y_test_path).squeeze()
    y_pred = pd.read_csv(y_pred_path).squeeze()
    matrix = confusion_matrix(y_test, y_pred)

    plt.clf()
    plt.figure()
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    plt.gcf().savefig(path)
