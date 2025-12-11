import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    """Form an object y_pred containing a list of your classifer predictions"""
    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, name='predicted_built_age')
    y_pred_series.to_csv(y_pred_path, index=False)
    """Form an object y_pred_prob containing a list of your classifer probabilities"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_prob_series = pd.Series(y_pred_prob, name='predicted_built_age')
    y_pred_prob_series.to_csv(y_pred_prob_path, index=False)


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'lda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred_prob.csv'
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'qda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred_prob.csv'
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_random_forest():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'random_forest_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test_raw.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'rf_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'rf_y_pred_prob.csv'
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    features = pd.Series(model.feature_importances_, index=X_test.columns)
    path = base_dir / 'data_cache' / 'models' / 'features.csv'
    features.to_csv(path)
