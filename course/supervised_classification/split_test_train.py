import pandas as pd
from sklearn.model_selection import train_test_split
from course.utils import find_project_root


def test_and_train():
    base_dir = find_project_root()
    base_data_path = base_dir / 'data_cache' / 'energy.csv'
    X_train_path = base_dir / 'data_cache' / 'energy_X_train.csv'
    y_train_path = base_dir / 'data_cache' / 'energy_y_train.csv'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path)


def split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path):
    df = pd.read_csv(base_data_path).dropna()
    y = df['built_age']
    X = df.drop(columns=['built_age'])
    """Form four dataframes, X_train, y_train, X_test, y_test with 30% of the data for testing"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
