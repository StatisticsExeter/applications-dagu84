import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from course.utils import find_project_root


def test_and_train():
    base_dir = find_project_root()
    base_data_path = base_dir / 'data_cache' / 'energy.csv'
    X_train_path = base_dir / 'data_cache' / 'energy_X_train.csv'
    y_train_path = base_dir / 'data_cache' / 'energy_y_train.csv'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    split_data_pre(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path)
    X_train_path_raw = base_dir / 'data_cache' / 'energy_X_train_raw.csv'
    y_train_path_raw = base_dir / 'data_cache' / 'energy_y_train_raw.csv'
    X_test_path_raw = base_dir / 'data_cache' / 'energy_X_test_raw.csv'
    y_test_path_raw = base_dir / 'data_cache' / 'energy_y_test_raw.csv'
    split_data(base_data_path, X_train_path_raw, y_train_path_raw, X_test_path_raw, y_test_path_raw)


def split_data_pre(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path):
    df = pd.read_csv(base_data_path).dropna()
    y = df['built_age']
    X = df.drop(columns=['built_age'])
    """Form four dataframes, X_train, y_train, X_test, y_test with 30% of the data for testing"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1999)
    """Apply a power transform and standard scaler to X_train and X_test"""
    X_train, X_test = _preprocess(X_train, X_test) # my code
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)


def split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path):
    df = pd.read_csv(base_data_path).dropna()
    y = df['built_age'].apply(lambda x: 1 if x == 'Pre-30s' else 0)
    X = df.drop(columns=['built_age'])
    """Form four dataframes, X_train, y_train, X_test, y_test with 30% of the data for testing"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1999)
    """Apply a power transform and standard scaler to X_train and X_test"""
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)


def _preprocess(X_train, X_test):
    transformer = PowerTransformer()
    scaler = StandardScaler()

    X_train_1 = transformer.fit_transform(X_train)
    X_train_1 = scaler.fit_transform(X_train_1)

    X_test_1 = transformer.transform(X_test)
    X_test_1 = scaler.transform(X_test_1)

    X_train = pd.DataFrame(X_train_1, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_1, columns=X_test.columns)

    return X_train, X_test
