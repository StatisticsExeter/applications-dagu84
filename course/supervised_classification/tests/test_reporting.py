import pandas as pd
import pytest  # noqa: F401
from course.supervised_classification.metrics import metric_report


def test_metric_report_creates_csv(tmp_path):
    # Create dummy y_test and y_pred
    y_test = pd.Series(['A', 'B', 'A', 'C'])
    y_pred = pd.Series(['A', 'B', 'C', 'C'])

    # Save to temporary files
    y_test_path = tmp_path / "y_test.csv"
    y_pred_path = tmp_path / "y_pred.csv"
    report_path = tmp_path / "report.csv"

    y_test.to_csv(y_test_path, index=False)
    y_pred.to_csv(y_pred_path, index=False)

    # Run the function
    metric_report(y_test_path, y_pred_path, report_path)

    # Check that the report file was created
    assert report_path.exists()

    # Check contents
    report_df = pd.read_csv(report_path, index_col=0)
    print(report_df)
    assert 'precision' in report_df.columns
    assert 'recall' in report_df.columns
    assert 'f1-score' in report_df.columns
