import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from course.utils import find_project_root
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def _get_roc_results(y_test_path, y_pred_prob_path):
    y_test = pd.read_csv(y_test_path)['built_age']
    y_pred_prob = pd.read_csv(y_pred_prob_path)['predicted_built_age']
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}


def plot_roc_curve():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred_prob.csv'
    lda_results = _get_roc_results(y_test_path, y_pred_prob_path)
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred_prob.csv'
    qda_results = _get_roc_results(y_test_path, y_pred_prob_path)
    fig = _plot_roc_curve(lda_results, qda_results)
    outpath = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'roc.html'
    fig.write_html(outpath)


def _plot_roc_curve(lda_roc, qda_roc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lda_roc['fpr'], y=lda_roc['tpr'],
                             mode='lines',
                             name=f'ROC curve from LDA (AUC = {lda_roc["roc_auc"]:.2f})'))
    fig.add_trace(go.Scatter(x=qda_roc['fpr'], y=qda_roc['tpr'],
                             mode='lines',
                             name=f'ROC curve from QDA (AUC = {qda_roc["roc_auc"]:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    return fig


def pca_check():
    # get the test set for X
    base_dir = find_project_root()

    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    X_test = pd.read_csv(X_test_path)
    # perform pca on it
    pca = PCA(n_components=2)
    pca_X_test = pca.fit_transform(X_test)
    # turn it into a dataframe
    final_df = pd.DataFrame(pca_X_test, columns=['PCA_1', 'PCA_2'], index=X_test.index)
    # get the classification from the model
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    y_pred = pd.read_csv(y_pred_path).squeeze()
    # input classification/cluster into dataframe
    final_df['class'] = y_pred
    final_df = final_df.drop(index=7895)
    # scatterplot with cluster as hue
    plt.clf()
    plt.figure()

    sns.scatterplot(data=final_df, x='PCA_1', y='PCA_2', hue='class')
    path = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'pca_plot.png'
    plt.gcf().savefig(path)
    return None
