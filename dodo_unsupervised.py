from pathlib import Path
from course.unsupervised_classification.eda import plot_scatter
from course.unsupervised_classification.tree import (hierarchical_groups,
                                                     hcluster_analysis)
from course.unsupervised_classification.kmeans import kmeans


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
        models_path = Path("data_cache/vignettes/unsupervised_classification")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
      'actions': [check_cache_results]
    }


# def task_crash_summaries():
#     def crash_summaries():
#         with open('sql/la_collision.sql', 'r') as file:
#             QUERY = file.read()
#         df = load_pg_data(QUERY)
#         df.set_index("lad_ons", inplace=True)
#         df[df.columns[1:7]] = df[df.columns[1:7]].astype(float)
#         df[df.columns[0]] = df[df.columns[0]].astype(float)
#         df.iloc[:, 1:7] = df.iloc[:, 1:7].div(df.iloc[:, 0], axis=0)
#         df.to_csv('data_cache/la_collision.csv', index=False)
#     return {
#         'actions': [crash_summaries],
#         'file_dep': ['sql/la_collision.sql'],
#         'targets': ['data_cache/la_collision.csv'],
#         'uptodate': [config_changed({'sql_file': open('sql/la_collision.sql').read()})],
#     }


def task_eda():
    return {
      'actions': [plot_scatter],
      'file_dep': ['data_cache/la_collision.csv',
                   'course/unsupervised_classification/eda.py'],
      'targets': ['data_cache/vignettes/supervised_classification/scatterplot.html']
    }


def task_hcluster_analysis():
    return {
      'actions': [hcluster_analysis],
      'file_dep': ['data_cache/la_collision.csv',
                   'course/unsupervised_classification/tree.py'],
      'targets': ['data_cache/vignettes/supervised_classification/dendrogram.html']
    }


def task_hierarchical_groups():
    return {
      'actions': [lambda: hierarchical_groups(20)],
      'file_dep': ['data_cache/la_collision.csv',
                   'course/unsupervised_classification/tree.py'],
      'targets': ['data_cache/vignettes/supervised_classification/hscatter.html']
    }


def task_kmeans():
    return {
      'actions': [lambda: kmeans(4)],
      'file_dep': ['data_cache/la_collision.csv',
                   'course/unsupervised_classification/tree.py'],
      'targets': ['data_cache/vignettes/supervised_classification/kscatter.html',
                  'data_cache/vignettes/supervised_classification/kcentroids1.html'
                  'data_cache/vignettes/supervised_classification/kcentroids2.html']
    }
