import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _fit_model(df):
    """Given data frame df containing columns 'shortfall', 'n_rooms', 'age' and
    'local_authority_code'
    Fit a linear mixed model with shortfall as the response variable
    n_rooms and age as fixed predictors
    with local_authority_code as a random effect"""
    model = smf.mixedlm("shortfall ~ n_rooms + age", df, groups=df["local_authority_code"])
    return model.fit()


def _save_model_summary(model, outpath):
    with open(outpath, "w") as f:
        f.write(model.summary().as_text())


def _random_effects(results):
    re_df = pd.DataFrame(results.random_effects).T
    re_df.columns = ['Intercept'] + [f"Slope_{i}" for i in range(len(re_df.columns)-1)]
    re_df['group'] = re_df.index
    stderr = np.sqrt(results.cov_re.iloc[0, 0])
    re_df['lower'] = re_df['Intercept'] - 1.96 * stderr
    re_df['upper'] = re_df['Intercept'] + 1.96 * stderr
    re_df = re_df.sort_values('Intercept')
    return re_df


def fit_model():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    results = _fit_model(df)
    outpath = VIGNETTE_DIR / 'model_fit.txt'
    _random_effects(results).to_csv(base_dir / 'data_cache' / 'models' / 'reffs.csv')
    _save_model_summary(results, outpath)
