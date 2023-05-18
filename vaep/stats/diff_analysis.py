from __future__ import annotations

import numpy as np
import pandas as pd

import pingouin as pg
import statsmodels


def ancova_pg(df_long: pd.DataFrame,
              feat_col: str,
              dv: str,
              between: str,
              covar: list[str]|str,
              fdr=0.05) -> pd.DataFrame:
    """ Analysis of covariance (ANCOVA) using pg.ancova
    https://pingouin-stats.org/generated/pingouin.ancova.html
    
    Adds multiple hypothesis testing correction by Benjamini-Hochberg
    (qvalue, rejected)

    Parameters
    ----------
    df_long : pd.DataFrame
        should be long data format
    feat_col : str
        feature column (or index) name
    dv : str
        Name of column containing the dependant variable, passed to pg.ancova
    between : str
        Name of column containing the between factor, passed to pg.ancova
    covar : list, str
        Name(s) of column(s) containing the covariate, passed to pg.ancova
    fdr : float, optional
        FDR treshold to apply, by default 0.05


    Returns
    -------
    pd.DataFrame
        Columns:  [ 'Source',
                    'SS',
                    'DF',
                    'F',
                    'p-unc',
                    'np2',
                    '{feat_col}',
                    '-Log10 pvalue',
                    'qvalue',
                    'rejected']
    """
    scores = []
    # num_covar = len(covar)

    for feat_name, data_feat in df_long.groupby(feat_col):
        ancova = pg.ancova(data=data_feat, dv=dv,
                           between=between, covar=covar)
        covar = [covar] if isinstance(covar, str) else covar
        N_used = data_feat[[dv, between, *covar]].dropna().shape[0]
        ancova[feat_col] = feat_name
        ancova['N'] = N_used
        scores.append(ancova)
    scores = pd.concat(scores)
    scores['-Log10 pvalue'] = -np.log10(scores['p-unc'])
    scores = scores[scores.Source != 'Residual']

    #FDR correction
    scores = add_fdr_scores(scores, random_seed=123)
    return scores


def add_fdr_scores(scores: pd.DataFrame, random_seed: int = None, alpha=0.05, method='indep') -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)
    reject, qvalue = statsmodels.stats.multitest.fdrcorrection(
        scores['p-unc'], alpha=alpha, method=method)
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores


def analyze(df_proteomics: pd.DataFrame,
            df_clinic: pd.DataFrame,
            target: str,
            covar: list[str],
            value_name: str='intensity') -> pd.DataFrame:
    """apply ancova and multiple test correction.

    Parameters
    ----------
    df_proteomics : pd.DataFrame
        proteomic measurements in wide format
    df_clinic : pd.DataFrame
        clinical data, containing `target` and `covar`
    target : str
        Variable for stratification contained in `df_clinic`
    covar : list[str]
        List of control varialbles contained in `df_clinic`
    value_name : str
        Name to be used for protemics measurements in long-format, default "intensity"

    Returns
    -------
    pd.DataFrame
        Columns = ['SS', 'DF', 'F', 'p-unc', 'np2', '-Log10 pvalue',
                   'qvalue', 'rejected']
    """

    data = (df_proteomics
            .loc[df_clinic[target].notna()]
            .stack()
            .to_frame(value_name)
            .join(df_clinic))
    feat_col = data.index.names[-1]
    scores = ancova_pg(data,
                       feat_col=feat_col,
                       dv=value_name,
                       between=target,
                       covar=covar)
    return scores.set_index([feat_col, 'Source'])
