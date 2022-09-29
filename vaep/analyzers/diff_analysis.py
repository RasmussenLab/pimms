from __future__ import annotations
from collections import namedtuple
import logging

import pandas as pd

logger = logging.getLogger()


Cutoffs = namedtuple('Cutoffs', 'feat_completness_over_samples min_feat_in_sample')


def select_raw_data(df: pd.DataFrame,
                    data_completeness: float,
                    frac_protein_groups: int) -> tuple[pd.DataFrame, Cutoffs]:
    msg = 'N samples: {}, M feat: {}'
    N, M = df.shape
    logger.info("Initally: " + msg.format(N, M))
    treshold_completeness = int(M * data_completeness)
    df = df.dropna(axis=1, thresh=treshold_completeness)
    logger.info(
        f"Dropped features quantified in less than {int(treshold_completeness)} samples.")
    N, M = df.shape
    logger.info("After feat selection: " + msg.format(N, M))
    min_n_protein_groups = int(M * frac_protein_groups)
    logger.info(
        f"Min No. of Protein-Groups in single sample: {min_n_protein_groups}")
    df = df.dropna(axis=0, thresh=min_n_protein_groups)
    logger.info("Finally: " + msg.format(*df.shape))

    return df, Cutoffs(treshold_completeness, min_n_protein_groups)


def select_feat(df_qc:pd.DataFrame, threshold:float=0.4, axis:int=0):
    qc_cv_feat = df_qc.std(axis=axis) / df_qc.mean(axis=axis)
    mask = qc_cv_feat < threshold
    return qc_cv_feat.loc[mask].index