
import logging
from typing import Tuple

import pandas as pd
# import explicit objects for functional annotations
from fastai.collab import *
from fastai.collab import (Categorify, IndexSplitter, TabularCollab,
                           TransformBlock)
from fastai.tabular.all import *

import vaep.io.dataloaders
import vaep.io.datasplits
from vaep.models import analysis

logger = logging.getLogger(__name__)


def combine_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Helper function to combine training and validation data in long-format. The
    training and validation data will be mixed up in CF training as the sample
    embeddings have to be trained for all samples. The returned frac can be used to have
    the same number of (non-missing) validation samples as before.

    Parameters
    ----------
    train_df : pd.DataFrame
        Consecutive training data in long-format, each row having (unit, feature, value)
    val_df : pd.DataFrame
        Consecutive training data in long-format, each row having (unit, feature, value)

    Returns
    -------
    Tuple[pd.DataFrame, List[list, list]]
        Pandas DataFrame of concatenated samples of training and validation data.
        Fraction of samples originally in validation data.
    """
    X = pd.concat([train_df, val_df]).reset_index()
    frac = len(val_df) / (len(train_df) + len(val_df))
    return X, frac


class CollabAnalysis(analysis.ModelAnalysis):

    def __init__(self,
                 datasplits: vaep.io.datasplits.DataSplits,
                 sample_column: str = 'Sample ID',
                 item_column: str = 'peptide',
                 target_column: str = 'intensity',
                 model_kwargs: dict = None,
                 batch_size: int = 1_024):
        if datasplits.val_y is not None:
            self.X, _ = combine_data(datasplits.train_X,
                                     datasplits.val_y)
        else:
            self.X, _ = datasplits.train_X.reset_index(), 0.0
        self.batch_size = batch_size
        user_name = sample_column
        item_name = item_column
        rating_name = target_column
        cat_names = [user_name, item_name]
        splits = None
        if datasplits.val_y is not None:
            idx_splitter = IndexSplitter(
                list(range(len(datasplits.train_X), len(self.X))))
            splits = idx_splitter(self.X)
        self.to = TabularCollab(
            self.X,
            procs=[Categorify],
            cat_names=cat_names,
            y_names=[rating_name],
            y_block=TransformBlock(),
            splits=splits)
        self.dls = self.to.dataloaders(path='.', bs=self.batch_size)
        self.params = {}
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.params['model_kwargs'] = self.model_kwargs

        self.transform = None  # No data transformation needed
        self.learn = None


def get_missing_values(df_train_long: pd.DataFrame,
                       val_idx: pd.Index,
                       test_idx: pd.Index,
                       analysis_collab: CollabAnalysis) -> pd.Series:
    """Helper function to get missing values from predictions.
    Excludes simulated missing values from validation and test data.

    Parameters
    ----------
    df_train_long : pd.DataFrame
        Training data in long-format, each row having (unit, feature, value)
    val_idx : pd.Index
        Validation index (unit, feature)
    test_idx : pd.Index
        Test index (unit, feature)
    analysis_collab : CollabAnalysis
        CollabAnalysis object

    Returns
    -------
    pd.Series
        Predicted values for missing values in training data (unit, feature, value)
    """
    mask = df_train_long.unstack().isna().stack()
    idx_real_na = mask.loc[mask].index
    idx_real_na = (idx_real_na
                   .drop(val_idx)
                   .drop(test_idx))
    dl_real_na = analysis_collab.dls.test_dl(idx_real_na.to_frame())
    pred_real_na, _ = analysis_collab.learn.get_preds(dl=dl_real_na)
    pred_real_na = pd.Series(pred_real_na, idx_real_na, name='intensity')
    return pred_real_na
