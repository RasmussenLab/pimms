
import pandas
import pandas as pd
from fastai.data.all import *
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from torch.utils.data import Dataset

from vaep.io import datasets
from vaep.io.datasets import DatasetWithTarget
from vaep.transform import VaepPipeline


def get_dls(train_X: pandas.DataFrame,
            valid_X: pandas.DataFrame,
            transformer: VaepPipeline,
            bs: int = 64,
            num_workers=0) -> DataLoaders:
    """Create training and validation dataloaders

    Parameters
    ----------
    train_X : pandas.DataFrame
        Training Data, index is ignored for data fetching
    valid_X : pandas.DataFrame
        Validation data, won't be shuffled.
    transformer : VaepPipeline
        Pipeline with separate encode and decode
    bs : int, optional
        batch size, by default 64
    num_workers : int, optional
        number of workers to use for data loading, by default 0

    Returns
    -------
    fastai.data.core.DataLoaders
        FastAI DataLoaders with train and valid Dataloder

    Example
    -------
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from vaep.dataloader import get_dls
    from vaep.transform import VaepPipeline

    dae_default_pipeline = sklearn.pipeline.Pipeline(
        [('normalize', StandardScaler()),
         ('impute', SimpleImputer(add_indicator=False))
         ])
    # train_X, val_X = None, None # pandas.DataFrames
    transforms = VaepPipeline(df_train=train_X,
                                  encode=dae_default_pipeline,
                                  decode=['normalize'])
    dls = get_dls(train_X, val_X, transforms, bs=4)
    """
    train_ds = datasets.DatasetWithTarget(df=train_X,
                                          transformer=transformer)
    if valid_X is not None:
        valid_ds = datasets.DatasetWithTargetSpecifyTarget(df=train_X,
                                                           targets=valid_X,
                                                           transformer=transformer)
    else:
        # empty dataset will be ignored by fastai in training loops
        valid_ds = datasets.DatasetWithTarget(df=pd.DataFrame())
    # ! Need for script exection (as plain python file)
    # https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    drop_last = False
    if (len(train_X) % bs) == 1:
        # Batch-Normalization does not work with batches of size one
        drop_last = True
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs, drop_last=drop_last,
                                  num_workers=num_workers)


# dls.test_dl
# needs to be part of setup procedure of a class
def get_test_dl(df: pandas.DataFrame,
                transformer: VaepPipeline,
                dataset: Dataset = DatasetWithTarget,
                bs: int = 64):
    """[summary]

    Parameters
    ----------
    df : pandas.DataFrame
        Test data in a DataFrame
    transformer : vaep.transform.VaepPipeline
        Pipeline with separate encode and decode
    dataset : torch.utils.data.Dataset, optional
        torch Dataset to yield encoded samples, by default DatasetWithTarget
    bs : int, optional
        batch size, by default 64

    Returns
    -------
    fastai.data.load.DataLoader
        DataLoader from fastai for test data.
    """
    ds = dataset(df, transformer)
    return DataLoader(ds, bs=bs, shuffle=False)
