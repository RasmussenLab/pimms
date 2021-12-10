import pandas
import torch
from typing import Tuple

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from fastai.data.core import DataLoaders

from vaep.io.datasets import DatasetWithTarget
from vaep.transform import VaepPipeline


class DataLoadersCreator():
    """DataLoader creator. For training or evaluation."""

    def __init__(self,
                 df_train: pandas.DataFrame,
                 df_valid: pandas.DataFrame,
                 scaler,
                 DataSetClass: torch.utils.data.Dataset,
                 batch_size: int
                 ):
        """Helper function to create from pandas.DataFrame(s) in memory datasets.

        Parameters
        ----------
        df_train : pandas.DataFrame
            [description]
        df_valid : pandas.DataFrame
            [description]
        scaler : [type]
            A pipeline of transform to apply to the dataset.
        DataSetClass : torch.utils.data.Dataset
            [description]
        batch_size : int
            [description]

        Returns
        -------
        Tuple[torch.utils.data.Dataloader, torch.utils.data.Dataloader]
            train and validation set dataloaders.
        """
        self.data_train = DataSetClass(
            data=scaler.transform(df_train))
        self.data_valid = DataSetClass(data=scaler.transform(df_valid))
        self.scaler = scaler
        self.batch_size = batch_size

    def get_dls(self, shuffle_train: bool = True, **kwargs) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        self.shuffle_train = shuffle_train
        dl_train = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size, shuffle=shuffle_train, **kwargs)

        dl_valid = DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size, shuffle=False, **kwargs)
        return dl_train, dl_valid

    def __repr__(self):
        return f"{self.__class__.__name__} for creating dataloaders with {self.batch_size}."


def get_dls(train_X: pandas.DataFrame,
            valid_X: pandas.DataFrame,
            transformer: VaepPipeline,
            dataset: Dataset = DatasetWithTarget,
            bs: int = 64) -> DataLoaders:
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
        number of samples in a single batch, by default 64

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
    train_ds, valid_ds = (dataset(train_X, transformer),
                          dataset(valid_X, transformer))
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs)
