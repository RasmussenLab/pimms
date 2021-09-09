import pandas
import torch
from typing import Tuple
from torch.utils.data import DataLoader

# This function temporaily.


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
        self.batch_size=batch_size

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
