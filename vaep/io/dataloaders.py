import pandas
import torch
from typing import Tuple
from torch.utils.data import DataLoader

# This function temporaily. 
def get_dataloaders(df_train: pandas.DataFrame,
                    df_valid: pandas.DataFrame,
                    scaler,
                    DataSetClass: torch.utils.data.Dataset,
                    batch_size: int,
                    **kwargs
                    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    data_train = DataSetClass(
        data=scaler.transform(df_train))
    data_valid = DataSetClass(data=scaler.transform(df_valid))

    dl_train = DataLoader(
        dataset=data_train,
        batch_size=batch_size, shuffle=True, **kwargs)

    dl_valid = DataLoader(
        dataset=data_valid,
        batch_size=batch_size, shuffle=False, **kwargs)

    return dl_train, dl_valid
