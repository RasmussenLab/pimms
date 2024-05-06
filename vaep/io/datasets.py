from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.pipeline
import torch
from torch.utils.data import Dataset

DEFAULT_DTYPE = torch.get_default_dtype()


class PeptideDatasetInMemory(Dataset):
    """Peptide Dataset fully in memory."""

    nan = torch.tensor(float('NaN'))

    def __init__(self, data: np.array, mask: np.array = None, fill_na=0.0):
        """Build torch.Tensors for DataLoader.

        Parameters
        ----------
        data : np.array
            Peptide data for training, potentially with missings.
        mask : [type], optional
            Mask selecting values for evaluation from data(y), by default None
            If no mask is provided, all non-missing values from `data`-array
            will be used.
        fill_na : int, optional
            value to replace missing values with, by default 0
        """
        self.peptides = torch.FloatTensor(data)
        if mask is None:
            self.mask = torch.from_numpy(np.isnan(data))
        else:
            self.mask = torch.from_numpy(mask)
        self.y = torch.where(~self.mask, self.peptides, self.nan)

        if mask is not None:
            self.peptides = torch.where(
                ~self.mask, self.nan, self.peptides)

        self.peptides = torch.where(self.peptides.isnan(),
                                    torch.FloatTensor([fill_na]), self.peptides)

        self.length_ = len(self.peptides)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.mask[idx], self.peptides[idx], self.y[idx]


def to_tensor(s: pd.Series) -> torch.Tensor:
    return torch.from_numpy(s.values).type(DEFAULT_DTYPE)


class DatasetWithMaskAndNoTarget(Dataset):

    # nan = torch.tensor(np.float32('NaN'))
    # if res.dtype is torch.float64: return res.float()

    def __init__(self, df: pd.DataFrame, transformer: sklearn.pipeline.Pipeline = None):
        if not issubclass(type(df), pd.DataFrame):
            raise ValueError(
                f'please pass a pandas DataFrame, not: {type(df) = }')
        self.mask_isna = df.isna()  # .astype('uint8') # in case 0,1 is preferred
        self.columns = df.columns
        self.transformer = transformer
        if transformer:
            if hasattr(transformer, 'transform'):
                df = transformer.transform(df)
            else:
                raise AttributeError(
                    f'{type(transformer)} is not sklearn compatible, has no inverse_transform.')
        self.data = df
        self.length_ = len(self.data)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_isna = self.mask_isna.iloc[idx]
        data = self.data.iloc[idx]
        mask_isna, data = to_tensor(mask_isna), to_tensor(data)
        return mask_isna, data


class DatasetWithTarget(DatasetWithMaskAndNoTarget):

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask, data = super().__getitem__(idx)
        return mask, data, data


class DatasetWithTargetSpecifyTarget(DatasetWithMaskAndNoTarget):

    def __init__(self, df: pd.DataFrame, targets: pd.DataFrame,
                 transformer: sklearn.pipeline.Pipeline = None):
        """Create a dataset for validation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame, indexed as targets
        targets : Targets to use for evaluation
            DataFrame, indexed as df
        transformer : sklearn.pipeline.Pipeline, optional
            transformation pipeline to use, by default None
        """
        if not issubclass(type(df), pd.DataFrame):
            raise ValueError(
                f'please pass a pandas DataFrame, not: {type(df) = }')
        self.mask_isna = targets.isna()
        self.columns = df.columns
        self.transformer = transformer

        self.target = df.fillna(targets)  # not really necessary, without mask would not be needed

        if transformer:
            if hasattr(transformer, 'transform'):
                df = transformer.transform(df)
                self.target = transformer.transform(self.target)
            else:
                raise AttributeError(
                    f'{type(transformer)} is not sklearn compatible, has no inverse_transform.')

        self.data = df
        self.length_ = len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_isna, data = super().__getitem__(idx)
        target = to_tensor(self.target.iloc[idx])
        return mask_isna, data, target
