import numpy as np
import pandas as pd

import sklearn.pipeline

import torch
from torch.utils.data import Dataset
from typing import Tuple


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
            self.mask = torch.from_numpy(np.isfinite(data))
        else:
            self.mask = torch.from_numpy(mask)
        self.y = torch.where(self.mask, self.peptides, self.nan)

        if mask is not None:
            self.peptides = torch.where(
                self.mask, self.nan, self.peptides)

        self.peptides = torch.where(self.peptides.isnan(),
                                    torch.FloatTensor([fill_na]), self.peptides)

        self.length_ = len(self.peptides)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.peptides[idx], self.mask[idx], self.y[idx]


def to_tensor(s: pd.Series) -> torch.Tensor:
    return torch.from_numpy(s.values)


class DatasetWithMaskAndNoTarget(Dataset):

    def __init__(self, df: pd.DataFrame, transformer: sklearn.pipeline.Pipeline = None):
        if not issubclass(type(df), pd.DataFrame):
            raise ValueError(
                f'please pass a pandas DataFrame, not: {type(df) = }')
        self.mask_obs = df.isna()  # .astype('uint8') # in case 0,1 is preferred
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
        mask = self.mask_obs.iloc[idx]
        data = self.data.iloc[idx]
        return to_tensor(mask), to_tensor(data)

# DatasetWithMaskAndNoTargetAndNanReplaced

class PeptideDatasetInMemoryMasked(DatasetWithMaskAndNoTarget):
    """Peptide Dataset fully in memory.
    
    Dataset: torch.utils.data.Dataset
    """

    def __init__(self, *args, fill_na=0, **kwargs):
        """[summary]

        Parameters
        ----------
        data : pandas.DataFrame
            Features. Each row contains a set of intensity values.
        fill_na : int, optional
            value to fill missing values, by default 0
        """
        self.fill_na = fill_na
        super().__init__(*args, **kwargs)
        self.data.fillna(self.fill_na, inplace=True)


class PeptideDatasetInMemoryNoMissings(Dataset):
    """Peptide Dataset fully in memory.
    
    Dataset: torch.utils.data.Dataset
    """

    def __init__(self, data: pd.DataFrame, transform=None):
        """Create {} instance.

        Parameters
        ----------
        data : pandas.DataFrame
            Features. Each row contains a set of intensity values.
            No missings expected.
        transform : Callable
            Series of transform to be performed on the training data
        """.format(self.__class__.__name__)
        assert np.isnan(data).sum().sum(
        ) == 0, f"There are {int(np.isnan(data).sum())} missing values."
        self.peptides = np.array(data)
        self.transform = transform
        self.length_ = len(data)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        _peptide_intensities = self.peptides[idx]
        if self.transform:
            _peptide_intensities = self.transform(_peptide_intensities)
        return _peptide_intensities
