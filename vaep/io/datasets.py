from attr import Attribute
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


class PeptideDatasetInMemoryMasked(Dataset):
    """Peptide Dataset fully in memory.
    
    Dataset: torch.utils.data.Dataset
    """
    def __init__(self, data: pd.DataFrame, fill_na=0, device=None):
        """[summary]

        Parameters
        ----------
        data : pandas.DataFrame
            Features. Each row contains a set of intensity values.
        fill_na : int, optional
            value to fill missing values, by default 0
        """
        assert np.isnan(data).sum() > 0, "There a no missing values in the data."
        # ensure copy? https://stackoverflow.com/a/52103839/9684872
        # https://numpy.org/doc/stable/reference/routines.array-creation.html#routines-array-creation
        self.mask_obs = torch.from_numpy(np.isfinite(data))
        # data = data.fillna(fill_na)
        self.peptides = torch.from_numpy(np.nan_to_num(data, nan=fill_na))
        self.length_ = len(data)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.peptides[idx], self.mask_obs[idx]

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
        assert np.isnan(data).sum().sum() == 0, f"There are {int(np.isnan(data).sum())} missing values."
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