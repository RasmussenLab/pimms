from attr import Attribute
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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