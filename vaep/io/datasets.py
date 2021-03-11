import pandas as pd
import torch
from torch.utils.data import Dataset

class PeptideDatasetInMemory(Dataset):
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
        self.mask_obs = torch.from_numpy(data.notna().values)
        data = data.fillna(fill_na)
        self.peptides = torch.from_numpy(data.values)
        self.length_ = len(data)
        # if device:
        #     assert isinstance(device, torch.device
        #     ), "Please pass a torch.device, not {}: {}".format(
        #         type(torch.device), torch.device
        #     )
        #     self.device = device
        # else:
        #     self.device = torch.device()

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.peptides[idx], self.mask_obs[idx]
