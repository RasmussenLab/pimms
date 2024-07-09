import logging

import numpy as np
import pandas as pd
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def build_df_from_pred_batches(pred, scaler=None, index=None, columns=None):
    pred = np.vstack(pred)
    if scaler:
        pred = scaler.inverse_transform(pred)
    pred = pd.DataFrame(pred, index=index, columns=columns)
    return pred


def get_latent_space(model_method_call: callable,
                     dl: torch.utils.data.DataLoader,
                     dl_index: pd.Index,
                     latent_tuple_pos: int = 0) -> pd.DataFrame:
    """Create a DataFrame of the latent space based on the model method call
    to be used (here: the model encoder or a latent space helper method)

    Parameters
    ----------
    model_method_call : callable
        A method call on a pytorch.Module to create encodings for a batch of data.
    dl : torch.utils.data.DataLoader
        The DataLoader to use, producing predictions in a non-random fashion.
    dl_index : pd.Index
        pandas Index
    latent_tuple_pos : int, optional
        if model_method_call returns a tuple from a batch,
        the tensor at the tuple position to selecet, by default 0

    Returns
    -------
    pd.DataFrame
        DataFrame of latent space indexed with dl_index.
    """
    latent_space = []
    for b in dl:
        model_input = b[1]
        res = model_method_call(model_input)
        # if issubclass(type(res), torch.Tensor):
        if isinstance(res, tuple):
            res = res[latent_tuple_pos]
        res = res.detach().numpy()
        latent_space.append(res)
    M = res.shape[-1]

    latent_space = build_df_from_pred_batches(latent_space,
                                              index=dl_index,
                                              columns=[f'latent dimension {i+1}'
                                                       for i in range(M)])
    return latent_space
