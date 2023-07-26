import logging
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

import fastai.collab as _fastai

logger = logging.getLogger(__name__)













def build_df_from_pred_batches(pred, scaler=None, index=None, columns=None):
    pred = np.vstack(pred)
    if scaler:
        pred = scaler.inverse_transform(pred)
    pred = pd.DataFrame(pred, index=index, columns=columns)
    return pred


def get_latent_space(model_method_call:callable,
                     dl:torch.utils.data.DataLoader,
                     dl_index:pd.Index,
                     latent_tuple_pos:int=0) -> pd.DataFrame:
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
        #if issubclass(type(res), torch.Tensor):
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




# # Defining the model manuelly

# import torch.nn as nn
# d = 3

# n_features= 10

# class VAE(nn.Module):
#     def __init__(self, d_input=n_features, d=d):
#         super().__init__()

#         self.d_input = d_input
#         self.d_hidden = d

#         self.encoder = nn.Sequential(
#             nn.Linear(d_input, d ** 2),
#             nn.ReLU(),
#             nn.Linear(d ** 2, d * 2)
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(d, d ** 2),
#             nn.ReLU(),
#             nn.Linear(d ** 2, self.d_input),
#             nn.Sigmoid(),
#         )

#     def reparameterise(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.data.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, x):
#         mu_logvar = self.encoder(x.view(-1, self.d_input)).view(-1, 2, d)
#         mu = mu_logvar[:, 0, :]
#         logvar = mu_logvar[:, 1, :]
#         z = self.reparameterise(mu, logvar)
#         return self.decoder(z), mu, logvar

# model = VAE().double().to(device)
# model

# # Training and testing the VAE

# def loss_function(recon_batch, batch, mu, logvar, beta=1):
#     BCE = nn.functional.binary_cross_entropy(
#         recon_batch, batch, reduction='sum'
#     )
#     KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

#     return BCE + beta * KLD

# epochs = 10
# codes = dict(μ=list(), logσ2=list())
# for epoch in range(0, epochs + 1):
#     # Training
#     if epoch > 0:  # test untrained net first
#         model.train()
#         train_loss = 0
#         for x in dl_train:
#             x = x.to(device)
#             # ===================forward=====================
#             x_hat, mu, logvar = model(x)
#             loss = loss_function(x_hat, x, mu, logvar)
#             train_loss += loss.item()
#             # ===================backward====================
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # ===================log========================
#         print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dl_train.dataset):.4f}')

#     # Testing

#     means, logvars = list(), list()
#     with torch.no_grad():
#         model.eval()
#         test_loss = 0
#         for x in dl_valid:
#             x = x.to(device)
#             # ===================forward=====================
#             x_hat, mu, logvar = model(x)
#             test_loss += loss_function(x_hat, x, mu, logvar).item()
#             # =====================log=======================
#             means.append(mu.detach())
#             logvars.append(logvar.detach())
#     # ===================log========================
#     codes['μ'].append(torch.cat(means))
#     codes['logσ2'].append(torch.cat(logvars))
#     test_loss /= len(dl_valid.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
