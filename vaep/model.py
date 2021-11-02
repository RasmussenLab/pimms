from vaep.models.ae import loss_function
import logging
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

import fastai.collab as _fastai
# from fastai.collab import sigmoid_range, Module, Embedding

logger = logging.getLogger(__name__)


# from IPython.core.debugger import set_trace # invoke debugging


# from fastai.losses import MSELossFlat
# from fastai.learner import Learner


# Reconstruction + β * KL divergence losses summed over all elements and batch
# def loss_function(recon_batch, batch, mu, logvar, beta=1):
#     BCE = nn.functional.binary_cross_entropy(
#         recon_batch, batch, reduction='sum'
#     )
#     KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

#     return {'loss':  BCE + beta * KLD, 'BCE': BCE, 'KLD': KLD}

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim,
          device, return_pred=False):
    """Train one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        [description]
    train_loader : torch.utils.data.DataLoader
        [description]
    optimizer : torch.optim
        [description]
    device : [type]
        [description]
    """
    model.train()
    batch_metrics = {}

    for batch_idx, batch in enumerate(train_loader):
        try:
            data, mask = batch
            batch = batch.to(device)
            mask = mask.to(device)
            batch = (data, mask)
        except:
            batch = batch.to(device)
            data = batch

        recon_batch, mu, logvar = model(data)
        _batch_metric = loss_function(
            # this needs to be just batch_data (which is then unpacked?) # could be a static Model function
            recon_batch=recon_batch,
            batch=data,
            mu=mu,
            reconstruction_loss=F.binary_cross_entropy,
            logvar=logvar)

        # train specific
        optimizer.zero_grad()
        loss = _batch_metric['loss']
        loss.backward()
        optimizer.step()

        batch_metrics[batch_idx] = {
            key: value.item() / len(data) for key, value in _batch_metric.items()}

    return batch_metrics


def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
             device,
             return_pred=False):
    """Evaluate all batches in data_loader

    Parameters
    ----------
    model : torch.nn.Module
        [description]
    data_loader : torch.utils.data.DataLoader
        [description]
    device : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    model.eval()
    assert model.training == False
    batch_metrics = {}
    if return_pred:
        pred = []
    for batch_idx, batch in enumerate(data_loader):
        try:
            if not isinstance(batch, torch.Tensor):
                data, mask = batch
            else:
                raise ValueError
            batch = batch.to(device)
            mask = mask.to(device)
            batch = (data, mask)
        except ValueError:
            batch = batch.to(device)
            data = batch

        recon_batch, mu, logvar = model(data)
        _batch_metric = loss_function(
            # this needs to be just batch_data (which is then unpacked?) # could be a static Model function
            recon_batch=recon_batch,
            batch=data,
            mu=mu,
            logvar=logvar)
        batch_metrics[batch_idx] = {
            key: value.item() / len(data) for key, value in _batch_metric.items()}
        if return_pred:
            pred.append(recon_batch.detach().numpy())
    return batch_metrics if not return_pred else (batch_metrics, pred)


def build_df_from_pred_batches(pred, scaler=None, index=None, columns=None):
    pred = np.vstack(pred)
    if scaler:
        pred = scaler.inverse_transform(pred)
    pred = pd.DataFrame(pred, index=index, columns=columns)
    return pred


def process_train_loss(d: dict, alpha=0.1):
    """Process training loss to DataFrame.

    Parameters
    ----------
    d: dict
        Dictionary of {'key': Iterable}
    alpha: float
        Smooting factor, default 0.1

    Returns
    -------
    df: pandas.DataFrame
        Pandas DataFrame including the loss and smoothed loss.
    """
    assert len(
        d) == 1, "Not supported here. Only one list-like loss with key {key: Iterable}."
    df = pd.DataFrame(d)
    key = next(iter(d.keys()))
    df = df.reset_index(drop=False).rename(columns={'index': 'steps'})
    key_new = f'{key} smoothed'
    df[key_new] = df[key].ewm(alpha=alpha).mean()
    return df


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
