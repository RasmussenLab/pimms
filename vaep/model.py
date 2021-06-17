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


class DotProductBias(_fastai.Module):
    def __init__(self, n_samples, n_peptides, dim_latent_factors, y_range=(14, 30)):
        self.sample_factors = _fastai.Embedding(n_samples, dim_latent_factors)
        self.sample_bias = _fastai.Embedding(n_samples, 1)
        self.peptide_factors = _fastai.Embedding(
            n_peptides, dim_latent_factors)
        self.peptide_bias = _fastai.Embedding(n_peptides, 1)
        self.y_range = y_range

    def forward(self, x):
        samples = self.sample_factors(x[:, 0])
        peptides = self.peptide_factors(x[:, 1])
        res = (samples * peptides).sum(dim=1, keepdim=True)
        res += self.sample_bias(x[:, 0]) + self.peptide_bias(x[:, 1])
        return _fastai.sigmoid_range(res, *self.y_range)



class Autoencoder(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_neurons: int,
                 activation=nn.Tanh,
                 last_encoder_activation=nn.Tanh,
                 last_decoder_activation=None,
                 dim_latent: int = 10):
        """Initialize an Autoencoder

        Parameters
        ----------
        n_features : int
            Input dimension
        n_neurons : int
            Hidden layer dimension in Encoder and Decoder
        activation : [type], optional
            Activatoin for hidden layers, by default nn.Tanh
        last_encoder_activation : [type], optional
            Optional last encoder activation, by default nn.Tanh
        last_decoder_activation : [type], optional
            Optional last decoder activation, by default None
        dim_latent : int, optional
            Hidden space dimension, by default 10
        """
        super().__init__()
        self.n_features, self.n_neurons = n_features, n_neurons
        self.dim_latent = dim_latent

        # Encoder
        self.encoder = [
            nn.Linear(n_features, n_neurons),
            activation(),
            nn.Linear(n_neurons, dim_latent)]
        if last_encoder_activation:
            self.encoder.append(last_encoder_activation())
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        self.decoder = [nn.Linear(dim_latent, n_neurons),
                        activation(),
                        nn.Linear(n_neurons, n_features)]
        if last_decoder_activation:
            self.decoder.append(last_decoder_activation())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# from fastai.losses import MSELossFlat
# from fastai.learner import Learner


class VAE(Autoencoder):
    """Variational Autoencoder. Latent dimension is composed of mean and log variance,
    so effecively the number of neurons are duplicated.
    """

    def __init__(self,
                 n_features: int,
                 n_neurons: int,
                 activation=nn.Tanh,
                 last_encoder_activation=nn.Tanh,
                 last_decoder_activation=None,
                 dim_latent: int = 10):

        # locals() need to be cleand, otherwise one runs into
        # ModuleAttributeError: 'VAE' object has no attribute '_modules'
        _locals = locals()
        args = {k: _locals[k] for k in _locals.keys() if k not in [
            'self', '__class__', '_locals']}

        super().__init__(**args)

        del self.encoder
        # this could also become a seperate encoder class
        self.encoder = [
            nn.Linear(n_features, n_neurons),
            activation(),
            nn.Linear(n_neurons, dim_latent*2)]
        if last_encoder_activation:
            self.encoder.append(last_encoder_activation())
        self.encoder = nn.Sequential(*self.encoder)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
            # std = torch.exp(0.5*logvar)  # will always be positive
            # eps = torch.randn_like(std)
            # return mu + eps*std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.get_mu_and_logvar(x)
        z = self.reparameterize(mu=mu, logvar=logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def get_mu_and_logvar(self, x):
        """Helper function to return mu and logvar"""
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.view(-1, 2, self.dim_latent)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar


def loss_function(recon_batch: torch.tensor,
                  batch: torch.tensor,
                  mu: torch.tensor,
                  logvar: torch.tensor,
                  reconstruction_loss=F.mse_loss,
                  t: float = 0.9):
    """Loss function only considering the observed values in the reconstruction loss.

    Reconstruction + KL divergence losses summed over all *non-masked* elements and batch.


    Parameters
    ----------
    recon_batch : torch.tensor
        Model output
    batch : Union[tuple, torch.tensor]
        Batch from dataloader. Either only data or tuple of (data, mask)
    mu : torch.tensor
        [description]
    logvar : [type]
        [description]
    t : float, optional
        [description], by default 0.9

    Returns
    -------
    dict
        Containing: {total: loss, recon: loss, kld: loss}

        total: float
            Total, weighted average loss for provided input and mask
        reconstruction_loss: float
            reconstruction loss for non-masked inputs
        kld: float
            unweighted Kullback-Leibler divergence between prior and empirical
            normal distribution (defined by encoded moments) on latent representation.
    """
    try:
        if isinstance(batch, torch.Tensor):
            raise ValueError
        X, mask = batch
        recon_batch = recon_batch*mask.float()  # recon_x.mask_select(mask)
        X = X*mask.float()  # x.mask_select(mask)
    except ValueError:
        X = batch
    recon_loss = reconstruction_loss(
        input=recon_batch, target=X, reduction='sum')

    # KL-divergence
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # there might be an error in the paper log(sigma^2) -> log(sigma)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + t*KLD
    return {'loss': total, 'recon_loss': recon_loss, 'KLD': KLD}


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
