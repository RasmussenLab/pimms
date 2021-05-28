import logging
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# from IPython.core.debugger import set_trace # invoke debugging


class Autoencoder(nn.Module):
    pass


class CollabFiltering(nn.Module):
    pass


class VAE(nn.Module):
    """Variational Autoencoder


    Attributes
    ----------
    compression_factor: int
        Compression factor for latent representation in comparison
        to input features, default 0.25
    """

    compression_factor = 0.25

    def __init__(self, n_features: int, n_neurons: int, dim_vae_latent: int = 10):
        """PyTorch model for Variational autoencoder

        Parameters
        ----------
        n_features : int
            number of input features.
        n_neurons : int
            number of neurons in encoder and decoder layer
        """
        super().__init__()

        self._n_neurons = n_neurons
        self._n_features = n_features

        self.dim_vae_latent = dim_vae_latent

        # ToDo: Create Encoder Module for creating encoders
        self.encoder = nn.Linear(n_features, n_neurons).double()
        # latent representation:
        self.mean = nn.Linear(n_neurons, dim_vae_latent).double()  # mean
        self.std = nn.Linear(n_neurons, dim_vae_latent).double()   # stdev

        self.decoder = nn.Linear(dim_vae_latent, n_neurons).double()

        self.out = nn.Linear(n_neurons, n_features).double()

    def encode(self, x):
        h1 = self.encoder(x)
        mu = self.mean(h1)
        # https://github.com/RasmussenLab/vamb/blob/734b741b85296377937de54166b7db274bc7ba9c/vamb/encode.py#L212-L221
        # Jacob retrains his to positive values. This should be garantued by exp-fct in reparameterize
        std = self.std(h1)
        return mu, std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # will always be positive
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # SigmoidRange to smooth gradients?
        # def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
        h3 = F.relu(self.decoder(z))
        return self.out(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_batch: torch.tensor, batch, mu: torch.tensor, logvar: torch.tensor, t: float = 0.9):
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
    total: float
        Total, weighted average loss for provided input and mask
    mse: float
        unweighted mean-squared-error for non-masked inputs
    kld: float
        unweighted Kullback-Leibler divergence between prior and empirical
        normal distribution (defined by encoded moments) on latent representation.
    """
    try:
        if isinstance(batch, torch.Tensor):
            raise ValueError
        X, mask = batch
        MSE = F.mse_loss(input=recon_batch*mask.float(),  # recon_x.mask_select(mask)
                         target=X*mask.float(),  # x.mask_select(mask)
                         reduction='sum')  # MSE of observed values
        # MSE per feature: try to use mean of summed sse per sample?
        # MSE /= mask.sum()  # only consider observed number of values
    except ValueError:
        X = batch
        MSE = F.mse_loss(input=recon_batch, target=X, reduction='sum')
        # MSE loss for each measurement
        # MSE = (x - recon_x).pow(2).mean(axis=0).sum()

    # KL-divergence
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD /= 100
    total = t*MSE + (1-t)*KLD
    return {'loss': total, 'MSE': MSE, 'KLD': KLD}


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
            logvar=logvar)
        batch_metrics[batch_idx] = {
            key: value.item() / len(data) for key, value in _batch_metric.items()}

        # train specific
        optimizer.zero_grad()
        loss = _batch_metric['loss']
        loss.backward()
        optimizer.step()

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


def build_df_from_pred_batches(pred, scaler, index=None, columns=None):
    pred = np.vstack(pred)
    pred = scaler.inverse_transform(pred)
    pred= pd.DataFrame(pred, index=index, columns=columns)
    return pred