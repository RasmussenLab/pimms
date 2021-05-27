import logging
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


def loss_function(recon_x, x, mask, mu, logvar, t=0.9, device=None):
    """Loss function only considering the observed values in the reconstruction loss.

    Reconstruction + KL divergence losses summed over all *non-masked* elements and batch.

    Default MSE loss would have a too big nominator (Would this matter?)
    MSE = F.mse_loss(input=recon_x, target=x, reduction='mean')


    Parameters
    ----------
    recon_x : [type]
        [description]
    x : [type]
        [description]
    mask : [type]
        [description]
    mu : [type]
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
    MSE = F.mse_loss(input=recon_x*mask,
                     target=x*mask,
                     reduction='sum')  # MSE of observed values
    MSE /= mask.sum()  # only consider observed number of values

    # KL-divergence
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = t*MSE + (1-t)*KLD
    return total, MSE, KLD


def train(epoch, model, train_loader, optimizer, device, writer=None):
    """Train one epoch.

    Parameters
    ----------
    epoch : [type]
        [description]
    model : [type]
        [description]
    train_loader : [type]
        [description]
    optimizer : [type]
        [description]
    device : [type]
        [description]
    writer : [type], optional
        [description], by default None

    Returns
    ------_
    loss : float
        Total loss for each epoch.

    """    
    model.train()
    train_loss = 0
 
    for batch_idx, (data, mask) in enumerate(train_loader):
        data = data.to(device)
        mask = mask.to(device)
        # assert data.is_cuda and mask.is_cuda
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, mse, kld = loss_function(
                            recon_x=recon_batch,
                            x=data,
                            mask=mask,
                            mu=mu,
                            logvar=logvar,
                            device=device)
        logger.debug("Epoch: {epoch:3}, Batch: {batch_idx:4}")
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader) 
    if epoch % 25 == 0:
        logger.info('====> Epoch: {epoch:3} Average loss: {avg_loss:10.4f}'.format(
            epoch=epoch, avg_loss=avg_loss))
    if writer is not None:
        writer.add_scalar('avg training loss',
                          avg_loss,
                          epoch)
    return avg_loss


def eval(model, data_loader, device):
    """Evaluate all batches in data_loader

    Parameters
    ----------
    model : [type]
        [description]
    data_loader : [type]
        [description]
    device : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    model.eval()
    metrics = {'loss': 0, 'mse': 0,  'kld': 0}

    for batch, mask in data_loader:
        batch = batch.to(device)
        mask = mask.to(device)
        batch_recon, mu, logvar = model(batch)
        loss, mse, kld = loss_function(
            recon_x=batch_recon, x=batch, mask=mask, mu=mu, logvar=logvar)
        metrics['loss'] += loss.item()
        metrics['mse'] += mse.item()
        metrics['kld'] += kld.item()
    return metrics

# namedtuple("EpochAverages", 'loss mse kld')
