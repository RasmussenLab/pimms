import logging
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# from IPython.core.debugger import set_trace # invoke debugging
class VAE(nn.Module):
    """Variational Autoencoder


    Attributes
    ----------
    compression_factor: int
        Compression factor for latent representation in comparison
        to input features, default 0.25
    """

    compression_factor = 0.25

    def __init__(self, n_features: int, n_neurons: int):
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

        dim_vae_latent = int(n_features * self.compression_factor)

        self.encoder = nn.Linear(n_features, n_neurons)
        # latent representation:
        self.mean = nn.Linear(n_neurons, dim_vae_latent)  # mean
        self.std = nn.Linear(n_neurons, dim_vae_latent)   # stdev

        self.decoder = nn.Linear(dim_vae_latent, n_neurons)
        self.out = nn.Linear(n_neurons, n_features)

    def encode(self, x):
        h1 = F.relu(self.encoder(x))
        return self.mean(h1), self.std(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.decoder(z))
        return self.out(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self._n_features))
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
    model.train()
    train_loss = 0
    n_samples = len(train_loader.dataset)
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

    avg_loss_per_sample = train_loss / n_samples
    if epoch % 25 == 0:
        logger.info('====> Epoch: {epoch:3} Average loss: {avg_loss:10.4f}'.format(
            epoch=epoch, avg_loss=avg_loss_per_sample))
    if writer is not None:
        writer.add_scalar('training loss',
                          avg_loss_per_sample,
                          epoch)
    return
