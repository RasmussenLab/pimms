from torch.nn import functional as F
from fastcore.basics import store_attr
from fastcore.imports import noop
from fastai.data.transforms import Normalize, broadcast_vec
from fastai.callback.core import Callback
from fastai.tabular.core import Tabular, TabularPandas

import pandas as pd

import torch
from torch import nn


def transform_preds(pred: torch.Tensor, index: pd.Index, normalizer) -> pd.Series:
    pred = pd.Series(pred, index).unstack()
    pred = TabularPandas(pred, cont_names=list(pred.columns))
    _ = normalizer.decode(pred)
    pred = pred.items.stack()
    return pred


class NormalizeShiftedMean(Normalize):
    "Normalize/denorm batch of `TensorImage` with shifted mean and scaled variance."

    def __init__(self, mean=None, std=None, axes=(0, 2, 3),
                 shift_mu=0.5, scale_var=2):
        store_attr()

    def setups(self, to: Tabular):
        store_attr(but='to', means=dict(getattr(to, 'train', to).conts.mean()),
                   stds=dict(getattr(to, 'train', to).conts.std(ddof=0)+1e-7))
        self.shift_mu = 0.5
        self.scale_var = 2
        return self(to)

    def encodes(self, to: Tabular):
        to.conts = (to.conts-self.means) / self.stds
        to.conts = to.conts / self.scale_var + self.shift_mu
        return to

    def decodes(self, to: Tabular):
        to.conts = (to.conts - self.shift_mu) * self.scale_var
        to.conts = (to.conts*self.stds) + self.means
        return to

    _docs = dict(encodes="Normalize batch with shifted mean and scaled variance",
                 decodes="Normalize batch with shifted mean and scaled variance")


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

    def get_mu_and_logvar(self, x, detach=False):
        """Helper function to return mu and logvar"""
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.view(-1, 2, self.dim_latent)
        if detach:
            mu_logvar = mu_logvar.detach().numpy()
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar


class ModelAdapter(Callback):
    """Models forward only expects on input matrix. 
    Apply mask from dataloader to both pred and targets."""

    def __init__(self, p=0.1):
        self.do = nn.Dropout(p=p)  # for denoising AE

    def before_batch(self):
        """Remove cont. values from batch (mask)"""
        mask, data = self.xb  # x_cat, x_cont (model could be adapted)
        self.learn._mask = mask != 1
        # dropout data using median
        self.learn.xb = (self.do(data),)

    def after_pred(self):
        M = self._mask.shape[-1]

        if len(self.yb):
            self.learn.yb = (self.y[self.learn._mask],)            

        self.learn.pred = self.pred[self._mask]


class ModelAdapterVAE(Callback):
    """Models forward only expects on input matrix. 
    Apply mask from dataloader to both pred and targets."""

    def before_batch(self):
        """Remove cont. values from batch (mask)"""
        # assert self.yb
        data, mask = self.xb
        self.learn._mask = mask  # mask is True for non-missing, measured values
        # dropout data using median
        self.learn.xb = (data,)

    def after_pred(self):
        M = self._mask.shape[-1]
#         self.learn.yb = (self.y[self.learn._mask],)
#         if not self.training:
        if len(self.yb):
            #             breakpoint()
            self.learn.yb = (self.y[self.learn._mask],)
# #                 self.val_targets.append(self.learn.yb[0])

        pred, mu, logvar = self.pred  # return predictions
        self.learn.pred = (pred[self._mask], mu, logvar)  # is this flat?
#         if not self.training:
#             self.val_preds.append(self.learn.pred)


# from fastai.losses import CrossEntropyLossFlat

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
        Containing: {'loss': total, 'recon_loss': recon_loss, 'KLD': KLD}
                    # {total: loss, recon: loss, kld: loss}

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


#self.loss_func(self.pred, *self.yb)
def loss_fct_vae(pred, y):
    recon_batch, mu, logvar = pred
    batch = y
    res = loss_function(recon_batch=recon_batch,
                        batch=batch,
                        mu=mu,
                        logvar=logvar,
                        reconstruction_loss=F.binary_cross_entropy)
    return res['loss']
