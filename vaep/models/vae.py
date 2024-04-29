"""VAE implementation based on https://github.com/ronaldiscool/VAETutorial

Adapted to the setup of learning missing values.

- funnel architecture (or fixed hidden layer layout)
- loss is adapted to Dataset and FastAI adaptions
- batchnorm1D for now (not weight norm)
"""
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

leaky_relu_default = nn.LeakyReLU(.1)


class VAE(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_neurons: List[int],
                 activation=leaky_relu_default,
                 #  last_encoder_activation=leaky_relu_default,
                 last_decoder_activation=None,
                 dim_latent: int = 10):
        super().__init__()
        # set up hyperparameters
        self.n_features, self.n_neurons = n_features, list(n_neurons)
        self.layers = [n_features, *self.n_neurons]
        self.dim_latent = dim_latent

        # define architecture hidden layer
        def build_layer(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(out_feat),
                    activation]

        # Encoder
        self.encoder = []

        for i in range(len(self.layers) - 1):
            in_feat, out_feat = self.layers[i:i + 2]
            self.encoder.extend(build_layer(in_feat=in_feat,
                                            out_feat=out_feat))
        self.encoder.append(nn.Linear(out_feat, dim_latent * 2))

        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        self.layers_decoder = self.layers[::-1]
        assert self.layers_decoder is not self.layers
        assert out_feat == self.layers_decoder[0]

        self.decoder = build_layer(in_feat=self.dim_latent,
                                   out_feat=out_feat)

        i = -1  # in case a single hidden layer is passed
        for i in range(len(self.layers_decoder) - 2):
            in_feat, out_feat = self.layers_decoder[i:i + 2]
            self.decoder.extend(build_layer(in_feat=in_feat,
                                            out_feat=out_feat))
        in_feat, out_feat = self.layers_decoder[i + 1:i + 3]

        self.decoder.append(nn.Linear(in_feat, out_feat * 2))
        if last_decoder_activation is not None:
            self.append(last_decoder_activation)

        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, x):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.dim_latent]
        z_logvar = z_params[:, self.dim_latent:]
        return z_mu, z_logvar

    def get_mu_and_logvar(self, x, detach=False):
        return self.encode(x)

    def decode(self, z):
        x_params = self.decoder(z)
        x_mu = x_params[:, :self.n_features]
        x_logvar = x_params[:, self.n_features:]
        return x_mu, x_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar


def compute_kld(z_mu, z_logvar):
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)


def gaussian_log_prob(z, mu, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu)**2 / torch.exp(logvar))


def loss_fct(pred, y, reduction='sum', results: List = None, freebits=0.1):
    x_mu, x_logvar, z_mu, z_logvar = pred
    batch = y

    l_rec = -torch.sum(gaussian_log_prob(batch, x_mu, x_logvar))
    l_reg = torch.sum((F.relu(compute_kld(z_mu, z_logvar)
                              - freebits * math.log(2))
                       + freebits * math.log(2)),
                      1)

    if results is not None:
        results.append((l_rec.item(), torch.mean(l_reg).item()))
    return l_rec / l_reg.shape[0] + torch.mean(l_reg)
