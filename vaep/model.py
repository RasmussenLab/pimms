import logging
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger()


class PeptideDatasetInMemory(Dataset):
    """Peptide Dataset fully in memory."""

    def __init__(self, data: pd.DataFrame, fill_na=0):
        self.mask_obs = torch.from_numpy(data.notna().values)
        data = data.fillna(fill_na)
        self.peptides = torch.from_numpy(data.values)
        self.length_ = len(data)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.peptides[idx], self.mask_obs[idx]


# from IPython.core.debugger import set_trace # invoke debugging
class VAE(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()

        self._n_neurons = n_neurons
        self._n_features = n_features

        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc21 = nn.Linear(n_neurons, 50)
        self.fc22 = nn.Linear(n_neurons, 50)
        self.fc3 = nn.Linear(50, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self._n_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mask, mu, logvar, t=0.9):
    """Loss function only considering the observed values in the
    reconstruction loss.

    Reconstruction + KL divergence losses summed over all *non-masked* elements and batch.

    Default MSE loss would have a too big nominator (Would this matter?)
    MSE = F.mse_loss(input=recon_x, target=x, reduction='mean')


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

    return t*MSE + (1-t)*KLD


def train(epoch, model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    N_SAMPLES = len(train_loader.dataset)
    for (data, mask) in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mask, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print(batch_idx)
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))
    # logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
    #     epoch, train_loss / N_SAMPLES))
