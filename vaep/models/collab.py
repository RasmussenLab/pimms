import logging

logger = logging.getLogger(__name__)

from fastai.collab import Module, Embedding, sigmoid_range
from fastai.collab import EmbeddingDotBias

class DotProductBias(Module):
    def __init__(self, n_samples, n_peptides, dim_latent_factors, y_range=(14, 30)):
        self.sample_factors = Embedding(n_samples, dim_latent_factors)
        self.sample_bias = Embedding(n_samples, 1)
        self.peptide_factors = Embedding(
            n_peptides, dim_latent_factors)
        self.peptide_bias = Embedding(n_peptides, 1)
        self.y_range = y_range

    def forward(self, x):
        samples = self.sample_factors(x[:, 0])
        peptides = self.peptide_factors(x[:, 1])
        res = (samples * peptides).sum(dim=1, keepdim=True)
        res += self.sample_bias(x[:, 0]) + self.peptide_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


