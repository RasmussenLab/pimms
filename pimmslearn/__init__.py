"""
pimmslearn: a package for imputation using self-supervised deep learning models:

1. Collaborative Filtering
2. Denoising Autoencoder
3. Variational Autoencoder

The package offers Imputation transformers in the style of scikit-learn.

PyPI package is called pimms-learn (with a hyphen).
"""
from __future__ import annotations

# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging
from importlib import metadata

import njab

import pimmslearn.logging
import pimmslearn.nb
import pimmslearn.pandas
import pimmslearn.plotting

_logging.getLogger(__name__).addHandler(_logging.NullHandler())


# put into some pandas_cfg.py file and import all


savefig = pimmslearn.plotting.savefig

__license__ = 'GPLv3'
__version__ = metadata.version("pimms-learn")

__all__ = ['logging', 'nb', 'pandas', 'plotting', 'savefig']

# set some defaults


njab.pandas.set_pandas_number_formatting(float_format='{:,.3f}')

pimmslearn.plotting.make_large_descriptors('x-large')
