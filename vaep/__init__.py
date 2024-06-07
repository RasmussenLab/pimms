"""
VAEP
Variatonal autoencoder for proteomics
"""
from __future__ import annotations

# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging
from importlib import metadata

import njab

import vaep.logging
import vaep.nb
import vaep.pandas
import vaep.plotting

_logging.getLogger(__name__).addHandler(_logging.NullHandler())


# put into some pandas_cfg.py file and import all


savefig = vaep.plotting.savefig

__license__ = 'GPLv3'
__version__ = metadata.version("pimms-learn")

__all__ = ['logging', 'nb', 'pandas', 'plotting', 'savefig']

# set some defaults


njab.pandas.set_pandas_number_formatting(float_format='{:,.3f}')

vaep.plotting.make_large_descriptors('x-large')
