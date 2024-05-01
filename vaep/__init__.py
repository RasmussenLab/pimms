"""
VAEP
Variatonal autoencoder for proteomics
"""
from __future__ import annotations

# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging
from importlib import metadata

import pandas as pd
import pandas.io.formats.format as pf

# from . import logging, nb, pandas, plotting
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


class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter

vaep.plotting.make_large_descriptors('x-large')
