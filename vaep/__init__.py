"""
VAEP
Variatonal autoencoder for proteomics
"""
from __future__ import annotations
# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

# put into some pandas_cfg.py file and import all
import pandas as pd
import pandas.io.formats.format as pf

import vaep.pandas
import vaep.plotting
import vaep.logging
import vaep.plotting

import vaep.nb

savefig = vaep.plotting.savefig

__license__ = 'GPLv3'
__version__ = (0, 1, 0)


# set some defaults
class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter

vaep.plotting.make_large_descriptors('x-large')
