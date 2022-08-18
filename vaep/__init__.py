"""
VAEP
Variatonal autoencoder for proteomics
"""

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
from vaep.plotting import savefig

from . import nb

class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values

pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter

vaep.plotting.make_large_descriptors()