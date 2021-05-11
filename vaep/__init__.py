"""
VAEP
Variatonal autoencoder for proteomics
"""

# Add imports here
# from . import *

# put into some pandas_cfg.py file and import all
import pandas as pd
import pandas.io.formats.format as pf

class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values

pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter

