from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src import config
from vaep.analyzers.analyzers import Analysis, AnalyzePeptides
from vaep.analyzers import analyzers

import vaep
from vaep.transform import StandardScaler, get_df_fitted_mean_std


plt.rcParams.update({'xtick.labelsize': 'xx-large',
                     'ytick.labelsize': 'xx-large',
                     'axes.titlesize' : 'xx-large',
                     'axes.labelsize' : 'xx-large',
                    })