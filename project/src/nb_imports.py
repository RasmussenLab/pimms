from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from src import config
from src.analyzers import Analysis, AnalyzePeptides
from src import analyzers

import vaep
from vaep.transform import StandardScaler, get_df_fitted_mean_std


