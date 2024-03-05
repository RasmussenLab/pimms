# %%  [markdown]
# # Compare setup of different samling strategies of simulated data
#
# 1. sampling from all samples
# 2. sampling from subset of samples

# %%
from pathlib import Path
import pandas as pd

import vaep.plotting
import vaep.pandas
import vaep.nb

import logging
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'), level=10)


# %%
# parameters
FOLDER = Path('runs/appl_ald_data_rev3/plasma/')

files_in = {'All': 'runs/appl_ald_data_2023_11/plasma/proteinGroups/01_2_performance_summary.xlsx',
            'Subset': FOLDER / 'proteinGroups_subset/01_2_performance_summary.xlsx'
            }

pred_in = {'All': 'runs/appl_ald_data_2023_11/plasma/proteinGroups/01_2_agg_pred_test.csv',
           'Subset': FOLDER / 'proteinGroups_subset/01_2_agg_pred_test.csv'
           }

# %%
fname = FOLDER / 'comparison.xlsx'
print(f"{fname = }")
writer = pd.ExcelWriter(fname)

# %%
cp_all = list()
for key, file_in in files_in.items():
    _ = (pd.read_excel(file_in, index_col=0, sheet_name='mae_stats_ordered_test').iloc[:3]
         .T
         .dropna()
         .astype({'count': int})
         )
    _.columns = pd.MultiIndex.from_tuples((key, k) for k in _.columns)
    cp_all.append(_)
cp_all = pd.concat(cp_all, axis=1)
cp_all

# %%
cp_all.to_excel(writer, sheet_name='all')

# %%
pred = list()
for key, file_in in pred_in.items():
    _ = (pd.read_csv(file_in, index_col=[0, 1])
         ).dropna(axis=1, how='all')
    _ = vaep.pandas.calc_errors.get_absolute_error(_)
    _.columns = pd.MultiIndex.from_tuples((key, k) for k in _.columns)
    pred.append(_)
pred = pd.concat(pred, axis=1)
pred

# %%
cp_top6_subset = (pred
                  .loc[:,
                       pd.IndexSlice[['All', 'Subset'],
                                     ['DAE', 'TRKNN', 'CF', 'RF', 'VAE', 'Median']]]
                  .dropna()
                  .describe()
                  .iloc[:3]
                  .T
                  .astype({'count': int})
                  .unstack(0)
                  .swaplevel(0, 1, axis=1)
                  .loc[:, pd.IndexSlice[['All', 'Subset'], ['count', 'mean', 'std']]]
                  )
cp_top6_subset

# %% [markdown]
# get indices of 1,086 protein groups which are shared between the two setups

# %%
idx_shared = (pred
              .loc[:,
                   pd.IndexSlice[['All', 'Subset'],
                                 ['DAE', 'TRKNN', 'CF', 'RF', 'VAE', 'Median']]]
              .dropna()).index

# %%
cp_subset = (pred
             .loc[idx_shared]
             .describe()
             .iloc[:3]
             .T
             .astype({'count': int})
             .unstack(0)
             .swaplevel(0, 1, axis=1)
             .loc[:, pd.IndexSlice[['All', 'Subset'], ['count', 'mean', 'std']]]
             ).loc[cp_all.index]
cp_subset

# %%
cp_subset.to_excel(writer, sheet_name='subset')

# %%
print(f"{fname = }")
writer.close()
