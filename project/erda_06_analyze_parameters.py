# %% [markdown]
# # Analyzse and rename dumped parameters

# %%
import vaep
import pandas as pd

fname_mq_params = 'data/all_parameter_files.csv'
fname_id_mappings = 'data/rename/selected_old_new_id_mapping.csv'

fname_out = 'data/selected_parameter_files.csv'

parameter_files = pd.read_csv(fname_mq_params, index_col=0, header=list(range(4))
                              )
parameter_files

# %%
# thread experiments...
vaep.pandas.show_columns_with_variation(
    parameter_files
    .loc[parameter_files.index.duplicated(keep=False)])

# %%
parameter_files = parameter_files.loc[~parameter_files.index.duplicated()]
parameter_files

# %%
id_mappings = pd.read_csv(fname_id_mappings, index_col=0, usecols=['Sample ID', 'new_sample_id'])
id_mappings.head()

# %%
parameter_files.loc[id_mappings.index]

# %%
sel = (parameter_files
       .loc[id_mappings.index]
       .drop('filePaths', axis=1)
       .rename(id_mappings['new_sample_id']))
sel.to_csv(fname_out)
sel

# %% [markdown]
# -inf and + inf cannot be handled correctly (fullMinMz, fullMaxMz)
# number of Threads differs as the setting was varied in the beginning (most runs used 4 threads)

# %%
sel_with_diffs = vaep.pandas.show_columns_with_variation(sel)
sel_with_diffs

# %%
sel_with_diffs.describe()

# %%
sel[('numThreads', 'nan', 'nan', 'nan')].value_counts()

# %%
# 388 columns are identical
sel.drop(sel_with_diffs.columns, axis=1
         ).drop_duplicates()
