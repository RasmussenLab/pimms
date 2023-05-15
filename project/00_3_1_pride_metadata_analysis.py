# %%

# %% [markdown]
# ## Output Excel for Analysis

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn


from vaep.io import thermo_raw_files
import vaep.pandas

plt.rcParams['figure.figsize'] = [4, 3]
vaep.plotting.make_large_descriptors(5)


# %% [tags=["parameters"]]
fn_meta = 'data/pride_metadata.csv'
date_col: str = 'Content Creation Date'
out_folder: str = 'data/dev_datasets/pride_upload'

# %% [markdown]
# ## Prepare outputs

# %%
out_folder = Path(out_folder)
out_folder.mkdir(exist_ok=True)
files_out = dict()

# %%
df_meta = pd.read_csv(fn_meta, index_col=0)
df_meta

# %%

df_meta['instrument_label'] = (
    df_meta["Thermo Scientific instrument model"].str.replace(' ', '-')
    + '_'
    + df_meta["instrument serial number"].str.split('#').str[-1]
)

# {k: k.replace('-Orbitrap_', ' ').replace('-', ' ').replace('_', ' ')
#  for k in df_meta['instrument_label'].unique()}
# further small changes applied manually
# based on https://www.ebi.ac.uk/ols4/
#
# Q Exactive HF-X MS:1002877
# Q Exactive HF MS:1002523
# Orbitrap Exploris 480 MS:1003028
# Exactive Plus MS:1002526
# Q Exactive MS:1001911
# Orbitrap Fusion Lumos MS:1002732

instrument_labels = {'Q-Exactive-Orbitrap_1': 'Q Exactive 1',
 'Q-Exactive-Plus-Orbitrap_1': 'Exactive Plus 1',
 'Q-Exactive-HF-Orbitrap_206': 'Q Exactive HF 206',
 'Q-Exactive-Plus-Orbitrap_143': 'Exactive Plus 143',
 'Q-Exactive-HF-Orbitrap_1': 'Q Exactive HF 1',
 'Q-Exactive-HF-Orbitrap_147': 'Q Exactive HF 147',
 'Q-Exactive-HF-Orbitrap_204': 'Q Exactive HF 204',
 'Q-Exactive-HF-Orbitrap_148': 'Q Exactive HF 148',
 'Q-Exactive-HF-Orbitrap_207': 'Q Exactive HF 207',
 'Q-Exactive-HF-Orbitrap_143': 'Q Exactive HF 143',
 'Orbitrap-Fusion-Lumos_FSN20115': 'Orbitrap Fusion Lumos FSN20115',
 'Q-Exactive-HF-Orbitrap_2612': 'Q Exactive HF 2612',
 'Q-Exactive-HF-X-Orbitrap_6016': 'Q Exactive HF-X 6016',
 'Q-Exactive-HF-X-Orbitrap_6004': 'Q Exactive HF-X 6004',
 'Q-Exactive-HF-X-Orbitrap_6075': 'Q Exactive HF-X 6075',
 'Q-Exactive-HF-X-Orbitrap_6078': 'Q Exactive HF-X 6078',
 'Q-Exactive-HF-X-Orbitrap_6070': 'Q Exactive HF-X 6070',
 'Q-Exactive-HF-X-Orbitrap_6071': 'Q Exactive HF-X 6071',
 'Q-Exactive-HF-X-Orbitrap_6011': 'Q Exactive HF-X 6011',
 'Q-Exactive-HF-X-Orbitrap_6073': 'Q Exactive HF-X 6073',
 'Q-Exactive-HF-X-Orbitrap_6101': 'Q Exactive HF-X 6101',
 'Q-Exactive-HF-X-Orbitrap_6096': 'Q Exactive HF-X 6096',
 'Exactive-Series-Orbitrap_6004': 'Exactive Series 6004',
 'Q-Exactive-HF-X-Orbitrap_6043': 'Q Exactive HF-X 6043',
 'Q-Exactive-HF-X-Orbitrap_6025': 'Q Exactive HF-X 6025',
 'Q-Exactive-HF-X-Orbitrap_6022': 'Q Exactive HF-X 6022',
 'Q-Exactive-HF-X-Orbitrap_6023': 'Q Exactive HF-X 6023',
 'Q-Exactive-HF-X-Orbitrap_6028': 'Q Exactive HF-X 6028',
 'Q-Exactive-HF-X-Orbitrap_6013': 'Q Exactive HF-X 6013',
 'Q-Exactive-HF-X-Orbitrap_6044': 'Q Exactive HF-X 6044',
 'Q-Exactive-HF-X-Orbitrap_6324': 'Q Exactive HF-X 6324',
 'Orbitrap-Exploris-480_Invalid_SN_0001': 'Orbitrap Exploris 480 Invalid SN 0001',
 'Orbitrap-Exploris-480_MA10134C': 'Orbitrap Exploris 480 MA10134C',
 'Orbitrap-Exploris-480_MA10132C': 'Orbitrap Exploris 480 MA10132C',
 'Orbitrap-Exploris-480_MA10130C': 'Orbitrap Exploris 480 MA10130C',
 'Orbitrap-Exploris-480_MA10215C': 'Orbitrap Exploris 480 MA10215C'}

df_meta["instrument_label"] = df_meta["instrument_label"].replace(instrument_labels)

# %%
writer_args = dict(float_format='%.3f')
fname = out_folder / 'pride_data_infos.xlsx'
files_out[fname.name] = fname.as_posix()
excel_writer = pd.ExcelWriter(fname)

# %% [markdown]
# ## Varying data between runs

# %%
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats.T.to_excel(excel_writer, sheet_name='des_stats', **writer_args)

view = meta_stats.loc[:, (meta_stats.loc['unique'] > 1)
                      | (meta_stats.loc['std'] > 0.01)].T
view.to_excel(excel_writer, sheet_name='des_stats_varying', **writer_args)

# %% [markdown]
# ## Instruments in selection

# %%
thermo_raw_files.cols_instrument

# %%
df_meta[date_col] = pd.to_datetime(df_meta[date_col])

counts_instrument = (df_meta
                     .groupby(thermo_raw_files.cols_instrument)[date_col]
                     .agg(['count', 'min', 'max'])
                     .sort_values(by=thermo_raw_files.cols_instrument[:2] + ['count'], ascending=False))

counts_instrument = counts_instrument.join(
    (df_meta
    [[*thermo_raw_files.cols_instrument, 'instrument_label']]
    .drop_duplicates()
    .set_index(thermo_raw_files.cols_instrument)
    )
    .set_index('instrument_label', append=True)
)
counts_instrument.to_excel(
    excel_writer, sheet_name='instruments', **writer_args)
counts_instrument

# %%
top10_instruments = counts_instrument['count'].nlargest(10)
top10_instruments

# %%
mask_top10_instruments = (df_meta[thermo_raw_files.cols_instrument]
                          .apply(
    lambda x: tuple(x) in top10_instruments.index, axis=1))
assert mask_top10_instruments.sum() == top10_instruments.sum()


# %%
# counts_instrument = (df_meta
#                      .groupby(['instrument_label'])[date_col]
#                      .agg(['count', 'min', 'max'])
#                      .sort_values('count', ascending=False)
#                      )
counts_instrument = (counts_instrument
                     .reset_index()
                     .set_index('instrument_label')
                     ['count']
                     .sort_values(ascending=False)
                     )
counts_instrument

# %%
fig, ax = plt.subplots()
ax = (counts_instrument
        .plot
        .bar(
                ax=ax,
        )
)
ax.set_xlabel('')
ax.set_ylabel('number of samples (runs)')
fname = out_folder / 'number_of_samples_per_instrument.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(fig, fname)

# %% [markdown]
# ## File size and number of identifications

# %%
cols = ['Peptide Sequences Identified', 'size_gb']

mask = ((df_meta[cols[0]] < 20_000) & (df_meta[cols[1]] > 3.5)
        | (df_meta[cols[1]] > 5)
        )

cols = ['Peptide Sequences Identified', 'size_gb']
ax = (df_meta
      .loc[~mask, cols]
      .plot
      .scatter(cols[0], cols[1],
               label='large files',
               s=2,
               )
      )
ax = (df_meta
      .loc[mask, cols]
      .plot
      .scatter(cols[0],  cols[1],
               color='orange',
               label='normal files',
               ylabel='filesize (in GB)',
               ax=ax,
               s=2,
               )
      )
ax.xaxis.set_major_formatter("{x:,.0f}")
fname = out_folder / 'filesize_vs_iden_peptides.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.get_figure(), fname)


view = df_meta.loc[mask].sort_values(by=cols)
view.to_excel(excel_writer, sheet_name='instrument_outliers', **writer_args)
view

# %%
cols = ['Number of MS1 spectra', 'Number of MS2 spectra',
        'Peptide Sequences Identified']
cols = vaep.pandas.get_columns_accessor_from_iterable(cols)

view = df_meta.loc[mask_top10_instruments]
view["instrument_label+N"] = view["instrument_label"].replace(counts_instrument.to_frame().apply( lambda s: f"{s.name} (N={s['count']:03d})" , axis=1))
view

# %%
fig, ax = plt.subplots()

ax = seaborn.scatterplot(view,
                         x=cols.Number_of_MS1_spectra,
                         y=cols.Number_of_MS2_spectra,
                         hue='instrument_label+N',
                         legend='brief',
                         ax=ax,
                         s=5,
                         palette='deep')
_ = ax.legend(fontsize=5,
              title_fontsize=5,
              markerscale=0.4,
              title='instrument label',
              loc='upper right',
              # alignment='left',
)
ax.xaxis.set_major_formatter("{x:,.0f}")
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = out_folder / 'ms1_to_ms2_top10_instruments.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(fig, fname)


# %%
fig, ax = plt.subplots()
ax = view.plot.scatter(x=cols.Peptide_Sequences_Identified,
                       y=cols.Number_of_MS1_spectra,
                       label=cols.Number_of_MS1_spectra,
                       s=2,
                       c='green',
                       ax=ax)
ax = view.plot.scatter(x=cols.Peptide_Sequences_Identified,
                       y=cols.Number_of_MS2_spectra,
                       label=cols.Number_of_MS2_spectra,
                       ylabel='# spectra',
                       s=2,
                       ax=ax)
fname = out_folder / 'ms1_vs_ms2.pdf'
ax.xaxis.set_major_formatter("{x:,.0f}")
ax.yaxis.set_major_formatter("{x:,.0f}")
files_out[fname.name] = fname.as_posix()
vaep.savefig(fig, fname)

# %% [markdown]
# ## run length to number of identified peptides

# %%
df_meta.filter(like='RT', axis=1).describe()

# %%
cols = ['MS max RT',
        'Peptide Sequences Identified']
cols = vaep.pandas.get_columns_accessor_from_iterable(cols)

fig, ax = plt.subplots()

ax = ax = seaborn.scatterplot(
    view,
    x=cols.MS_max_RT,
    y=cols.Peptide_Sequences_Identified,
    hue='instrument_label+N',
    legend='brief',
    ax=ax,
    s=5,
    palette='deep')
_ = ax.legend(fontsize=5,
              title_fontsize=5,
              markerscale=0.4,
              title='instrument label',
              )
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = out_folder / 'RT_vs_identified_peptides_top10_instruments.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ## Outputs

# %%
excel_writer.close()

# %%
files_out
# %%
