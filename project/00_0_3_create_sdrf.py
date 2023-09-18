# %% [markdown]
# # Create SDRF file
# - [example](https://github.com/bigbio/proteomics-sample-metadata/blob/6f31044f0bcf545ae2da6e853f8ccad011ea4703/annotated-projects/PXD000895/PXD000895.sdrf.tsv)

# %%
from pathlib import Path
import pandas as pd


# %%
fn_sdrf_cellline_template = Path('data') / 'sdrf-cell-line-template.tsv'
fn_meta = Path('data/rename') / 'selected_old_new_id_mapping.csv'


# %%
df_meta = pd.read_csv(fn_meta, index_col='new_sample_id')
df_meta

# %%
sdrf = pd.DataFrame()  # pd.read_table(fn_sdrf_cellline_template)
sdrf['source name'] = df_meta.index
sdrf = sdrf.set_index('source name')
sdrf['characteristics[organism]'] = 'Homo sapiens'
sdrf['characteristics[organism part]'] = 'cervex'
sdrf['characteristics[ancestry category]'] = 'Black'
sdrf['characteristics[age]'] = '31Y'
sdrf['characteristics[developmental stage]'] = 'adult'
sdrf['characteristics[sex]'] = 'female'
sdrf['characteristics[cell line]'] = 'HeLa cells'
sdrf['characteristics[cell type]'] = 'epithelial'
sdrf['characteristics[disease]'] = 'adenocarcinoma'
sdrf['characteristics[cell line]'] = 'HeLa cells'
sdrf['characteristics[biological replicate]'] = 1
sdrf['assay name'] = sdrf.index
sdrf['technology type'] = 'proteomic profiling by mass spectrometer'
sdrf['comment[technical replicate]'] = range(0, len(sdrf))
sdrf['comment[data file]'] = sdrf.index + '.raw'
sdrf['comment[fraction identifier]'] = 1
sdrf['comment[label]'] = 'NT=label free sample;AC=MS:1002038'  # To check
sdrf['comment[cleavage agent details]'] = 'NT=Trypsin;AC=MS:1001251'
sdrf['comment[instrument]'] = df_meta['Instrument_name']

sdrf
# %%
# based on https://www.ebi.ac.uk/ols4/
#
# Q Exactive HF-X MS:1002877
# Q Exactive HF MS:1002523
# Orbitrap Exploris 480 MS:1003028
# Exactive Plus MS:1002526
# Q Exactive MS:1001911
# Orbitrap Fusion Lumos MS:1002732


instrument_ms_mapping = {
    'Q-Exactive-HF-X-Orbitrap_6070': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6071': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6075': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6101': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-Orbitrap_207': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-HF-X-Orbitrap_6096': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6078': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-Orbitrap_147': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-Orbitrap_1': 'NT=Q Exactive;AC=MS:1001911',
    'Q-Exactive-HF-Orbitrap_143': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-HF-Orbitrap_204': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-HF-X-Orbitrap_6011': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-Orbitrap_206': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-HF-X-Orbitrap_6073': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-Orbitrap_1': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-HF-Orbitrap_148': 'NT=Q Exactive HF;AC=MS:1002523',
    'Orbitrap-Fusion-Lumos_FSN20115': 'NT=Orbitrap Fusion Lumos;AC=MS:1002732',
    'Q-Exactive-HF-X-Orbitrap_6016': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6004': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Orbitrap-Exploris-480_MA10132C': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
    'Q-Exactive-HF-X-Orbitrap_6028': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6044': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6025': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6324': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Orbitrap-Exploris-480_MA10134C': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
    'Q-Exactive-HF-X-Orbitrap_6022': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6043': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6013': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Q-Exactive-HF-X-Orbitrap_6023': 'NT=Q Exactive HF-X;AC=MS:1002877:',
    'Exactive-Series-Orbitrap_6004': 'NT=Q Exactive;AC=MS:1001911',
    'Orbitrap-Exploris-480_Invalid_SN_0001': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
    'Orbitrap-Exploris-480_MA10215C': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
    'Q-Exactive-HF-Orbitrap_2612': 'NT=Q Exactive HF;AC=MS:1002523',
    'Q-Exactive-Plus-Orbitrap_1': 'NT=Exactive Plus;AC=MS:1002526',
    'Q-Exactive-Plus-Orbitrap_143': 'NT=Exactive Plus;AC=MS:1002526',
    'Orbitrap-Exploris-480_MA10130C': 'NT=Orbitrap Exploris 480;AC=MS:1003028',
}
sdrf['comment[instrument]'] = sdrf['comment[instrument]'].replace(
    instrument_ms_mapping)

# %%
# change order: The column `technology type`` cannot be before the `assay name`` -- ERROR
# template has wrong order (open PR)
# -> done now above
# order = ['characteristics[organism]',
#          'characteristics[organism part]',
#          'characteristics[ancestry category]',
#          'characteristics[cell type]',
#          'characteristics[disease]',
#          'characteristics[cell line]',
#          'characteristics[biological replicate]',
#          'assay name',
#          'technology type',
#          'comment[technical replicate]',
#          'comment[data file]',
#          'comment[fraction identifier]',
#          'comment[label]',
#          'comment[cleavage agent details]',
#          'comment[instrument]']

# sdrf = sdrf[order]

# %%
fname = Path('data') / 'dev_datasets' / 'Experimental-Design.sdrf.tsv'
sdrf.to_csv(fname, sep='\t')
fname
# %% [markdown]
# ## Validate SDRF file
# ```
# pip install sdrf-pipelines
# parse_sdrf validate-sdrf --sdrf_file project\data\dev_datasets\sdrf.tsv
# ```
