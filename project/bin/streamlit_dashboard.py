import os
import streamlit as st
import pandas as pd
import numpy as np
from vaep.utils import load_data
from IPython.display import display


FOLDER = 'data'
FILE = 'Mann_Hepa_data.tsv'
_file = os.path.join(FOLDER, FILE)

st.title('Proteomics Viewer')
st.sidebar.title(
    "Hela cellline data aggregated by MaxQuant. ")
st.sidebar.markdown(    
    "Relative abundance data based on quantification "
    "by sample")

data_load_state = st.text('Loading data...')
normalize = st.checkbox('Normalize rel. abundance',
            value=False, key='cb_normalize')
meta_data, proteins = load_data(_file, normalize=normalize)
NSAMPLES = len(meta_data)
NPROTEINS= proteins.shape[-1]
data_load_state.text(f'Loaded {NSAMPLES} samples.')
info_na = proteins.isna().sum()
info_na_percentiles = info_na.describe(
    percentiles=[x/100 for x in range(0,100,1)]).to_frame(
        name='freq NA'
    ).iloc[4:]

info_notna = proteins.notna().sum()
info_na_percentiles['freq not NA'] = info_notna.describe(
    percentiles=[x/100 for x in range(0,100,1)]).iloc[4:]
info_notna_proportion = info_notna / NSAMPLES

st.subheader('Raw meta data')
st.write(meta_data)
st.subheader('Raw protein data (created by MaxQuant)')
st.write(proteins.head())
st.subheader('Percentiles of NAs per protein')
st.write(info_na_percentiles.transpose())
st.write(info_notna_proportion)
st.subheader('Proteins being present at least in specified share of samples.')
share = st.slider('Proportion of samples sharing a protein.', 
                   min_value=0.0, max_value=1.0,
                   value=0.7,
                   step=0.01)

mask = info_notna_proportion >= share
n_proteins_shared = info_notna_proportion.loc[mask].notna().sum()
st.write(f"Number of proteins being present in at least {share:.2f}: "
         f"{n_proteins_shared} of {NPROTEINS}")