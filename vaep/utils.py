import pandas as pd
import streamlit as st

@st.cache
def load_data(filepath, normalize=False):
    """
    Load data processed by Annelaura.

    Parameters
    ----------
    filepath: filelink (csv)
        Reference to file. 
    normalize: Boolean
        Whether to normalize relative protein abundance data 
        per sample to zero mean and standard deviation of one.
    
    Returns
    -------
    tuple(pandas.Dataframe, pandas.DataFrame)
        Tuple of meta_data and protein data. 
        Protein data the is the normalized relative abundance 
        by sample.
    """
    df = pd.read_csv(filepath, sep='\t', index_col='index')
    meta_data = df.iloc[:,:5]
    proteins = df.iloc[:,5:]
    if normalize:
        _mean = proteins.mean(axis=1)  # emp. mean
        _std  = proteins.std(axis=1, ddof=1) # emp. std. dev
        proteins = proteins.sub(_mean, axis=0) # substract row means
        proteins = proteins.div(_std, axis=0)  # devide by emp. std. dev.
    return meta_data, proteins