"""
Functionality to handle protein and peptide datasets. 
"""

import pandas as pd

#coverage
def coverage(X:pd.DataFrame, coverage_col:float, coverage_row:float):
    """Select proteins by column depending on their coverage. 
    Of these selected proteins, where the rows have a certain number of overall proteins.
    """
    mask_col = X.isnull().mean() <= 1-coverage_col
    df = X.loc[:,mask_col]
    mask_row = df.isnull().mean(axis=1) <= 1-coverage_row
    df = df.loc[mask_row,:]
    return df