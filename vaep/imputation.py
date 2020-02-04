"""
Reduce number of missing values of DDA massspectromety.

Imputation can be down by column.


"""
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

RANDOMSEED = 123

def impute_missing(protein_values, mean=None, std=None):
    """
    Imputation is based on the mean and standard deviation 
    from the protein_values.
    If mean and standard deviation (std) are given, 
    missing values are imputed and protein_values are returned imputed.
    If no mean and std are given, the mean and std are computed from
    the non-missing protein_values.

    Parameters
    ----------
    protein_values: Iterable
    mean: float
    std: float

    Returns
    ------
    protein_values: pandas.Series
    """
    raise NotImplementedError('Will be the main function combining features')
    #clip by zero?


# define imputation methods
def imputation_KNN(data, alone=True, threshold=0.5):
    """
    

    Parameters
    ----------
    data: pandas.DataFrame
    alone: bool  # is not used
    threshold: float
        Threshold of missing data by column in interval (0, 1)
    """
    columns_selection = data.notnull().mean() > threshold
    missDf = data.loc[:, columns_selection]
    # prepare data type to fit imputation function
    X = np.array(missDf.values, dtype=np.float64)
    # impute
    X_trans = NearestNeighbors(k=3).fit_transform(X)
    # regenerate dataframe
    missingdata_df = missDf.columns.tolist()
    dfm = pd.DataFrame(X_trans, index=list(
        missDf.index), columns=missingdata_df)
    # replace nan values in original dataframe with imputed values
    data.update(dfm)
    return data


def imputation_normal_distribution(data, shift=1.8, nstd=0.3):
    np.random.seed(RANDOMSEED)
    data_imputed = data.transpose().copy()
    for i in data_imputed.loc[:, data_imputed.isnull().any()]:
        missing = data_imputed[i].isnull()
        std = data_imputed[i].std()
        mean = data_imputed[i].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        data_imputed.loc[missing, i] = np.random.normal(
            mu, sigma, size=len(data_imputed[missing]))
    return data_imputed.transpose()


def imputation_mixed_norm_KNN(data):
    # impute columns with less than 50% missing values with KNN
    data = imputation_KNN(data, alone=False) # ToDo: Alone is not used.
    # impute remaining columns based on the distribution of the protein
    data = imputation_normal_distribution(data, shift=1.8, nstd=0.3)
    return data
