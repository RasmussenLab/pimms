"""
Reduce number of missing values of DDA massspectromety.

Imputation can be down by column.


"""
import scipy
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger()

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsTransformer

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
    columns_to_impute = data.notnull().mean() >= threshold
    data_selected = data.loc[:, columns_to_impute].copy()
    
    indices = np.nonzero(~np.isnan(data_selected.to_numpy()))
    data_selected_sparse = data_selected.to_numpy()
    data_selected_sparse = scipy.sparse.coo_matrix(
                    (data_selected_sparse[indices], indices), 
                    shape=data_selected_sparse.shape)
     
    # impute
    knn_fitted = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(
                                   data_selected_sparse)
    fit_distances, fit_neighbors = knn_fitted.kneighbors(data_selected_sparse)
   
    for i , (weights, ids) in enumerate(zip(fit_distances, fit_neighbors)):
    
        def get_weighted_mean(weights, data):
            """Compute weighted mean ignoring
            identical entries"""
            weighted_sum = data.mul(weights, axis=0)
            mask = weighted_sum.sum(axis=1) > 0.0
            mean_imputed = weighted_sum.loc[mask].sum() / sum(mask)
            return mean_imputed
        mean_imputed = get_weighted_mean(weights, data_selected.loc[ids])
        if all(weights == 0.0):
            logger.warning(f"Did not find any neighbor for int-id: {i}")
        else:
            assert i == ids[weights == 0.0], (   
            "None or more then one identical data points "
            "for ids: {}".format(ids[weights == 0.0])
            )
        mask = data_selected.iloc[i].isna()
        data_selected.loc[i, mask] = mean_imputed.loc[mask]

    data.update(data_selected)
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
