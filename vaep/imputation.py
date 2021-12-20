"""
Reduce number of missing values of DDA massspectromety.

Imputation can be down by column.


"""
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


def _select_data(data: pd.DataFrame, threshold: float):
    """Select (protein-) columns for imputation.

    Based on the threshold representing the minimum proportion of available
    data per protein, the columns of a `pandas.DataFrame` are selected.

    Parameters
    ----------
    data: pandas.DataFrame
    threshold: float
        Threshold of percentage of non-missing values to select a column/feature.
    """
    columns_to_impute = data.notnull().mean() >= threshold
    return columns_to_impute


def _sparse_coo_array(data: pd.DataFrame):
    """Return a sparse scipy matrix from dense `pandas.DataFrame` with many 
    missing values.
    """
    indices = np.nonzero(~np.isnan(data.to_numpy()))
    data_selected_sparse = data.to_numpy()
    data_selected_sparse = scipy.sparse.coo_matrix(
        (data_selected_sparse[indices], indices),
        shape=data_selected_sparse.shape)
    return data_selected_sparse


def _get_weighted_mean(distances, data):
    """Compute weighted mean ignoring
    identical entries"""
    mask = distances > 0.0
    weights = distances[mask] / distances[mask].sum()
    weighted_sum = data.loc[mask].mul(weights, axis=0)
    mean_imputed = weighted_sum.sum() / sum(mask)
    return mean_imputed


# define imputation methods
# could be done in PCA transformed space
def imputation_KNN(data, alone=True, threshold=0.5):
    """
    

    Parameters
    ----------
    data: pandas.DataFrame
    alone: bool  # is not used
    threshold: float
        Threshold of missing data by column in interval (0, 1)
    """
    mask_selected = _select_data(data=data, threshold=threshold)
    data_selected = data.loc[:, mask_selected].copy()
    data_selected_sparse = _sparse_coo_array(data_selected)
    # impute
    knn_fitted = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(
        data_selected_sparse)
    fit_distances, fit_neighbors = knn_fitted.kneighbors(data_selected_sparse)
    for i, (distances, ids) in enumerate(zip(fit_distances, fit_neighbors)):
        mean_imputed = _get_weighted_mean(distances, data_selected.loc[ids])
        if all(distances == 0.0):
            logger.warning(f"Did not find any neighbor for int-id: {i}")
        else:
            assert i == ids[distances == 0.0], (
                "None or more then one identical data points "
                "for ids: {}".format(ids[distances == 0.0])
            )
        mask = data_selected.iloc[i].isna()
        data_selected.loc[i, mask] = mean_imputed.loc[mask] # SettingWithCopyError

    data.update(data_selected)
    return data


def imputation_normal_distribution(log_intensities: pd.Series, mean_shift=1.8, std_shrinkage=0.3):
    """Impute missing log-transformed intensity values of DDA run.

    Parameters
    ----------
    log_intensities: pd.Series
        Series of normally distributed values. Here usually log-transformed
        protein intensities.
    mean_shift: integer, float
        Shift the mean of the log_intensities by factors of their standard
        deviation to the negative.
    std_shrinkage: float
        Value greater than zero by which to shrink (or inflate) the 
        standard deviation of the log_intensities.
    """
    np.random.seed(RANDOMSEED)
    if not isinstance(log_intensities, pd.Series):
        try:
            log_intensities.Series(log_intensities)
            logger.warning("Series created of Iterable.")
        except:
            raise ValueError(
                "Plese provided data which is a pandas.Series or an Iterable")
    if mean_shift < 0:
        raise ValueError(
            "Please specify a positive float as the std.-dev. is non-negative.")
    if std_shrinkage <= 0:
        raise ValueError(
            "Please specify a positive float as shrinkage factor for std.-dev.")
    if std_shrinkage >= 1:
        logger.warning("Standard Deviation will increase for imputed values.")

    mean = log_intensities.mean()
    std = log_intensities.std()

    mean_shifted = mean - (std * mean_shift)
    std_shrinked = std * std_shrinkage

    return log_intensities.where(log_intensities.notna(),
                                 np.random.normal(mean_shifted, std_shrinked))


def imputation_mixed_norm_KNN(data):
    # impute columns with less than 50% missing values with KNN
    data = imputation_KNN(data, alone=False)  # ToDo: Alone is not used.
    # impute remaining columns based on the distribution of the protein
    data = imputation_normal_distribution(
        data, mean_shift=1.8, std_shrinkage=0.3)
    return data
