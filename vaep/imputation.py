

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
    pass
    #clip by zero?

