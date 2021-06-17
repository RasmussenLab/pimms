import pandas as pd
import numpy as np
from sklearn import preprocessing


def log(row: pd.Series):
    """Apply log Transformation to values."""
    return np.log(row.where(row != 0.0))


# ## Avoid tedious post-processing:
# ## - numpy array variant
# analysis.corr_linear_vs_log = pd.DataFrame(
#     scaler.transform(X=analysis.df),
#     columns = analysis.df.columns
# ).corrwith(
#     other=pd.DataFrame(
#         scaler_log.transform(X_log10),
#         columns = analysis.df.columns
#     ),
#     axis=0)
# analysis.corr_linear_vs_log.describe()

# Can this be a MixIn class?
class StandardScaler(preprocessing.StandardScaler):
    def transform(self, X, copy=None):
        res = super().transform(X, copy)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(res, columns=X.columns, index=X.index)
        return res

    def inverse_transform(self, X, copy=None):
        res = super().inverse_transform(X, copy)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(res, columns=X.columns, index=X.index)
        return res

msg_return_docstring = """

        Returns
        -------
        Y: array-like
            If X is a pandas DataFrame, Y will be a DataFrame with the initial 
            Indix and column Index objects.
"""

StandardScaler.transform.__doc__ = preprocessing.StandardScaler.transform.__doc__ + \
    msg_return_docstring
StandardScaler.inverse_transform.__doc__ = preprocessing.StandardScaler.inverse_transform.__doc__ + msg_return_docstring

# # this could be a class method

# @make_pandas_compatible
# class MinMaxScaler(preprocessing.MinMaxScaler):
#     pass



class ShiftedStandardScaler(StandardScaler):

    def __init__(self, shift_mu=0.5, scale_var=2.0, **kwargs):
        """Augmented StandardScaler, shift the standard normalized data 
        by mu and scales the variance by a scale factor.

        Parameters
        ----------
        shift_mu : float, optional
            shift mean, by default 0.5
        scale_var : float, optional
            scale variance, by default 2.0
        """
        super().__init__(**kwargs if kwargs else {})
        self.shift_mu, self.scale_var = shift_mu, scale_var

    def transform(self, X, copy=None):
        res = super().transform(X, copy)
        res /= self.scale_var
        res += self.shift_mu
        return res

    def inverse_transform(self, X, copy=None):
        X -= self.shift_mu
        X *= self.scale_var
        res = super().inverse_transform(X, copy)
        return res


def get_df_fitted_mean_std(self, index):
    return pd.DataFrame({'mean': self.mean_, 'stddev': self.scale_}, index=index)
