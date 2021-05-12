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

msg_return_docstring  =  """

        Returns
        -------
        Y: array-like
            If X is a pandas DataFrame, Y will be a DataFrame with the initial 
            Indix and column Index objects.
"""

StandardScaler.transform.__doc__ = preprocessing.StandardScaler.transform.__doc__ + msg_return_docstring
StandardScaler.inverse_transform.__doc__ = preprocessing.StandardScaler.inverse_transform.__doc__ + msg_return_docstring


def get_df_fitted_mean_std(self, index):
    return pd.DataFrame({'mean': self.mean_, 'stddev': self.scale_}, index=index)