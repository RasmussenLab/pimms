from typing import List

import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


import torch

from vaep.io.datasets import to_tensor

import logging
logger = logging.getLogger(__name__)


def log(row: pd.Series):
    """Apply log Transformation to values setting zeros to NaN."""
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

# ? Can this be a MixIn class?
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

# # look at fastcore to see if **kwargs could be replaced with original
# # arguments, see https://fastcore.fast.ai/meta.html#Metaprogramming
# # decorate()

def transform(self, X, **kwargs):
    res = super(self.__class__, self).transform(X, **kwargs)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(res, columns=X.columns, index=X.index)
    return res


def inverse_transform(self, X, **kwargs):
    res = super(self.__class__, self).inverse_transform(X, **kwargs)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(res, columns=X.columns, index=X.index)
    return res
# could become factory function, build args dictionary


def make_pandas_compatible(cls):
    """Patch transform and inverse_transform."""
    _fcts = ['transform', 'inverse_transform']
    for _fct in _fcts:
        if not hasattr(cls, _fct):
            raise ValueError(f"no {_fct} method for {cls.__name__}")
    new_class = type(cls.__name__, (cls,), dict(
        transform=transform, inverse_transform=inverse_transform))

    new_class.transform.__doc__ = cls.transform.__doc__ + msg_return_docstring
    new_class.inverse_transform.__doc__ = cls.inverse_transform.__doc__ + msg_return_docstring
    return new_class


MinMaxScaler = make_pandas_compatible(preprocessing.MinMaxScaler)


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


class VaepPipeline():
    """Custom Pipeline combining a pandas.DataFrame and a sklearn.pipeline.Pipleine."""
    def __init__(self, df_train:pd.DataFrame, encode:sklearn.pipeline.Pipeline,
                        decode:List[str] =None):
        """[summary]

        Parameters
        ----------
        df_train : pd.DataFrame
            pandas.DataFrame to which the data should be fitted.
        encode : sklearn.pipeline.Pipeline, optional
            sklearn.pipeline to fit with df_train, by default None
        decode : List[str], optional
            subset of transforms (their string name) as an Iterable, by default None, i.e.
            the same as encode
        """        
        self.columns = df_train.columns
        self.M = len(df_train.columns)
        self.encode = encode
        self.encode.fit(df_train)
        if decode:
            self.decode = list()
            for d in decode:
                self.decode.append(
                    (d, self.encode.named_steps[d])
                    )

            self.decode = sklearn.pipeline.Pipeline(self.decode)
        else:
            self.decode = self.encode
        
        
    
    def transform(self, X):
        res = self.encode.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(res, columns=X.columns, index=X.index)
        return res
    
    # Option: single-dispatch based on type of X
    def inverse_transform(self, X, index=None):
        columns = self.columns
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        if isinstance(X, pd.Series):
            columns = X.index
            index = [X.name]
            X = X.values
        elif isinstance(X, torch.Tensor):
            X = X.numpy()
        if len(X.shape) == 1:
            logger.warning("Reshape")
            X = X.reshape(-1, self.M)
        res = self.decode.inverse_transform(X)
        res = pd.DataFrame(res, columns=columns, index=index)
        return res