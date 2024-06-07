"""Scikit-learn related functions for the project for ALD part.

Might be moved to a separate package in the future.
"""
import logging

from njab.sklearn import run_pca
from sklearn.impute import SimpleImputer

from vaep.io import add_indices

logger = logging.getLogger(__name__)


def get_PCA(df, n_components=2, imputer=SimpleImputer):
    imputer_ = imputer()
    X = imputer_.fit_transform(df)
    X = add_indices(X, df)
    assert all(X.notna())

    PCs, _ = run_pca(X, n_components=n_components)
    return PCs
