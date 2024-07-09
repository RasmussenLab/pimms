"""Test scikit-learn transformers provided by PIMMS."""
import numpy as np
import pandas as pd
import pytest

from pimmslearn.sklearn.ae_transformer import AETransformer
from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer

test_data = 'project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50_M227.csv'
index_name = 'Sample ID'
column_name = 'protein group'
value_name = 'intensity'


def test_CollaborativeFilteringTransformer():
    model = CollaborativeFilteringTransformer(
        target_column=value_name,
        sample_column=index_name,
        item_column=column_name,)
    # read data, name index and columns
    df = pd.read_csv(test_data, index_col=0)
    df = np.log2(df + 1)
    df.index.name = index_name  # already set
    df.columns.name = column_name  # not set due to csv disk file format
    series = df.stack()
    series.name = value_name  # ! important
    # run for 2 epochs
    model.fit(series, cuda=False, epochs_max=2)
    df_imputed = model.transform(series).unstack()
    assert df_imputed.isna().sum().sum() == 0


@pytest.mark.parametrize("model", ['DAE', 'VAE'])
def test_AETransformer(model):
    df = pd.read_csv(test_data, index_col=0)
    df = np.log2(df + 1)

    df.index.name = index_name  # already set
    df.columns.name = column_name  # not set due to csv disk file format
    model = AETransformer(
        model=model,
        hidden_layers=[512,],
        latent_dim=50,
        out_folder='runs/scikit_interface',
        batch_size=10,
    )
    model.fit(df,
              cuda=False,
              epochs_max=2,
              )
    df_imputed = model.transform(df)
    assert df_imputed.isna().sum().sum() == 0
