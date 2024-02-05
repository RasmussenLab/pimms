import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from vaep.transform import VaepPipeline
from vaep.io.dataloaders import get_dls
from vaep.utils import create_random_df


def test_get_dls():
    N, M = 23, 11
    X_train = create_random_df(N, M)
    N_valid = int(N * 0.3)
    X_valid = create_random_df(
        N_valid, M, prop_na=.1, start_idx=len(X_train))

    dae_default_pipeline = sklearn.pipeline.Pipeline(
        [('normalize', StandardScaler()),
         ('impute', SimpleImputer(add_indicator=False))])
    transforms = VaepPipeline(df_train=X_train,
                              encode=dae_default_pipeline,
                              decode=['normalize'])
    BS = 4
    dls = get_dls(train_X=X_train, valid_X=X_valid, transformer=transforms, bs=BS)
    assert len(dls.train_ds) == N
    assert len(dls.valid_ds) == N
    batch = dls.one_batch()
    assert batch[0].shape == (BS, M)
