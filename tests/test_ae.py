import vaep
from vaep.models import ae


expected_repr = """Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=100, out_features=30, bias=True)
    (1): Dropout(p=0.2, inplace=False)
    (2): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): LeakyReLU(negative_slope=0.1)
    (4): Linear(in_features=30, out_features=10, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=10, out_features=30, bias=True)
    (1): Dropout(p=0.2, inplace=False)
    (2): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): LeakyReLU(negative_slope=0.1)
    (4): Linear(in_features=30, out_features=100, bias=True)
  )
)"""


def test_basic_repr():
    model = ae.Autoencoder(n_features=100, n_neurons=30)
    actual_repr = repr(model)
    assert actual_repr == expected_repr
    assert model.dim_latent == 10
    assert model.n_neurons == [30]
    assert model.n_features == 100
