import vaep
from vaep.models import ae


expected_repr = """Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=100, out_features=30, bias=True)
    (1): Tanh()
    (2): Linear(in_features=30, out_features=10, bias=True)
    (3): Tanh()
  )
  (decoder): Sequential(
    (0): Linear(in_features=10, out_features=30, bias=True)
    (1): Tanh()
    (2): Linear(in_features=30, out_features=100, bias=True)
  )
)"""

def test_basic_repr():
    model = ae.Autoencoder(n_features=100, n_neurons=30)
    actual_repr = repr(model)
    assert actual_repr == expected_repr
    assert model.dim_latent == 10
    assert model.n_neurons == [30]
    assert model.n_features == 100


def test_get_funnel_layers():
    actual = ae.get_funnel_layers(900, 100, 3)
    expected = [700, 500, 300]
    assert actual == expected

