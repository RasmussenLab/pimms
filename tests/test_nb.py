import pytest
from pimmslearn.nb import Config


def test_Config():
    cfg = Config()
    cfg.test = 'test'
    with pytest.raises(AttributeError):
        cfg.test = 'raise AttributeError'
