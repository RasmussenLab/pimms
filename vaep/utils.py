from random import sample


def sample_iterable(iterable: dict, n=10):
    """Sample some keys from a given dictionary."""
    n_examples_ = n if len(iterable) > n else len(iterable)
    sample_ = sample(iterable, n_examples_)
    return sample_
