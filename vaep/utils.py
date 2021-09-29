from random import sample
import pathlib


def sample_iterable(iterable: dict, n=10):
    """Sample some keys from a given dictionary."""
    n_examples_ = n if len(iterable) > n else len(iterable)
    sample_ = sample(iterable, n_examples_)
    return sample_


def append_to_filepath(filepath: pathlib.Path,
                       to_append: str,
                       sep: str = '_',
                       new_suffix: str = None):
    """Append filepath with specified to_append using a seperator. 
    
    Example: `data.csv` to data_processed.csv
    """
    suffix = filepath.suffix
    if new_suffix:
        suffix = f".{new_suffix}"
    new_fp = filepath.parent / f'{filepath.stem}{sep}{to_append}{suffix}'
    return new_fp
