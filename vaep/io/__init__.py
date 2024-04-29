import json
import logging
import pickle
from collections import namedtuple
from pathlib import Path, PurePath, PurePosixPath
from typing import Tuple, Union

import numpy as np
import pandas as pd

import vaep.pandas

PathsList = namedtuple('PathsList', ['files', 'folder'])


logger = logging.getLogger(__name__)
logger.info(f"Calling from {__name__}")


def search_files(path='.', query='.txt'):
    """Uses Pathlib to find relative to path files
    with the query text in their file names. Returns
    the path relative to the specified path.

    Parameters
    ----------
    path : str, optional
        Path to search, by default '.'
    query : str, optional
        query string for for filenames, by default '.txt'

    Returns
    -------
    list
        list with files as string containig query key.
    """
    path = Path(path)
    files = []
    for p in path.rglob("*"):
        if query in p.name:
            files.append(str(p.relative_to(path)))
    return PathsList(files=files, folder=path)


def search_subfolders(path='.', depth: int = 1, exclude_root: bool = False):
    """Search subfolders relative to given path."""
    if not isinstance(depth, int) and depth > 0:
        raise ValueError(
            f"Please provide an strictly positive integer, not {depth}")
    EXCLUDED = ["*ipynb_checkpoints*"]

    path = Path(path)
    directories = [path]

    def get_subfolders(path):
        return [x for x in path.iterdir()
                if x.is_dir() and not any(x.match(excl) for excl in EXCLUDED)
                ]

    directories_previous = directories.copy()
    while depth > 0:
        directories_new = list()
        for p in directories_previous:
            directories_new.extend(
                get_subfolders(p))
        directories.extend(directories_new)
        directories_previous = directories_new.copy()
        depth -= 1

    if exclude_root:
        directories.pop(0)
    return directories


def resolve_path(path: Union[str, Path], to: Union[str, Path] = '.') -> Path:
    """Resolve a path partly overlapping with to another path."""
    pwd = Path(to).absolute()
    pwd = [p for p in pwd.parts]
    ret = [p for p in Path(path).parts if p not in pwd]
    return Path('/'.join(ret))


def get_fname_from_keys(keys, folder='.', file_ext='.pkl', remove_duplicates=True):
    if remove_duplicates:
        # https://stackoverflow.com/a/53657523/9684872
        keys = list(dict.fromkeys(keys))
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    fname_dataset = folder / '{}{}'.format(vaep.pandas.replace_with(
        ' '.join(keys), replace='- ', replace_with='_'), file_ext)
    return fname_dataset


def dump_to_csv(df: pd.DataFrame,
                folder: Path,
                outfolder: Path,
                parent_folder_fct=None
                ) -> None:
    fname = f"{folder.stem}.csv"
    if parent_folder_fct is not None:
        outfolder = outfolder / parent_folder_fct(folder)
    outfolder.mkdir(exist_ok=True, parents=True)
    fname = outfolder / fname
    logger.info(f"Dump to file: {fname}")
    df.to_csv(fname)
    return fname


def dump_json(data_dict: dict, filename: Union[str, Path]):
    """Dump dictionary as JSON.

    Parameters
    ----------
    data_dict : dict
        Dictionary with valid JSON entries to dump.
    filename : Union[str, Path]
        Filepath to save dictionary as JSON.
    """
    with open(filename, 'w') as f:
        json.dump(obj=data_dict, fp=f, indent=4)


def to_pickle(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def from_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_json(fname: Union[str, Path]) -> dict:
    """Load JSON from disc.

    Parameters
    ----------
    fname : Union[str, Path]
        Filepath to JSON on disk.

    Returns
    -------
    dict
        Loaded JSON file.
    """
    with open(Path(fname)) as f:
        d = json.load(f)
    return d


def parse_dict(input_dict: dict,
               types: Tuple[Tuple] = ((PurePath, lambda p: str(PurePosixPath(p))),
                                      (np.ndarray, lambda a: a.to_list()))):
    """Transform a set of items (instances) to their string representation"""
    d = dict()
    for k, v in input_dict.items():
        for (old_type, fct) in types:
            if isinstance(v, old_type):
                v = fct(v)
        d[k] = v
    return d


def extend_name(fname: Union[str, Path], extend_by: str, ext: str = None) -> Path:
    """Extend the name of a file.

    Parameters
    ----------
    fname : Union[str, Path]
        Filepath to file to rename.
    extend_by : str
        Extend file stem by string

    Returns
    -------
    Path
        Changed filepath with extension
    """
    fname = Path(fname)
    if ext is None:
        ext = fname.suffix
    fname = fname.parent / f"{fname.stem}{extend_by}"
    fname = fname.with_suffix(ext)
    return fname


def add_indices(array: np.array, original_df: pd.DataFrame,
                index_only: bool = False) -> pd.DataFrame:
    """Add indices to array using provided origional DataFrame.

    Parameters
    ----------
    array : np.array
        Array of data to add indices to.
    original_df : pd.DataFrame
        Original DataFrame data was generated from.
    index_only : bool, optional
        Only add row index, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with array data and original indices.
    """

    index = original_df.index
    columns = None
    if not index_only:
        columns = original_df.columns
    return pd.DataFrame(array, index=index, columns=columns)
