from collections import namedtuple
import json
from pathlib import Path

PathsList = namedtuple('PathsList', ['files', 'folder'])


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
                if x.is_dir() and
                not any(x.match(excl) for excl in EXCLUDED)
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


def dump_json(data_dict:dict, filename):
    """Dump dictionary as json.

    Parameters
    ----------
    data_dict : dict
        [description]
    filename : [type]
        [description]
    """    
    with open(filename, 'w') as f:
        json.dump(obj=data_dict, fp=f)
    