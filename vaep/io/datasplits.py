from __future__ import annotations
from dataclasses import dataclass
import logging
from functools import partial
from pathlib import Path

import pandas as pd

logger = logging.getLogger('vaep')


def long_format(df: pd.DataFrame,
                colname_values: str = 'intensity',
                # index_name: str = 'Sample ID'
                ) -> pd.DataFrame:
    # ToDo: Docstring as in class when finalized
    df_long = df.stack().to_frame(colname_values)
    return df_long


to_long_format = partial(
    long_format, colname_values="intensity"  # , index_name="Sample ID"
)


def wide_format(df: pd.DataFrame,
                columns: str = 'Sample ID',
                name_values: str = 'intensity') -> pd.DataFrame:
    # ToDo: Docstring as in class when finalized
    df_wide = df.pivot(columns=columns, values=name_values)
    df_wide = df_wide.T
    return df_wide


@dataclass
class DataSplits():
    train_X: pd.DataFrame = None
    val_X: pd.DataFrame = None
    val_y: pd.DataFrame = None
    test_X: pd.DataFrame = None
    test_y: pd.DataFrame = None

    def __post_init__(self):
        self._items = sorted(self.__dict__)

    def __getitem__(self, index):
        return (self._items[index], getattr(self, self._items[index]))

    def __dir__(self):
        return self._items

    def dump(self, folder='data'):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        n_dumped = 0
        for (_attr, _df) in self:
            if _df is None:
                logger.info(f"Missing attribute: {_attr}")
                continue

            _dim = _df.shape
            if len(_dim) == 1:
                logger.info(f"'{_attr}' has shape: {_df.shape}")
            elif len(_dim) == 2:
                logger.info(f"'{_attr}' had old shape: {_df.shape}")
                _df = to_long_format(_df)
                logger.info(f"'{_attr}' has new shape: {_df.shape}")
            else:
                raise ValueError()
            fname = folder / _attr
            logger.info(f"save '{_attr}' to file: {fname}")
            _df.to_csv(fname)
            n_dumped += 1
        if not n_dumped:
            raise ValueError(f'Nothing to dump, all None: {self}')
            # _df.to_json(fname) # does not work easily for series

    def load(self, folder: str) -> None:
        args = load_items(folder=folder, items=self.__annotations__)
        for _attr, _df in args.items():
            setattr(self, _attr, _df)
        return None  # could also be self

    @classmethod
    def from_folder(cls, folder: str) -> DataSplits:
        args = load_items(folder=folder, items=cls.__annotations__)
        return cls(**args)


def load_items(folder: str, items: dict) -> dict:
    folder = Path(folder)
    assert folder.exists(), 'Could not find folder: {folder}'
    args = {}
    for _attr, _cls in items.items():
        # assert issubclass(_cls, (pd.DataFrame, pd.Series)) # now strings, see
        # https://docs.python.org/3/whatsnew/3.7.html#pep-563-postponed-evaluation-of-annotations
        fname = folder / _attr
        if not fname.exists():
            raise FileNotFoundError(f"Missing file requested for attr '{_attr}', missing {fname}")
        _df = pd.read_csv(fname)
        _df.set_index(list(_df.columns[:-1]), inplace=True)
        logger.info(f"Loaded '{_attr}' from file: {fname}")
        args[_attr] = _df.squeeze()
    return args
