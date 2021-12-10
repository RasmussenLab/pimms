from __future__ import annotations
from dataclasses import dataclass
import logging
from functools import partial
from pathlib import Path

import pandas as pd
from pandas.core.algorithms import isin

from vaep.pandas import interpolate
from vaep.io.format import classname, class_full_module

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
        self._is_wide = None

    def __getitem__(self, index):
        return (self._items[index], getattr(self, self._items[index]))

    def __dir__(self):
        return self._items

    def dump(self, folder='data'):
        """dump in long format."""
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

    def load(self, folder: str, use_wide_format=False) -> None:
        """Load data in place from folder"""
        args = load_items(folder=folder, items=self.__annotations__, use_wide_format=use_wide_format)
        for _attr, _df in args.items():
            setattr(self, _attr, _df)
        self._is_wide = use_wide_format
        return None  # could also be self

    @classmethod
    def from_folder(cls, folder: str, use_wide_format=False) -> DataSplits:
        """Build DataSplits instance from folder."""
        args = load_items(folder=folder, items=cls.__annotations__, use_wide_format=use_wide_format)
        _data_splits = cls(**args)
        _data_splits._is_wide = use_wide_format
        return _data_splits

    def to_wide_format(self):
        if self._is_wide:
            return

        for _attr, _series in self:
            _df = _series.unstack()
            setattr(self, _attr, _df)
        self._is_wide = True
    
    def to_long_format(self):
        if not self._is_wide: 
            return
        
        for _attr, _df in self:
            index_name = _df.columns.name
            _series = _df.stack()
            _series.index.name = index_name
            setattr(self, _attr, _series)
        self._is_wide = False

    # singledispatch possible
    def interpolate(self, dataset:Union[str, pd.DataFrame]):
        if issubclass(type(dataset), pd.DataFrame):
            ds = dataset
        elif issubclass(type(dataset), pd.Series):
            ds = dataset.unstack()
        elif issubclass(type(dataset), str):
            try:
                ds = getattr(self, dataset)
            except AttributeError:
                raise AttributeError(f"Please provide a valid attribute, not '{dataset}'. "
                "Valid attributes are {}".format(', '.join(x for x in self._items)))
            if dataset[-1] in ['y', 'Y']:
                logger.warning(
                    f'Attempting to interpolate target: {dataset} '
                    '(this might make sense, but a warning')
            if ds is None:
                raise ValueError(f'Attribute is None: {dataset!r}.')
            if not self._is_wide:
                ds = ds.unstack() # series is unstack to DataFrame
        else:
            raise TypeError(f"Unknown type: {classname(dataset)}."
            f" None of str, {class_full_module(pd.DataFrame)}, {class_full_module(pd.Series)}"
            )
 
        return interpolate(wide_df=ds)




def load_items(folder: str, items: dict, use_wide_format=False) -> dict:
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
        cols = list(_df.columns)
        if use_wide_format:
            # ToDo: Add warning for case of more than 3 columns
            _df = wide_format(_df.set_index(cols[1]), columns=cols[0], name_values=cols[-1])
        else:
            _df.set_index(cols[:-1], inplace=True)
        logger.info(f"Loaded '{_attr}' from file: {fname}")
        args[_attr] = _df.squeeze()
    return args
