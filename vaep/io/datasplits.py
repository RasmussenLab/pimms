from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd

from vaep.io.format import class_full_module, classname
from vaep.pandas import interpolate

logger = logging.getLogger(__name__)

FILE_FORMAT_TO_DUMP_FCT = {'pkl': ('to_pickle', 'read_pickle'),
                           # 'pickle': 'to_pickle',
                           'csv': ('to_csv', 'read_csv')}


def long_format(df: pd.DataFrame,
                colname_values: str = 'intensity',
                # index_name: str = 'Sample ID'
                ) -> pd.DataFrame:
    # ToDo: Docstring as in class when finalized
    names = df.columns.names
    if None in names:
        raise ValueError(f"Column names must not be None: {names}")
    df_long = df.stack(names).to_frame(colname_values)
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
    is_wide_format: bool = field(init=True, repr=False)
    train_X: pd.DataFrame = None
    val_y: pd.DataFrame = None
    test_y: pd.DataFrame = None

    def __post_init__(self):
        self._items = sorted(self.__dict__)
        self._is_wide = self.is_wide_format
        self._items.remove('is_wide_format')

    def __getitem__(self, index):
        return (self._items[index], getattr(self, self._items[index]))

    def __dir__(self):
        # return self._items
        return ['dump', 'from_folder', 'interpolate', 'load', 'test_X', 'test_y',
                'to_long_format', 'to_wide_format', 'train_X', 'val_X', 'val_y']

    def dump(self, folder='data', file_format='csv') -> dict:
        """dump in long format."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if file_format not in FILE_FORMAT_TO_DUMP_FCT:
            raise ValueError(f"Select one of these formats: {', '.join(FILE_FORMAT_TO_DUMP_FCT.keys())}")
        dumps = {}
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
            fname = folder / f"{_attr}.{file_format}"
            dumps[fname.name] = fname.as_posix()
            logger.info(f"save '{_attr}' to file: {fname}")
            dump_fct = getattr(_df, FILE_FORMAT_TO_DUMP_FCT[file_format][0])
            dump_fct(fname)
            n_dumped += 1
        if not n_dumped:
            raise ValueError(f'Nothing to dump, all None: {self}')
            # _df.to_json(fname) # does not work easily for series
        return dumps

    def load(self, folder: str, use_wide_format=False, file_format='csv') -> None:
        """Load data in place from folder"""
        items = dict(self.__annotations__)
        del items['is_wide_format']
        args = load_items(folder=folder, items=items, use_wide_format=use_wide_format, file_format=file_format)
        for _attr, _df in args.items():
            setattr(self, _attr, _df)
        self._is_wide = use_wide_format
        return None  # could also be self

    @classmethod
    def from_folder(cls, folder: str, use_wide_format=False, file_format='csv') -> DataSplits:
        """Build DataSplits instance from folder."""
        items = dict(cls.__annotations__)
        del items['is_wide_format']
        args = load_items(folder=folder, items=items, use_wide_format=use_wide_format, file_format=file_format)
        _data_splits = cls(**args, is_wide_format=use_wide_format)
        _data_splits._is_wide = use_wide_format
        return _data_splits

    def to_wide_format(self):
        if self._is_wide:
            return

        for _attr, _series in self:
            if _series is None:
                continue
            _df = _series.unstack()
            setattr(self, _attr, _df)
        self._is_wide = True

    def to_long_format(self, name_values: str = 'intensity'):
        if not self._is_wide:
            return

        for _attr, _df in self:
            if _df is None:
                continue
            _series = _df.stack()
            _series.name = name_values
            setattr(self, _attr, _series)
        self._is_wide = False

    # singledispatch possible
    def interpolate(self, dataset: Union[str, pd.DataFrame]):
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
                ds = ds.unstack()  # series is unstack to DataFrame
        else:
            raise TypeError(f"Unknown type: {classname(dataset)}."
                            f" None of str, {class_full_module(pd.DataFrame)}, {class_full_module(pd.Series)}"
                            )

        return interpolate(wide_df=ds)


def load_items(folder: str, items: dict, use_wide_format=False, file_format='csv') -> dict:
    folder = Path(folder)
    assert folder.exists(), f'Could not find folder: {folder}'
    args = {}
    for _attr, _cls in items.items():
        # assert issubclass(_cls, (pd.DataFrame, pd.Series)) # now strings, see
        # https://docs.python.org/3/whatsnew/3.7.html#pep-563-postponed-evaluation-of-annotations
        fname = folder / f"{_attr}.{file_format}"
        if not fname.exists():
            raise FileNotFoundError(f"Missing file requested for attr '{_attr}', missing {fname}")
        read_fct = getattr(pd, FILE_FORMAT_TO_DUMP_FCT[file_format][1])
        _df = read_fct(fname)
        # logic below is suited for csv reader -> maybe split up loading and saving later?
        if len(_df.shape) == 1:
            _df = _df.to_frame().reset_index()  # in case Series was pickled
        cols = list(_df.columns)
        if use_wide_format:
            _df = wide_format(_df.set_index(cols[1:-1]), columns=cols[0], name_values=cols[-1])
        else:
            _df.set_index(cols[:-1], inplace=True)
        logger.info(f"Loaded '{_attr}' from file: {fname}")
        args[_attr] = _df.squeeze()
    return args


# set default file name -> intergrate into DataSplits?
def load_freq(folder: str, file='freq_features.pkl'):
    folder = Path(folder)
    fname = folder / file
    if fname.suffix == '.json':
        freq_per_feature = pd.read_json(fname, orient='index').squeeze()
        freq_per_feature.name = 'freq'
    elif fname.suffix == '.pkl':
        freq_per_feature = pd.read_pickle(fname)
    else:
        raise ValueError(f"Unknown Fileextension: {fname.suffix}")
    return freq_per_feature
