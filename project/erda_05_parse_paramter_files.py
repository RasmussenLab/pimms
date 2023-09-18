# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parse parameter files

# %%
from pprint import pprint
import collections
from pathlib import Path
from tqdm.notebook import tqdm

import pandas as pd

# %%
import logging

import xml.etree.ElementTree as ET

logger = logging.getLogger()

test_file = 'data/mqpar_example.xml'


# %%
def extend_tuple(t, target_length: int):
    if not isinstance(t, tuple):
        raise TypeError(
            f"Wrong type provided. Expected tuple, got {type(t)} : {t!r}")
    if len(t) > target_length:
        raise ValueError(
            f"Tuple is too long (got {len(t)}, expected {target_length}: {t!r}")
    return t + (None,) * (target_length - len(t))
# extend_tuple("test", 4)
# extend_tuple(('k1', 'k2'), 1)


# %%
def extend_tuples_with_none(list_of_tuples, target_length):
    extended_tuples = []
    for tuple_ in list_of_tuples:
        # if len(tuple_) > target_length:
        #     raise ValueError(f"tuple is too long: {len(tuple_)}")
        extended_tuple = extend_tuple(tuple_, target_length)
        extended_tuples.append(extended_tuple)
    return extended_tuples


list_of_tuples = [(1, 2), (3, 4, 5), (6,)]
extend_tuples_with_none(list_of_tuples, 3)

# %%


def add_record(data, tag, record):
    if tag in data:
        if isinstance(data[tag], list):
            data[tag].append(record)
        else:
            data[tag] = [data[tag], record]
    else:
        data[tag] = record
    return data


def read_xml_record(element):
    data = dict()
    for child in element:
        if len(child) > 1 and child.tag:
            # if there is a list, process each element one by one
            # either nested or a plain text
            data[child.tag] = [add_record(dict(), tag=child.tag, record=read_xml_record(child) if not (
                child.text and child.text.strip()) else child.text.strip()) for child in child]
        elif child.text and child.text.strip():
            # just plain text record
            data = add_record(data=data, tag=child.tag,
                              record=child.text.strip())
        else:
            record = read_xml_record(child)
            data = add_record(data, child.tag, record)
    if not data:
        # empty strings and None are normalzied to None
        return None
    return data


tree = ET.parse(test_file)
root = tree.getroot()

record_example = read_xml_record(root)
record_example

# %%


def flatten_dict_of_dicts(d: dict, parent_key: str = '') -> dict:
    """Build tuples for nested dictionaries for use as `pandas.MultiIndex`.

    Parameters
    ----------
    d : dict
        Nested dictionary for which all keys are flattened to tuples.
    parent_key : str, optional
        Outer key (used for recursion), by default ''

    Returns
    -------
    dict
        Flattend dictionary with tuple keys: {(outer_key, ..., inner_key) : value}
    """
    # simplified and adapted from: https://stackoverflow.com/a/6027615/9684872
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,) if parent_key else (k,)
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict_of_dicts(v, parent_key=new_key))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, collections.abc.MutableMapping):
                    items.extend(flatten_dict_of_dicts(
                        item, parent_key=new_key))
                elif isinstance(item, str):
                    items.append((new_key, item))
                else:
                    raise ValueError(f"Unknown item: {item:r}")
        else:
            items.append((new_key, v))
    return items


case_1 = {'k': 'v'}
case_2 = {'k1': {'k2': 'v1', 'k3': 'v2'}}
case_3 = {'k1': {'k2': [{'k4': 'v1'}, {'k4': 'v2'}]}}
case_4 = {'k1': [{'k2': {'k4': 'v1', 'k5': 'v2'}},
                 {'k2': {'k4': 'v1', 'k5': 'v2'}}]}
case_5 = {'restrictMods': [{'string': 'Oxidation (M)'},
                           {'string': 'Acetyl (Protein N-term)'}]}
case_6 = {'variableModifications': {
    'string': ['Oxidation (M)',
               'Acetyl (Protein N-term)']}}

test_cases = [case_1, case_2, case_3, case_4, case_5, case_6]

for case in (test_cases):
    pprint(flatten_dict_of_dicts(case))

# %%
entries = list()
for case in test_cases:
    entries.extend(flatten_dict_of_dicts(case))
[(extend_tuple(k, 4), v) for (k, v) in entries]


# %%
def build_Series_from_records(records, index_length=4):
    records = flatten_dict_of_dicts(records)
    idx = pd.MultiIndex.from_tuples(
        (extend_tuple(k, index_length) for (k, v) in records))
    return pd.Series((v for (k, v) in records), index=idx)


tree = ET.parse(test_file)
root = tree.getroot()

record_example = read_xml_record(root)
flattend = build_Series_from_records(record_example, 4)
flattend.to_frame('example')

# %% [markdown]
# ## Parameters

# %%
# folders to check
folder_w_params = Path('/home/jovyan/work/mqpar_files')
root = Path('/home/jovyan/work/')
dumped_folder = 'mq_out'
dumped_folder_names = 'mq_out_folder.txt'
# out
fname_out = 'data/all_parameter_files.csv'


# %% [markdown]
# ## Dump of some parameter files

# %%
def read_file(file, name, idx_levels=4) -> pd.Series:
    tree = ET.parse(file)
    root = tree.getroot()
    record = read_xml_record(root)
    s = build_Series_from_records(record, idx_levels)
    s.name = name
    return s


# %%
parameter_files_part_1 = list()
for file in tqdm(folder_w_params.iterdir()):
    s_parameters = read_file(file, name=file.stem[6:])
    parameter_files_part_1.append(s_parameters)

parameter_files_part_1 = pd.concat(parameter_files_part_1, axis=1).T
parameter_files_part_1

# %% [markdown]
# ## Search for parameter files in output folders
#
# - read folders from dump (for stable execution on erda)

# %%
# # read as generator if file does not exist:
# folders = list(Path('/home/jovyan/work/mq_out').iterdir())

root = Path('/home/jovyan/work/')
with open(root / dumped_folder_names) as f:
    folders = list()
    for line in f:
        fpath = root / dumped_folder / line.strip()
        folders.append(fpath)

# %% [markdown]
# read paramter files:

# %%
parameter_files_part_2 = list()
i = 0
for folder in tqdm(folders):
    for file in folder.iterdir():
        if file.suffix == '.xml':
            s_parameters = read_file(file, file.parent.name)
            parameter_files_part_2.append(s_parameters)
            i += 1

parameter_files_part_2 = pd.concat(parameter_files_part_2, axis=1).T
parameter_files_part_2

# %%
print(f"Found {i} parameter files")

# %% [markdown]
# ## Combine both sets

# %%
parameter_files = pd.concat([parameter_files_part_1, parameter_files_part_2])
# del parameter_files_part_1, parameter_files_part_2
parameter_files

# %%
# 11066

# %%
parameter_files = parameter_files.infer_objects()
parameter_files.dtypes.value_counts()

# %%
parameter_files.to_csv(fname_out)

# %% [markdown]
# Read aggregated parameters dump

# %%
parameter_files = pd.read_csv(fname_out, index_col=0, header=list(range(4)))
parameter_files

# %%
parameter_files.dtypes.value_counts()

# %%
parameter_files.loc[:, parameter_files.dtypes == 'object']

# %%
parameter_files['fastaFiles']

# %%
parameter_files.droplevel(-1, axis=1)['fastaFiles']

# %%
parameter_files.columns.to_list()

# %%
