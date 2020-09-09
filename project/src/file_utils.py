import os
from pathlib import Path
import logging

logger = logging.getLogger('root')

from tqdm.notebook import trange, tqdm
import pandas as pd
from pandas.errors import EmptyDataError
import ipywidgets as widgets

def check_for_key(iterable, key):
    hits = [x for x in iterable if key in x]
    n_hits = len(hits)
    if n_hits == 1:
        return hits[0]
    elif n_hits == 0:
        return 0
    elif n_hits > 1:
        return '_'.join(iterable)
    
from collections import namedtuple
PathsList = namedtuple('PathsList', ['files', 'folder'])
    
def search_files(path='.', query='.txt'):
    path = Path(path)
    files= []
    for p in path.rglob("*"):
         if query in p.name:
            files.append(str(p.relative_to(path)))
    return PathsList(files=files, folder=path)

def search_subfolders(path='.', depth : int=1):
    """Search subfolders relative to given path."""
    if not isinstance(depth, int) and depth>0:
        raise ValueError(f"Please provide an strictly positive integer, not {depth}")
        
    path = Path(path)
    directories = [path]
    def get_subfolders(path):
        return [x for x in path.iterdir() if x.is_dir()]
    
    directories_previous = directories.copy()
    while depth > 0:
        directories_new = list()
        for p in directories_previous:
            directories_new.extend(
                get_subfolders(p))
        directories.extend(directories_new)
        directories_previous = directories_new.copy()
        depth -= 1
    return directories

#can file-loading be made concurrent?
#check tf.data
def process_files(handler_fct, filepaths, key=None, relative_to=None):
    """Process a list of filepaths using a `handler_fct`.
    `Handler_fct`s have to return a `pandas.DataFrame`.
    
    handle_fct: function
        Function returning a DataFrame for a filepath from `filepaths`
    key_lookup: function, dict
    filepaths: Iterable
        List, tuple, etc. containing filepath to iteratore over.
    """
    names = []
    failed = []
    for i, _file in enumerate(tqdm(filepaths, position=0, leave=True)):
        if relative_to:
            _file = os.path.join(relative_to, _file)
            logger.debug(f"New File path: {_file}")
        if i == 0:
            df = handler_fct(_file) # throws an error if the first file cannot be read-in 
            if key:
                names.append(check_for_key(iterable=_file.split(os.sep), key=key))
            else:
                names.append(os.path.basename(os.path.dirname(_file)))
        else:
            try:
                df = df.join(handler_fct(filepath=_file), how='outer', rsuffix=i)
                if key:
                    names.append(check_for_key(iterable=_file.split(os.sep), key=key))
                else:
                    names.append(os.path.basename(os.path.dirname(_file)))
            except EmptyDataError:
                logger.warning('\nEmpty DataFrame: {}'.format(_file))
                failed.append(_file)
    return df, names, failed


def load_summary(filepath:str='summary.txt')-> pd.DataFrame:
    """Load MaxQuant summary.txt file"""
    df = pd.read_table(str(filepath))
    df = df.T
    df = df.iloc[:,:-1]
    return df

import xmltodict
def load_mqpar_xml(filepath):
    """Load mqpar.xml file. Stores parameters of a MaxQuant run, including version numbers."""
    with open(filepath) as f:
        _ = f.readline()
        xml = f.read()
        return pd.DataFrame(xmltodict.parse(xml))

from numpy import dtype
types_peptides = {'N-term cleavage window': dtype('O'),
 'C-term cleavage window': dtype('O'),
 'Amino acid before': dtype('O'),
 'First amino acid': dtype('O'),
 'Second amino acid': dtype('O'),
 'Second last amino acid': dtype('O'),
 'Last amino acid': dtype('O'),
 'Amino acid after': dtype('O'),
 'A Count': dtype('int64'),
 'R Count': dtype('int64'),
 'N Count': dtype('int64'),
 'D Count': dtype('int64'),
 'C Count': dtype('int64'),
 'Q Count': dtype('int64'),
 'E Count': dtype('int64'),
 'G Count': dtype('int64'),
 'H Count': dtype('int64'),
 'I Count': dtype('int64'),
 'L Count': dtype('int64'),
 'K Count': dtype('int64'),
 'M Count': dtype('int64'),
 'F Count': dtype('int64'),
 'P Count': dtype('int64'),
 'S Count': dtype('int64'),
 'T Count': dtype('int64'),
 'W Count': dtype('int64'),
 'Y Count': dtype('int64'),
 'V Count': dtype('int64'),
 'U Count': dtype('int64'),
 'O Count': dtype('int64'),
 'Length': dtype('int64'),
 'Missed cleavages': dtype('int64'),
 'Mass': dtype('float64'),
 'Proteins': dtype('O'),
 'Leading razor protein': dtype('O'),
 'Start position': dtype('float64'),
 'End position': dtype('float64'),
 'Gene names': dtype('O'),
 'Protein names': dtype('O'),
 'Unique (Groups)': dtype('O'),
 'Unique (Proteins)': dtype('O'),
 'Charges': dtype('O'),
 'PEP': dtype('float64'),
 'Score': dtype('float64'),
 'Intensity': dtype('int64'),
 'Reverse': dtype('O'),
 'Potential contaminant': dtype('O'),
 'id': dtype('int64'),
 'Protein group IDs': dtype('O'),
 'Mod. peptide IDs': dtype('O'),
 'Evidence IDs': dtype('O'),
 'MS/MS IDs': dtype('O'),
 'Best MS/MS': dtype('float64'),
 'Oxidation (M) site IDs': dtype('O'),
 'MS/MS Count': dtype('int64')}

def load_peptide_intensities(filepath):
    """Load Intensities from `peptides.txt`."""
    df = pd.read_table(filepath, index_col='Sequence', dtype=types_peptides)
    return df[['Intensity']]

dtypes_proteins = {'Protein IDs': dtype('O'),
 'Majority protein IDs': dtype('O'),
 'Peptide counts (all)': dtype('O'),
 'Peptide counts (razor+unique)': dtype('O'),
 'Peptide counts (unique)': dtype('O'),
 'Protein names': dtype('O'),
 'Gene names': dtype('O'),
 'Fasta headers': dtype('O'),
 'Number of proteins': dtype('int64'),
 'Peptides': dtype('int64'),
 'Razor + unique peptides': dtype('int64'),
 'Unique peptides': dtype('int64'),
 'Sequence coverage [%]': dtype('float64'),
 'Unique + razor sequence coverage [%]': dtype('float64'),
 'Unique sequence coverage [%]': dtype('float64'),
 'Mol. weight [kDa]': dtype('float64'),
 'Sequence length': dtype('int64'),
 'Sequence lengths': dtype('O'),
 'Q-value': dtype('float64'),
 'Score': dtype('float64'),
 'Intensity': dtype('int64'),
 'MS/MS count': dtype('int64'),
 'Only identified by site': dtype('O'),
 'Reverse': dtype('O'),
 'Potential contaminant': dtype('O'),
 'id': dtype('int64'),
 'Peptide IDs': dtype('O'),
 'Peptide is razor': dtype('O'),
 'Mod. peptide IDs': dtype('O'),
 'Evidence IDs': dtype('O'),
 'MS/MS IDs': dtype('O'),
 'Best MS/MS': dtype('O'),
 'Oxidation (M) site IDs': dtype('O'),
 'Oxidation (M) site positions': dtype('O'),
 'Taxonomy IDs': dtype('O')}
    
def load_protein_intensities(filepath):
    """Load Intensities from `proteins.txt`."""
    df = pd.read_table(filepath, index_col='Majority protein IDs', dtype=dtypes_proteins)
    return df[['Intensity']]
