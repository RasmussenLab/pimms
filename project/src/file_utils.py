import os
from collections import namedtuple
from pathlib import Path
import logging

from tqdm import tqdm

import pandas as pd
from pandas.errors import EmptyDataError

import xmltodict
from numpy import dtype

logger = logging.getLogger('src.file_utils.py')


MQ_VERSION = '1.6.12.0'


def check_for_key(iterable, key):
    """Check for key in items of Iterable
    using `in`(`__contains__`).

    Parameters
    ----------
    iterable : Iterable of Strings
        Iterable of items which the key can be checked for
    key : String
        key to check for using `key in item` of Iterable.

    Returns
    -------
    string, int
        Returns zero if nothing is found, otherwise a string.
        If only one item is found containing the key, return this.
        Multiple hits are returned connacotaed using an underscore.
    """
    hits = [x for x in iterable if key in x]
    n_hits = len(hits)
    if n_hits == 1:
        return hits[0]
    elif n_hits == 0:
        return 0
    elif n_hits > 1:
        return '_'.join(iterable)


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

# can file-loading be made concurrent?
# check tf.data


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
            # throws an error if the first file cannot be read-in
            df = handler_fct(_file)
            if key:
                names.append(check_for_key(
                    iterable=_file.split(os.sep), key=key))
            else:
                names.append(os.path.basename(os.path.dirname(_file)))
        else:
            try:
                df = df.join(handler_fct(filepath=_file),
                             how='outer', rsuffix=i)
                if key:
                    names.append(check_for_key(
                        iterable=_file.split(os.sep), key=key))
                else:
                    names.append(os.path.basename(os.path.dirname(_file)))
            except EmptyDataError:
                logger.warning('\nEmpty DataFrame: {}'.format(_file))
                failed.append(_file)
    return df, names, failed


def load_summary(filepath: str = 'summary.txt') -> pd.DataFrame:
    f"""Load MaxQuant {MQ_VERSION} summary.txt file.

    Parameters
    ----------
    filepath : str, optional
        filepath, by default 'summary.txt'

    Returns
    -------
    pd.DataFrame
        Text-File is returned as pandas.DataFrame 
    """
    df = pd.read_table(filepath)
    df = df.T
    df = df.iloc[:, :-1]
    return df


def load_mqpar_xml(filepath):
    f"""Load MaxQuant {MQ_VERSION}parameter file in xml format which stores parameters for MaxQuant run,
    including version numbers.

    Parameters
    ----------
    filepath : str, optional
        filepath to xml- parameter file

    Returns
    -------
    pd.DataFrame
        XML-File is returned as pandas.DataFrame     
    """
    with open(filepath) as f:
        _ = f.readline()
        xml = f.read()
        return pd.DataFrame(xmltodict.parse(xml))


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
    f"""Load Intensities from `peptides.txt`.
    Data types of columns as of in MaxQuant {MQ_VERSION}

    Parameters
    ----------
    filepath : str
        filepath (rel or absolute) to MQ peptides.txt

    Returns
    -------
    pandas.DataFrame
        Return text file as DataFrame.
    """
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
    f"""Load Intensities from `proteins.txt`.
    Data types of columns as of in MaxQuant {MQ_VERSION}

    Parameters
    ----------
    filepath : str
        filepath (rel or absolute) to MQ proteins.txt

    Returns
    -------
    pandas.DataFrame
        Return text file as DataFrame.
    """
    df = pd.read_table(
        filepath, index_col='Majority protein IDs', dtype=dtypes_proteins)
    return df[['Intensity']]




class MaxQuantOutput:
    """Class assisting with MaxQuant txt output folder.

    Parameters
    ----------
    folder: pathlib.Path, str
        Path to Maxquant `txt` output folder.

   
    Attributes
    ----------
    self.files : list 
        list of files in `folder`.
    _inital_attritubutes : list
        Initial set of non-magic attributes 
    NAME_FILE_MAP : dict
        Keys for known MaxQuant output files.
    """
    NAME_FILE_MAP = {'allPeptides': 'allPeptides.txt',
             'evidence': 'evidence.txt',
             'matchedFeatures': 'matchedFeatures.txt',
             'modificationSpecificPeptides': 'modificationSpecificPeptides.txt',
             'ms3Scans': 'ms3Scans.txt',
             'msms': 'msms.txt',
             'msmsScans': 'msmsScans.txt',
             'mzRange': 'mzRange.txt',
             'OxidationSites': 'Oxidation (M)Sites.txt',
             'parameters': 'parameters.txt',
             'peptides': 'peptides.txt',
             'proteinGroups': 'proteinGroups.txt',
             'summary': 'summary.txt'}
   
    def __init__(self, folder):
        self.folder = Path(folder)
        self.files = self.get_files()
        
    
    def get_files(self):
        """Get all txt files in output folder
        
        Attributes
        ---------
        paths: NamedTuple
        """
        self.paths = search_files(path=self.folder, query='.txt')
        return self.paths.files
    
    @classmethod
    def register_file(cls, filename):
        
        @property
        def fct(cls):
            return cls.find_attribute(f'_{filename}')
        
        return fct
    
    def find_attribute(self, filename):
        """Look up or load attribute."""
        if not hasattr(self, filename):
            df = self.load(filename[1:])
            setattr(self, filename, df)
        return getattr(self, filename)
        
    def load(self, file):
        """Load a specified file into memory and return it.
        Can be used """
        filepath = self.folder / self.NAME_FILE_MAP[file]
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No such file: {file}.txt: Choose one of the following {', '.join(self.files)}")
        
        return pd.read_table(filepath, index_col=0) 
    
    # needed to reset attributes on instance creation.
    _inital_attritubutes = [x for x in dir() if not x.startswith('__')]

    def get_list_of_attributes(self):
        """Return current list on non-magic instance attributes."""
        return [x for x in dir(self) if not x.startswith('__')]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.folder!r})'

# register all properties
# Would be great to be able to do this at runtime based on the files actually present.
for filename in MaxQuantOutput.NAME_FILE_MAP.keys():
    setattr(MaxQuantOutput, filename, MaxQuantOutput.register_file(filename))

# This version offers less inspection possibilities as the attributes are only set when they are looked up.
class MaxQuantOutputDynamic:
    """Class assisting with MaxQuant txt output folder. Fetches only availabe txt files.
    
    Parameters
    ----------
    folder: pathlib.Path, str
        Path to Maxquant `txt` output folder.

    Attributes
    ---------
    files : list
        file names on disk
    file_keys : list
        keys for file name on disk to use for lookup
    name_file_map : dict
        Keys for known MaxQuant output files.
    _inital_attritubutes : list
        Initial set of non-magic attributes 
    """  
    def __init__(self, folder):
        self.folder = Path(folder)
        self.files = self.get_files()
        
        # patch properties at instance creation?
        self.name_file_map = {}
        for file in self.files:
             file_key = Path(file).stem
             for symbol in " ()":
                 file_key = file_key.replace(symbol, '')
             self.name_file_map[file_key] = file
        self.file_keys = list(self.name_file_map)
    
    def get_files(self):
        """Get all txt files in output folder
        
        Attributes
        ---------
        paths: NamedTuple
        """
        self.paths = search_files(path=self.folder, query='.txt')
        return self.paths.files
    
        
    def load(self, filename):
        """Load a specified file into memory and return it.
        Can be used """
        filepath = self.folder / self.name_file_map[filename]
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No such file: {filename}.txt: Choose one of the following:\n{', '.join(self.files)}")
        
        return pd.read_table(filepath, index_col=0) 
    
    # needed to reset attributes on instance creation.
    _inital_attritubutes = [x for x in dir() if not x.startswith('__')]

    def get_list_of_attributes(self):
        """Return current list on non-magic instance attributes."""
        return [x for x in dir(self) if not x.startswith('__')]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.folder!r})'
    
    def __getattr__(self, filename):
        if filename in self.name_file_map:
            df = self.load(filename)
            setattr(self, filename, df)
        else:
            msg = f"No such file: {filename}.txt: Choose one of the following:\n{', '.join(self.file_keys)}"
            raise AttributeError(msg)
        return df