import logging
from collections import Counter, namedtuple
from pathlib import Path
from typing import Iterable
import omegaconf

import pandas as pd
from pandas import Int64Dtype, StringDtype, Float64Dtype

import vaep.io

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


mq_use_columns = ['Gene names',
                  'Intensity',
                  'Retention time',
                  'Calibrated retention time',
                  'Sequence',
                  'Leading razor protein',
                  'Proteins'
                  ]

MqColumnsUsed = namedtuple(typename='MqColumns',
                           field_names=[
                               s.upper().replace(' ', '_') for s in mq_use_columns
                           ])
mq_col = MqColumnsUsed(*mq_use_columns)

FASTA_KEYS = ["Proteins", "Gene names"]

mq_evidence_cols = {'Sequence': 'Sequence',
                    'Length': 'Length',
                    'Modifications': 'Modifications',
                    'Modified_sequence': 'Modified sequence',
                    'Oxidation_M_Probabilities': 'Oxidation (M) Probabilities',
                    'Oxidation_M_Score_Diffs': 'Oxidation (M) Score Diffs',
                    'Acetyl_Protein_N-term': 'Acetyl (Protein N-term)',
                    'Oxidation_M': 'Oxidation (M)',
                    'Missed_cleavages': 'Missed cleavages',
                    'Proteins': 'Proteins',
                    'Leading_proteins': 'Leading proteins',
                    'Leading_razor_protein': 'Leading razor protein',
                    'Gene_names': 'Gene names',
                    'Protein_names': 'Protein names',
                    'Type': 'Type',
                    'Raw_file': 'Raw file',
                    'MSMS_mz': 'MS/MS m/z',
                    'Charge': 'Charge',
                    'mz': 'm/z',
                    'Mass': 'Mass',
                    'Uncalibrated_-_Calibrated_mz_[ppm]': 'Uncalibrated - Calibrated m/z [ppm]',
                    'Uncalibrated_-_Calibrated_mz_[Da]': 'Uncalibrated - Calibrated m/z [Da]',
                    'Mass_error_[ppm]': 'Mass error [ppm]',
                    'Mass_error_[Da]': 'Mass error [Da]',
                    'Uncalibrated_mass_error_[ppm]': 'Uncalibrated mass error [ppm]',
                    'Uncalibrated_mass_error_[Da]': 'Uncalibrated mass error [Da]',
                    'Max_intensity_mz_0': 'Max intensity m/z 0',
                    'Retention_time': 'Retention time',
                    'Retention_length': 'Retention length',
                    'Calibrated_retention_time': 'Calibrated retention time',
                    'Calibrated_retention_time_start':
                    'Calibrated retention time start',
                    'Calibrated_retention_time_finish': 'Calibrated retention time finish',
                    'Retention_time_calibration': 'Retention time calibration',
                    'Match_time_difference': 'Match time difference',
                    'Match_mz_difference': 'Match m/z difference',
                    'Match_q-value': 'Match q-value',
                    'Match_score': 'Match score',
                    'Number_of_data_points': 'Number of data points',
                    'Number_of_scans': 'Number of scans',
                    'Number_of_isotopic_peaks': 'Number of isotopic peaks',
                    'PIF': 'PIF',
                    'Fraction_of_total_spectrum': 'Fraction of total spectrum',
                    'Base_peak_fraction': 'Base peak fraction',
                    'PEP': 'PEP',
                    'MSMS_count': 'MS/MS count',
                    'MSMS_scan_number': 'MS/MS scan number',
                    'Score': 'Score',
                    'Delta_score': 'Delta score',
                    'Combinatorics': 'Combinatorics',
                    'Intensity': 'Intensity',
                    'Reverse': 'Reverse',
                    'Potential_contaminant': 'Potential contaminant',
                    'id': 'id',
                    'Protein_group_IDs': 'Protein group IDs',
                    'Peptide_ID': 'Peptide ID',
                    'Mod._peptide_ID': 'Mod. peptide ID',
                    'MSMS_IDs': 'MS/MS IDs',
                    'Best_MSMS': 'Best MS/MS',
                    'Oxidation_M_site_IDs': 'Oxidation (M) site IDs',
                    'Taxonomy_IDs': 'Taxonomy IDs'}

mq_evidence_cols = omegaconf.OmegaConf.create(mq_evidence_cols)


mq_evidence_dtypes = {'Length': Int64Dtype(),
                      'Modifications': StringDtype,
                      'Modified sequence': StringDtype,
                      'Oxidation (M) Probabilities': StringDtype,
                      'Oxidation (M) Score Diffs': StringDtype,
                      'Acetyl (Protein N-term)': Int64Dtype(),
                      'Oxidation (M)': Int64Dtype(),
                      'Missed cleavages': Int64Dtype(),
                      'Proteins': StringDtype,
                      'Leading proteins': StringDtype,
                      'Leading razor protein': StringDtype,
                      'Gene names': StringDtype,
                      'Protein names': StringDtype,
                      'Type': StringDtype,
                      'Raw file': StringDtype,
                      'MS/MS m/z': Float64Dtype(),
                      'm/z': Float64Dtype(),
                      'Mass': Float64Dtype(),
                      'Uncalibrated - Calibrated m/z [ppm]': Float64Dtype(),
                      'Uncalibrated - Calibrated m/z [Da]': Float64Dtype(),
                      'Mass error [ppm]': Float64Dtype(),
                      'Mass error [Da]': Float64Dtype(),
                      'Uncalibrated mass error [ppm]': Float64Dtype(),
                      'Uncalibrated mass error [Da]': Float64Dtype(),
                      'Max intensity m/z 0': Float64Dtype(),
                      'Retention time': Float64Dtype(),
                      'Retention length': Float64Dtype(),
                      'Calibrated retention time': Float64Dtype(),
                      'Calibrated retention time start': Float64Dtype(),
                      'Calibrated retention time finish': Float64Dtype(),
                      'Retention time calibration': Float64Dtype(),
                      'Match time difference': Int64Dtype(),
                      'Match m/z difference': Int64Dtype(),
                      'Match q-value': Int64Dtype(),
                      'Match score': Int64Dtype(),
                      'Number of data points': Int64Dtype(),
                      'Number of scans': Int64Dtype(),
                      'Number of isotopic peaks': Int64Dtype(),
                      'PIF': Int64Dtype(),
                      'Fraction of total spectrum': Int64Dtype(),
                      'Base peak fraction': Int64Dtype(),
                      'PEP': Float64Dtype(),
                      'MS/MS count': Int64Dtype(),
                      'MS/MS scan number': Int64Dtype(),
                      'Score': Float64Dtype(),
                      'Delta score': Float64Dtype(),
                      'Combinatorics': Int64Dtype(),
                      'Intensity': Int64Dtype(),
                      'Reverse': Int64Dtype(),
                      'Potential contaminant': Int64Dtype(),
                      'id': Int64Dtype(),
                      'Protein group IDs': StringDtype,
                      'Peptide ID': Int64Dtype(),
                      'Mod. peptide ID': Int64Dtype(),
                      'MS/MS IDs': StringDtype,
                      'Best MS/MS': Int64Dtype(),
                      'Oxidation (M) site IDs': StringDtype,
                      'Taxonomy IDs': StringDtype,
                      }


mq_protein_groups_cols = {'Protein_IDs': 'Protein IDs',
                          'Majority_protein_IDs': 'Majority protein IDs',
                          'Peptide_counts_all': 'Peptide counts (all)',
                          'Peptide_counts_razor+unique': 'Peptide counts (razor+unique)',
                          'Peptide_counts_unique': 'Peptide counts (unique)',
                          'Protein_names': 'Protein names',
                          'Gene_names': 'Gene names',
                          'Fasta_headers': 'Fasta headers',
                          'Number_of_proteins': 'Number of proteins',
                          'Peptides': 'Peptides',
                          'Razor_+_unique_peptides': 'Razor + unique peptides',
                          'Unique_peptides': 'Unique peptides',
                          'Sequence_coverage_[%]': 'Sequence coverage [%]',
                          'Unique_+_razor_sequence_coverage_[%]': 'Unique + razor sequence coverage [%]',
                          'Unique_sequence_coverage_[%]': 'Unique sequence coverage [%]',
                          'Mol._weight_[kDa]': 'Mol. weight [kDa]',
                          'Sequence_length': 'Sequence length',
                          'Sequence_lengths': 'Sequence lengths',
                          'Q_value': 'Q-value',
                          'Score': 'Score',
                          'Intensity': 'Intensity',
                          'MSMS_count': 'MS/MS count',
                          'Only_identified_by_site': 'Only identified by site',
                          'Reverse': 'Reverse',
                          'Potential_contaminant': 'Potential contaminant',
                          'id': 'id',
                          'Peptide_IDs': 'Peptide IDs',
                          'Peptide_is_razor': 'Peptide is razor',
                          'Mod._peptide_IDs': 'Mod. peptide IDs',
                          'Evidence_IDs': 'Evidence IDs',
                          'MSMS_IDs': 'MS/MS IDs',
                          'Best_MSMS': 'Best MS/MS',
                          'Oxidation_M_site_IDs': 'Oxidation (M) site IDs',
                          'Oxidation_M_site_positions': 'Oxidation (M) site positions',
                          'Taxonomy_IDs': 'Taxonomy IDs'}

mq_protein_groups_cols = omegaconf.OmegaConf.create(mq_protein_groups_cols)

##########################################################################################
##########################################################################################
# import abc # abc.ABCMeta ?


class MaxQuantOutput():
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
        self.paths = vaep.io.search_files(path=self.folder, query='.txt')
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
            raise FileNotFoundError(
                f"No such file: {file}.txt: Choose one of the following {', '.join(self.files)}")

        return pd.read_table(filepath, index_col=0)

    # def dump_training_data(self, )

    def get_list_of_attributes(self):
        """Return current list on non-magic instance attributes."""
        return [x for x in dir(self) if not x.startswith('__')]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.folder!r})'

    def dump_intensity(self, folder='.'):
        """Dump all intensity values from peptides.txt"""
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        fname = folder / f"{self.folder.stem}.json"
        vaep.io.dump_json(
            data_dict=self.peptides.Intensity.dropna().to_dict(),
            filename=fname)
        logger.info(f'Dumped intensities in peptides.txt: {fname}.')

    # needed to reset attributes on instance creation.
    _inital_attritubutes = [x for x in dir() if not x.startswith('__')]


# register all properties
# Would be great to be able to do this at runtime based on the files actually present.
for filename in MaxQuantOutput.NAME_FILE_MAP.keys():
    setattr(MaxQuantOutput, filename, MaxQuantOutput.register_file(filename))

# This version offers less inspection possibilities as the attributes are only set when they are looked up.


class MaxQuantOutputDynamic(MaxQuantOutput):
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
        super().__init__(folder)

        # patch properties at instance creation?
        self.name_file_map = {}
        for file in self.files:
            file_key = Path(file).stem
            for symbol in " ()":
                file_key = file_key.replace(symbol, '')
            self.name_file_map[file_key] = file
        self.file_keys = list(self.name_file_map)

    def __getattr__(self, filename):
        if filename in self.name_file_map:
            df = self.load(filename)
            setattr(self, filename, df)
        else:
            msg = f"No such file: {filename}.txt: Choose one of the following:\n{', '.join(self.file_keys)}"
            raise AttributeError(msg)
        return df

    # needed to reset attributes on instance creation.
    _inital_attritubutes = [x for x in dir() if not x.startswith('__')]

##########################################################################################
##########################################################################################


def check_df(df, columns):
    """Check DataFrame for specified columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for which should contain `columns`.
    columns : Iterable
        Iterable of column names.

    Raises
    ------
    AttributeError
        One or more `columns` are missing. Specifies which.
    """

    missing = []
    for col in columns:
        if not col in df:
            missing.append(col)

    if missing:
        raise AttributeError(f'Missing column(s): {", ".join(missing)}')


COLS_ = [mq_col.INTENSITY, mq_col.LEADING_RAZOR_PROTEIN] + FASTA_KEYS


def get_peptides_with_single_gene(peptides, keep_columns=COLS_, gene_column=mq_col.GENE_NAMES):
    """Get long-data-format. Ungroup gene names. Peptides "shared" by genes
    are assigned individual rows. retains only cases with full list of 
    features provided by `keep_columns`.

    Parameters
    ----------
    peptides: pandas.DataFrame
        MaxQuant txt output loaded as `pandas.DataFrame`.
    keep_columns: list
        List of columns to keep from the `peptides`.txt, default 
        {cols_}
    gene_column: str
        Column containing group information of format "group1;group2",
        i.e. in MQ for genes "gene1;gene2".
    """.format(cols_=COLS_)
    if gene_column not in keep_columns:
        keep_columns.append(gene_column)
    check_df(peptides, COLS_)
    peptides_with_single_gene = peptides[COLS_].dropna(how='any')
    if len(peptides) < len(peptides_with_single_gene):
        logger.warning('Removed {} of {} entries due to missing values.'.format(
            len(peptides) - len(peptides_with_single_gene),
            len(peptides)
        ))
    peptides_with_single_gene[gene_column] = peptides_with_single_gene[gene_column].str.split(
        ';')
    peptides_with_single_gene = peptides_with_single_gene.explode(
        column=gene_column)
    return peptides_with_single_gene


def get_set_of_genes(iterable, sep_in_str: str = ';'):
    "Return the set of unique strings for an Iterable of strings (gene names)."
    genes_single_unique = set()
    for gene_iterable in pd.Series(iterable).str.split(sep_in_str):
        try:
            genes_single_unique.update(gene_iterable)
        except TypeError:
            pass
    return genes_single_unique


def validate_gene_set(n_gene_single_unique, n_gene_sets):
    """Compare N single geens to number of unqiue gene sets.

    Parameters
    ----------
    n_gene_single_unique : int
        Count in set.
    n_gene_sets : int
        Count in set.

    Raises
    ------
    ValueError
        [description]
    """
    if n_gene_single_unique < n_gene_sets:
        print(
            f'There are however less unique-single genes {n_gene_single_unique} than sets.')
    elif n_gene_single_unique == n_gene_sets:
        print(f'Only realy unique gene (sets)')
    else:
        raise ValueError(
            f'There are more gene-sets than unique genes: {n_gene_sets} vs. {n_gene_single_unique}.')


def count_genes_in_sets(gene_sets, sep=';'):
    """Count for an Iterable of gene_sets

    Parameters
    ----------
    gene_sets : Iterable
        Iterable of gene_sets which entries are separated with `sep`
    sep : str
        Seperator of gene sets, default ';'

    Returns
    -------
    collections.Counter
        Counter with keys as genes and counts as value.
    """
    genes_counted_each_in_unique_sets = Counter()

    for gene in pd.Series(gene_sets).dropna():
        try:
            gene_iterable = gene.split(sep)
            genes_counted_each_in_unique_sets.update(gene_iterable)
        except TypeError:
            print(f"Error on: {gene}")

    return genes_counted_each_in_unique_sets


def get_identifier_from_column(df: pd.DataFrame, identifier_col: str):
    """Get unique identifier in a column of a DataFrame. 

    Parameters
    ----------
    df : pd.DataFrame
        (Sub-) DataFrame with data for a gene.
    identifier_col : str
        Column name in which unique identifier is suspected

    Returns
    -------
    Any
        unique identifier in `identifier_col`

    Raises
    ------
    ValueError
        Non-unique identifier in column
    """
    identifier = df[identifier_col].unique()
    if len(identifier) == 1:
        identifier = identifier[0]
    else:
        raise ValueError(
            f"Found {len(identifier)} non-unique identifier: {identifier}")
    return identifier


def find_exact_cleaved_peptides_for_razor_protein(gene_data, fasta_db, gene_id: str = None):
    """Find exactly cleaved peptides based on razor protein in provided data-set

    Parameters
    ----------
    gene_data : pandas.DataFrame
        Pandas DataFrame with information from MQ peptides.txt output table.
        gene_data.columns.name should be set to gene names of gene_data.
    fasta_db : dict
        Created fasta database with specific scheme.
    gene_id : str, optional
        gene name, by default None

    Returns
    -------
    list
        list of exact peptides for the razor protein of the gene.

    Raises
    ------
    ValueError
        Raised if no unique gene identifier could be inferred from the data if no gene-id
        was set.
    KeyError
        If no protein could be found in fasta_db for specified gene.
    """
    # ToDo: Replace with config from package
    KEY_PEPTIDES = 'peptides'

    if not isinstance(gene_data.columns.name, str) or not gene_id:
        try:
            gene_id = get_identifier_from_column(gene_data, mq_col.GENE_NAMES)
        except ValueError as e:
            raise ValueError(
                f"Could not identify single, unique identifier from {gene_id} column: {e}"
                "Please set columns.name feature to a string-identifier (for genes separated by ;)"
                f" not of {type(gene_data.columns.name)}: {gene_data.columns.name}")
    protein_id = gene_data[mq_col.LEADING_RAZOR_PROTEIN].unique()

    # ToDo: Check for all proteins and decide on the best?
    if len(protein_id) != 1:
        logger.warning("- Gene: {:8}: More than one razor protein (try first): {} (Gene: {}) ".format(
            gene_id, ", ".join(x for x in protein_id), gene_data.columns.name))
    protein_id = protein_id[0]
    try:
        peps_exact_cleaved = fasta_db[protein_id][KEY_PEPTIDES][0]
    except KeyError:
        # MQ marks potential contaminent proteins
        if 'CON__' in protein_id:
            logger.info(
                f"- Gene: {gene_id:8}: "
                f"Potential contaminent protein is leading razor protein: {protein_id}"
                f" (Gene: {gene_data.columns.name})")
        elif 'REV__' in protein_id:
            logger.info(
                f"- Gene: {gene_id:8}: "
                f"Reversed protein is leading razor protein: {protein_id}"
                f" (Gene: {gene_data.columns.name})")
        else:
            raise ValueError(f'Check case for {gene_id} on {protein_id}.')
        # assert len(gene_data[mq_col.PROTEINS].unique()) == 1, f"{gene_data[mq_col.PROTEINS].unique()}"
        protein_sets = gene_data[mq_col.PROTEINS].unique()
        if len(protein_sets) > 1:
            logger.warning(
                f"More than one set of genes: {gene_data[mq_col.PROTEINS].unique()}")
            # ToDo: find intersection of proteins between all sequences.

        # Enforce: proteins have to be share between all peptides
        protein_sets = [set.split(';') for set in protein_sets]
        # ToDo: Check if ordering is relevant (if not all proteins are checked)
        proteins_shared_by_all = set(
            protein_sets.pop()).intersection(*protein_sets)
        # ToDo: Some CON_ proteins are also present in the fasta and appear twice.
        #       Remove all CON__ proteins from data globally, including their fasta
        #       pendants (e.g. Keratin: Q04695;CON__Q04695)
        # exclude potential other contaminents
        protein_sets = [
            x for x in proteins_shared_by_all if not 'CON__' in x]  # .sorted()
        if len(protein_sets) == 0:
            # raise KeyError("No other overall protein found for sequences.")
            logger.warning(
                f'No good protein found for gene ({gene_id:8}). Return empty list.')
            return []
        if len(protein_sets) > 1:
            logger.warning(
                f"- Gene: {gene_id:8}: "
                "Non-unique other protein set found (select first): {}".format(
                    ', '.join(protein_sets)
                ))
        protein_id = protein_sets.pop()
        peps_exact_cleaved = fasta_db[protein_id][KEY_PEPTIDES][0]
    return peps_exact_cleaved


def calculate_completness_for_sample(
        peps_exact_cleaved: Iterable[str],
        peps_in_data: Iterable[str]):
    """Calculate completeness for set of peptides.

    Parameters
    ----------
    peps_exact_cleaved : Iterable[str]
        Iterable of peptides exactly cleaved
    peps_in_data : Iterable[str]
        Iterable of peptides found during a run / in a sample. Check if peptides 
        overlap with any of the exact peptides.

    Returns
    -------
    float
        proportion of exact peptides for which some evidence was found.
    """
    c = 0
    if not peps_exact_cleaved:
        return 0  # no exact peptides
    for i, _pep in enumerate(peps_exact_cleaved):
        logger.debug(f"Check if exact peptide matches: {_pep}")
        for _found_pep in peps_in_data:
            logger.debug(f"Check for peptide: {_found_pep}")
            if _pep in _found_pep:
                c += 1
                break
        if c == len(peps_in_data):
            logger.debug(f"Last checked peptides in position {i:2}: {_pep}")
            logger.debug(
                f"Searched therfore {i+1:2} out of {len(peps_exact_cleaved)} peptides, "
                f"i.e. a share of {(i+1)/len(peps_exact_cleaved):.3f}")
            break
    return c / len(peps_exact_cleaved)


class ExtractFromPeptidesTxt():
    """Strategy to extract Intensity measurements from MaxQuant txt output peptides.txt.
    Creates dump of Training Data.
    """

    def __init__(self,
                 out_folder,
                 mq_output_object: MaxQuantOutput,
                 # Could be made a certain type -> ensure schema is met.
                 fasta_db: dict
                 ):
        # # ToDo: make this check work
        assert isinstance(mq_output_object, MaxQuantOutput)
        self._mq_output = mq_output_object
        self.out_folder = Path(out_folder) / mq_output_object.folder.stem
        self.out_folder.mkdir(exist_ok=True, parents=True)
        self.fname_template = '{gene}.json'
        self.fasta_db = fasta_db

    def __call__(self):
        """Dump valid cases to file.

        Returns:
        collections.Counter
            Counter with gene IDs as key and completeness as value.
        """
        _counter = 0
        _genes = dict()
        peptides_with_single_gene = get_peptides_with_single_gene(
            peptides=self._mq_output.peptides)
        for gene_names, data_gene in peptides_with_single_gene.groupby(mq_col.GENE_NAMES):
            data_gene.columns.name = gene_names  # ToDo: Find better solution
            peps_exact_cleaved = find_exact_cleaved_peptides_for_razor_protein(
                data_gene, fasta_db=self.fasta_db)
            c = calculate_completness_for_sample(peps_exact_cleaved=peps_exact_cleaved,
                                                 peps_in_data=data_gene.index)
            assert gene_names not in _genes
            _genes[gene_names] = c
            # ToDo check completeness for each shared protein in list
            if c >= .6:
                fname = self.out_folder / \
                    self.fname_template.format(gene=gene_names)
                with open(fname, 'w') as f:
                    data_gene.to_json(f)
                _counter += 1
        logger.info(
            f'Dumped {_counter} genes from {self._mq_output.folder.stem}')
        fname = self.out_folder / '0_completeness_all_genes.json'
        vaep.io.dump_json(_genes, fname)
        logger.info(f'Dumped files to: {str(self.out_folder)}')
        return _genes

    def __repr__(self):
        return f"{self.__class__.__name__}(out_folder={self.out_folder}, mq_output_object={repr(self._mq_output)}, fasta_db)"


# so MaxQuantOutput could know which strategy to apply for which file-type?
STRATEGIES = {'peptides.txt': '',
              'evidence.txt': ''}
