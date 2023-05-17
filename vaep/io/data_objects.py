from collections import Counter
import os
import sys
import logging
import json
from pathlib import Path
import multiprocessing
from types import SimpleNamespace
from typing import Callable, Iterable, List, Union

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

from fastcore.meta import delegates

from vaep.io import dump_json, dump_to_csv
import vaep.io.mq as mq
from vaep.io.mq import MaxQuantOutputDynamic
import vaep.pandas
from vaep.plotting import plot_feat_counts
# from .config import FOLDER_MQ_TXT_DATA, FOLDER_PROCESSED

logger = logging.getLogger(__name__)
logger.info(f"Calling from {__name__}")

FOLDER_DATA = Path('data')
FOLDER_DATA.mkdir(exist_ok=True)

FOLDER_PROCESSED = FOLDER_DATA / 'processed'
FOLDER_PROCESSED.mkdir(exist_ok=True)

FOLDER_MQ_TXT_DATA = FOLDER_DATA / 'mq_out'


# from vaep.cfg import DEFAULTS
DEFAULTS = SimpleNamespace()
DEFAULTS.ALL_SUMMARIES = Path(FOLDER_PROCESSED) / 'all_summaries.json'
DEFAULTS.COUNT_ALL_PEPTIDES = FOLDER_PROCESSED / 'count_all_peptides.json'

# fastcore.imports has in_notebook, etc functionality
from fastcore.imports import IN_IPYTHON, IN_JUPYTER, IN_COLAB, IN_NOTEBOOK
# IN_IPYTHON,IN_JUPYTER,IN_COLAB,IN_NOTEBOOK = in_ipython(),in_jupyter(),in_colab(),in_notebook()

N_WORKERS_DEFAULT = os.cpu_count() - 1 if os.cpu_count() <= 16 else 16
if sys.platform == 'win32' and IN_NOTEBOOK:
    N_WORKERS_DEFAULT = 1
    logger.warn(
        "only use main process due to issue with ipython and multiprocessing on Windows")


manager = multiprocessing.Manager()


def _convert_dtypes(df):
    """Convert dtypes automatically and make string columns categories."""
    df = df.convert_dtypes()
    l_string_columns = df.columns[df.dtypes == 'string']
    if not l_string_columns.empty:
        df[l_string_columns] = df[l_string_columns].astype('category')
    return df


def calc_chunksize(n_workers, len_iterable, factor=4):
    """Calculate chunksize argument for Pool-methods.

    Source and reference: https://stackoverflow.com/a/54813527/9684872
    """
    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    return chunksize


class col_summary:
    MS = 'MS'
    MS2 = 'MS/MS Identified'


class MqAllSummaries():

    def __init__(self, fp_summaries=DEFAULTS.ALL_SUMMARIES):
        fp_summaries = Path(fp_summaries)
        if fp_summaries.exists():
            self.df = _convert_dtypes(
                pd.read_json(fp_summaries, orient='index'))
            print(
                f"{self.__class__.__name__}: Load summaries of {len(self.df)} folders.")
        else:
            if not fp_summaries.parent.exists():
                raise FileNotFoundError(
                    f'Folder of filename not found: {fp_summaries.parent}')
            self.df = None
        self.fp_summaries = fp_summaries
        self.usecolumns = col_summary()

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        else:
            raise ValueError("No data loaded yet.")

    def load_summary(self, folder):
        folder_name = folder.stem
        try:
            mq_output = MaxQuantOutputDynamic(folder)
            # to_dict not very performant
            return {folder_name: mq_output.summary.iloc[0].to_dict()}
        except FileNotFoundError as e:
            if not mq_output.files and len(list(mq_output.folder.iterdir())) == 0:
                mq_output.folder.rmdir()
                logger.warning(f'Remove empty folder: {mq_output}')
                self.empty_folders.append(f"{folder_name}\n")
            else:
                logger.error(f"{mq_output}, No summary and not empty.")
        return {}

    def load_new_samples(self, folders, workers: int = 1):
        if self.df is not None:
            d_summaries = self.df.to_dict(orient='index')
            samples = set(folder.stem for folder in folders) - \
                set(d_summaries.keys())
            samples = [folder for folder in folders if folder.stem in samples]
        else:
            d_summaries = {}
            samples = folders
        if not hasattr(self, 'empty_folders'):
            # should this depend on multiprocessing?
            self.empty_folders = manager.list()

        if samples:
            # process_chunch_fct = self.load_summary
            # workers=workers
            # desc = 'Load summaries'
            if workers > 1:
                with multiprocessing.Pool(workers) as p:
                    # set chunksize: https://stackoverflow.com/a/49533645/9684872
                    chunksize = calc_chunksize(workers, len(samples), factor=2)
                    list_of_updates = list(
                        tqdm(
                            p.imap(self.load_summary, samples,
                                   chunksize=chunksize),
                            total=len(samples),
                            desc='Load summaries'))
            else:
                list_of_updates = [self.load_summary(
                    folder) for folder in tqdm(samples)]

            print("Newly loaded samples:", len(list_of_updates))

            for d in list_of_updates:
                d_summaries.update(d)

            self.df = _convert_dtypes(
                pd.DataFrame.from_dict(d_summaries, orient='index'))
            self.save_state()
        else:
            print("No new sample added.")
        return self.df

    def save_state(self):
        """Save summaries DataFrame as json and pickled object."""
        self.df.to_json(self.fp_summaries, orient='index')
        self.df.to_pickle(self.fp_summaries.parent /
                          f"{self.fp_summaries.stem}.pkl")

    def get_files_w_min_MS2(self, threshold=10_000, relativ_to=FOLDER_MQ_TXT_DATA):
        """Get a list of file ids with a minimum MS2 observations."""
        threshold_ms2_identified = threshold
        mask = self.df[self.usecolumns.MS2] > threshold_ms2_identified
        print(f"Selected  {mask.sum()} of {len(mask)} folders.")
        return [Path(relativ_to) / folder for folder in self.df.loc[mask].index]

# maybe move functions related to fnames
def get_fname(N, M):
    """Helper function to get file for intensities"""
    return f'df_intensities_N{N:05d}_M{M:05d}'


def get_folder_names(folders: Iterable[str]):
    return set(Path(folder).stem for folder in folders)


def create_parent_folder_name(folder: Path) -> str:
    return folder.stem[:4]


## plotting function for value_counts from FeatureCounter.get_df_counts



def collect_in_chuncks(paths: Iterable[Union[str, Path]],
                       process_chunk_fct: Callable,
                       n_workers: int = N_WORKERS_DEFAULT,
                       chunks=10,
                       desc='Run chunks in parallel') -> List:
    """collect the results from process_chunk_fct (chunk of files to loop over).
    The idea is that process_chunk_fct creates a more memory-efficient intermediate
    result than possible if only callling single fpaths in paths. 

    Parameters
    ----------
    paths : Iterable
        Iterable of paths
    process_chunk_fct : Iterable[str, Path]
        Callable which takes a chunk of paths and returns an result to collect, e.g. a dict
    n_workers : int, optional
        number of processes, by default N_WORKERS_DEFAULT

    Returns
    -------
    List
        List of results returned by process_chunk_fct
    """
    paths_splits = np.array_split(paths, min(chunks, len(paths)))
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as p:
            collected = list(tqdm(p.imap(process_chunk_fct, paths_splits),
                                             total=len(paths_splits),
                                             desc=desc))
    else:
        collected = map(process_chunk_fct, paths_splits)
    return collected


class FeatureCounter():
    def __init__(self, fp_counter: str, counting_fct: Callable[[List], Counter],
                idx_names:Union[List, None]=None,
                feature_name='feature',
                overwrite=False):
        self.fp = Path(fp_counter)
        self.counting_fct = counting_fct
        self.idx_names = idx_names
        self.feature_name = feature_name
        if self.fp.exists() and not overwrite:
            d = self.load(self.fp)
            self.counter = d['counter']
            self.loaded = set(folder for folder in d['based_on'])
            self.dumps = d['dumps']
        else:
            self.loaded = set() # None
            self.counter = Counter()
            self.dumps = dict()

    def __repr__(self):
        return f"{self.__class__.__name__}(fp_counter={str(self.fp)})"

    def get_new_folders(self, folders: List[str]):
        ret = get_folder_names(folders) - self.loaded
        return ret

    # combine multiprocessing into base class?
    def sum_over_files(self, folders: List[Path], n_workers=N_WORKERS_DEFAULT, save=True):
        if self.loaded:
            new_folder_names = self.get_new_folders(folders)
            logger.info(f'{len(new_folder_names)} new folders to process.')
            if new_folder_names:
                folders = [
                    folder for folder in folders if folder.stem in new_folder_names]
            else:
                folders = []

        if folders:
            list_of_sample_dicts = collect_in_chuncks(folders,
                       process_chunk_fct=self.counting_fct,
                       n_workers = n_workers,
                         chunks=n_workers*3,
                       desc = 'Count features in 100 chunks')

            for d in tqdm(list_of_sample_dicts,
                          total=len(list_of_sample_dicts),
                          desc='combine counters from chunks'):
                self.counter += d['counter']
                self.dumps.update(d['dumps'])

            if self.loaded:
                self.loaded |= new_folder_names
            else:
                self.loaded = get_folder_names(folders)
            if save:
                self.save()
        else:
            logger.info('Nothing to process.')
        return self.counter

    @property
    def n_samples(self):
        return len(self.loaded)
    

    def get_df_counts(self) -> pd.DataFrame:
        """Counted features as DataFrame with proportion values.

        Returns
        -------
        pd.DataFrame
            _description_
        """
        feat_counts = (pd.Series(self.counter)
                    .sort_values(ascending=False)
                    .to_frame('counts'))
        feat_counts['proportion'] = feat_counts / self.n_samples
        if self.idx_names: 
            feat_counts.index.names = self.idx_names
        feat_counts.reset_index(inplace=True)
        feat_counts.index.name = 'consecutive count'
        return feat_counts  

    def plot_counts(self, df_counts: pd.DataFrame = None, ax=None, prop_feat=0.25, min_feat_prop=.01):
        """Plot counts based on get_df_counts."""
        if df_counts is None:
            df_counts = self.get_df_counts()
        ax = plot_feat_counts(df_counts,
                              feat_name=self.feature_name,
                              n_samples=self.n_samples,
                              count_col='counts',
                              ax=ax)
        n_feat_cutoff = vaep.pandas.get_last_index_matching_proportion(
            df_counts=df_counts, prop=prop_feat)
        n_samples_cutoff = df_counts.loc[n_feat_cutoff, 'counts']
        logger.info(f'{n_feat_cutoff = }, {n_samples_cutoff = }')
        x_lim_max = vaep.pandas.get_last_index_matching_proportion(
            df_counts, min_feat_prop)
        logger.info(f'{x_lim_max = }')
        ax.set_xlim(-1, x_lim_max)
        ax.axvline(n_feat_cutoff, c='red')

        # ax.text(n_feat_cutoff + 0.03 * x_lim_max,
        #         n_samples_cutoff, '25% cutoff',
        #         style='italic', fontsize=12,
        #         bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})

        ax.annotate(f'{prop_feat*100}% cutoff',
                    xy=(n_feat_cutoff, n_samples_cutoff),
                    xytext=(n_feat_cutoff + 0.1 * x_lim_max, n_samples_cutoff),
                    fontsize=16,
                    arrowprops=dict(facecolor='black', shrink=0.05))

        return ax

    def save(self):
        """Save state

        {
         'counter': Counter,
         'based_on': list,
         'dumps: dict,
         }
        """
        d = {'counter': self.counter,
            'based_on': list(self.loaded),
            'dumps': {k: str(v) for k, v in self.dumps.items()}}
        logger.info(f"Save to: {self.fp}")
        dump_json(d, filename=self.fp)

    def load(self, fp):
        with open(self.fp) as f:
            d = json.load(f)
        d['counter'] = Counter(d['counter'])
        d['dumps'] = {k: Path(v) for k,v in d['dumps'].items()}
        return d

    def load_dump(self, fpath, fct=pd.read_csv, use_cols=None):
        return fct(fpath, index=self.idx_names, usecols=None)


class Count():

    def __init__(self,
                 process_folder_fct: Callable,
                 use_cols=None,
                 parent_folder_fct: Callable = create_parent_folder_name,
                 outfolder=FOLDER_PROCESSED / 'dumps',
                 dump=False):
        self.outfolder = Path(outfolder)
        self.outfolder.mkdir(exist_ok=True, parents=True)
        self.use_cols = use_cols
        self.process_folder_fct = process_folder_fct
        self.parent_folder_fct = parent_folder_fct
        self.dump = dump

    def __call__(self, folders,
                 **fct_args):
        logging.debug(
            f"Passed function arguments for process_folder_fct Callable: {fct_args}")
        c = Counter()
        fpath_dict = {}
        for folder in tqdm(folders):
            folder = Path(folder)
            df = self.process_folder_fct(
                folder=folder, use_cols=self.use_cols, **fct_args)
            c.update(df.index)
            if self.dump:
                fpath_dict[folder.stem] = dump_to_csv(df, folder=folder, outfolder=self.outfolder,
                            parent_folder_fct=self.parent_folder_fct)
        ret = {'counter': c, 'dumps': fpath_dict}
        return ret

### aggregated peptides

# # check df for redundant information (same feature value for all entries)
usecols = mq.COLS_ + ['Potential contaminant', mq.mq_col.SEQUENCE]


def count_peptides(folders: List[Path], dump=True,
                   usecols=usecols,
                   parent_folder_fct: Callable = create_parent_folder_name,
                   outfolder=FOLDER_PROCESSED / 'agg_peptides_dumps'):
    c = Counter()
    fpath_dict = {}
    for folder in folders:
        peptides = pd.read_table(folder / 'peptides.txt',
                                 usecols=usecols,
                                 index_col=0)
        mask = (peptides[mq.mq_col.INTENSITY] == 0) | (
            peptides["Potential contaminant"] == '+')
        peptides = peptides.loc[~mask]
        c.update(peptides.index)
        if dump:
            fpath_dict[folder.stem] = dump_to_csv(peptides.drop('Potential contaminant', axis=1),
                             folder=folder, outfolder=outfolder,
                    parent_folder_fct=parent_folder_fct)
    ret = {'counter': c, 'dumps': fpath_dict}
    return ret

d_dtypes_training_sample = {
    'Sequence': pd.StringDtype(),
    'Proteins': pd.StringDtype(),
    'Leading razor protein': pd.StringDtype(),
    'Gene names': pd.StringDtype(),
    'Intensity': pd.Int64Dtype()
}


def load_agg_peptide_dump(fpath):
    fpath = Path(fpath)
    peptides = pd.read_csv(fpath, index_col=0, dtype=d_dtypes_training_sample)
    return peptides

@delegates()
class PeptideCounter(FeatureCounter):

    def __init__(self,
                 fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_peptides,
                 idx_names=['Sequence'],
                 feature_name='aggregated peptide',
                  **kwargs):
        super().__init__(fp_counter, counting_fct=counting_fct,
                         idx_names=idx_names, feature_name=feature_name, **kwargs)

    @staticmethod
    def load_dump(fpath):
        return load_agg_peptide_dump(fpath)



### Evidence
evidence_cols = mq.mq_evidence_cols


def select_evidence(df_evidence: pd.DataFrame) -> pd.DataFrame:
    mask = (df_evidence[evidence_cols.Potential_contaminant]
            == '+') | (df_evidence[evidence_cols.Intensity] == 0)
    evidence = df_evidence.loc[~mask].drop(
        evidence_cols.Potential_contaminant, axis=1)
    evidence = evidence.dropna(subset=[evidence_cols.Intensity])
    return evidence


idx_columns_evidence = [evidence_cols.Sequence, evidence_cols.Charge]


def load_process_evidence(folder: Path, use_cols, select_by):
    evidence = pd.read_table(folder / 'evidence.txt',
                             usecols=idx_columns_evidence + use_cols)
    evidence = select_evidence(evidence)
    evidence = vaep.pandas.select_max_by(
        evidence, grouping_columns=idx_columns_evidence, selection_column=select_by)
    evidence = evidence.set_index(idx_columns_evidence).sort_index()
    return evidence


def count_evidence(folders: List[Path],
                   select_by: str = 'Score',
                   dump=True,
                   use_cols=[evidence_cols.mz,
                             evidence_cols.Protein_group_IDs,
                             evidence_cols.Intensity,
                             evidence_cols.Score,
                             evidence_cols.Potential_contaminant],
                   parent_folder_fct: Callable = create_parent_folder_name,
                   outfolder=FOLDER_PROCESSED / 'evidence_dumps'):
    outfolder = Path(outfolder)
    outfolder.mkdir(exist_ok=True, parents=True)
    c = Counter()
    if dump:
        fpath_dict = {}
    for folder in tqdm(folders):
        folder = Path(folder)
        evidence = load_process_evidence(
            folder=folder, use_cols=use_cols, select_by=select_by)
        c.update(evidence.index)
        if dump:
            fpath_dict[folder.stem] = dump_to_csv(evidence, folder=folder, outfolder=outfolder,
            parent_folder_fct=parent_folder_fct)
    ret = {'counter': c, 'dumps': fpath_dict}
    return ret


@delegates()
class EvidenceCounter(FeatureCounter):

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_evidence,
                 idx_names=['Sequence', 'Charge'],
                 feature_name='charged peptide',
                 **kwargs):
        super().__init__(fp_counter, counting_fct,
                         idx_names=idx_names, feature_name=feature_name, **kwargs)

    # Methods should use super, otherwise non-specific duplication is needed.
    def save(self):
        """Save state

        {
         'counter': Counter with tuple keys,
         'based_on': list
         }
        """
        d = {'counter': vaep.pandas.create_dict_of_dicts(self.counter),
             'based_on': list(self.loaded), 
             'dumps': {k: str(v) for k, v in self.dumps.items()}}
        print(f"Save to: {self.fp}")
        dump_json(d, filename=self.fp)

    def load(self, fp):
        with open(self.fp) as f:
            d = json.load(f)
        d['counter'] = Counter(
            vaep.pandas.flatten_dict_of_dicts(d['counter']))
        d['dumps'] = {k: Path(v) for k,v in d['dumps'].items()}
        return d


def load_evidence_dump(fpath, index_col=['Sequence', 'Charge']):
    df = pd.read_csv(fpath, index_col=index_col)
    return df

### Protein Groups


pg_cols = mq.mq_protein_groups_cols

# def load_process_evidence(folder: Path, use_cols, select_by):


def load_and_process_proteinGroups(folder: Union[str, Path],
                                   #use_cols not really a parameter (or needs asserts?)
                                   use_cols: List = [
                                       pg_cols.Protein_IDs,
                                       pg_cols.Majority_protein_IDs,
                                       pg_cols.Gene_names,
                                       pg_cols.Evidence_IDs,
                                       pg_cols.Q_value,
                                       pg_cols.Score,
                                       pg_cols.Only_identified_by_site,
                                       pg_cols.Reverse,
                                       pg_cols.Potential_contaminant,
                                       pg_cols.Intensity,
]):
    folder = Path(folder)
    pg = pd.read_table(folder / 'proteinGroups.txt',
                       usecols=use_cols)
    mask = pg[[pg_cols.Only_identified_by_site, pg_cols.Reverse,
               pg_cols.Potential_contaminant]].notna().sum(axis=1) > 0
    pg = pg.loc[~mask]
    mask = pg[pg_cols.Intensity] > 1
    pg = pg.loc[mask]
    gene_set = pg[pg_cols.Gene_names].str.split(';')
    col_loc_gene_names = pg.columns.get_loc(pg_cols.Gene_names)
    _ = pg.insert(col_loc_gene_names+1, 'Number of Genes',
                  gene_set.apply(vaep.pandas.length))
    mask_no_gene = pg[pg_cols.Gene_names].isna()
    pg_no_gene = pg.loc[mask_no_gene]
    logger.debug(f"Entries without any gene annotation: {len(pg_no_gene)}")
    pg = vaep.pandas.select_max_by(df=pg.loc[~mask_no_gene],
                                   grouping_columns=[pg_cols.Gene_names],
                                   selection_column=pg_cols.Score)
    pg = pg.append(pg_no_gene)
    pg = pg.set_index(pg_cols.Protein_IDs)
    return pg





count_protein_groups = Count(load_and_process_proteinGroups,
                             use_cols=[
                                 pg_cols.Protein_IDs,
                                 pg_cols.Majority_protein_IDs,
                                 pg_cols.Gene_names,
                                 pg_cols.Evidence_IDs,
                                 pg_cols.Q_value,
                                 pg_cols.Score,
                                 pg_cols.Only_identified_by_site,
                                 pg_cols.Reverse,
                                 pg_cols.Potential_contaminant,
                                 pg_cols.Intensity,
                             ],
                             outfolder=FOLDER_PROCESSED / 'proteinGroups_dumps',
                             dump=True)


@delegates()
class ProteinGroupsCounter(FeatureCounter):

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List],
                                        Counter] = count_protein_groups,
                 idx_names=[pg_cols.Protein_IDs], # mq_specfic
                 feature_name='protein group',
                 **kwargs):
        super().__init__(fp_counter, counting_fct, idx_names=idx_names,
                         feature_name=feature_name, **kwargs)


def load_pg_dump(folder, use_cols=None):
    logger.debug(f"Load: {folder}")
    df = pd.read_csv(folder, index_col=pg_cols.Protein_IDs, usecols=use_cols)
    return df

## Gene Counter

def pg_idx_gene_fct(folder:Union[str, Path], use_cols=None):
    folder = Path(folder)
    logger.debug(f"Load: {folder}")
    df = pd.read_csv(folder, index_col=pg_cols.Gene_names, usecols=use_cols)
    return df


count_genes = Count(pg_idx_gene_fct,
                    use_cols=[
                        pg_cols.Protein_IDs,
                        pg_cols.Gene_names,
                        pg_cols.Intensity,
                    ],
                    outfolder=FOLDER_PROCESSED / 'gene_dumps',  # don't dump, only read
                    dump=False)


#summing needs to be done over processed proteinGroup dumps
@delegates()
class GeneCounter(FeatureCounter):
    """Gene Counter to count gene in dumped proteinGroups."""

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_genes,
                 feature_name='gene',
                 idx_names=['Gene names'], **kwargs):
        super().__init__(fp_counter, counting_fct, idx_names=idx_names,
                         feature_name=feature_name, **kwargs)
