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

from vaep.io import dump_json
import vaep.io.mq as mq
from vaep.io.mq import MaxQuantOutputDynamic
import vaep.pandas
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

            if workers > 1:
                with multiprocessing.Pool(workers) as p:
                    # set chunksize: https://stackoverflow.com/a/49533645/9684872
                    chunksize = calc_chunksize(workers, len(samples), factor=2)
                    list_of_updates = list(tqdm(p.imap(
                        self.load_summary, samples, chunksize=chunksize), total=len(samples), desc='Load summaries'))
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


def get_fname(N, M):
    """Helper function to get file for intensities"""
    return f'df_intensities_N{N:05d}_M{M:05d}'


def get_folder_names(folders: Iterable[str]):
    return set(Path(folder).stem for folder in folders)


class FeatureCounter():
    def __init__(self, fp_counter: str, counting_fct: Callable[[List], Counter]):
        self.fp = Path(fp_counter)
        self.counting_fct = counting_fct
        if self.fp.exists():
            d = self.load(self.fp)
            self.counter = d['counter']
            self.loaded = set(folder for folder in d['based_on'])
        else:
            self.loaded = None
            self.counter = None

    def __repr__(self):
        return f"{self.__class__.__name__}(fp_counter={str(self.fp)})"

    def get_new_folders(self, folders: List[str]):
        ret = get_folder_names(folders) - self.loaded
        return ret

    # combine multiprocessing into base class?
    def sum_over_files(self, folders: List[Path], n_workers=N_WORKERS_DEFAULT, save=True):
        if self.loaded:
            new_folder_names = self.get_new_folders(folders)
            print(f'{len(new_folder_names)} new folders to process.')
            if new_folder_names:
                folders = [
                    folder for folder in folders if folder.stem in new_folder_names]
            else:
                folders = []

        if folders:
            folder_splits = np.array_split(folders, min(100, len(folders)))
            if n_workers > 1:
                with multiprocessing.Pool(n_workers) as p:
                    list_of_sample_dicts = list(tqdm(p.imap(self.counting_fct, folder_splits),
                                                     total=len(folder_splits),
                                                     desc='Count peptides in 100 chunks'))
            else:
                list_of_sample_dicts = map(self.counting_fct, folder_splits)
            if not self.counter:
                self.counter = Counter()
            for d in tqdm(list_of_sample_dicts,
                          total=len(folder_splits),
                          desc='combine counters from chunks'):
                self.counter += d

            if self.loaded:
                self.loaded |= new_folder_names
            else:
                self.loaded = get_folder_names(folders)
            if save:
                self.save()
        else:
            print('Nothing to process.')
        return self.counter

    def save(self):
        """Save state

        {
         'counter': Counter,
         'based_on': list
         }
        """
        d = {'counter': self.counter, 'based_on': list(self.loaded)}
        print(f"Save to: {self.fp}")
        dump_json(d, filename=self.fp)

    def load(self, fp):
        with open(self.fp) as f:
            d = json.load(f)
        d['counter'] = Counter(d['counter'])
        return d


# aggregated peptides

# # check df for redundant information (same feature value for all entries)
usecols = mq.COLS_ + ['Potential contaminant', mq.mq_col.SEQUENCE]


def count_peptides(folders: List[Path], dump=True):
    c = Counter()
    for folder in folders:
        peptides = pd.read_table(folder / 'peptides.txt',
                                 usecols=usecols,
                                 index_col=0)
        mask = (peptides[mq.mq_col.INTENSITY] == 0) | (
            peptides["Potential contaminant"] == '+')
        peptides = peptides.loc[~mask]
        c.update(peptides.index)
        if dump:
            # change into subfolder structure:
            folder_out = FOLDER_PROCESSED / folder.stem[:4]
            folder_out.mkdir(exist_ok=True, parents=True)
            fpath = folder_out / f"{folder.stem}.csv"
            logger.info(f"Dump file: {fpath}")
            peptides.drop('Potential contaminant', axis=1).to_csv(fpath)
    return c


class PeptideCounter(FeatureCounter):

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_peptides):
        super().__init__(fp_counter, counting_fct)


# Evidence
evidence_cols = mq.mq_evidence_cols


def select_evidence(df_evidence: pd.DataFrame) -> pd.DataFrame:
    mask = (df_evidence[evidence_cols.Potential_contaminant]
            == '+') | (df_evidence[evidence_cols.Intensity] == 0)
    evidence = df_evidence.loc[~mask].drop(
        evidence_cols.Potential_contaminant, axis=1)
    evidence = evidence.dropna(subset=[evidence_cols.Intensity])
    return evidence


idx_columns_evidence = [evidence_cols.Sequence, evidence_cols.Charge]


def create_parent_folder_name(folder: Path) -> str:
    return folder.stem[:4]


def dump_to_csv(df: pd.DataFrame,
                folder: Path,
                outfolder: Path,
                parent_folder_fct=None
                ) -> None:
    fname = f"{folder.stem}.csv"
    if parent_folder_fct is not None:
        outfolder = outfolder / parent_folder_fct(folder)
    outfolder.mkdir(exist_ok=True)
    fname = outfolder / fname
    logging.info(f"Dump to file: {fname}")
    df.to_csv(fname)


def load_process_evidence(folder: Path, use_cols, select_by):
    evidence = pd.read_table(folder / 'evidence.txt',
                             usecols=idx_columns_evidence + use_cols)
    evidence = select_evidence(evidence)
    evidence = vaep.pandas.select_max_by(
        evidence, index_columns=idx_columns_evidence, selection_column=select_by)
    evidence = evidence.sort_index()
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

    for folder in tqdm(folders):
        folder = Path(folder)
        evidence = load_process_evidence(
            folder=folder, use_cols=use_cols, select_by=select_by)
        c.update(evidence.index)
        if dump:
            dump_to_csv(evidence, folder=folder, outfolder=outfolder,
            parent_folder_fct=parent_folder_fct)
    return c


class EvidenceCounter(FeatureCounter):

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_evidence):
        super().__init__(fp_counter, counting_fct)

    def save(self):
        """Save state

        {
         'counter': Counter with tuple keys,
         'based_on': list
         }
        """
        d = {'counter': vaep.pandas.create_dict_of_dicts(self.counter),
             'based_on': list(self.loaded)}
        print(f"Save to: {self.fp}")
        dump_json(d, filename=self.fp)

    def load(self, fp):
        with open(self.fp) as f:
            d = json.load(f)
        d['counter'] = Counter(
            vaep.pandas.flatten_dict_of_dicts(d['counter']))
        return d

# Protein Groups


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
                      #index_col=pg_cols.Protein_IDs,
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
    pg = vaep.pandas.select_max_by(df=pg,
                                   index_columns=[pg_cols.Gene_names],
                                   selection_column=pg_cols.Score)
    pg = pg.reset_index().set_index(pg_cols.Protein_IDs)
    return pg


class Count():

    def __init__(self,
                 process_folder_fct: Callable,
                 use_cols=None,
                 parent_folder_fct: Callable = create_parent_folder_name,
                 outfolder=FOLDER_PROCESSED / 'dumps',
                 dump=True):
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

        for folder in tqdm(folders):
            folder = Path(folder)
            df = self.process_folder_fct(
                folder=folder, use_cols=self.use_cols, **fct_args)
            c.update(df.index)
            if self.dump:
                dump_to_csv(df, folder=folder, outfolder=self.outfolder,
                            parent_folder_fct=self.parent_folder_fct)
        return c


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
                             outfolder=FOLDER_PROCESSED / 'proteinGroups_dumps')


class ProteinGroupsCounter(FeatureCounter):

    def __init__(self, fp_counter: str,
                 counting_fct: Callable[[List], Counter] = count_protein_groups):
        super().__init__(fp_counter, counting_fct)
