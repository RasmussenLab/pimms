from collections import Counter
import os
import logging
from pathlib import Path
import multiprocessing
from types import SimpleNamespace

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

from vaep.io import dump_json
import vaep.io.mq as mq
from vaep.io.mq import MaxQuantOutputDynamic
from config import FOLDER_MQ_TXT_DATA, FOLDER_PROCESSED


# from vaep.cfg import DEFAULTS
DEFAULTS = SimpleNamespace()
DEFAULTS.ALL_SUMMARIES =  Path(FOLDER_PROCESSED) / 'all_summaries.json'
DEFAULTS.COUNT_ALL_PEPTIDES = FOLDER_PROCESSED / 'count_all_peptides.json'

N_WORKERS_DEFAULT = os.cpu_count() - 1 if os.cpu_count() <= 8 else 8 

logger = logging.getLogger(__name__)
logger.info(f"Calling from {__name__}")
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
            self.df = _convert_dtypes(pd.read_json(fp_summaries, orient='index'))
            print(f"{self.__class__.__name__}: Load summaries of {len(self.df)} folders.")
        else:
            if not fp_summaries.parent.exists():
                raise FileNotFoundError(f'Folder of filename not found: {fp_summaries.parent}')
            self.df = None
        self.fp_summaries = fp_summaries
        self.usecolumns= col_summary()
    
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        else:
            raise ValueError("No data loaded yet.")
    
    def load_summary(self, folder):
        folder_name = folder.stem
        try:
            mq_output = MaxQuantOutputDynamic(folder)
            return {folder_name : mq_output.summary.iloc[0].to_dict()}
        except FileNotFoundError as e:
            if not mq_output.files and len(list(mq_output.folder.iterdir())) == 0 :
                mq_output.folder.rmdir()
                logger.warning(f'Remove empty folder: {mq_output}')
                self.empty_folders.append(f"{folder_name}\n")
            else:
                logger.error(f"{mq_output}, No summary and not empty.")
        return {}
    
    def load_new_samples(self, folders, workers=N_WORKERS_DEFAULT):
        if self.df is not None:
            d_summaries = self.df.to_dict(orient='index')
            samples = set(folder.stem for folder in folders) - set(d_summaries.keys())
            samples = [folder for folder in folders if folder.stem in samples]
        else:
            d_summaries = {}
            samples = folders
        if not hasattr(self, 'empty_folders'):
            # should this depend on multiprocessing?
            self.empty_folders = manager.list()
        
        if samples:
            with multiprocessing.Pool(workers) as p:
                # set chunksize: https://stackoverflow.com/a/49533645/9684872
                chunksize = calc_chunksize(workers, len(samples), factor=2)
                list_of_updates = list(tqdm(p.imap(self.load_summary, samples, chunksize=chunksize), total=len(samples), desc='Load summaries'))
                
            print("Newly loaded samples:", len(list_of_updates))

            for d in list_of_updates:
                d_summaries.update(d)
            
            self.df = _convert_dtypes(pd.DataFrame.from_dict(d_summaries, orient='index'))
            self.save_state()
        else:
            print("Now new sample added.")
        return self.df
    
    def save_state(self):
        """Save summaries DataFrame as json and pickled object."""
        self.df.to_json(self.fp_summaries, orient='index')
        self.df.to_pickle(self.fp_summaries.parent / f"{self.fp_summaries.stem}.pkl")
        
    def get_files_w_min_MS2(self, threshold=10_000, relativ_to=FOLDER_MQ_TXT_DATA):
        """Get a list of file ids with a minimum MS2 observations."""
        threshold_ms2_identified = threshold
        mask  = self.df[self.usecolumns.MS2] > threshold_ms2_identified
        print(f"Selected  {mask.sum()} of {len(mask)} folders.")
        return [Path(FOLDER_MQ_TXT_DATA) / folder for folder in self.df.loc[mask].index]
    
class PeptideCounter():
    def __init__(self, fp_count_all_peptides=DEFAULTS.COUNT_ALL_PEPTIDES):
        self.fp = fp_count_all_peptides
        
    # # add directly dumping of folder? unique peptides?    
    # # reduce storage for potential download? which columns to retain
    # # check df for redundant information (same feature value for all entries)
    def count_peptides(self, folders):
        c = Counter()
        for folder in folders:
            peptides = pd.read_table(folder / 'peptides.txt',
                                     usecols=[mq.mq_col.SEQUENCE, mq.mq_col.INTENSITY, "Potential contaminant"],
                                     index_col=0)
            mask = (peptides[mq.mq_col.INTENSITY] == 0) | (peptides["Potential contaminant"] == '+')
            c.update(peptides.loc[~mask, mq.mq_col.INTENSITY].index)

        return c

    # combine multiprocessing into base class?
    def sum_over_files(self, folders, n_workers=N_WORKERS_DEFAULT):
        with multiprocessing.Pool(n_workers) as p:
            list_of_sample_dicts = list(tqdm(p.imap(self.count_peptides, np.array_split(folders, 100)), 
                                             total=100, desc='Count peptides in 100 chunks'))

        c = Counter()
        for d in tqdm(list_of_sample_dicts, desc='combine counters from chunks'):
            c += d
        self.counter = c
        return self.counter

    def save(self):
        print(f"Save to: {self.fp}")
        dump_json(self.counter, filename=self.fp)
