# %% [markdown]
# # Check if filesizes of local and uploaded files match
# - could be replaced with checksums, but it's too slow on erda
# - instead: compare if filesizes in bytes more or less match (tolerance of 5 bytes)
#
# many things could be refactored in case a tool should be created from this

# %%
from collections import namedtuple
from pathlib import Path, PurePosixPath
import pandas as pd


# %%
# Parameters
FOLDER = Path('data/rename')
fname_mq_out_pride = FOLDER / 'mq_out_filesizes_pride.log'
fname_mq_out_erda = FOLDER / 'mq_out_filesizes_erda.log'
fname_rawfiles_pride = FOLDER / 'rawfiles_filesizes_pride.log'
fname_rawfiles_erda = FOLDER / 'rawfiles_filesizes_erda.log'
fname_filenames_mapping = FOLDER / 'selected_old_new_id_mapping.csv'


# %%
df_meta = pd.read_csv(fname_filenames_mapping, index_col='Path_old')
df_meta

# %%
df_meta['path_pride'] = 'raw_files/' + df_meta['Instrument_name'] + '/' + df_meta["new_sample_id"] + '.raw'

# %%
entries = list()
Entry = namedtuple('Entry', 'size_erda fname name_erda')
with open(fname_rawfiles_erda) as f:
    for line in f:
        size, fname = line.strip().split('\t')
        fname = PurePosixPath(fname)
        if fname.suffix:
            entry = Entry(int(size), str(fname).replace('share_hela_raw/', './'), fname.name)
            if entry.fname in df_meta.index:
                entries.append(entry)
print(f"{len(entries) =: }")
entries[:3]

# %%
entries = pd.DataFrame(entries).set_index('fname')
entries = (entries
           .join(df_meta.loc[entries.index, 'path_pride'])
           .reset_index()
           .set_index('path_pride')
           .sort_index())


# %%
entries_pride = list()
Entry = namedtuple('Entry', ['size_pride', 'path_pride', 'name_pride', 'instrument'])
with open(fname_rawfiles_pride) as f:
    for line in f:
        size, fname = line.strip().split()
        fname = PurePosixPath(fname)
        if fname.suffix:
            entry = Entry(int(size), str(fname), fname.name, fname.parent.name)
            entries_pride.append(entry)
print(f"{len(entries_pride) =: }")
entries_pride[:3]

# %%
entries_pride = pd.DataFrame(entries_pride).set_index('path_pride').sort_index()
entries_pride

# %%
entries = entries.join(entries_pride, on='path_pride', how='left')

# %%
mask = (entries['size_pride'] - entries['size_erda']).abs() > 5
to_redo = entries.loc[mask].reset_index()
to_redo

# %%
commands = 'put ' + to_redo['fname'] + ' -o ' + to_redo['path_pride']
print(commands.to_csv(header=False, index=False))

# %% [markdown]
# ## Check MaxQuant output filesizes

# %%
df_meta = df_meta.reset_index().set_index('Sample ID')


# %%
files = list()
folder = set()
Entry = namedtuple('Entry', 'size_erda path_erda id_old filename')
with open(fname_mq_out_erda) as f:
    for line in f:
        size, fname = line.strip().split('\t')
        fname = PurePosixPath(fname)
        if fname.suffix and fname.suffix != '.pdf':
            entry = Entry(int(size), str(fname), fname.parent.name, fname.name)
            if entry.id_old in df_meta.index:
                files.append(entry)
                if entry.id_old not in folder:
                    folder.add(entry.id_old)

print(f"{len(folder) =: }")
print(f"{len(files) =: }")
files[:3]

# %%
files = pd.DataFrame(files).set_index('id_old')
files = files.join(df_meta[['Instrument_name', 'new_sample_id']])
files

# %%
files['path_pride'] = ('MQ_tables/'
                       + files['Instrument_name']
                       + '/'
                       + files["new_sample_id"]
                       + '/'
                       + files["filename"])
files['path_pride'].iloc[:4].to_list()


# %%
files['filename'].value_counts()  # except mqpar.xml all present on erda

# %%
files_pride = list()
Entry = namedtuple('Entry', ['size_pride', 'path_pride', 'id_new', 'instrument'])
with open(fname_mq_out_pride) as f:
    for line in f:
        size, fname = line.strip().split('\t')
        fname = PurePosixPath(fname)
        if fname.suffix:
            entry = Entry(int(size), str(fname), fname.parent.name, fname.parent.parent.name)
            files_pride.append(entry)
print(f"{len(files_pride) =: }")
files_pride[:3]

# %%
files_pride = pd.DataFrame(files_pride).set_index('path_pride')
files_pride
# %%
files = files.set_index('path_pride').join(files_pride, how='left')
# %%
missing_on_pride = files.loc[files['size_pride'].isna()]
missing_on_pride
# %%
missing_on_pride['filename'].value_counts()


# %%
files['size_diff'] = files['size_pride'] - files['size_erda']
files['size_diff'].abs().describe()
# %%
files_redo = files.loc[files['size_diff'].abs() > 5]
files_redo

# %% [markdown]
# ensure quoted paths as they might contain whitespaces

# %%
to_do = pd.concat([missing_on_pride, files_redo])
commands = 'put -e \'' + to_do['path_erda'] + "' -o '" + to_do.index + "'"
commands.to_csv(FOLDER / 'mq_out_remaining.txt', header=False, index=False)
