# %% [markdown]
# # Rename file names in MaxQuant output files
# renaming the folder of outputs does not delete all occurences of the names
# in the text files. This needs to be done manually by the PRIDE team using a shell script
# that uses `sed` to replace the old names with the new ones.
#
# uses the list of output as stored on pride dropbox server and meta data of old and new name

# %%
from collections import defaultdict
from pathlib import Path, PurePosixPath
import pandas as pd


# %%
FOLDER = Path('data/rename')
meta_in = FOLDER / 'selected_old_new_id_mapping.csv'
fn_server_log: str = 'data/rename/mq_out_server.log'  # server log of all uploaded files

# %%
df_meta = pd.read_csv(meta_in, index_col='new_sample_id')
df_meta

# %% [makrdown]
# ## Create commands to rename file names in text files itself
# - only subset of files contain original file names on exection of MaxQuant

# %%
files_types = ["modificationSpecificPeptides.txt",
               "mqpar.xml",
               "mzRange.txt",
               "Oxidation (M)Sites.txt",
               "summary.txt",]

# %%
name_lookup = df_meta["Sample ID"].reset_index().set_index("new_sample_id")
name_lookup

# %%
to_rename = list()
command_template = 'sed -i "s/{old_name}/{new_name}/g" "{fn}"'
counter = defaultdict(int)

with open(fn_server_log) as f:
    for line in f:
        fname = PurePosixPath(line.strip())
        if fname.name in files_types:
            new_name = fname.parent.name
            old_name = name_lookup.loc[new_name, 'Sample ID']
            command = command_template.format(old_name=old_name,
                                              new_name=new_name,
                                              fn=fname)
            to_rename.append(command)

            counter[fname.name] += 1
len(to_rename)

# %%
# mqpar.xml missing in some folders
pd.Series(counter)  # maybe one folder has some missing?

# %%
with open(FOLDER / 'sed_rename_commands.sh', 'w') as f:
    f.writelines('\n'.join(to_rename))
