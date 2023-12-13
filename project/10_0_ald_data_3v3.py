# %%
from pathlib import Path
import pandas as pd

# %%
FN_INTENSITIES = "data/ALD_study/processed/ald_plasma_proteinGroups.pkl"
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"

FN_INTENSITIES = Path(FN_INTENSITIES)

# %%
df = pd.read_pickle(FN_INTENSITIES)
df

# %%
meta = pd.read_csv(fn_clinical_data, index_col=0)
meta

# %%
sel = pd.concat(
    [df.loc[meta['kleiner'] == 0].sample(3),
     df.loc[meta['kleiner'] == 4].sample(3),
     ])
sel

# %%
fname = FN_INTENSITIES.parent / f'{FN_INTENSITIES.stem}_3v3.pkl'
sel.to_pickle(fname)
fname.as_posix()

# %%
