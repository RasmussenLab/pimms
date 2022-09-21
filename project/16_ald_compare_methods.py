# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare outcomes from differential analysis based on different imputation methods
#
# - load scores based on `16_ald_diff_analysis`

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import vaep
logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"
model_key = 'vae'
target = 'kleiner'
out_folder='diff_analysis'

# %%
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args, out_root=args.folder_experiment/params["out_folder"]/params["target"]/params["model_key"])
args.update_from_dict(params)
args

# %% [markdown]
# # Load scores 

# %%
[x for x in args.out_folder.iterdir() if 'scores' in str(x)]

# %%
fname = args.out_folder / f'diff_analysis_scores.pkl'
fname

# %%
scores = pd.read_pickle(fname)
scores

# %%
models = vaep.nb.Config.from_dict(vaep.pandas.index_to_dict(scores.columns.levels[0]))
vars(models)

# %%
assert args.model_key in models.keys(), f"Missing model key which was expected: {args.model_key}"

# %%
scores.describe()

# %%
scores = scores.loc[pd.IndexSlice[:, args.target], :]
scores

# %%
scores.describe()

# %% [markdown]
# ## Load frequencies of observed features

# %%
fname = args.folder_experiment / 'freq_features_observed.csv'
freq_feat = pd.read_csv(fname, index_col=0)
freq_feat

# %% [markdown]
# # Compare shared features

# %%
scores_common = scores.dropna().reset_index(-1, drop=True)
scores_common


# %%
def annotate_decision(scores, model):
    return scores[(model, 'rejected')].replace({False: f'{model} ->  no', True: f'{model} -> yes'})

annotations = None
for model, model_column in models.items():
    if not annotations is None:
        annotations += ' - '
        annotations += scores_common[(model_column, 'rejected')].replace({False: f'{model} ->  no', True: f'{model} -> yes'})
    else:
        annotations= scores_common[(model_column, 'rejected')].replace({False: f'{model} ->  no', True: f'{model} -> yes'})
annotations.name = 'Differential Analysis Comparison'
annotations.value_counts()

# %%
mask_different = ( (scores_common.loc[:, pd.IndexSlice[:, 'rejected']].any(axis=1)) & 
 ~(scores_common.loc[:, pd.IndexSlice[:, 'rejected']].all(axis=1))
)

scores_common.loc[mask_different]

# %%
fname = args.out_folder / f'diff_analysis_differences.xlsx'
scores_common.loc[mask_different].to_excel(fname)
fname

# %%
var = 'qvalue'
to_plot = [scores_common[v][var] for k,v in models.items()]
for s, k in zip(to_plot, models.keys()):
    s.name = k.replace('_', ' ') 
to_plot.append(freq_feat.loc[scores_common.index])
to_plot.append(annotations)
to_plot = pd.concat(to_plot, axis=1)
to_plot

# %% [markdown]
# ## Differences plotted
#
# - first only using created annotations

# %%
figsize = (10, 10)
fig, ax = plt.subplots(figsize=figsize)
x_col = to_plot.columns[0]
y_col = to_plot.columns[1]
ax = sns.scatterplot(data=to_plot,
                     x=x_col,
                     y=y_col,
                     hue='Differential Analysis Comparison',
                     ax=ax)
ax.set_xlabel(f"qvalue for {x_col}")
ax.set_ylabel(f"qvalue for {y_col}")
ax.hlines(0.05, 0, 1, color='grey', linestyles='dotted')
ax.vlines(0.05, 0, 1, color='grey', linestyles='dotted')
sns.move_legend(ax, "upper center")
fname = args.out_folder / f'diff_analysis_comparision_1_{args.model_key}'
vaep.savefig(fig, name=fname)

# %% [markdown]
# - showing how many features were measured ("observed")

# %%
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=to_plot, x=to_plot.columns[0], y=to_plot.columns[1],
                     palette='Set2',
                     size='frequency', hue='Differential Analysis Comparison')
ax.set_xlabel(f"qvalue for {x_col}")
ax.set_ylabel(f"qvalue for {y_col}")
ax.hlines(0.05, 0, 1, color='grey', linestyles='dotted')
ax.vlines(0.05, 0, 1, color='grey', linestyles='dotted')
sns.move_legend(ax, "upper center")
fname = args.out_folder / f'diff_analysis_comparision_2_{args.model_key}'
vaep.savefig(fig, name=fname)

# %% [markdown]
# # Only features contained in model

# %%
scores_model_only = scores.reset_index(level=-1, drop=True)
scores_model_only = (scores_model_only
                     .loc[
                         scores_model_only.index.difference(scores_common.index),
                         args.model_key]
                     .sort_values(by='qvalue', ascending=True)
                     .join(freq_feat)
                     )
scores_model_only

# %%
scores_model_only.rejected.value_counts()

# %%
fname = args.out_folder / 'diff_analysis_only_model.xlsx'
scores_model_only.to_excel(fname)
fname

# %% [markdown] tags=[]
# # Feature lookup
#
# - [x] look-up ids and diseases, manually (uncomment)
# - [ ] automatically by querying `api.jensenlab.org`: see if disease (`DOID` needed) has associations to gene found

# %%
# from IPython.display import IFrame
# display(IFrame('https://www.uniprot.org/', width=900,height=500))

# %%
# # %%html
# <iframe 
#   style="transform-origin: 0px 0px 0px; transform: scale(1.5); width: 600px; height: 600px;" 
#   src='https://diseases.jensenlab.org/Search'
#   name="iFrame"
#   scrolling="no">
# </iframe>
