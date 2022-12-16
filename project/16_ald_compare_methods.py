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
import vaep.databases.diseases
logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
folder_experiment = 'runs/appl_ald_data/plasma/proteinGroups'
model_key = 'vae'
target = 'kleiner'
out_folder = 'diff_analysis'

disease_ontology = 5082  # code from https://disease-ontology.org/
f_annotations = 'data/single_datasets/ald_plasma_proteinGroups_annotations.csv' # snakemake -> copy to experiment folder
annotaitons_gene_col = 'PG.Genes'

# %%
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args, out_root=args.folder_experiment /
                                 params["out_folder"]/params["target"]/params["model_key"])
args.update_from_dict(params)
args

# %%
files_in = {'freq_features_observed.csv': args.folder_experiment / 'freq_features_observed.csv',
            'diff_analysis_scores.pkl': args.out_folder / f'diff_analysis_scores.pkl'}
files_in

# %% [markdown]
# ## Excel file for exports

# %%
files_out = dict()

# %%
writer_args = dict(float_format='%.3f')

files_out['diff_analysis_compare_methods.xlsx'] = (args.out_folder /
                                                   f'diff_analysis_compare_methods.xlsx')
writer = pd.ExcelWriter(files_out['diff_analysis_compare_methods.xlsx'])

# %% [markdown]
# # Load scores 

# %%
[x for x in args.out_folder.iterdir() if 'scores' in str(x)]

# %%
scores = pd.read_pickle(files_in['diff_analysis_scores.pkl'])
scores

# %%
# ToDo: change in library
names = {'vae': 'VAE', 'random shifted_imputation': 'RSN'}
scores = scores.rename(names, axis=1)
scores.head()

# %%
models = vaep.nb.Config.from_dict(
    vaep.pandas.index_to_dict(scores.columns.levels[0]))
vars(models)

# %%
assert args.model_key.upper() in models.keys(
), f"Missing model key which was expected: {args.model_key}"

# %%
scores.describe()

# %%
scores = scores.loc[pd.IndexSlice[:, args.target], :]
scores.to_excel(writer, 'scores', **writer_args)
scores

# %%
scores.describe()

# %%
scores.describe(include=['bool', 'O'])

# %% [markdown]
# ## Load frequencies of observed features

# %%
freq_feat = pd.read_csv(files_in['freq_features_observed.csv'], index_col=0)
freq_feat

# %% [markdown]
# # Compare shared features

# %%
scores_common = scores.dropna().reset_index(-1, drop=True)
scores_common[('data', 'freq')] = freq_feat
scores_common


# %%
def annotate_decision(scores, model, model_column):
    return scores[(model_column, 'rejected')].replace({False: f'{model} (no) ', True: f'{model} (yes)'})


annotations = None
for model, model_column in models.items():
    if not annotations is None:
        annotations += ' - '
        annotations += annotate_decision(scores_common,
                                         model=model, model_column=model_column)
    else:
        annotations = annotate_decision(
            scores_common, model=model, model_column=model_column)
annotations.name = 'Differential Analysis Comparison'
annotations.value_counts()

# %%
mask_different = ((scores_common.loc[:, pd.IndexSlice[:, 'rejected']].any(axis=1)) &
                  ~(scores_common.loc[:, pd.IndexSlice[:, 'rejected']].all(
                      axis=1))
                  )

scores_common.loc[mask_different]

# %%
scores_common.loc[mask_different].to_excel(
    writer, 'differences', **writer_args)

# %%
var = 'qvalue'
to_plot = [scores_common[v][var] for k, v in models.items()]
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
figsize = (8, 8)
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
sns.move_legend(ax, "upper right")
files_out[f'diff_analysis_comparision_1_{args.model_key}'] = (args.out_folder /
                                                              f'diff_analysis_comparision_1_{args.model_key}')
fname = files_out[f'diff_analysis_comparision_1_{args.model_key}']
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
sns.move_legend(ax, "upper right")
files_out[f'diff_analysis_comparision_2_{args.model_key}'] = (
    args.out_folder / f'diff_analysis_comparision_2_{args.model_key}')
vaep.savefig(
    fig, name=files_out[f'diff_analysis_comparision_2_{args.model_key}'])

# %% [markdown]
# # Only features contained in model

# %%
scores_model_only = scores.reset_index(level=-1, drop=True)
scores_model_only = (scores_model_only
                     .loc[
                         scores_model_only.index.difference(
                             scores_common.index),
                         args.model_key.upper()]
                     .sort_values(by='qvalue', ascending=True)
                     .join(freq_feat)
                     )
scores_model_only

# %%
scores_model_only.rejected.value_counts()

# %%
scores_model_only.to_excel(writer, 'only_model', **writer_args)

# %%
scores_model_only_rejected = scores_model_only.loc[scores_model_only.rejected]
scores_model_only_rejected.to_excel(
    writer, 'only_model_rejected', **writer_args)

# %% [markdown] tags=[]
# # Feature lookup

# %%
data = vaep.databases.diseases.get_disease_association(
    doid=args.disease_ontology, limit=10000)
data = pd.DataFrame.from_dict(data, orient='index').rename_axis('ENSP', axis=0)
data = data.rename(columns={'name': args.annotaitons_gene_col}).reset_index(
).set_index(args.annotaitons_gene_col)
data

# %% [markdown]
# ## Shared features

# %%
feat_name = scores_common.index.name
annotations = pd.read_csv(args.f_annotations, usecols=[
                          feat_name, args.annotaitons_gene_col])
annotations = annotations.drop_duplicates().set_index(args.annotaitons_gene_col)
annotations

# %%
disease_associations_all = data.join(
    annotations).dropna().reset_index().set_index(feat_name)
disease_associations_all

# %% [markdown]
# ## only by model

# %%
idx = disease_associations_all.index.intersection(scores_model_only.index)
disease_assocications_new = disease_associations_all.loc[idx].sort_values(
    'score', ascending=False)
disease_assocications_new.head(20)

# %%
mask = disease_assocications_new.loc[idx, 'score'] >= 2.0
disease_assocications_new.loc[idx].loc[mask]

# %% [markdown]
# ## Only by model which were significant

# %%
idx = disease_associations_all.index.intersection(
    scores_model_only_rejected.index)
disease_assocications_new_rejected = disease_associations_all.loc[idx].sort_values(
    'score', ascending=False)
disease_assocications_new_rejected.head(20)

# %%
mask = disease_assocications_new_rejected.loc[idx, 'score'] >= 2.0
disease_assocications_new_rejected.loc[idx].loc[mask]

# %% [markdown]
# ## Write to excel

# %%
disease_associations_all.to_excel(
    writer, sheet_name='disease_assoc_all', **writer_args)
disease_assocications_new.to_excel(
    writer, sheet_name='disease_assoc_new', **writer_args)
disease_assocications_new_rejected.to_excel(
    writer, sheet_name='disease_assoc_new_rejected', **writer_args)

# %% [markdown]
# # Outputs

# %%
files_out
