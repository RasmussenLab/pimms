# %%
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import vaep
from vaep.analyzers.compare_predictions import load_single_csv_pred_file

plt.rcParams['figure.figsize'] = (4, 2)
vaep.plotting.make_large_descriptors(5)


def load_pred_from_run(run_folder: Path,
                       model_keys,
                       template_pred='pred_real_na_{}.csv'):
    pred_real_na = dict()
    for method in model_keys:
        fname = run_folder / 'preds' / template_pred.format(method)
        print(f"missing values pred. by {method}: {fname}")
        pred_real_na[method] = load_single_csv_pred_file(fname)
    pred_real_na = pd.DataFrame(pred_real_na)
    return pred_real_na


# %%
reps_folder = 'runs/appl_ald_data/plasma/proteinGroups/reps'
template_pred = 'pred_real_na_{}.csv'  # fixed, do not change
model_keys = ['CF', 'DAE', 'KNN', 'Median', 'RSN', 'VAE', 'rf']


# %%
reps_folder = Path(reps_folder)
excel_out = reps_folder / 'imputed_stats.xlsx'
writer = pd.ExcelWriter(excel_out)

# %%
run_folders = [f for f in reps_folder.iterdir() if 'run_' in f.name]
run_folders

# %%
pred_real_na = pd.concat([load_pred_from_run(f, model_keys)
                         for f in run_folders])
pred_real_na.shape

# %%
pred_real_na = pred_real_na.groupby(
    by=list(pred_real_na.index.names)).agg(['mean', 'std'])
pred_real_na.to_excel(writer, float_format='%.3f', sheet_name='mean_std')
pred_real_na

# %%'
pred_real_na_cvs = pd.DataFrame()
for method in model_keys:
    pred_real_na_cvs[method] = pred_real_na[(
        method, 'std')] / pred_real_na[(method, 'mean')]

pred_real_na_cvs.to_excel(writer, float_format='%.3f', sheet_name='CVs')

ax = pred_real_na_cvs.plot.hist(bins=15,
                                color=vaep.plotting.defaults.assign_colors(model_keys),
                                alpha=0.5)
ax.yaxis.set_major_formatter('{x:,.0f}')
ax.set_xlabel(f'Coefficient of variation of imputed intensites (N={len(pred_real_na):,d})')
fname = reps_folder / 'pred_real_na_cvs.png'
vaep.savefig(ax.get_figure(), name=fname)

# %%
writer.close()
excel_out.as_posix()

# %%
