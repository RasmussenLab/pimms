# %%
from itertools import product

# import subprocess
mnar_mcar = [25, 50, 75]
datasets = ["pg_m", "pg_l", "pep_m", "evi_m", "pep_l", "evi_l"]

for dataset, perc in product(datasets, mnar_mcar):
    print(f"# {dataset  = } # {perc = }")
    cmd = (
        "qsub bin/run_snakemake_cluster.sh"
        f" -N sm_{dataset}_{perc}"
        f" -v configfile=config/single_dev_dataset/mnar_mcar/{dataset}.yaml,prefix={dataset}_{perc},"
        f"frac_mnar={perc/100:.2f},"
        f"config_split=runs/mnar_mcar/{dataset}_{perc}MNAR/01_0_split_data.yaml,"
        f"config_train=runs/mnar_mcar/{dataset}_{perc}MNAR/train_{{model}}.yaml,"
        f"folder_experiment=runs/mnar_mcar/{dataset}_{perc}MNAR"
    )
    print(cmd)
    # subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE)

# %% [markdown]
# Create local command to run on interactive node
print()
print("#" * 80)
print()
# %%
for dataset, perc in product(datasets, mnar_mcar):
    cmd = (
        "snakemake -s workflow/Snakefile_v2.smk"
        f" --configfile config/single_dev_dataset/mnar_mcar/{dataset}.yaml"
        f" --config frac_mnar={perc/100:.2f}"
        f" config_split=runs/mnar_mcar/{dataset}_{perc}MNAR/01_0_split_data.yaml"
        f" config_train=runs/mnar_mcar/{dataset}_{perc}MNAR/train_{{model}}.yaml"
        f" folder_experiment=runs/mnar_mcar/{dataset}_{perc}MNAR"
        " -c1"
    )
    print(cmd)
# %%
