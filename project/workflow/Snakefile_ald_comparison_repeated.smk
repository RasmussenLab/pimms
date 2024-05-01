"""
Try to execute several time the same Snakemake workflow using another Snakemake workflow.

- one by one? (-> one process at a time?)
"""
folder_experiment = "runs/appl_ald_data_2023_11/reps/plasma/proteinGroups"
folder_run = folder_experiment + "/run_{run}"
out_folder = folder_run + "/{sub_folder}/{target}"

target = "kleiner"
sub_folder = "diff_analysis"
N = 10
make_plots = False


rule all:
    input:
        f"{folder_experiment}/agg_differences_compared.xlsx",


rule compare_repetitions:
    input:
        qvalues=expand(
            f"{out_folder}/qvalues_target.pkl",
            target=target,
            sub_folder=sub_folder,
            run=range(N),
        ),
        equality_rejected_target=expand(
            f"{out_folder}/equality_rejected_target.pkl",
            target=target,
            sub_folder=sub_folder,
            run=range(N),
        ),
    output:
        f"{folder_experiment}/agg_differences_compared.xlsx",
    log:
        notebook=f"{folder_experiment}/10_5_comp_diff_analysis_repetitions.ipynb",
    params:
        folder_experiment=folder_experiment,
    notebook:
        "../10_5_comp_diff_analysis_repetitions.ipynb"


rule run_comparison_workflow:
    input:
        f"{folder_run}/figures/2_1_test_errors_binned_by_feat_medians.pdf",
    output:
        excel=f"{out_folder}/equality_rejected_target.pkl",
        qvalues=f"{out_folder}/qvalues_target.pkl",
    params:
        workflow="workflow/Snakefile_ald_comparison.smk",
        folder_experiment=folder_run,
    shell:
        "snakemake -s workflow\Snakefile_ald_comparison.smk"
        " --config folder_experiment={params.folder_experiment}"
        f" make_plots={make_plots}"
        " --drop-meta"
        " -p"
        " -c1"


rule run_models:
    output:
        f"{folder_run}/figures/2_1_test_errors_binned_by_feat_medians.pdf",
    params:
        configfile="config/appl_ald_data/plasma/proteinGroups/config_reps.yaml",
        folder_experiment=folder_run,
    shell:
        "snakemake --configfile {params.configfile}"
        " --config folder_experiment={params.folder_experiment}"
        " --drop-meta"
        " -p"
        " -c1"
