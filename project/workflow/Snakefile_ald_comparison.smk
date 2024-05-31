""" Snakemake file for the ALD study workflow

- needs that data was created (could be added)
- performs differential analysis for a set of methods/models.
  - currently need to be explicitly specified
- select one base model for comparison (RSN was used in original study)
- plot observed and imputed values for selected features with different diff. analysis
  outcomes (not all the same)
"""


configfile: "config/appl_ald_data/plasma/proteinGroups/comparison.yaml"


folder_experiment = config["folder_experiment"]

out_folder = folder_experiment + "/{out_folder}/{target}/"

out_folder_two_methods_cp = out_folder + "{baseline}_vs_{model}/"

target = config["target"]

all_methods = [config["baseline"], "None", *config["methods"]]


wildcard_constraints:
    target=target,
    baseline=config["baseline"],
    out_folder=config["out_folder"],
    model="|".join(all_methods),


rule all:
    input:
        expand(
            out_folder + "diff_analysis_compare_DA.xlsx",
            target=target,
            out_folder=config["out_folder"],
        ),
        expand(
            [
                out_folder_two_methods_cp + "diff_analysis_comparision_2_{model}.pdf",
                out_folder_two_methods_cp + "mrmr_feat_by_model.xlsx",
            ],
            target=[target],
            baseline=config["baseline"],
            model=config["methods"],
            out_folder=config["out_folder"],
        ),


##########################################################################################
# Create plots for featues where decisions between model differ (if computed)

nb = "10_4_ald_compare_single_pg.ipynb"


rule plot_intensities_for_diverging_results:
    input:
        expand(
            folder_experiment + "/preds/pred_real_na_{method}.csv",
            method=[config["baseline"], *config["methods"]],
        ),
        expand(
            [
                out_folder + "scores/diff_analysis_scores_{model}.pkl",
            ],
            target=[target],
            baseline=config["baseline"],
            model=all_methods,
            out_folder=config["out_folder"],
        ),
        nb=nb,
        # replace with config
        fn_clinical_data=f"{folder_experiment}/data/clinical_data.csv",
    output:
        diff_da=out_folder + "diff_analysis_compare_DA.xlsx",
        qvalues=out_folder + "qvalues_target.pkl",
        nb=out_folder + nb,
    params:
        baseline=config["baseline"],
        cutoff=lambda wildcards: config["cutoffs"][wildcards.target],
        make_plots=config["make_plots"],
        ref_method_score=config["ref_method_score"],  # None, 
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r baseline {params.baseline}"
        " -r out_folder {wildcards.out_folder}"
        " -p cutoff_target {params.cutoff}"
        " -p make_plots {params.make_plots}"
        " -p ref_method_score {params.ref_method_score}"
        " -r fn_clinical_data {input.fn_clinical_data}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# Compare performance of models (methods) on prediction task
nb = "10_3_ald_ml_new_feat.ipynb"


rule ml_comparison:
    input:
        nb=nb,
        pred_base=folder_experiment + "/preds/pred_real_na_{baseline}.csv",
        pred_model=folder_experiment + "/preds/pred_real_na_{model}.csv",
        fn_clinical_data=f"{folder_experiment}/data/clinical_data.csv",
    output:
        sel_feat=out_folder_two_methods_cp + "mrmr_feat_by_model.xlsx",
        nb=out_folder_two_methods_cp + nb,
    params:
        cutoff=lambda wildcards: config["cutoffs"][wildcards.target],
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r baseline {wildcards.baseline}"
        " -r model_key {wildcards.model}"
        " -r out_folder {wildcards.out_folder}"
        " -p cutoff_target {params.cutoff}"
        " -r fn_clinical_data {input.fn_clinical_data}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# basemethod vs other methods
nb = "10_2_ald_compare_methods.ipynb"
nb_stem = "10_2_ald_compare_methods"


rule compare_diff_analysis:
    input:
        nb=nb,
        score_base=out_folder + "scores/diff_analysis_scores_{baseline}.pkl",
        score_model=out_folder + "scores/diff_analysis_scores_{model}.pkl",
    output:
        nb=out_folder_two_methods_cp + nb,
        figure=out_folder_two_methods_cp + "diff_analysis_comparision_2_{model}.pdf",
    params:
        disease_ontology=lambda wildcards: config["disease_ontology"][wildcards.target],
        annotaitons_gene_col=config["annotaitons_gene_col"],
    benchmark:
        out_folder_two_methods_cp + f"{nb_stem}.tsv"
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r baseline {wildcards.baseline}"
        " -r model_key {wildcards.model}"
        " -r out_folder {wildcards.out_folder}"
        " -p disease_ontology {params.disease_ontology}"
        " -r annotaitons_gene_col {params.annotaitons_gene_col}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# Scores for each model (method)
nb_stem = "10_1_ald_diff_analysis"


rule differential_analysis:
    input:
        nb=f"{nb_stem}.ipynb",
        fn_clinical_data=f"{folder_experiment}/data/clinical_data.csv",
    output:
        score=out_folder + "scores/diff_analysis_scores_{model}.pkl",
        nb=out_folder + f"scores/{nb_stem}_{{model}}.ipynb",
    params:
        covar=lambda wildcards: config["covar"][wildcards.target],
        f_annotations=config["f_annotations"],
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r fn_clinical_data {input.fn_clinical_data}"
        " -p f_annotations {params.f_annotations}"
        " -r target {wildcards.target}"
        " -r covar {params.covar}"
        " -r model_key {wildcards.model}"
        " -r out_folder {wildcards.out_folder}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# Save clinical metadata to data folder of experimental folder
# Makes it possible to have remote clincial data

rule copy_clinical_data:
    output:
        local_clincial_data = f"{folder_experiment}/data/clinical_data.csv",
    params:
        fn_clinical_data = config["fn_clinical_data"],
    run:
        import pandas as pd
        # could be extended for several file-types
        df = pd.read_csv(params.fn_clinical_data)
        df.to_csv(output.local_clincial_data, index=False)
        # , index_col=0)             
        # usecols=[args.sample_id_col, args.target])
