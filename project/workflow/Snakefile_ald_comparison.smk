configfile: "config/appl_ald_data/plasma/proteinGroups/comparison.yaml"

folder_experiment = config['folder_experiment']

stem = folder_experiment+"/{out_folder}/{target}/{model}/"

target_cutoff = dict(kleiner="2")

rule all:
    input:
        expand([stem+"diff_analysis_comparision_2_{model}.pdf",
        stem+'mrmr_feat_by_model.xlsx'],
        target=['kleiner'],
        model=['vae', 'collab'],
        out_folder='diff_analysis')

nb='16_ald_compare_methods.ipynb'
rule compare_diff_analysis:
    input:
        nb=nb,
        scores=stem+"diff_analysis_scores.pkl"
    output:
        nb=stem+nb,
        figure=stem+"diff_analysis_comparision_2_{model}.pdf"
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r model_key {wildcards.model}"
        " && jupyter nbconvert --to html {output.nb}"

nb='16_ald_diff_analysis.ipynb'
rule differential_analysis:
    input:
        nb=nb,
        pred_real_na=folder_experiment+"/preds/pred_real_na_{model}.csv"
    output:
        scores=stem+"diff_analysis_scores.pkl",
        nb=stem+nb,
    params: covar = lambda wildcards: config["covar"][wildcards.target]
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r covar {params.covar}"
        " -r model_key {wildcards.model}"
        " && jupyter nbconvert --to html {output.nb}"

nb='16_ald_ml_new_feat.ipynb'
rule ml_comparison:
    input:
        nb=nb,
        fn_clinical_data="data/single_datasets/ald_metadata_cli.csv",
        
    output:
        sel_feat=stem+'mrmr_feat_by_model.xlsx',
        nb=stem+nb,
    params:
        model_key='{model}',
        # cutoff_target="2",
        cutoff=lambda wildcards: config["cutoffs"][wildcards.target],
        out_folder='{out_folder}',
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -p cutoff_target {params.cutoff}"
        " -r fn_clinical_data {input.fn_clinical_data}"
        " -r out_folder {params.out_folder}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"
