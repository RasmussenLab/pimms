configfile: "config/appl_ald_data/plasma/proteinGroups/comparison.yaml"

folder_experiment = config['folder_experiment']

stem = folder_experiment+"/diff_analysis/{target}/{model}/"

rule all:
    input:
        expand(stem+"diff_analysis_comparision_2_{model}.pdf",
        target=['kleiner'],
        model=['vae'])

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
        #/diff_analysis_differences.xlsx",
        # "diff_analysis_only_model.xlsx"
    params: covar = lambda wildcards: config["covar"][wildcards.target]
    shell:
        "papermill {input.nb} {output.nb}"
        f" -r folder_experiment {folder_experiment}"
        " -r target {wildcards.target}"
        " -r covar {params.covar}"
        " -r model_key {wildcards.model}"
        " && jupyter nbconvert --to html {output.nb}"


