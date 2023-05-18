"""Workflow to produce global analysis for the project."""
nb_outfolder = "runs"

DATASETS=[ "df_intensities_proteinGroups_long",
           "df_intensities_peptides_long",
           "df_intensities_evidence_long"
           ]

OUT_INFO = "dataset_info"

rule all:
    input:
        "data/files_per_instrument.yaml",  # nested: model, attribute, serial number
        "data/files_selected_metadata.csv",
        "data/files_selected_per_instrument.yaml",
        "data/files_selected_per_instrument_counts.csv",  # counts
        f'{nb_outfolder}/{"00_2_hela_all_raw_files.ipynb"}',
        "data/samples_selected.yaml",
        expand(
            "data/single_datasets/{dataset}/{OUT_INFO}.json",
            dataset=DATASETS,
            OUT_INFO=OUT_INFO,
        ),


nb = "00_2_hela_all_raw_files.ipynb"


rule metadata:
    input:
        nb=nb,
        # meta='../workflows/metadata/rawfile_metadata.json',
        meta="data/all_raw_files_dump_2021_10_29.txt",
        summaries="data/processed/all_summaries.json",
        intensities="data/df_intensities_N07285_M01000",  # csv
    output:
        nb=f"{nb_outfolder}/{nb}",
        # # final config
        # {'FN_ALL_RAW_FILES': 'data/all_raw_files_dump_2021_10_29.txt', # input
        # 'FN_ALL_SUMMARIES': 'data/processed/all_summaries.json', # input
        # 'FN_ALL_RAW_FILES_UNIQUE': 'data/all_raw_files_dump_2021_10_29_unique_N50521_M00003.csv',
        # 'FN_ALL_RAW_FILES_DUPLICATED': 'data/all_raw_files_dump_2021_10_29_duplicated.txt',
        # 'raw_file_overview': 'Figures/raw_file_overview.pdf',
        # 'fname_1000_most_common_peptides': 'data/df_intensities_N07285_M01000',
        # 'figure_1': 'Figures/figure_1.pdf',
        # 'remote_files': 'data/remote_files.yaml'}
    shell:
        "papermill {input.nb} {output.nb}"
        " -p FN_ALL_RAW_FILES {input.meta}"
        " -p FN_ALL_SUMMARIES {input.summaries}"
        " -p FN_PEPTIDE_INTENSITIES {input.intensities}"
        " && jupyter nbconvert --to html {output.nb}"


nb = "00_1_hela_MQ_summaries.ipynb"


rule summaries:
    input:
        nb=nb,
        summaries="data/processed/all_summaries.json",
    output:
        nb=f"{nb_outfolder}/{nb}",
        selected="data/samples_selected.yaml",
    shell:
        "papermill {input.nb} {output.nb}"
        " -r FN_ALL_SUMMARIES {input.summaries}"
        " && jupyter nbconvert --to html {output.nb}"


nb = "00_0_hela_metadata_rawfiles.ipynb"


rule metadata_rawfiles:
    input:
        "data/rawfile_metadata.csv",
        "data/samples_selected.yaml",
        nb=nb,
    output:
        "data/files_per_instrument.yaml",  # nested: model, attribute, serial number
        "data/files_selected_metadata.csv",
        "data/files_selected_per_instrument.yaml",
        "data/files_selected_per_instrument_counts.csv",  # counts
        nb=f"{nb_outfolder}/{nb}",
    shell:
        "papermill {input.nb} {output.nb}"
        " && jupyter nbconvert --to html {output.nb}"  # run with defaults



nb='00_3_hela_development_dataset_splitting.ipynb'
outfolder=f'dev_datasets'
ROOT_DUMPS = "C:/Users/kzl465/OneDrive - University of Copenhagen/vaep/project/data"


rule split_data:
    input:
        nb=nb,
        data=f"{ROOT_DUMPS}/{{dataset}}.pkl",
    output:
        nb=f"data/dev_datasets/{{dataset}}/{nb}",
        json=f'data/dev_datasets/{{dataset}}/{OUT_INFO}.xlsx'
    params:
        folder_datasets="single_datasets/{dataset}",
    shell:
        # papermill parameters with whitespaces > 
        "papermill {input.nb} {output.nb}"
        ' -r DUMP "{input.data}" '
        " -r FILE_EXT pkl"
        " -r FOLDER_DATASETS {params.folder_datasets}"
        ' -r SAMPLE_ID "Sample ID" '
        f" -r OUT_INFO {OUT_INFO} "
        " && jupyter nbconvert --to html {output.nb}"
