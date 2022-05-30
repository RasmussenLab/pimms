"""
Document how all the notebooks are connected.
"""

nb_outfolder = 'runs'

DATASETS=[ "df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547",
            # "df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725",
            # "df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321"
            ]

rule all:
    input: 
        'data/files_per_instrument.yaml', # nested: model, attribute, serial number
        'data/files_selected_metadata.csv',
        'data/files_selected_per_instrument.yaml',
        'data/files_selected_per_instrument_counts.csv', # counts
        f'{nb_outfolder}/{"04_all_raw_files.ipynb"}',
        'data/samples_selected.yaml',
        expand(f"data/single_datasets/{{d}}/{'13_training_data_splitting.ipynb'}", 
            d=DATASETS,
            ),

# # rule template

# nb = '.ipynb'
# rule name:
#     input:
#         nb=nb,
#     output:
#         nb=f"{nb_outfolder}/{nb}",
#     shell:
#         "papermill {input.nb} {output.nb}"



nb='04_all_raw_files.ipynb'
rule metadata:
    input:
        nb=nb,
        # meta='../workflows/metadata/rawfile_metadata.json',
        meta='data/all_raw_files_dump_2021_10_29.txt',
        summaries='data/processed/all_summaries.json'
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
        "papermill {input.nb} {output.nb} -p FN_ALL_RAW_FILES {input.meta} -p FN_ALL_SUMMARIES {input.summaries}"

nb = "02_summaries.ipynb"
rule summaries:
    input:
        nb=nb,
        summaries='data/processed/all_summaries.json'
    output:
        nb=f"{nb_outfolder}/{nb}",
        selected='data/samples_selected.yaml'
    shell:
        "papermill {input.nb} {output.nb}"

nb = "02_metadata_rawfiles.ipynb"
rule metadata_rawfiles:
    input:
        'data/rawfile_metadata.csv',
        'data/samples_selected.yaml',
        nb=nb,
    output:
        'data/files_per_instrument.yaml', # nested: model, attribute, serial number
        'data/files_selected_metadata.csv',
        'data/files_selected_per_instrument.yaml',
        'data/files_selected_per_instrument_counts.csv', # counts
        nb=f"{nb_outfolder}/{nb}"
    shell:
        "papermill {input.nb} {output.nb}" # run with defaults



nb='13_training_data_splitting.ipynb'
outfolder=f'single_datasets'
ROOT_DUMPS = "C:/Users/kzl465/OneDrive - University of Copenhagen/vaep/project/data"
OUT_INFO = "dataset_info"

rule split_data:
    input:
        nb=nb,
        data=f'{ROOT_DUMPS}/{{dataset}}.pkl'
    output:
        nb=f"data/single_datasets/{{dataset}}/{nb}",
        json=f'data/single_datasets/{{dataset}}/{OUT_INFO}.json'
    params:
        folder_datasets = "single_datasets/{dataset}"
    shell:
        # papermill parameters with whitespaces > 
       f'papermill {{input.nb}} {{output.nb}} -r DUMP "{{input.data}}" -p FILE_EXT pkl -r FOLDER_DATASETS {{params.folder_datasets}} -r SAMPLE_ID "Sample ID" -p OUT_INFO {OUT_INFO}'

# rule experiment:

# types_data = [
    # 'proteinGroup',
    # 'peptide',
    # 'evidence',
    # ]

# separate workflow by level -> provide custom configs
nb = '14_experiment_03_data.ipynb'
rule create_splits:
    input:
        nb=nb,
        intensities='data\single_datasets\df_intensities_{type}_long_2017_2018_2019_2020_N05015_M04547\Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.csv',
        metadata='data/files_selected_metadata.csv',
        
    output:
        data=f"{nb_outfolder}/experiment_03/df_intensities_{{type}}_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070/data/train_X.pkl",
        nb=f"{nb_outfolder}/experiment_03/df_intensities_{{type}}_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070/{nb}",
    shell:
        """
        papermill {input.nb} {output.nb} -p FN_INTENSITIES {input.intensities} -p fn_rawfile_metadata {input.metadata}
        """