from pathlib import Path
ROOT_DUMPS = Path("path/to/data")

FN_PROTEIN_GROUPS = ROOT_DUMPS / 'df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547.pkl'
FN_PEPTIDES = ROOT_DUMPS / 'df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725.pkl'
FN_EVIDENCE = ROOT_DUMPS / 'df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321.pkl'

ERDA_DUMPS = [FN_EVIDENCE, FN_PEPTIDES, FN_PROTEIN_GROUPS]
