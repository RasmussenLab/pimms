from .defaults import FOLDER_PROCESSED 

FNAME_C_PEPTIDES = FOLDER_PROCESSED / 'count_all_peptides.json' # aggregated peptides
FNAME_C_EVIDENCE = FOLDER_PROCESSED / 'count_all_evidences.json' # evidence peptides (sequence, charge, modification)

FNAME_C_PG = FOLDER_PROCESSED / 'count_all_protein_groups.json'
FNAME_C_GENES = FOLDER_PROCESSED / 'count_all_genes.json'
