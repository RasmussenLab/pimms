import pandas as pd
# could be extended for several file-types
df = pd.read_csv(snakemake.params.fn_clinical_data)
df.to_csv(snakemake.output.local_clincial_data, index=False)
# , index_col=0)             
# usecols=[args.sample_id_col, args.target])