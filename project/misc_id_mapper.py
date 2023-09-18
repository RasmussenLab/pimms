# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Analyse peptides
#
# ## Specification
# - access different levels of peptides easily
# - select training data per gene easily
#

# %%
import json
import logging
logging.basicConfig(level=logging.INFO) # configures root logger
logger = logging.getLogger()
logger.info("test")

# %%
import pandas as pd
from config import FN_FASTA_DB, FN_ID_MAP, FN_PEPTIDE_INTENSITIES

# %%
id_map = pd.read_json(FN_ID_MAP, orient="split")

mask_no_gene = id_map.gene.isna()
id_map.loc[mask_no_gene, "gene"] = "-"


with open(FN_FASTA_DB) as f:
    data_fasta = json.load(f)

# %%
data_peptides = pd.read_pickle(FN_PEPTIDE_INTENSITIES)

# %%
set_peptides = set(data_peptides.columns)

# %% [markdown]
# - switch between list of proteins with any support and non
#     - set threshold of number of peptides per protein over all samples (some peptides uniquely matched to one protein in on sample is just noise -> check razor peptides)
# - show support

# %%
from collections import defaultdict
import ipywidgets as w
from config import KEY_FASTA_HEADER, KEY_FASTA_SEQ, KEY_PEPTIDES, KEY_GENE_NAME, KEY_GENE_NAME_FASTA

TGREEN = "\033[32m"  # Green Text
RESET = "\033[0;0m"

w_first_letter = w.Dropdown(
    options=id_map[KEY_GENE_NAME_FASTA].str[0].unique())
w_genes = w.Dropdown(
    options=id_map.gene.loc[id_map[KEY_GENE_NAME_FASTA].str[0] == w_first_letter.value].unique(),
    value='ACTB'
)

mask = id_map.gene == w_genes.value
selected = id_map.loc[mask, "protein"]


w_proteins_ids = w.Dropdown(options=selected.index)
w_protein = w.Dropdown(options=selected.unique())


def update_gene_list(first_letter):
    """Update proteins when new gene is selected"""
    mask_selected_genes = id_map[KEY_GENE_NAME_FASTA].str[0] == w_first_letter.value
    w_genes.options = id_map.gene.loc[mask_selected_genes].unique()


_ = w.interactive_output(update_gene_list, {"first_letter": w_first_letter})


def update_protein_list(gene):
    mask = id_map[KEY_GENE_NAME_FASTA] == gene
    selected = id_map.loc[mask, "protein"]
    w_protein.options = selected.unique()
#     w_proteins_ids.options = selected.loc[selected == w_protein.value].index


_ = w.interactive_output(update_protein_list, {"gene": w_genes})
    

def update_protein_id_list(protein):
    """Update isotope list when protein is selected"""
    mask = id_map.protein == w_protein.value
    selected = id_map.protein.loc[mask]
    w_proteins_ids.options = selected.index

_ = w.interactive_output(update_protein_id_list, {'protein': w_protein})

d_peptides_observed_prot_id = defaultdict(list)

def show_sequences(prot_id):
    _data = data_fasta[prot_id]
    print(f"Protein_ID on Uniport: {prot_id}")
    print(f"HEADER: {_data[KEY_FASTA_HEADER]}")
#     print(f"Seq  : {_data[KEY_FASTA_SEQ]}")
    annotate_seq = "Peptides: "
    global d_peptides_observed_prot_id
    for i, _l in enumerate(_data[KEY_PEPTIDES]):
        annotate_seq += f"\nNo. of missed K or R: {i}"
        prot_seq_annotated = _data[KEY_FASTA_SEQ]
        for j, _pep in enumerate(_l):
            if _pep in set_peptides:
                d_peptides_observed_prot_id[prot_id].append(_pep)
                _pep_in_green = TGREEN + f"{_pep}" + RESET
                prot_seq_annotated = prot_seq_annotated.replace(_pep, _pep_in_green)
                _pep = _pep_in_green
            if j==0:
                annotate_seq += "\n\t" + _pep
            else:
                annotate_seq += ",\n\t" + _pep
        print(f"Seq {i}: {prot_seq_annotated}")
    print(annotate_seq)
    
    
    display(data_peptides[d_peptides_observed_prot_id[prot_id]].dropna(how='all'))

w_out = w.interactive_output(show_sequences, {"prot_id": w_proteins_ids})

label_first_letter = w.Label(value='First letter of Gene')
label_genes = w.Label('Gene')
label_protein = w.Label('Protein')
label_proteins_ids = w.Label('Protein Isotopes')

panel_levels = w.VBox([
         w.HBox([
            w.VBox([label_first_letter, w_first_letter]),
            w.VBox([label_genes, w_genes]),
            w.VBox([label_protein, w_protein]),
            w.VBox([label_proteins_ids, w_proteins_ids])
            ]),
         w_out]
)
panel_levels

# %% [markdown]
# - relatively short peptides resulting from one missed cleaveage, do not appear in the upper part.

# %% [markdown]
# - `gene` `->` `Protein_ID` (contains information of `gene` `->` `protein_isotopes`
# - `protein_ID` `->` `sequences` (`FN_FASTA_DB`)

# %%
import pickle
from tqdm.notebook import tqdm
from config import FN_PROTEIN_SUPPORT_MAP, FN_PROTEIN_SUPPORT_FREQ
try:
    df_protein_support = pd.read_pickle(FN_PROTEIN_SUPPORT_MAP)
    with open(FN_PROTEIN_SUPPORT_FREQ, 'rb') as f:
        d_protein_support_freq = pickle.load(f)
except FileNotFoundError:
    from vaep.utils import sample_iterable
    d_protein_support = {}
    d_protein_support_freq = {}
    for prot_id in tqdm(data_fasta.keys()):
        _data = data_fasta[prot_id]
        peptides_measured = []
        for i, _l in enumerate(_data[KEY_PEPTIDES]):
            for _pep in _l:
                if _pep in set_peptides:
                    peptides_measured.append(_pep)
        _d_protein_support = {}
        _df_support_protein = data_peptides[peptides_measured].dropna(how='all')

        _n_samples = len(_df_support_protein)
        if _n_samples > 0:
            _d_protein_support['N_samples'] = _n_samples
            d_protein_support_freq[prot_id] = _df_support_protein.notna().sum().to_dict()
            d_protein_support[prot_id] = _d_protein_support
        else:
            d_protein_support[prot_id] = None
        
    df_protein_support = pd.DataFrame(d_protein_support).T.dropna()
    df_protein_support = df_protein_support.join(id_map)
    df_protein_support.to_pickle(FN_PROTEIN_SUPPORT_MAP)
    
    with open(FN_PROTEIN_SUPPORT_FREQ, 'wb') as f:
        pickle.dump(d_protein_support_freq, f)

# %%
l_proteins_good_support = df_protein_support.sort_values(by='N_samples').tail(100).index.to_list()

# %%
d_protein_support_freq['I3L3I0']

# %% [markdown]
# ## Connect to experimental peptide data
#
# Check if counts by `data_fasta`.

# %%
from tqdm.notebook import tqdm

counts_observed_by_missed_cleavages = {}
for _protein_id, _data in tqdm(data_fasta.items()):
    _peptides = _data[KEY_PEPTIDES]
    _counts = {}
    for i, _l in enumerate(_peptides):
        _counts[i] = 0
        for _pep in _l:
            if _pep in set_peptides:
                _counts[i] += 1
    counts_observed_by_missed_cleavages[_protein_id] = _counts

# %%
df_counts_observed_by_missed_cleavages = pd.DataFrame(
    counts_observed_by_missed_cleavages
).T

# %%
import matplotlib.pyplot as plt
from matplotlib import table

fig, axes = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [5, 1], "wspace": 0.2}, figsize=(10,4))

_counts_summed = df_counts_observed_by_missed_cleavages.sum()
_counts_summed.name = "frequency"

ax = axes[0]
_ = _counts_summed.plot(kind="bar", ax=ax)
ax.set_xlabel("peptides from n miscleavages")
ax.set_ylabel("frequency")

ax = axes[1]
ax.axis("off")
_ = pd.plotting.table(ax=ax, data=_counts_summed, loc="best", colWidths=[1], edges='open')
_ = fig.suptitle('Peptides frequencies')

# %% [markdown]
# These are unnormalized counts in the meaning of that _razor_ peptides are counted as often as they are matched.

# %%
mask = df_counts_observed_by_missed_cleavages != 0
df_prot_observed = df_counts_observed_by_missed_cleavages.replace(0, pd.NA)

# %%
df_prot_observed = df_prot_observed.dropna(axis=0, how="all")
df_prot_observed = df_prot_observed.fillna(0)
df_prot_observed = df_prot_observed.convert_dtypes()

# %%
from vaep.pandas import combine_value_counts

combine_value_counts(df_prot_observed)

# %%
freq_pep_mapped_to_protID = df_prot_observed.sum(axis=1).value_counts()
freq_pep_mapped_to_protID = freq_pep_mapped_to_protID.sort_index()

# %%
freq_pep_mapped_to_protID

# %% [markdown]
# ### Genes with support in data
#
# try software to identify the _most likely_ protein. OpenMS or russian alternative?  

# %%

# %% [markdown]
# ## Imputation: Train model
#
# > Select Gene or Protein
#
# As the samples are all obtained from the same biological sample (in principal), the single run should somehow be comparable.
# An description of variablity (from the Data Scientist perspective) can highlight some commenly known facts about proteomics experiments:
#  - batch effects: Measurements on consecutive days are have to be normalized to each other
#  - scoring: PSM are assigned to a peptide based on a score. Small variations can lead to different assignments
#  
# Can a complex representation of a sample level out experimental variation on an in principle comparable data. 
#
# ### Strategy
# - first start using peptides from single Protein_IDs
# - then move to all models from genes
# - explore structure

# %%
import torch

# %%
d_peptides_observed_prot_id

# %%
w_select_proteins_good_support = w.Dropdown(options=l_proteins_good_support)
w_select_proteins_queried = w.Dropdown(options=list(d_peptides_observed_prot_id.keys()))
w.HBox(
    [
        w.VBox(
            [
                w.Label(f"Top {len(l_proteins_good_support)} covered proteins"),
                w_select_proteins_good_support,
            ]
        ),
        w.VBox([w.Label("Queried proteins from above"), w_select_proteins_queried]),
    ]
)
# select from top100 or above selection

# %% [markdown]
# Idea: Select a protein which leads to training. Each selection will create a dump of the selected data, which can be used in the `XZY.ipynb` for model fine-tuning.

# %%
prot_id = w_select_proteins_good_support.value
id_map.loc[prot_id]

# %%
prot_id = 'P00338' # 'I3L3I0' # w_select_proteins_queried.value # 
_protein, _gene, _ = id_map.loc[prot_id]
# _gene_fasta

# %%
w_first_letter.value = _gene[0]
w_genes.value = _gene
w_protein.value = _protein
w_proteins_ids.value = prot_id

# %%
peptides_measured = d_peptides_observed_prot_id[prot_id]
n_peptides_in_selection = len(peptides_measured)
print(f"Selected a total of {n_peptides_in_selection} peptides.") 

# %%
data_peptides[peptides_measured].notna().sum(axis=1).value_counts().sort_index()

# %%
PROP_DATA_COMPLETENESS = 0.75
mask_samples_selected = data_peptides[peptides_measured].notna().sum(axis=1) >= int(n_peptides_in_selection * 0.75)
print(f"Using a share of at least {PROP_DATA_COMPLETENESS}, i.e. at least {int(n_peptides_in_selection * 0.75)} out of {n_peptides_in_selection}.",
     f"In total {mask_samples_selected.sum()} samples.", sep="\n")

# %%
from config import PROTEIN_DUMPS
_ = data_peptides.loc[mask_samples_selected, peptides_measured]
_.to_json(PROTEIN_DUMPS / f"{prot_id}.pkl")
_

# %%
import vaep
from vaep.transform import log

peptides_selected_log10 = data_peptides.loc[mask_samples_selected, peptides_measured].apply(log) # selected in widget overview above
peptides_selected_log10

# %% [markdown]
# > The data to be seen here should be **assigned** peptides. Razor peptides are for now not put to one or the other protein (focus only on unique peptides?).

# %% [markdown]
# ### Hyperparameters

# %%
n_samples, n_features = peptides_selected_log10.shape

# %%
from vaep.models.cmd import parser

BATCH_SIZE = 16
EPOCHS = 600
args = ['--batch-size', str(BATCH_SIZE), '--seed', '43', '--epochs', str(EPOCHS), '--log-interval', str(BATCH_SIZE)]
args = parser.parse_args(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
args

# %%
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cpu")

# %%
# torch.device?

# %% [markdown]
# ### Dataset and DataLoader
#
# The `torch.utils.data.Dataset` can load data into memory, or just create a mapping to data somewhere to be continously loaded by the `torch.utils.data.DataLoader`.

# %%
peptide_intensities = peptides_selected_log10
detection_limit = float(int(peptide_intensities.min().min()))
detection_limit 

# %%
# from vaep.model import PeptideDatasetInMemory

from torch.utils.data import Dataset
class PeptideDatasetInMemory(Dataset):
    """Peptide Dataset fully in memory."""

    def __init__(self, data: pd.DataFrame, fill_na=0):
        self.mask_obs = torch.from_numpy(data.notna().values)
        data = data.fillna(fill_na)
        self.peptides = torch.from_numpy(data.values)
        self.length_ = len(data)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        return self.peptides[idx], self.mask_obs[idx]


dataset_in_memory = PeptideDatasetInMemory(peptide_intensities.copy(), detection_limit)

# %%
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
train_loader = torch.utils.data.DataLoader(
    dataset=dataset_in_memory,
    batch_size=args.batch_size, shuffle=True, **kwargs)

# %%
for i, (data, mask) in enumerate(train_loader):
    print("Nummber of samples in mini-batch: {}".format(len(data)),
          "\tObject-Type: {}".format(type(mask)))
#     print(data)
#     print(mask)
    break

# %%
data[~mask] = 0
plt.imshow(data)

# %% [markdown]
# create logged information for tensorboard, see tutorial and docs.

# %%
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/{prot_id}_{format(datetime.now(), "%y%m%d_%H%M")}')

# %%
writer.add_image(f'{len(data)} samples heatmap', data, dataformats='HW')

# %%
# import importlib; importlib.reload(vaep.model)
from IPython.core.debugger import set_trace

from torch import optim
from vaep.models.ae import VAE
from vaep.models.ae import loss_function

model = VAE(n_features=n_features, n_neurons=30).double().to(device)
writer.add_graph(model, input_to_model=data)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
