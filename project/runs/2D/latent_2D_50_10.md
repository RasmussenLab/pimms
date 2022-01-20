# Experiment 02


The current approach for creating train, validation and test data sets is to split 
the data in long-format, i.e. one observation is an intensity value from one sample representing one peptide, into the desired splits. In this process missing values are not regarded.

- [x] mask entries in larger dataset in long-format
- [x] mask peptides based on their frequency in samples (probability of being observed)
- [x] create *long-format* training data set without masked values for each model
    - FNN based on embeddings of peptides and samples (long-format **without** missing values)
    - Denoising AE (wide-format **with** missing values)
    - VAE (wide-format **with** missing values)
- [ ] restrict to only a training data split of consective data: Increase number of samples.
    - focus on best reconstruction performance
    - mean comparison

### Collaborative Filtering model

- Cannot accomodate iid assumption of statistical test in current setup for embedding vectors.
  - if pretrained model should be applied to an new batch of replicates (with a certain condition) one would need to find a way to initialize the sample embeddings without fine-tuning the model


```python
import logging
from pprint import pprint
import seaborn
import numpy.testing as npt  # fastcore.test functionality

from pathlib import Path


import vaep.io_images
from vaep.pandas import interpolate
from vaep.model import build_df_from_pred_batches

from src.nb_imports import *
from src import metadata
from src.logging import setup_logger

logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info("Experiment 02")

figures = {}  # collection of ax or figures
```

    FOLDER_MQ_TXT_DATA = data\mq_out
    vaep - INFO     Experiment 02
    


```python
# None takes all
N_SAMPLES: int = 1000
n_features: int = 50
ADD_TENSORBOARD: bool = False
FN_PEPTIDE_INTENSITIES: Path = (
    config.FOLDER_DATA / 'df_intensities_N07285_M01000')  # 90%
epochs_max = 10
batch_size = 32
latent_dim = 2
most_common: bool = False
most_uncommon: bool = False
out_folder: str = 'poster'
# write to read only config ? namedtuple?
```


```python
# Parameters
n_feat = 50
n_epochs = 10
out_folder = "runs/2D"

```


```python
BATCH_SIZE, EPOCHS = batch_size, epochs_max
folder = Path(out_folder) / f'feat_{n_features:04d}_epochs_{epochs_max:03d}'
print(f"{folder = }")

if most_common and most_uncommon:
    raise ValueError(f"Cannot be both True: {most_common = } and {most_uncommon = }")
```

    folder = Path('runs/2D/feat_0050_epochs_010')
    

## Raw data


```python
FN_PEPTIDE_INTENSITIES = Path(FN_PEPTIDE_INTENSITIES)
```


```python
analysis = AnalyzePeptides(fname=FN_PEPTIDE_INTENSITIES, nrows=None)
analysis.df.columns.name = 'peptide'
analysis.log_transform(np.log2)
analysis
```




    AnalyzePeptides with attributes: M, N, df, index_col, is_log_transformed, is_wide_format, log_fct, stats




```python
# some date are not possible in the indices
rename_indices_w_wrong_dates = {'20161131_LUMOS1_nLC13_AH_MNT_HeLa_long_03': '20161130_LUMOS1_nLC13_AH_MNT_HeLa_long_03',
                                '20180230_QE10_nLC0_MR_QC_MNT_Hela_12': '20180330_QE10_nLC0_MR_QC_MNT_Hela_12',
                                '20161131_LUMOS1_nLC13_AH_MNT_HeLa_long_01': '20161130_LUMOS1_nLC13_AH_MNT_HeLa_long_01',
                                '20180230_QE10_nLC0_MR_QC_MNT_Hela_11': '20180330_QE10_nLC0_MR_QC_MNT_Hela_11',
                                '20161131_LUMOS1_nLC13_AH_MNT_HeLa_long_02': '20161130_LUMOS1_nLC13_AH_MNT_HeLa_long_02'}
analysis.df.rename(index=rename_indices_w_wrong_dates, inplace=True)
```

### Select N consecutive samples


```python
analysis.get_consecutive_dates(n_samples=N_SAMPLES)
```

    Get 1000 samples.
    Training data referenced unter: df_1000
    Updated attribute: df
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAAAAAALQAK</th>
      <th>AAFDDAIAELDTLSEESYK</th>
      <th>AAHSEGNTTAGLDMR</th>
      <th>AAVATFLQSVQVPEFTPK</th>
      <th>AAVEEGIVLGGGCALLR</th>
      <th>AAVPSGASTGIYEALELR</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>ACGLVASNLNLKPGECLR</th>
      <th>ADLINNLGTIAK</th>
      <th>...</th>
      <th>VVFVFGPDK</th>
      <th>VVFVFGPDKK</th>
      <th>VYALPEDLVEVKPK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YDDMAAAMK</th>
      <th>YDDMAACMK</th>
      <th>YDDMATCMK</th>
      <th>YLAEVACGDDRK</th>
      <th>YLDEDTIYHLQPSGR</th>
      <th>YRVPDVLVADPPIAR</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>29.591</td>
      <td>31.264</td>
      <td>28.098</td>
      <td>28.694</td>
      <td>30.208</td>
      <td>30.134</td>
      <td>31.358</td>
      <td>30.430</td>
      <td>29.440</td>
      <td>31.035</td>
      <td>...</td>
      <td>29.047</td>
      <td>27.827</td>
      <td>27.149</td>
      <td>28.446</td>
      <td>29.136</td>
      <td>28.936</td>
      <td>27.337</td>
      <td>28.873</td>
      <td>27.724</td>
      <td>28.059</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>29.817</td>
      <td>31.354</td>
      <td>28.132</td>
      <td>28.508</td>
      <td>30.171</td>
      <td>30.189</td>
      <td>31.311</td>
      <td>30.412</td>
      <td>29.410</td>
      <td>31.034</td>
      <td>...</td>
      <td>28.830</td>
      <td>27.624</td>
      <td>27.150</td>
      <td>28.537</td>
      <td>28.643</td>
      <td>29.347</td>
      <td>27.980</td>
      <td>29.179</td>
      <td>27.444</td>
      <td>28.035</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>29.982</td>
      <td>31.211</td>
      <td>24.357</td>
      <td>28.795</td>
      <td>29.833</td>
      <td>28.291</td>
      <td>32.533</td>
      <td>31.257</td>
      <td>30.998</td>
      <td>32.087</td>
      <td>...</td>
      <td>29.325</td>
      <td>29.346</td>
      <td>28.907</td>
      <td>29.588</td>
      <td>28.434</td>
      <td>28.369</td>
      <td>27.397</td>
      <td>29.143</td>
      <td>27.233</td>
      <td>29.432</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>29.974</td>
      <td>30.238</td>
      <td>28.022</td>
      <td>28.917</td>
      <td>30.113</td>
      <td>30.597</td>
      <td>31.781</td>
      <td>30.620</td>
      <td>27.377</td>
      <td>31.457</td>
      <td>...</td>
      <td>29.409</td>
      <td>27.696</td>
      <td>27.622</td>
      <td>29.104</td>
      <td>28.151</td>
      <td>29.017</td>
      <td>27.815</td>
      <td>29.317</td>
      <td>28.027</td>
      <td>28.516</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>29.552</td>
      <td>31.966</td>
      <td>28.165</td>
      <td>29.985</td>
      <td>31.660</td>
      <td>31.787</td>
      <td>31.915</td>
      <td>31.792</td>
      <td>30.077</td>
      <td>31.479</td>
      <td>...</td>
      <td>29.341</td>
      <td>NaN</td>
      <td>27.327</td>
      <td>29.091</td>
      <td>29.422</td>
      <td>29.057</td>
      <td>26.068</td>
      <td>29.129</td>
      <td>27.474</td>
      <td>28.216</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>31.517</td>
      <td>32.278</td>
      <td>30.832</td>
      <td>31.541</td>
      <td>32.803</td>
      <td>33.676</td>
      <td>31.927</td>
      <td>31.906</td>
      <td>27.631</td>
      <td>33.747</td>
      <td>...</td>
      <td>31.452</td>
      <td>28.512</td>
      <td>29.634</td>
      <td>31.134</td>
      <td>30.690</td>
      <td>30.142</td>
      <td>29.881</td>
      <td>31.508</td>
      <td>29.884</td>
      <td>29.468</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>31.104</td>
      <td>31.747</td>
      <td>28.999</td>
      <td>28.458</td>
      <td>30.588</td>
      <td>28.800</td>
      <td>32.415</td>
      <td>31.270</td>
      <td>30.077</td>
      <td>32.357</td>
      <td>...</td>
      <td>29.504</td>
      <td>29.435</td>
      <td>28.424</td>
      <td>29.489</td>
      <td>29.702</td>
      <td>29.877</td>
      <td>28.777</td>
      <td>30.471</td>
      <td>27.793</td>
      <td>29.185</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>31.248</td>
      <td>31.923</td>
      <td>28.643</td>
      <td>28.257</td>
      <td>31.068</td>
      <td>28.864</td>
      <td>32.340</td>
      <td>31.672</td>
      <td>31.636</td>
      <td>32.466</td>
      <td>...</td>
      <td>29.728</td>
      <td>29.741</td>
      <td>28.667</td>
      <td>29.647</td>
      <td>29.555</td>
      <td>29.709</td>
      <td>28.592</td>
      <td>30.776</td>
      <td>27.955</td>
      <td>29.272</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>31.122</td>
      <td>31.802</td>
      <td>28.260</td>
      <td>28.620</td>
      <td>30.936</td>
      <td>NaN</td>
      <td>32.558</td>
      <td>29.854</td>
      <td>31.661</td>
      <td>32.413</td>
      <td>...</td>
      <td>29.755</td>
      <td>29.772</td>
      <td>28.764</td>
      <td>29.507</td>
      <td>29.443</td>
      <td>29.544</td>
      <td>28.643</td>
      <td>30.481</td>
      <td>27.837</td>
      <td>29.167</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>30.814</td>
      <td>31.892</td>
      <td>28.546</td>
      <td>28.408</td>
      <td>30.727</td>
      <td>28.683</td>
      <td>32.734</td>
      <td>31.699</td>
      <td>31.578</td>
      <td>32.284</td>
      <td>...</td>
      <td>29.747</td>
      <td>29.809</td>
      <td>29.158</td>
      <td>29.274</td>
      <td>29.575</td>
      <td>29.545</td>
      <td>28.487</td>
      <td>30.467</td>
      <td>27.935</td>
      <td>29.139</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 1000 columns</p>
</div>




```python
assert not analysis.df._is_view
```

## Long format

- Data in long format: (peptide, sample_id, intensity)
- no missing values kept
- 


```python
analysis.df_long.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAAAAAALQAK</th>
      <td>29.591</td>
    </tr>
    <tr>
      <th>AAFDDAIAELDTLSEESYK</th>
      <td>31.264</td>
    </tr>
    <tr>
      <th>AAHSEGNTTAGLDMR</th>
      <td>28.098</td>
    </tr>
    <tr>
      <th>AAVATFLQSVQVPEFTPK</th>
      <td>28.694</td>
    </tr>
    <tr>
      <th>AAVEEGIVLGGGCALLR</th>
      <td>30.208</td>
    </tr>
  </tbody>
</table>
</div>




```python
assert analysis.df_long.isna().sum().sum(
) == 0, "There are still missing values in the long format."
```


```python
analysis.df_wide.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAAAAAALQAK</th>
      <th>AAFDDAIAELDTLSEESYK</th>
      <th>AAHSEGNTTAGLDMR</th>
      <th>AAVATFLQSVQVPEFTPK</th>
      <th>AAVEEGIVLGGGCALLR</th>
      <th>AAVPSGASTGIYEALELR</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>ACGLVASNLNLKPGECLR</th>
      <th>ADLINNLGTIAK</th>
      <th>...</th>
      <th>VVFVFGPDK</th>
      <th>VVFVFGPDKK</th>
      <th>VYALPEDLVEVKPK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YDDMAAAMK</th>
      <th>YDDMAACMK</th>
      <th>YDDMATCMK</th>
      <th>YLAEVACGDDRK</th>
      <th>YLDEDTIYHLQPSGR</th>
      <th>YRVPDVLVADPPIAR</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>29.591</td>
      <td>31.264</td>
      <td>28.098</td>
      <td>28.694</td>
      <td>30.208</td>
      <td>30.134</td>
      <td>31.358</td>
      <td>30.430</td>
      <td>29.440</td>
      <td>31.035</td>
      <td>...</td>
      <td>29.047</td>
      <td>27.827</td>
      <td>27.149</td>
      <td>28.446</td>
      <td>29.136</td>
      <td>28.936</td>
      <td>27.337</td>
      <td>28.873</td>
      <td>27.724</td>
      <td>28.059</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>29.817</td>
      <td>31.354</td>
      <td>28.132</td>
      <td>28.508</td>
      <td>30.171</td>
      <td>30.189</td>
      <td>31.311</td>
      <td>30.412</td>
      <td>29.410</td>
      <td>31.034</td>
      <td>...</td>
      <td>28.830</td>
      <td>27.624</td>
      <td>27.150</td>
      <td>28.537</td>
      <td>28.643</td>
      <td>29.347</td>
      <td>27.980</td>
      <td>29.179</td>
      <td>27.444</td>
      <td>28.035</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>29.982</td>
      <td>31.211</td>
      <td>24.357</td>
      <td>28.795</td>
      <td>29.833</td>
      <td>28.291</td>
      <td>32.533</td>
      <td>31.257</td>
      <td>30.998</td>
      <td>32.087</td>
      <td>...</td>
      <td>29.325</td>
      <td>29.346</td>
      <td>28.907</td>
      <td>29.588</td>
      <td>28.434</td>
      <td>28.369</td>
      <td>27.397</td>
      <td>29.143</td>
      <td>27.233</td>
      <td>29.432</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>29.974</td>
      <td>30.238</td>
      <td>28.022</td>
      <td>28.917</td>
      <td>30.113</td>
      <td>30.597</td>
      <td>31.781</td>
      <td>30.620</td>
      <td>27.377</td>
      <td>31.457</td>
      <td>...</td>
      <td>29.409</td>
      <td>27.696</td>
      <td>27.622</td>
      <td>29.104</td>
      <td>28.151</td>
      <td>29.017</td>
      <td>27.815</td>
      <td>29.317</td>
      <td>28.027</td>
      <td>28.516</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>29.552</td>
      <td>31.966</td>
      <td>28.165</td>
      <td>29.985</td>
      <td>31.660</td>
      <td>31.787</td>
      <td>31.915</td>
      <td>31.792</td>
      <td>30.077</td>
      <td>31.479</td>
      <td>...</td>
      <td>29.341</td>
      <td>NaN</td>
      <td>27.327</td>
      <td>29.091</td>
      <td>29.422</td>
      <td>29.057</td>
      <td>26.068</td>
      <td>29.129</td>
      <td>27.474</td>
      <td>28.216</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1000 columns</p>
</div>




```python
assert analysis.df_wide.isna().sum().sum(
) > 0, "There are no missing values left in the wide format"
```

### Sampling peptides by their frequency (important for later)

- higher count, higher probability to be sampled into training data
- missing peptides are sampled both into training as well as into validation dataset
- everything not in training data is validation data


```python
# freq_per_peptide = analysis.df.unstack().to_frame('intensity').reset_index(1, drop=True)
freq_per_peptide = analysis.df_long['intensity']
freq_per_peptide = freq_per_peptide.notna().groupby(level=1).sum()
print(f"{n_features = }")

```

    n_features = 50
    

### Selecting N
 - most common
 - most uncommon


```python
freq_per_pepitde = freq_per_peptide.sort_values(ascending=False)

if most_common:
    freq_per_pepitde = freq_per_pepitde.iloc[:n_features]
elif most_uncommon:
    freq_per_pepitde = freq_per_pepitde.iloc[-n_features:]
else:
    freq_per_pepitde = freq_per_pepitde.sample(n_features)
    
assert len(freq_per_pepitde.index) == n_features

freq_per_pepitde
```




    peptide
    DLEEDHACIPIK               998
    TLTAVHDAILEDLVFPSEIVGK     960
    EQISDIDDAVR                992
    IGDLQAFQGHGAGNLAGLK      1,000
    TVLMNPNIASVQTNEVGLK        999
    MALIGLGVSHPVLK             999
    TYFSCTSAHTSTGDGTAMITR      994
    EMNDAAMFYTNR               990
    EAAENSLVAYK                952
    ASNGDAWVEAHGK              984
    KTEAPAAPAAQETK             994
    GANDFMCDEMER               975
    FNADEFEDMVAEK              991
    SLEDQVEMLR                 982
    LGQSDPAPLQHQMDIYQK         999
    AAVPSGASTGIYEALELRDNDK     954
    IIAPPERK                   940
    HLAGLGLTEAIDK              998
    PLRLPLQDVYK                947
    SEIDLFNIRK                 951
    ILLTEPPMNPTK               971
    TATPQQAQEVHEK              997
    IYVDDGLISLQVK              995
    FGYVDFESAEDLEK             989
    SAYDSTMETMNYAQIR           991
    DSYVGDEAQSK                999
    NSSYFVEWIPNNVK             984
    YNILGTNTIMDK               998
    QAQIEVVPSASALIIK           983
    AHSSMVGVNLPQK            1,000
    LLLGAGAVAYGVR              977
    LMDVGLIAIR                 996
    IWHHTFYNELR                992
    KQELEEICHDLEAR             978
    NYIQGINLVQAK               982
    ALPAVQQNNLDEDLIRK          992
    VLSAPPHFHFGQTNR            913
    DHENIVIAK                  986
    ALTSEIALLQSR               995
    FEELNMDLFR               1,000
    SSEHINEGETAMLVCK           997
    NLDIERPTYTNLNR             998
    ALLFVPR                    990
    FQSSHHPTDITSLDQYVER        998
    SHTILLVQPTK                993
    HEQNIDCGGGYVK              998
    ESEPQAAAEPAEAK             970
    AIVAIENPADVSVISSR          995
    TVTAMDVVYALK             1,000
    NSNLVGAAHEELQQSR           998
    Name: intensity, dtype: int64




```python
analysis.df = analysis.df[freq_per_pepitde.index]
# ToDo: clean-up other attributes needs to be integrated
del analysis._df_long  # , analysis._df_wide
analysis.df_long
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
    </tr>
    <tr>
      <th>EQISDIDDAVR</th>
      <td>29.423</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
    </tr>
    <tr>
      <th>TVLMNPNIASVQTNEVGLK</th>
      <td>28.762</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>HEQNIDCGGGYVK</th>
      <td>30.912</td>
    </tr>
    <tr>
      <th>ESEPQAAAEPAEAK</th>
      <td>30.445</td>
    </tr>
    <tr>
      <th>AIVAIENPADVSVISSR</th>
      <td>29.348</td>
    </tr>
    <tr>
      <th>TVTAMDVVYALK</th>
      <td>34.581</td>
    </tr>
    <tr>
      <th>NSNLVGAAHEELQQSR</th>
      <td>29.348</td>
    </tr>
  </tbody>
</table>
<p>49254 rows × 1 columns</p>
</div>



- biological stock differences in PCA plot. Show differences in models. Only see biological variance

## PCA plot of raw data


```python
fig = analysis.plot_pca()
```

    vaep - ERROR    Could not find LC instrument in 20190418_QX8_JuSc_MA_HeLa_500ng_1
    vaep - ERROR    Could not find LC instrument in 20190422_QX8_JuSc_MA_HeLa_500ng_1
    vaep - ERROR    Could not find LC instrument in 20190425_QX8_JuSc_MA_HeLa_500ng_1
    vaep - ERROR    Could not find LC instrument in 20190501_QX8_MiWi_MA_HeLa_500ng_new
    vaep - ERROR    Could not find LC instrument in 20190502_QX7_ChDe_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190502_QX8_MiWi_MA_HeLa_500ng_new
    vaep - ERROR    Could not find LC instrument in 20190502_QX8_MiWi_MA_HeLa_500ng_old
    vaep - ERROR    Could not find LC instrument in 20190506_QX7_ChDe_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190506_QX8_MiWi_MA_HeLa_500ng_new
    vaep - ERROR    Could not find LC instrument in 20190506_QX8_MiWi_MA_HeLa_500ng_old
    vaep - ERROR    Could not find LC instrument in 20190514_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190520_QX4_JoSw_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190521_QX4_JoSw_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190524_QX4_JoSw_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190526_QX4_LiSc_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190527_QX4_IgPa_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190530_QX4_IgPa_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190603_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190606_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190607_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190611_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190615_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190617_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190617_QX8_IgPa_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190618_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190618_QX4_JiYu_MA_HeLa_500ng_190618125902
    vaep - ERROR    Could not find LC instrument in 20190618_QX4_JiYu_MA_HeLa_500ng_190619010035
    vaep - ERROR    Could not find LC instrument in 20190618_QX4_JiYu_MA_HeLa_500ng_centroid
    vaep - ERROR    Could not find LC instrument in 20190621_QX4_JoMu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190621_QX4_JoMu_MA_HeLa_500ng_190621161214
    vaep - ERROR    Could not find LC instrument in 20190624_QX4_JiYu_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED
    vaep - ERROR    Could not find LC instrument in 20190701_QX4_MePh_MA_HeLa_500ng_MAX_ALLOWED
    vaep - ERROR    Could not find LC instrument in 20190703_QX4_MaTa_MA_HeLa_500ng_MAX_ALLOWED
    vaep - ERROR    Could not find LC instrument in 20190706_QX4_MiWi_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190706_QX4_MiWi_MA_HeLa_500ng_190707003046
    vaep - ERROR    Could not find LC instrument in 20190717_QX8_ChSc_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190718_QX8_ChSc_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190719_QX8_ChSc_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190722_QX4_StEb_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190722_QX8_ChSc_MA_HeLa_500ng_190722174431
    vaep - ERROR    Could not find LC instrument in 20190725_QX2_MePh_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190726_QX8_ChSc_MA_HeLa_500ng
    vaep - ERROR    Could not find LC instrument in 20190731_QX8_ChSc_MA_HeLa_500ng
    Created metadata DataFrame attribute `df_meta`.
    Added proportion of not NA values based on `df` intensities.
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_24_1.png)
    



```python
# ToDo add df_meta property
analysis.df_meta.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prop_not_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1,000.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.985</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.860</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.980</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
vaep.io_images._savefig(fig, folder /
                        f'pca_plot_raw_data_{analysis.fname_stub}')
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\pca_plot_raw_data_N01000_M00050
    

## Train and Validation data

- use mulitindex for obtaining validation split


```python
# analysis._df_long = analysis.df_long.reset_index(
# ).set_index(['Sample ID', 'peptide'])
analysis.df_long
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
    </tr>
    <tr>
      <th>EQISDIDDAVR</th>
      <td>29.423</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
    </tr>
    <tr>
      <th>TVLMNPNIASVQTNEVGLK</th>
      <td>28.762</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>HEQNIDCGGGYVK</th>
      <td>30.912</td>
    </tr>
    <tr>
      <th>ESEPQAAAEPAEAK</th>
      <td>30.445</td>
    </tr>
    <tr>
      <th>AIVAIENPADVSVISSR</th>
      <td>29.348</td>
    </tr>
    <tr>
      <th>TVTAMDVVYALK</th>
      <td>34.581</td>
    </tr>
    <tr>
      <th>NSNLVGAAHEELQQSR</th>
      <td>29.348</td>
    </tr>
  </tbody>
</table>
<p>49254 rows × 1 columns</p>
</div>




```python
# df_long = analysis.df.unstack().to_frame('intensity').reset_index(1)
analysis.df_train = analysis.df_long.reset_index(0).groupby(
    by='Sample ID',
    level=0
).sample(frac=0.90,
         weights=freq_per_peptide,
         random_state=42)
analysis.df_train = analysis.df_train.reset_index().set_index([
    'Sample ID', 'peptide'])
analysis.df_train
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20190408_QE8_nLC14_AGF_QC_MNT_HeLa_50cm_01</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>33.111</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>32.762</td>
    </tr>
    <tr>
      <th>20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>32.950</td>
    </tr>
    <tr>
      <th>20190603_QE3_nLC3_DS_QC_MNT_HeLa_03</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>32.237</td>
    </tr>
    <tr>
      <th>20190207_QE8_nLC0_ASD_QC_HeLa_43cm3</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>32.427</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190626_QX8_ChDe_MA_HeLa_BR14_500ng</th>
      <th>YNILGTNTIMDK</th>
      <td>31.008</td>
    </tr>
    <tr>
      <th>20190502_QX7_ChDe_MA_HeLa_500ng</th>
      <th>YNILGTNTIMDK</th>
      <td>31.911</td>
    </tr>
    <tr>
      <th>20190408_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-006</th>
      <th>YNILGTNTIMDK</th>
      <td>30.754</td>
    </tr>
    <tr>
      <th>20190701_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>YNILGTNTIMDK</th>
      <td>30.501</td>
    </tr>
    <tr>
      <th>20190219_QE3_Evo2_UHG_QC_MNT_HELA_01_190219173213</th>
      <th>YNILGTNTIMDK</th>
      <td>30.382</td>
    </tr>
  </tbody>
</table>
<p>44331 rows × 1 columns</p>
</div>




```python
analysis.indices_valid = analysis.df_long.index.difference(
    analysis.df_train.index)
analysis.df_valid = analysis.df_long.loc[analysis.indices_valid]
```


```python
assert len(analysis.df_long) == len(analysis.df_train) + len(analysis.df_valid)
```

Check that all samples are also in the validation data


```python
assert analysis.df_train.index.levshape == (N_SAMPLES, n_features)

try:
    assert analysis.df_valid.index.levshape == (N_SAMPLES, n_features)
except AssertionError:
    print(f'Expected shape in validation: {(N_SAMPLES, n_features)}')
    print(f'Shape in validation: {analysis.df_valid.index.levshape}')

analysis.df_train = analysis.df_train.loc[analysis.df_valid.index.levels[0]]
analysis.df_train = analysis.df_train.reset_index().set_index(
    ['Sample ID', 'peptide'])  # update index categories (there is probably a better way)
N_SAMPLES = analysis.df_valid.index.levshape[0]
analysis.df_train.index.levshape, analysis.df_valid.index.levshape
```

    Expected shape in validation: (1000, 50)
    Shape in validation: (995, 50)
    




    ((995, 50), (995, 50))



## Setup DL

- [ ] move all above to separate data notebook


```python
import vaep.models as models
from vaep.models.cmd import get_args
from vaep.models import ae

args = get_args(batch_size=BATCH_SIZE, epochs=EPOCHS,
                no_cuda=False)  # data transfer to GPU seems slow
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

# torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
device

print(f"{args = }", f"{device = }", sep='\n')
```

    args = Namespace(batch_size=32, cuda=True, epochs=10, log_interval=10, no_cuda=False, seed=43)
    device = device(type='cuda')
    

Fastai default device for computation


```python
import fastai.torch_core
print(f"{torch.cuda.is_available() = }")  # self-documenting python 3.8
fastai.torch_core.defaults
```

    torch.cuda.is_available() = True
    




    namespace(cpus=12,
              use_cuda=None,
              activation=torch.nn.modules.activation.ReLU,
              callbacks=[fastai.callback.core.TrainEvalCallback,
                         fastai.learner.Recorder,
                         fastai.callback.progress.ProgressCallback],
              lr=0.001)



### Comparison data

- first impute first and last row (using n=3 replicate)
- use pandas interpolate


```python
analysis.median_train = analysis.df_train['intensity'].unstack().median()
analysis.median_train.name = 'train_median'
analysis.averag_train = analysis.df_train['intensity'].unstack().mean()
analysis.averag_train.name = 'train_average'

df_pred = analysis.df_valid.copy()

df_pred = df_pred.join(analysis.median_train, on='peptide')
df_pred = df_pred.join(analysis.averag_train, on='peptide')


_ = interpolate(wide_df=analysis.df_train['intensity'].unstack())
df_pred = df_pred.join(_)

df_pred
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>train_median</th>
      <th>train_average</th>
      <th>replicates</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>31.358</td>
      <td>32.424</td>
      <td>32.114</td>
      <td>31.311</td>
    </tr>
    <tr>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
      <td>30.036</td>
      <td>30.528</td>
      <td>29.605</td>
    </tr>
    <tr>
      <th>FGYVDFESAEDLEK</th>
      <td>29.106</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>29.449</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
      <td>30.666</td>
      <td>30.489</td>
      <td>29.785</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>28.240</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>28.558</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <th>SLEDQVEMLR</th>
      <td>28.973</td>
      <td>28.941</td>
      <td>28.820</td>
      <td>29.179</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ALLFVPR</th>
      <td>31.110</td>
      <td>31.005</td>
      <td>30.981</td>
      <td>30.844</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>31.690</td>
      <td>31.268</td>
      <td>30.996</td>
      <td>31.645</td>
    </tr>
    <tr>
      <th>QAQIEVVPSASALIIK</th>
      <td>30.516</td>
      <td>30.019</td>
      <td>29.921</td>
      <td>30.723</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>29.820</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>29.760</td>
    </tr>
  </tbody>
</table>
<p>4923 rows × 4 columns</p>
</div>




```python
if any(df_pred.isna()):
    print("Consecutive NaNs are not imputed using replicates.")
    display(df_pred.loc[df_pred.isna().any(axis=1)])
```

    Consecutive NaNs are not imputed using replicates.
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>train_median</th>
      <th>train_average</th>
      <th>replicates</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03</th>
      <th>IIAPPERK</th>
      <td>33.672</td>
      <td>33.604</td>
      <td>33.312</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE8_nLC1_ASD_QC_HeLa_01</th>
      <th>SSEHINEGETAMLVCK</th>
      <td>29.033</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190115_QE2_NLC10_TW_QC_MNT_HeLa_01</th>
      <th>IWHHTFYNELR</th>
      <td>32.853</td>
      <td>32.452</td>
      <td>32.051</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190118_QE1_nLC2_ANHO_QC_MNT_HELA_03</th>
      <th>TATPQQAQEVHEK</th>
      <td>29.824</td>
      <td>31.988</td>
      <td>31.839</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190122_QE6_nLC6_SIS_QC_MNT_HeLa_01</th>
      <th>FGYVDFESAEDLEK</th>
      <td>30.404</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190731_QE8_nLC14_ASD_QC_MNT_HeLa_03</th>
      <th>FGYVDFESAEDLEK</th>
      <td>29.393</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QX8_ChSc_MA_HeLa_500ng</th>
      <th>EMNDAAMFYTNR</th>
      <td>29.605</td>
      <td>27.322</td>
      <td>27.360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_01_20190801145136</th>
      <th>NLDIERPTYTNLNR</th>
      <td>32.004</td>
      <td>31.915</td>
      <td>31.857</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05</th>
      <th>DLEEDHACIPIK</th>
      <td>32.550</td>
      <td>30.036</td>
      <td>30.528</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_300ng</th>
      <th>KQELEEICHDLEAR</th>
      <td>29.195</td>
      <td>29.413</td>
      <td>29.176</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>83 rows × 4 columns</p>
</div>


## Collaboritive filtering model


```python
from fastai.collab import CollabDataLoaders, MSELossFlat, Learner
from fastai.collab import EmbeddingDotBias

analysis.collab = Analysis()
collab = analysis.collab
collab.columns = 'peptide,Sample ID,intensity'.split(',')
```

Create data view for collaborative filtering

- currently a bit hacky as the splitter does not support predefinded indices (create custum subclass providing splits to internal methods?)

- Use the [`CollabDataLoaders`](https://docs.fast.ai/collab.html#CollabDataLoaders)  similar to the [`TabularDataLoaders`](https://docs.fast.ai/tabular.data.html#TabularDataLoaders).
- Use the [`IndexSplitter`](https://docs.fast.ai/data.transforms.html#IndexSplitter) and provide splits to whatever is used in `CollabDataLoaders`



```python
collab.df_train = analysis.df_train.reset_index()
collab.df_valid = analysis.df_valid.reset_index()
collab.df_train.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>peptide</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AHSSMVGVNLPQK</td>
      <td>29.899</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AIVAIENPADVSVISSR</td>
      <td>29.292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALLFVPR</td>
      <td>30.215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALTSEIALLQSR</td>
      <td>27.244</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ASNGDAWVEAHGK</td>
      <td>24.793</td>
    </tr>
  </tbody>
</table>
</div>




```python
collab.df_valid.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>peptide</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AAVPSGASTGIYEALELRDNDK</td>
      <td>31.358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DLEEDHACIPIK</td>
      <td>29.764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>FGYVDFESAEDLEK</td>
      <td>29.106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>IGDLQAFQGHGAGNLAGLK</td>
      <td>29.867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>SSEHINEGETAMLVCK</td>
      <td>28.240</td>
    </tr>
  </tbody>
</table>
</div>




```python
assert (collab.df_train.intensity.isna().sum(),
        collab.df_valid.intensity.isna().sum()) == (0, 0), "Remove missing values."
```

Hacky part uses training data `Datasets` from dataloaders to recreate a custom `DataLoaders` instance


```python
collab.dl_train = CollabDataLoaders.from_df(
    collab.df_train, valid_pct=0.0, user_name='Sample ID', item_name='peptide', rating_name='intensity', bs=args.batch_size, device=device)
collab.dl_valid = CollabDataLoaders.from_df(
    collab.df_valid, valid_pct=0.0, user_name='Sample ID', item_name='peptide', rating_name='intensity', bs=args.batch_size,
    shuffle=False, device=device)
collab.dl_train.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>peptide</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20190115_QE2_NLC10_TW_QC_MNT_HeLa_01</td>
      <td>IYVDDGLISLQVK</td>
      <td>31.065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190221_QE8_nLC9_JM_QC_MNT_HeLa_01_20190222005035</td>
      <td>ASNGDAWVEAHGK</td>
      <td>29.153</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190121_QE2_NLC1_GP_QC_MNT_HELA_02</td>
      <td>DHENIVIAK</td>
      <td>29.485</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_1</td>
      <td>KQELEEICHDLEAR</td>
      <td>25.993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190717_QE6_LC4_SCL_QC_MNT_Hela_04</td>
      <td>SHTILLVQPTK</td>
      <td>29.245</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190425_QX8_JuSc_MA_HeLa_500ng_1</td>
      <td>TLTAVHDAILEDLVFPSEIVGK</td>
      <td>31.826</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190425_QE9_nLC0_LiNi_QC_45cm_HeLa_01</td>
      <td>LMDVGLIAIR</td>
      <td>28.017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190729_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05</td>
      <td>SLEDQVEMLR</td>
      <td>28.351</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</td>
      <td>MALIGLGVSHPVLK</td>
      <td>28.828</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190712_QE1_nLC13_ANHO_QC_MNT_HELA_03</td>
      <td>LLLGAGAVAYGVR</td>
      <td>26.696</td>
    </tr>
  </tbody>
</table>



```python
from fastai.data.core import DataLoaders
collab.dls = DataLoaders(collab.dl_train.train, collab.dl_valid.train)
if args.cuda:
    collab.dls.cuda()
```


```python
collab.dl_valid.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>peptide</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20190219_QE10_nLC14_FaCo_QC_HeLa_50cm_20190221093339</td>
      <td>MALIGLGVSHPVLK</td>
      <td>26.963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190527_QX4_IgPa_MA_HeLa_500ng</td>
      <td>NYIQGINLVQAK</td>
      <td>30.123</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190420_QE8_nLC14_RG_QC_HeLa_01</td>
      <td>LLLGAGAVAYGVR</td>
      <td>28.125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190228_QE1_nLC2_ANHO_QC_MNT_HELA_01</td>
      <td>IYVDDGLISLQVK</td>
      <td>28.075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190726_QX8_ChSc_MA_HeLa_500ng</td>
      <td>YNILGTNTIMDK</td>
      <td>31.638</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190702_QE3_nLC5_GF_QC_MNT_Hela_01</td>
      <td>FQSSHHPTDITSLDQYVER</td>
      <td>25.505</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_02</td>
      <td>IGDLQAFQGHGAGNLAGLK</td>
      <td>30.671</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190225_QE10_PhGe_Evosep_88min_HeLa_9</td>
      <td>TYFSCTSAHTSTGDGTAMITR</td>
      <td>28.220</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190205_QE7_nLC7_MEM_QC_MNT_HeLa_02</td>
      <td>SLEDQVEMLR</td>
      <td>29.713</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190425_QX8_JuSc_MA_HeLa_500ng_1</td>
      <td>ALLFVPR</td>
      <td>32.193</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (996, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1377, 154)



Alternatively to the hacky version, one could use a factory method, but there the sampling/Splitting methods would need to be implemented (not using [`RandomSplitter`](https://docs.fast.ai/data.transforms.html#RandomSplitter) somehow)

 - [`TabDataLoader`](https://docs.fast.ai/tabular.core.html#TabDataLoader)
 - uses [`TabularPandas`](https://docs.fast.ai/tabular.core.html#TabularPandas)
 
 > Current problem: No custom splitter can be provided

### Model


```python
collab.model_args = {}
collab.model_args['n_samples'] = len(collab.dls.classes['Sample ID'])
collab.model_args['n_peptides'] = len(collab.dls.classes['peptide'])
collab.model_args['dim_latent_factors'] = latent_dim
collab.model_args['y_range'] = (
    int(analysis.df_train['intensity'].min()), int(analysis.df_train['intensity'].max())+1)

print("Args:")
pprint(collab.model_args)


# from vaep.models.collab import DotProductBias
# model = DotProductBias(**collab.model_args)
model = EmbeddingDotBias.from_classes(
    n_factors=collab.model_args['dim_latent_factors'], classes=collab.dls.classes, y_range=collab.model_args['y_range'])
learn = Learner(dls=collab.dls, model=model, loss_func=MSELossFlat())
if args.cuda:
    learn.cuda()
learn.summary()
```

    Args:
    {'dim_latent_factors': 2,
     'n_peptides': 51,
     'n_samples': 996,
     'y_range': (20, 36)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1992       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 996        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,141
    Total trainable params: 3,141
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001E86F527040>
    Loss function: FlattenedLoss of MSELoss()
    
    Callbacks:
      - TrainEvalCallback
      - Recorder
      - ProgressCallback



### Training


```python
learn.fit_one_cycle(epochs_max, 5e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.149426</td>
      <td>1.949176</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.960952</td>
      <td>0.932606</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.667276</td>
      <td>0.766322</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.615933</td>
      <td>0.677470</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.699376</td>
      <td>0.644258</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.610011</td>
      <td>0.619976</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.628306</td>
      <td>0.609396</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.545831</td>
      <td>0.602385</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.553660</td>
      <td>0.598564</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.625164</td>
      <td>0.598479</td>
      <td>00:08</td>
    </tr>
  </tbody>
</table>



```python
from vaep.models import plot_loss
from fastai import learner
learner.Recorder.plot_loss = plot_loss

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('Collab loss: Reconstruction loss.')
learn.recorder.plot_loss(skip_start=5, ax=ax)
vaep.io_images._savefig(fig, name='collab_training',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_training
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_58_1.png)
    


### Evaluation


```python
collab.dls.valid_ds.items
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>peptide</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>923</th>
      <td>185</td>
      <td>32</td>
      <td>26.963</td>
    </tr>
    <tr>
      <th>2,935</th>
      <td>594</td>
      <td>36</td>
      <td>30.123</td>
    </tr>
    <tr>
      <th>2,009</th>
      <td>405</td>
      <td>30</td>
      <td>28.125</td>
    </tr>
    <tr>
      <th>1,238</th>
      <td>247</td>
      <td>26</td>
      <td>28.075</td>
    </tr>
    <tr>
      <th>4,619</th>
      <td>936</td>
      <td>50</td>
      <td>31.638</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4,761</th>
      <td>964</td>
      <td>42</td>
      <td>30.201</td>
    </tr>
    <tr>
      <th>2,889</th>
      <td>585</td>
      <td>25</td>
      <td>29.412</td>
    </tr>
    <tr>
      <th>3,877</th>
      <td>787</td>
      <td>23</td>
      <td>33.311</td>
    </tr>
    <tr>
      <th>2,699</th>
      <td>542</td>
      <td>45</td>
      <td>32.888</td>
    </tr>
    <tr>
      <th>1,396</th>
      <td>281</td>
      <td>13</td>
      <td>27.884</td>
    </tr>
  </tbody>
</table>
<p>4923 rows × 3 columns</p>
</div>




```python
df_pred = df_pred.reset_index()
pred, target = learn.get_preds()
df_pred['intensity_pred_collab'] = pd.Series(
    pred.flatten().numpy(), index=collab.dls.valid.items.index)

npt.assert_almost_equal(
    actual=collab.dls.valid.items.intensity.to_numpy(),
    desired=target.numpy().flatten()
)


df_pred = analyzers.cast_object_to_category(df_pred)
df_pred.set_index(['Sample ID', 'peptide'], inplace=True)
df_pred
```








<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>train_median</th>
      <th>train_average</th>
      <th>replicates</th>
      <th>intensity_pred_collab</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>31.358</td>
      <td>32.424</td>
      <td>32.114</td>
      <td>31.311</td>
      <td>31.694</td>
    </tr>
    <tr>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
      <td>30.036</td>
      <td>30.528</td>
      <td>29.605</td>
      <td>29.387</td>
    </tr>
    <tr>
      <th>FGYVDFESAEDLEK</th>
      <td>29.106</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>29.449</td>
      <td>29.100</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
      <td>30.666</td>
      <td>30.489</td>
      <td>29.785</td>
      <td>29.671</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>28.240</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>28.558</td>
      <td>28.256</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <th>SLEDQVEMLR</th>
      <td>28.973</td>
      <td>28.941</td>
      <td>28.820</td>
      <td>29.179</td>
      <td>28.797</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ALLFVPR</th>
      <td>31.110</td>
      <td>31.005</td>
      <td>30.981</td>
      <td>30.844</td>
      <td>30.968</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>31.690</td>
      <td>31.268</td>
      <td>30.996</td>
      <td>31.645</td>
      <td>31.645</td>
    </tr>
    <tr>
      <th>QAQIEVVPSASALIIK</th>
      <td>30.516</td>
      <td>30.019</td>
      <td>29.921</td>
      <td>30.723</td>
      <td>30.192</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>29.820</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>29.760</td>
      <td>29.393</td>
    </tr>
  </tbody>
</table>
<p>4923 rows × 5 columns</p>
</div>




```python
assert (abs(target.reshape(-1) - pred.reshape(-1))).sum() / len(target) - \
    (df_pred.intensity - df_pred.intensity_pred_collab).abs().sum() / \
    len(df_pred) < 0.00001
```

### Plot biases and embedding weigths

- visualize relative order of samples and peptides


```python
from collections import namedtuple
def get_bias(learner, indices, is_item=True) -> pd.Series:
    ret = learner.model.bias(indices.values, is_item=is_item) # user=sample
    return pd.Series(ret, index=indices)

# def get_weigths

CollabIDs = namedtuple("CollabIDs", "sample peptide")

collab.biases = CollabIDs(
    sample=get_bias(learn, indices=analysis.df_train.index.levels[0], is_item=False), # item=peptide
    peptide=get_bias(learn, indices=analysis.df_train.index.levels[1] )
)
collab.biases.sample.head()
```




    Sample ID
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.157
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.172
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.248
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.260
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.221
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_66_1.png)
    



```python
def get_weight(learner, indices, is_item=True) -> pd.Series:
    ret = learner.model.weight(indices.values, is_item=is_item) # user=sample
    return pd.DataFrame(ret, index=indices, columns=[f'latent dimension {i+1}' for i in range(ret.shape[-1])])

collab.embeddings = CollabIDs(
    sample=get_weight(learn, indices=analysis.df_train.index.levels[0], is_item=False), # item=peptide
    peptide=get_weight(learn, indices=analysis.df_train.index.levels[1] )
)
collab.embeddings.sample.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latent dimension 1</th>
      <th>latent dimension 2</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.130</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.150</td>
      <td>0.122</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.134</td>
      <td>-0.049</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.099</td>
      <td>0.037</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.119</td>
      <td>0.203</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15, 15))
analyzers.plot_date_map(df=collab.embeddings.sample, fig=fig, ax=ax,
                        dates=analysis.df_meta.date.loc[collab.embeddings.sample.index])
vaep.io_images._savefig(fig, name='collab_latent_by_date',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_latent_by_date
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_68_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
meta_col = 'ms_instrument'

df_ = collab.embeddings.sample
analyzers.seaborn_scatter(df=df_,
                          fig=fig,
                          ax=ax,
                          meta=analysis.df_meta[meta_col].loc[df_.index],
                          title='2D sample embedding weights by MS instrument')

vaep.io_images._savefig(fig, name='collab_latent_by_ms_instrument',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_latent_by_ms_instrument
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_69_1.png)
    


## Denoising Autoencoder (DAE)

### Custom Transforms

- [x] Shift standard normalized data around
    - Error metrics won't be directly comparable afterwards


```python
from fastai.tabular.all import *
from vaep.models import ae

from fastai.tabular.core import TabularPandas

# from fastai.callback.core import Callback

from fastai.data.core import DataLoaders

from fastai.learner import Learner
from fastai.losses import MSELossFlat


# https://docs.fast.ai/tabular.core.html#FillStrategy
# from fastai.tabular.core import FillMissing
# from fastai.tabular.core import TabularPandas
```

### DataLoaders


```python
# revert format
# undo using `stack`
analysis.df_train = analysis.df_train['intensity'].unstack()
analysis.df_valid = analysis.df_valid['intensity'].unstack()
analysis.df_valid.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK</th>
      <th>SLEDQVEMLR</th>
      <th>SSEHINEGETAMLVCK</th>
      <th>TATPQQAQEVHEK</th>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <th>TVLMNPNIASVQTNEVGLK</th>
      <th>TVTAMDVVYALK</th>
      <th>TYFSCTSAHTSTGDGTAMITR</th>
      <th>VLSAPPHFHFGQTNR</th>
      <th>YNILGTNTIMDK</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>31.358</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.764</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.240</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>30.042</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.627</td>
      <td>NaN</td>
      <td>25.114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>32.533</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.960</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>31.781</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.176</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.326</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



Mean and std. dev. from training data


```python
# norm = Normalize.from_stats(analysis.df_train.mean(), analysis.df_valid.std()) # copy interface?
NORMALIZER = Normalize  # dae.NormalizeShiftedMean
```

#### Training data

procs passed to TabluarPandas are handled internally 
  1. not necessarily in order
  2. with setup call (using current training data)


```python
procs = [NORMALIZER, FillMissing(add_col=True)]
cont_names = list(analysis.df_train.columns)

to = TabularPandas(analysis.df_train, procs=procs, cont_names=cont_names)
print("Tabular object:", type(to))

to.items  # items reveals data in DataFrame
```

    Tabular object: <class 'fastai.tabular.core.TabularPandas'>
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK_na</th>
      <th>SLEDQVEMLR_na</th>
      <th>SSEHINEGETAMLVCK_na</th>
      <th>TATPQQAQEVHEK_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVLMNPNIASVQTNEVGLK_na</th>
      <th>TVTAMDVVYALK_na</th>
      <th>TYFSCTSAHTSTGDGTAMITR_na</th>
      <th>VLSAPPHFHFGQTNR_na</th>
      <th>YNILGTNTIMDK_na</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.227</td>
      <td>-0.320</td>
      <td>-0.770</td>
      <td>-0.706</td>
      <td>0.078</td>
      <td>-1.041</td>
      <td>-2.142</td>
      <td>-0.486</td>
      <td>-0.250</td>
      <td>-0.571</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.724</td>
      <td>0.015</td>
      <td>-0.828</td>
      <td>-0.724</td>
      <td>0.078</td>
      <td>-1.227</td>
      <td>-0.016</td>
      <td>-0.295</td>
      <td>-0.275</td>
      <td>-0.479</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.227</td>
      <td>-0.252</td>
      <td>0.039</td>
      <td>-0.475</td>
      <td>-0.227</td>
      <td>-0.475</td>
      <td>-1.770</td>
      <td>-0.864</td>
      <td>-0.710</td>
      <td>0.125</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.227</td>
      <td>0.101</td>
      <td>-0.286</td>
      <td>0.020</td>
      <td>-0.342</td>
      <td>-0.848</td>
      <td>-0.303</td>
      <td>-2.240</td>
      <td>-1.596</td>
      <td>-0.068</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.207</td>
      <td>-0.145</td>
      <td>-0.481</td>
      <td>-0.474</td>
      <td>0.078</td>
      <td>-0.873</td>
      <td>-0.203</td>
      <td>-0.549</td>
      <td>-0.315</td>
      <td>-0.406</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>-0.197</td>
      <td>-0.226</td>
      <td>0.988</td>
      <td>0.590</td>
      <td>0.605</td>
      <td>0.013</td>
      <td>1.312</td>
      <td>0.531</td>
      <td>-0.250</td>
      <td>0.187</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.220</td>
      <td>0.578</td>
      <td>-1.399</td>
      <td>-0.622</td>
      <td>-0.179</td>
      <td>-0.176</td>
      <td>0.165</td>
      <td>-0.035</td>
      <td>-0.358</td>
      <td>0.231</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.156</td>
      <td>0.386</td>
      <td>-0.901</td>
      <td>-0.134</td>
      <td>-3.349</td>
      <td>-0.160</td>
      <td>0.164</td>
      <td>0.333</td>
      <td>-0.287</td>
      <td>0.441</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.342</td>
      <td>-1.706</td>
      <td>-1.019</td>
      <td>-0.122</td>
      <td>0.128</td>
      <td>-0.014</td>
      <td>-0.016</td>
      <td>0.619</td>
      <td>-0.234</td>
      <td>0.365</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.492</td>
      <td>-1.503</td>
      <td>-0.719</td>
      <td>0.020</td>
      <td>0.065</td>
      <td>-0.043</td>
      <td>-0.082</td>
      <td>0.685</td>
      <td>-0.303</td>
      <td>0.139</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>995 rows × 100 columns</p>
</div>



Better manuelly apply `Transforms` on `Tabluar` type


```python
cont_names = list(analysis.df_train.columns)
to = TabularPandas(analysis.df_train, cont_names=cont_names, do_setup=False)

tf_norm = NORMALIZER()
_ = tf_norm.setups(to)  # returns to
tf_fillna = FillMissing(add_col=True)
_ = tf_fillna.setup(to)

print("Tabular object:", type(to))
# _ = (procs[0]).encodes(to)
to.items  # items reveals data in DataFrame
```

    Tabular object: <class 'fastai.tabular.core.TabularPandas'>
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK_na</th>
      <th>SLEDQVEMLR_na</th>
      <th>SSEHINEGETAMLVCK_na</th>
      <th>TATPQQAQEVHEK_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVLMNPNIASVQTNEVGLK_na</th>
      <th>TVTAMDVVYALK_na</th>
      <th>TYFSCTSAHTSTGDGTAMITR_na</th>
      <th>VLSAPPHFHFGQTNR_na</th>
      <th>YNILGTNTIMDK_na</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.246</td>
      <td>-0.302</td>
      <td>-0.732</td>
      <td>-0.664</td>
      <td>0.083</td>
      <td>-0.984</td>
      <td>-2.017</td>
      <td>-0.462</td>
      <td>-0.265</td>
      <td>-0.522</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.639</td>
      <td>0.015</td>
      <td>-0.787</td>
      <td>-0.681</td>
      <td>0.083</td>
      <td>-1.160</td>
      <td>-0.017</td>
      <td>-0.282</td>
      <td>-0.289</td>
      <td>-0.434</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.246</td>
      <td>-0.237</td>
      <td>0.033</td>
      <td>-0.446</td>
      <td>-0.206</td>
      <td>-0.448</td>
      <td>-1.668</td>
      <td>-0.818</td>
      <td>-0.702</td>
      <td>0.139</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.246</td>
      <td>0.097</td>
      <td>-0.273</td>
      <td>0.021</td>
      <td>-0.314</td>
      <td>-0.801</td>
      <td>-0.287</td>
      <td>-2.114</td>
      <td>-1.545</td>
      <td>-0.044</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.158</td>
      <td>-0.136</td>
      <td>-0.459</td>
      <td>-0.445</td>
      <td>0.083</td>
      <td>-0.825</td>
      <td>-0.193</td>
      <td>-0.521</td>
      <td>-0.327</td>
      <td>-0.365</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>-0.148</td>
      <td>-0.213</td>
      <td>0.932</td>
      <td>0.559</td>
      <td>0.581</td>
      <td>0.013</td>
      <td>1.233</td>
      <td>0.496</td>
      <td>-0.265</td>
      <td>0.198</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.240</td>
      <td>0.550</td>
      <td>-1.327</td>
      <td>-0.584</td>
      <td>-0.160</td>
      <td>-0.165</td>
      <td>0.154</td>
      <td>-0.037</td>
      <td>-0.367</td>
      <td>0.240</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.180</td>
      <td>0.368</td>
      <td>-0.856</td>
      <td>-0.124</td>
      <td>-3.156</td>
      <td>-0.150</td>
      <td>0.152</td>
      <td>0.309</td>
      <td>-0.300</td>
      <td>0.439</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.353</td>
      <td>-1.616</td>
      <td>-0.967</td>
      <td>-0.113</td>
      <td>0.130</td>
      <td>-0.011</td>
      <td>-0.017</td>
      <td>0.579</td>
      <td>-0.250</td>
      <td>0.367</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.493</td>
      <td>-1.424</td>
      <td>-0.684</td>
      <td>0.021</td>
      <td>0.070</td>
      <td>-0.040</td>
      <td>-0.079</td>
      <td>0.641</td>
      <td>-0.315</td>
      <td>0.153</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>995 rows × 100 columns</p>
</div>



Check mean and standard deviation after normalization


```python
to.items.iloc[:, :10].describe()  # not perferct anymore as expected
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
      <td>995.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.035</td>
      <td>0.002</td>
      <td>-0.003</td>
      <td>0.002</td>
      <td>0.009</td>
      <td>0.001</td>
      <td>-0.002</td>
      <td>-0.004</td>
      <td>-0.027</td>
      <td>0.021</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.931</td>
      <td>0.949</td>
      <td>0.947</td>
      <td>0.944</td>
      <td>0.946</td>
      <td>0.947</td>
      <td>0.941</td>
      <td>0.943</td>
      <td>0.951</td>
      <td>0.950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.199</td>
      <td>-4.651</td>
      <td>-4.364</td>
      <td>-5.286</td>
      <td>-4.314</td>
      <td>-4.594</td>
      <td>-2.939</td>
      <td>-4.518</td>
      <td>-3.095</td>
      <td>-6.824</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.150</td>
      <td>-0.446</td>
      <td>-0.458</td>
      <td>-0.392</td>
      <td>-0.321</td>
      <td>-0.410</td>
      <td>-0.452</td>
      <td>-0.419</td>
      <td>-0.554</td>
      <td>-0.208</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.246</td>
      <td>0.015</td>
      <td>-0.030</td>
      <td>0.021</td>
      <td>0.083</td>
      <td>0.013</td>
      <td>-0.017</td>
      <td>-0.037</td>
      <td>-0.265</td>
      <td>0.203</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.556</td>
      <td>0.578</td>
      <td>0.498</td>
      <td>0.533</td>
      <td>0.508</td>
      <td>0.488</td>
      <td>0.529</td>
      <td>0.533</td>
      <td>0.830</td>
      <td>0.550</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.702</td>
      <td>2.215</td>
      <td>2.348</td>
      <td>1.916</td>
      <td>1.763</td>
      <td>2.650</td>
      <td>1.976</td>
      <td>2.417</td>
      <td>1.929</td>
      <td>1.715</td>
    </tr>
  </tbody>
</table>
</div>



Mask is added as type bool


```python
to.items.dtypes.value_counts()
```




    float64   50
    bool      50
    dtype: int64



with the suffix `_na` where `True` is indicating a missing value replaced by the `FillMissing` transformation


```python
to.cont_names, to.cat_names
```




    ((#50) ['AAVPSGASTGIYEALELRDNDK','AHSSMVGVNLPQK','AIVAIENPADVSVISSR','ALLFVPR','ALPAVQQNNLDEDLIRK','ALTSEIALLQSR','ASNGDAWVEAHGK','DHENIVIAK','DLEEDHACIPIK','DSYVGDEAQSK'...],
     (#50) ['AAVPSGASTGIYEALELRDNDK_na','AHSSMVGVNLPQK_na','AIVAIENPADVSVISSR_na','ALLFVPR_na','ALPAVQQNNLDEDLIRK_na','ALTSEIALLQSR_na','ASNGDAWVEAHGK_na','DHENIVIAK_na','DLEEDHACIPIK_na','DSYVGDEAQSK_na'...])




```python
assert len(to.valid) == 0
```

#### Validation data

- reuse training data with different mask for evaluation
- target data is the validation data
    - switch between training and evaluation mode for setting comparison


```python
_df_valid = TabularPandas(
    analysis.df_valid, cont_names=analysis.df_valid.columns.tolist())
# assert analysis.df_valid.isna().equals(y_valid.items.isna())
_df_valid = tf_norm.encodes(_df_valid)
```


```python
_df_valid.items.iloc[:, :10].describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.132</td>
      <td>0.062</td>
      <td>0.146</td>
      <td>-0.101</td>
      <td>0.025</td>
      <td>0.102</td>
      <td>-0.169</td>
      <td>-0.071</td>
      <td>0.001</td>
      <td>-0.249</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.888</td>
      <td>0.958</td>
      <td>1.001</td>
      <td>1.002</td>
      <td>0.930</td>
      <td>1.088</td>
      <td>1.188</td>
      <td>0.974</td>
      <td>1.002</td>
      <td>1.097</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.948</td>
      <td>-2.954</td>
      <td>-3.097</td>
      <td>-4.975</td>
      <td>-3.556</td>
      <td>-4.307</td>
      <td>-2.725</td>
      <td>-3.751</td>
      <td>-1.785</td>
      <td>-5.136</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.197</td>
      <td>-0.323</td>
      <td>-0.551</td>
      <td>-0.579</td>
      <td>-0.368</td>
      <td>-0.481</td>
      <td>-1.219</td>
      <td>-0.551</td>
      <td>-0.689</td>
      <td>-0.424</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.340</td>
      <td>0.076</td>
      <td>0.087</td>
      <td>0.021</td>
      <td>0.094</td>
      <td>0.049</td>
      <td>-0.077</td>
      <td>-0.163</td>
      <td>-0.293</td>
      <td>0.032</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.689</td>
      <td>0.720</td>
      <td>0.876</td>
      <td>0.494</td>
      <td>0.473</td>
      <td>0.925</td>
      <td>0.859</td>
      <td>0.598</td>
      <td>1.093</td>
      <td>0.331</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.772</td>
      <td>1.987</td>
      <td>1.997</td>
      <td>2.133</td>
      <td>1.531</td>
      <td>2.079</td>
      <td>1.640</td>
      <td>2.061</td>
      <td>1.726</td>
      <td>1.244</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Validation dataset
# build validation DataFrame with mask according to validation data
# FillNA values in data as before, but do not add categorical columns (as this is done manuelly)
_valid_df = to.conts  # same data for predictions
_valid_df = _valid_df.join(analysis.df_valid.isna(), rsuffix='_na')  # mask
_valid_df = _valid_df.join(_df_valid.items, rsuffix='_val')  # target
_valid_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK_val</th>
      <th>SLEDQVEMLR_val</th>
      <th>SSEHINEGETAMLVCK_val</th>
      <th>TATPQQAQEVHEK_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVLMNPNIASVQTNEVGLK_val</th>
      <th>TVTAMDVVYALK_val</th>
      <th>TYFSCTSAHTSTGDGTAMITR_val</th>
      <th>VLSAPPHFHFGQTNR_val</th>
      <th>YNILGTNTIMDK_val</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.246</td>
      <td>-0.302</td>
      <td>-0.732</td>
      <td>-0.664</td>
      <td>0.083</td>
      <td>-0.984</td>
      <td>-2.017</td>
      <td>-0.462</td>
      <td>-0.265</td>
      <td>-0.522</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.616</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.639</td>
      <td>0.015</td>
      <td>-0.787</td>
      <td>-0.681</td>
      <td>0.083</td>
      <td>-1.160</td>
      <td>-0.017</td>
      <td>-0.282</td>
      <td>-0.289</td>
      <td>-0.434</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.246</td>
      <td>-0.237</td>
      <td>0.033</td>
      <td>-0.446</td>
      <td>-0.206</td>
      <td>-0.448</td>
      <td>-1.668</td>
      <td>-0.818</td>
      <td>-0.702</td>
      <td>0.139</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.400</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.246</td>
      <td>0.097</td>
      <td>-0.273</td>
      <td>0.021</td>
      <td>-0.314</td>
      <td>-0.801</td>
      <td>-0.287</td>
      <td>-2.114</td>
      <td>-1.545</td>
      <td>-0.044</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.966</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.158</td>
      <td>-0.136</td>
      <td>-0.459</td>
      <td>-0.445</td>
      <td>0.083</td>
      <td>-0.825</td>
      <td>-0.193</td>
      <td>-0.521</td>
      <td>-0.327</td>
      <td>-0.365</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>-0.148</td>
      <td>-0.213</td>
      <td>0.932</td>
      <td>0.559</td>
      <td>0.581</td>
      <td>0.013</td>
      <td>1.233</td>
      <td>0.496</td>
      <td>-0.265</td>
      <td>0.198</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.240</td>
      <td>0.550</td>
      <td>-1.327</td>
      <td>-0.584</td>
      <td>-0.160</td>
      <td>-0.165</td>
      <td>0.154</td>
      <td>-0.037</td>
      <td>-0.367</td>
      <td>0.240</td>
      <td>...</td>
      <td>-0.194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.180</td>
      <td>0.368</td>
      <td>-0.856</td>
      <td>-0.124</td>
      <td>-3.156</td>
      <td>-0.150</td>
      <td>0.152</td>
      <td>0.309</td>
      <td>-0.300</td>
      <td>0.439</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.353</td>
      <td>-1.616</td>
      <td>-0.967</td>
      <td>-0.113</td>
      <td>0.130</td>
      <td>-0.011</td>
      <td>-0.017</td>
      <td>0.579</td>
      <td>-0.250</td>
      <td>0.367</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.493</td>
      <td>-1.424</td>
      <td>-0.684</td>
      <td>0.021</td>
      <td>0.070</td>
      <td>-0.040</td>
      <td>-0.079</td>
      <td>0.641</td>
      <td>-0.315</td>
      <td>0.153</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.673</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>995 rows × 150 columns</p>
</div>




```python
# [norm, FillMissing(add_col=False)]  # mask is provided explicitly
procs = None

cont_names = list(analysis.df_train.columns)
cat_names = [f'{s}_na' for s in cont_names]
y_names = [f'{s}_val' for s in cont_names]

splits = None
y_block = None
to_valid = TabularPandas(_valid_df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                         y_names=y_names, splits=splits, y_block=y_block, do_setup=True)
to_valid.items
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK_val</th>
      <th>SLEDQVEMLR_val</th>
      <th>SSEHINEGETAMLVCK_val</th>
      <th>TATPQQAQEVHEK_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVLMNPNIASVQTNEVGLK_val</th>
      <th>TVTAMDVVYALK_val</th>
      <th>TYFSCTSAHTSTGDGTAMITR_val</th>
      <th>VLSAPPHFHFGQTNR_val</th>
      <th>YNILGTNTIMDK_val</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.246</td>
      <td>-0.302</td>
      <td>-0.732</td>
      <td>-0.664</td>
      <td>0.083</td>
      <td>-0.984</td>
      <td>-2.017</td>
      <td>-0.462</td>
      <td>-0.265</td>
      <td>-0.522</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.616</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.639</td>
      <td>0.015</td>
      <td>-0.787</td>
      <td>-0.681</td>
      <td>0.083</td>
      <td>-1.160</td>
      <td>-0.017</td>
      <td>-0.282</td>
      <td>-0.289</td>
      <td>-0.434</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.246</td>
      <td>-0.237</td>
      <td>0.033</td>
      <td>-0.446</td>
      <td>-0.206</td>
      <td>-0.448</td>
      <td>-1.668</td>
      <td>-0.818</td>
      <td>-0.702</td>
      <td>0.139</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.400</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.246</td>
      <td>0.097</td>
      <td>-0.273</td>
      <td>0.021</td>
      <td>-0.314</td>
      <td>-0.801</td>
      <td>-0.287</td>
      <td>-2.114</td>
      <td>-1.545</td>
      <td>-0.044</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.966</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.158</td>
      <td>-0.136</td>
      <td>-0.459</td>
      <td>-0.445</td>
      <td>0.083</td>
      <td>-0.825</td>
      <td>-0.193</td>
      <td>-0.521</td>
      <td>-0.327</td>
      <td>-0.365</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>-0.148</td>
      <td>-0.213</td>
      <td>0.932</td>
      <td>0.559</td>
      <td>0.581</td>
      <td>0.013</td>
      <td>1.233</td>
      <td>0.496</td>
      <td>-0.265</td>
      <td>0.198</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.240</td>
      <td>0.550</td>
      <td>-1.327</td>
      <td>-0.584</td>
      <td>-0.160</td>
      <td>-0.165</td>
      <td>0.154</td>
      <td>-0.037</td>
      <td>-0.367</td>
      <td>0.240</td>
      <td>...</td>
      <td>-0.194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.180</td>
      <td>0.368</td>
      <td>-0.856</td>
      <td>-0.124</td>
      <td>-3.156</td>
      <td>-0.150</td>
      <td>0.152</td>
      <td>0.309</td>
      <td>-0.300</td>
      <td>0.439</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.353</td>
      <td>-1.616</td>
      <td>-0.967</td>
      <td>-0.113</td>
      <td>0.130</td>
      <td>-0.011</td>
      <td>-0.017</td>
      <td>0.579</td>
      <td>-0.250</td>
      <td>0.367</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.493</td>
      <td>-1.424</td>
      <td>-0.684</td>
      <td>0.021</td>
      <td>0.070</td>
      <td>-0.040</td>
      <td>-0.079</td>
      <td>0.641</td>
      <td>-0.315</td>
      <td>0.153</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.673</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>995 rows × 150 columns</p>
</div>




```python
stats_valid = to_valid.targ.iloc[:, :100].describe()
stats_valid
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK_val</th>
      <th>AHSSMVGVNLPQK_val</th>
      <th>AIVAIENPADVSVISSR_val</th>
      <th>ALLFVPR_val</th>
      <th>ALPAVQQNNLDEDLIRK_val</th>
      <th>ALTSEIALLQSR_val</th>
      <th>ASNGDAWVEAHGK_val</th>
      <th>DHENIVIAK_val</th>
      <th>DLEEDHACIPIK_val</th>
      <th>DSYVGDEAQSK_val</th>
      <th>...</th>
      <th>SHTILLVQPTK_val</th>
      <th>SLEDQVEMLR_val</th>
      <th>SSEHINEGETAMLVCK_val</th>
      <th>TATPQQAQEVHEK_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVLMNPNIASVQTNEVGLK_val</th>
      <th>TVTAMDVVYALK_val</th>
      <th>TYFSCTSAHTSTGDGTAMITR_val</th>
      <th>VLSAPPHFHFGQTNR_val</th>
      <th>YNILGTNTIMDK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>91.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.132</td>
      <td>0.062</td>
      <td>0.146</td>
      <td>-0.101</td>
      <td>0.025</td>
      <td>0.102</td>
      <td>-0.169</td>
      <td>-0.071</td>
      <td>0.001</td>
      <td>-0.249</td>
      <td>...</td>
      <td>-0.065</td>
      <td>-0.162</td>
      <td>0.178</td>
      <td>0.003</td>
      <td>-0.145</td>
      <td>0.031</td>
      <td>0.106</td>
      <td>-0.205</td>
      <td>-0.184</td>
      <td>0.056</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.888</td>
      <td>0.958</td>
      <td>1.001</td>
      <td>1.002</td>
      <td>0.930</td>
      <td>1.088</td>
      <td>1.188</td>
      <td>0.974</td>
      <td>1.002</td>
      <td>1.097</td>
      <td>...</td>
      <td>1.071</td>
      <td>1.100</td>
      <td>1.008</td>
      <td>0.915</td>
      <td>1.086</td>
      <td>0.979</td>
      <td>1.100</td>
      <td>1.009</td>
      <td>0.958</td>
      <td>0.985</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.948</td>
      <td>-2.954</td>
      <td>-3.097</td>
      <td>-4.975</td>
      <td>-3.556</td>
      <td>-4.307</td>
      <td>-2.725</td>
      <td>-3.751</td>
      <td>-1.785</td>
      <td>-5.136</td>
      <td>...</td>
      <td>-2.920</td>
      <td>-3.281</td>
      <td>-5.625</td>
      <td>-2.865</td>
      <td>-3.388</td>
      <td>-2.990</td>
      <td>-3.800</td>
      <td>-3.795</td>
      <td>-2.295</td>
      <td>-2.378</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.197</td>
      <td>-0.323</td>
      <td>-0.551</td>
      <td>-0.579</td>
      <td>-0.368</td>
      <td>-0.481</td>
      <td>-1.219</td>
      <td>-0.551</td>
      <td>-0.689</td>
      <td>-0.424</td>
      <td>...</td>
      <td>-0.747</td>
      <td>-0.922</td>
      <td>-0.066</td>
      <td>-0.469</td>
      <td>-0.673</td>
      <td>-0.421</td>
      <td>-0.129</td>
      <td>-0.702</td>
      <td>-0.811</td>
      <td>-0.630</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.340</td>
      <td>0.076</td>
      <td>0.087</td>
      <td>0.021</td>
      <td>0.094</td>
      <td>0.049</td>
      <td>-0.077</td>
      <td>-0.163</td>
      <td>-0.293</td>
      <td>0.032</td>
      <td>...</td>
      <td>-0.331</td>
      <td>0.027</td>
      <td>0.285</td>
      <td>0.100</td>
      <td>0.149</td>
      <td>0.000</td>
      <td>0.423</td>
      <td>0.077</td>
      <td>-0.555</td>
      <td>0.106</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.689</td>
      <td>0.720</td>
      <td>0.876</td>
      <td>0.494</td>
      <td>0.473</td>
      <td>0.925</td>
      <td>0.859</td>
      <td>0.598</td>
      <td>1.093</td>
      <td>0.331</td>
      <td>...</td>
      <td>1.063</td>
      <td>0.662</td>
      <td>0.812</td>
      <td>0.644</td>
      <td>0.497</td>
      <td>0.515</td>
      <td>0.826</td>
      <td>0.478</td>
      <td>0.150</td>
      <td>0.901</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.772</td>
      <td>1.987</td>
      <td>1.997</td>
      <td>2.133</td>
      <td>1.531</td>
      <td>2.079</td>
      <td>1.640</td>
      <td>2.061</td>
      <td>1.726</td>
      <td>1.244</td>
      <td>...</td>
      <td>1.896</td>
      <td>1.708</td>
      <td>1.614</td>
      <td>1.665</td>
      <td>2.054</td>
      <td>1.868</td>
      <td>1.772</td>
      <td>1.474</td>
      <td>1.721</td>
      <td>1.900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 50 columns</p>
</div>




```python
# True = training data ("fill_na" transform sets mask to true in training data where values are replaced)
to_valid.cats
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK_na</th>
      <th>AHSSMVGVNLPQK_na</th>
      <th>AIVAIENPADVSVISSR_na</th>
      <th>ALLFVPR_na</th>
      <th>ALPAVQQNNLDEDLIRK_na</th>
      <th>ALTSEIALLQSR_na</th>
      <th>ASNGDAWVEAHGK_na</th>
      <th>DHENIVIAK_na</th>
      <th>DLEEDHACIPIK_na</th>
      <th>DSYVGDEAQSK_na</th>
      <th>...</th>
      <th>SHTILLVQPTK_na</th>
      <th>SLEDQVEMLR_na</th>
      <th>SSEHINEGETAMLVCK_na</th>
      <th>TATPQQAQEVHEK_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVLMNPNIASVQTNEVGLK_na</th>
      <th>TVTAMDVVYALK_na</th>
      <th>TYFSCTSAHTSTGDGTAMITR_na</th>
      <th>VLSAPPHFHFGQTNR_na</th>
      <th>YNILGTNTIMDK_na</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>995 rows × 50 columns</p>
</div>




```python
assert list(to_valid.cat_names) == list(
    _valid_df.select_dtypes(include='bool').columns)  # 'object'
assert to_valid.cats.equals(analysis.df_valid.isna().add_suffix('_na'))
```

### Mix and match dataloaders

- train dataloader in both TabularPandas objects used
- train dataloader in dataloaders used in both case


```python
args.batch_size
dl_train = to.dataloaders(shuffle_train=True, shuffle=False,
                          bs=args.batch_size).train  # , after_batch=after_batch)
dl_valid = to_valid.dataloaders(
    shuffle_train=False, shuffle=False, bs=args.batch_size).train
```


```python
dls = DataLoaders(dl_train, dl_valid)
b = dls.train.one_batch()
[x.shape for x in b]  # cat, cont, target
```




    [torch.Size([32, 50]), torch.Size([32, 50]), torch.Size([32, 0])]




```python
dls = DataLoaders(dl_train, dl_valid)
b = dls.valid.one_batch()
[x.shape for x in b]  # cat, cont, target
```




    [torch.Size([32, 50]), torch.Size([32, 50]), torch.Size([32, 50])]



### Model

- standard PyTorch Model from before


```python
M = analysis.df_train.shape[-1]
model = ae.Autoencoder(n_features=M, n_neurons=int(
    M/2), last_decoder_activation=None, dim_latent=latent_dim)
```

### Callbacks

- controll training loop
    - set what is data
    - what should be used for evaluation (differs for training and evaluation mode)


```python
ae.ModelAdapter
```




    vaep.models.ae.ModelAdapter



### Learner: Fastai Training Loop


```python
learn = Learner(dls=dls, model=model,
                loss_func=MSELossFlat(), cbs=ae.ModelAdapter())
```


```python
learn.show_training_loop()
```

    Start Fit
       - before_fit     : [TrainEvalCallback, Recorder, ProgressCallback]
      Start Epoch Loop
         - before_epoch   : [Recorder, ProgressCallback]
        Start Train
           - before_train   : [TrainEvalCallback, Recorder, ProgressCallback]
          Start Batch Loop
             - before_batch   : [ModelAdapter]
             - after_pred     : [ModelAdapter]
             - after_loss     : []
             - before_backward: []
             - before_step    : []
             - after_step     : []
             - after_cancel_batch: []
             - after_batch    : [TrainEvalCallback, Recorder, ProgressCallback]
          End Batch Loop
        End Train
         - after_cancel_train: [Recorder]
         - after_train    : [Recorder, ProgressCallback]
        Start Valid
           - before_validate: [TrainEvalCallback, ModelAdapter, Recorder, ProgressCallback]
          Start Batch Loop
             - **CBs same as train batch**: []
          End Batch Loop
        End Valid
         - after_cancel_validate: [Recorder]
         - after_validate : [Recorder, ProgressCallback]
      End Epoch Loop
       - after_cancel_epoch: []
       - after_epoch    : [Recorder]
    End Fit
     - after_cancel_fit: []
     - after_fit      : [ProgressCallback]
    


```python
learn.summary()
```








    Autoencoder (Input shape: 32 x 50)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 25             
    Linear                                    1275       True      
    Tanh                                                           
    ____________________________________________________________________________
                         32 x 2              
    Linear                                    52         True      
    Tanh                                                           
    ____________________________________________________________________________
                         32 x 25             
    Linear                                    75         True      
    Tanh                                                           
    ____________________________________________________________________________
                         32 x 50             
    Linear                                    1300       True      
    ____________________________________________________________________________
    
    Total params: 2,702
    Total trainable params: 2,702
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001E86F527040>
    Loss function: FlattenedLoss of MSELoss()
    
    Callbacks:
      - TrainEvalCallback
      - ModelAdapter
      - Recorder
      - ProgressCallback




```python
suggested_lr = learn.lr_find()
suggested_lr
```








    SuggestedLRs(valley=0.010964781977236271)




    
![png](latent_2D_50_10_files/latent_2D_50_10_108_2.png)
    


### Training


```python
learn.fit_one_cycle(epochs_max, lr_max=suggested_lr.valley)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.968499</td>
      <td>0.807930</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.695304</td>
      <td>0.445684</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.530367</td>
      <td>0.402602</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.450427</td>
      <td>0.388584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.409539</td>
      <td>0.390250</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.388776</td>
      <td>0.380822</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.374364</td>
      <td>0.381169</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.363574</td>
      <td>0.370051</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.352860</td>
      <td>0.368671</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.348941</td>
      <td>0.367079</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
# learn.val_preds, learn.val_targets #
```


```python
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('DAE loss: Reconstruction loss')
learn.recorder.plot_loss(skip_start=5, ax=ax)
vaep.io_images._savefig(fig, name='dae_training',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\dae_training
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4923, 4923)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3709)




```python
# check target is in expected order
Y = dls.valid.targ

npt.assert_almost_equal(
    actual=target.numpy(),
    desired=Y.stack().to_numpy()
)
```


```python
# import torch
# from fastai.tabular.core import TabularPandas

df_pred['intensity_pred_dae'] = ae.transform_preds(
    pred=pred, index=analysis.df_valid.stack().index, normalizer=tf_norm)
df_pred
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>train_median</th>
      <th>train_average</th>
      <th>replicates</th>
      <th>intensity_pred_collab</th>
      <th>intensity_pred_dae</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>31.358</td>
      <td>32.424</td>
      <td>32.114</td>
      <td>31.311</td>
      <td>31.694</td>
      <td>31.590</td>
    </tr>
    <tr>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
      <td>30.036</td>
      <td>30.528</td>
      <td>29.605</td>
      <td>29.387</td>
      <td>29.077</td>
    </tr>
    <tr>
      <th>FGYVDFESAEDLEK</th>
      <td>29.106</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>29.449</td>
      <td>29.100</td>
      <td>28.918</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
      <td>30.666</td>
      <td>30.489</td>
      <td>29.785</td>
      <td>29.671</td>
      <td>29.630</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>28.240</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>28.558</td>
      <td>28.256</td>
      <td>28.317</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <th>SLEDQVEMLR</th>
      <td>28.973</td>
      <td>28.941</td>
      <td>28.820</td>
      <td>29.179</td>
      <td>28.797</td>
      <td>28.616</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ALLFVPR</th>
      <td>31.110</td>
      <td>31.005</td>
      <td>30.981</td>
      <td>30.844</td>
      <td>30.968</td>
      <td>30.900</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>31.690</td>
      <td>31.268</td>
      <td>30.996</td>
      <td>31.645</td>
      <td>31.645</td>
      <td>31.267</td>
    </tr>
    <tr>
      <th>QAQIEVVPSASALIIK</th>
      <td>30.516</td>
      <td>30.019</td>
      <td>29.921</td>
      <td>30.723</td>
      <td>30.192</td>
      <td>30.066</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>29.820</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>29.760</td>
      <td>29.393</td>
      <td>29.319</td>
    </tr>
  </tbody>
</table>
<p>4923 rows × 6 columns</p>
</div>



### 2D plot of latent space

- 2 dimensional latent space: just plot
- more than 2 dimensional: PCA, etc


```python
latent_space = []
for b in dls.valid:
    model_input = b[1]
    latent_space.append(model.encoder(model_input).detach().numpy())

df_dae_latent = build_df_from_pred_batches(latent_space,
                                           index=_df_valid.items.index,
                                           columns=[f'latent dimension {i+1}' for i in range(latent_dim)])
df_dae_latent.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latent dimension 1</th>
      <th>latent dimension 2</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.672</td>
      <td>-0.453</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.232</td>
      <td>-0.232</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.856</td>
      <td>-0.672</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.770</td>
      <td>-0.689</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.439</td>
      <td>-0.532</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15, 15))
analyzers.plot_date_map(df=df_dae_latent, fig=fig, ax=ax,
                        dates=analysis.df_meta.date.loc[df_dae_latent.index])
vaep.io_images._savefig(fig, name='dae_latent_by_date',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\dae_latent_by_date
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_122_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))

meta_col = 'ms_instrument'

analyzers.seaborn_scatter(df=df_dae_latent,
                          fig=fig,
                          ax=ax,
                          meta=analysis.df_meta[meta_col].loc[df_dae_latent.index],
                          title='by MS instrument')

vaep.io_images._savefig(
    fig, name=f'dae_latent_by_{meta_col}', folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\dae_latent_by_ms_instrument
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_123_1.png)
    


## Variational Autoencoder (VAE)

### Scikit Learn MinMaxScaler

- [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)


```python
from vaep.transform import MinMaxScaler

args_vae = {}
args_vae['SCALER'] = MinMaxScaler
# select initial data: transformed vs not log transformed
scaler = args_vae['SCALER']().fit(analysis.df_train)
scaler.transform(analysis.df_valid.iloc[:5])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <th>AHSSMVGVNLPQK</th>
      <th>AIVAIENPADVSVISSR</th>
      <th>ALLFVPR</th>
      <th>ALPAVQQNNLDEDLIRK</th>
      <th>ALTSEIALLQSR</th>
      <th>ASNGDAWVEAHGK</th>
      <th>DHENIVIAK</th>
      <th>DLEEDHACIPIK</th>
      <th>DSYVGDEAQSK</th>
      <th>...</th>
      <th>SHTILLVQPTK</th>
      <th>SLEDQVEMLR</th>
      <th>SSEHINEGETAMLVCK</th>
      <th>TATPQQAQEVHEK</th>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <th>TVLMNPNIASVQTNEVGLK</th>
      <th>TVTAMDVVYALK</th>
      <th>TYFSCTSAHTSTGDGTAMITR</th>
      <th>VLSAPPHFHFGQTNR</th>
      <th>YNILGTNTIMDK</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>0.708</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.534</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.620</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>0.651</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.595</td>
      <td>NaN</td>
      <td>0.220</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.827</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.751</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.637</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.533</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



### DataLoaders

- follow instructions for using plain PyTorch Datasets, see [tutorial](https://docs.fast.ai/tutorial.siamese.html#Preparing-the-data)



```python
assert all(analysis.df_train.columns == analysis.df_valid.columns)
if not all(analysis.df.columns == analysis.df_train.columns):
    print("analysis.df columns are not the same as analysis.df_train")
    # ToDo: DataLoading has to be cleaned up
    # analysis.df = analysis.df_train.fillna(analysis.df_valid)
```

    analysis.df columns are not the same as analysis.df_train
    


```python
from vaep.io.datasets import PeptideDatasetInMemory

FILL_NA = 0.0

train_ds = PeptideDatasetInMemory(data=scaler.transform(
    analysis.df_train).to_numpy(dtype=None), fill_na=FILL_NA)
valid_ds = PeptideDatasetInMemory(data=scaler.transform(analysis.df_train.fillna(analysis.df_valid)).to_numpy(dtype=None),
                                  mask=analysis.df_valid.notna().to_numpy(), fill_na=FILL_NA)

assert (train_ds.peptides == valid_ds.peptides).all()
```


```python
dls = DataLoaders.from_dsets(train_ds, valid_ds, n_inp=2)
```

    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
    

### Model


```python
from torch.nn import Sigmoid

M = analysis.df_train.shape[-1]
model = ae.VAE(n_features=M, n_neurons=int(
    M/2), last_encoder_activation=None, last_decoder_activation=Sigmoid, dim_latent=latent_dim)
```

### Learner


```python
learn = Learner(dls=dls,
                model=model,
                loss_func=ae.loss_fct_vae,
                cbs=ae.ModelAdapterVAE())

learn.show_training_loop()
learn.summary()
```

    Start Fit
       - before_fit     : [TrainEvalCallback, Recorder, ProgressCallback]
      Start Epoch Loop
         - before_epoch   : [Recorder, ProgressCallback]
        Start Train
           - before_train   : [TrainEvalCallback, Recorder, ProgressCallback]
          Start Batch Loop
             - before_batch   : [ModelAdapterVAE]
             - after_pred     : [ModelAdapterVAE]
             - after_loss     : []
             - before_backward: []
             - before_step    : []
             - after_step     : []
             - after_cancel_batch: []
             - after_batch    : [TrainEvalCallback, Recorder, ProgressCallback]
          End Batch Loop
        End Train
         - after_cancel_train: [Recorder]
         - after_train    : [Recorder, ProgressCallback]
        Start Valid
           - before_validate: [TrainEvalCallback, Recorder, ProgressCallback]
          Start Batch Loop
             - **CBs same as train batch**: []
          End Batch Loop
        End Valid
         - after_cancel_validate: [Recorder]
         - after_validate : [Recorder, ProgressCallback]
      End Epoch Loop
       - after_cancel_epoch: []
       - after_epoch    : [Recorder]
    End Fit
     - after_cancel_fit: []
     - after_fit      : [ProgressCallback]
    








    VAE (Input shape: 64 x 50)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         64 x 25             
    Linear                                    75         True      
    Tanh                                                           
    ____________________________________________________________________________
                         64 x 50             
    Linear                                    1300       True      
    Sigmoid                                                        
    ____________________________________________________________________________
                         64 x 25             
    Linear                                    1275       True      
    Tanh                                                           
    ____________________________________________________________________________
                         64 x 4              
    Linear                                    104        True      
    ____________________________________________________________________________
    
    Total params: 2,754
    Total trainable params: 2,754
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001E86F527040>
    Loss function: <function loss_fct_vae at 0x000001E86F545940>
    
    Callbacks:
      - TrainEvalCallback
      - ModelAdapterVAE
      - Recorder
      - ProgressCallback



### Training


```python
suggested_lr = learn.lr_find()
suggested_lr
```








    SuggestedLRs(valley=0.00363078061491251)




    
![png](latent_2D_50_10_files/latent_2D_50_10_136_2.png)
    



```python
learn.fit_one_cycle(epochs_max, lr_max=suggested_lr.valley)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2008.388062</td>
      <td>217.378143</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1963.695312</td>
      <td>207.495682</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1892.144043</td>
      <td>197.567886</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1833.081299</td>
      <td>192.132599</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1795.163574</td>
      <td>189.988235</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1769.522827</td>
      <td>189.317657</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1752.220337</td>
      <td>189.198853</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1740.677002</td>
      <td>189.216080</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1731.904785</td>
      <td>189.102783</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1726.687500</td>
      <td>189.116043</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('VAE loss: Reconstruction loss and Kullback-Leiber-Divergence for latent space')
learn.recorder.plot_loss(skip_start=5, ax=ax)
vaep.io_images._savefig(fig, name='vae_training',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\vae_training
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4923)




```python
len(pred[0])
```




    4923




```python
learn.loss_func(pred, target)
```




    tensor(2985.9663)




```python
_pred = pd.Series(pred[0], index=analysis.df_valid.stack().index).unstack()
_pred = scaler.inverse_transform(_pred).stack()

df_pred['intensity_pred_vae'] = _pred
df_pred
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>train_median</th>
      <th>train_average</th>
      <th>replicates</th>
      <th>intensity_pred_collab</th>
      <th>intensity_pred_dae</th>
      <th>intensity_pred_vae</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELRDNDK</th>
      <td>31.358</td>
      <td>32.424</td>
      <td>32.114</td>
      <td>31.311</td>
      <td>31.694</td>
      <td>31.590</td>
      <td>32.341</td>
    </tr>
    <tr>
      <th>DLEEDHACIPIK</th>
      <td>29.764</td>
      <td>30.036</td>
      <td>30.528</td>
      <td>29.605</td>
      <td>29.387</td>
      <td>29.077</td>
      <td>30.649</td>
    </tr>
    <tr>
      <th>FGYVDFESAEDLEK</th>
      <td>29.106</td>
      <td>30.022</td>
      <td>30.048</td>
      <td>29.449</td>
      <td>29.100</td>
      <td>28.918</td>
      <td>30.260</td>
    </tr>
    <tr>
      <th>IGDLQAFQGHGAGNLAGLK</th>
      <td>29.867</td>
      <td>30.666</td>
      <td>30.489</td>
      <td>29.785</td>
      <td>29.671</td>
      <td>29.630</td>
      <td>30.623</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>28.240</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>28.558</td>
      <td>28.256</td>
      <td>28.317</td>
      <td>29.155</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <th>SLEDQVEMLR</th>
      <td>28.973</td>
      <td>28.941</td>
      <td>28.820</td>
      <td>29.179</td>
      <td>28.797</td>
      <td>28.616</td>
      <td>28.982</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ALLFVPR</th>
      <td>31.110</td>
      <td>31.005</td>
      <td>30.981</td>
      <td>30.844</td>
      <td>30.968</td>
      <td>30.900</td>
      <td>31.087</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>31.690</td>
      <td>31.268</td>
      <td>30.996</td>
      <td>31.645</td>
      <td>31.645</td>
      <td>31.267</td>
      <td>31.007</td>
    </tr>
    <tr>
      <th>QAQIEVVPSASALIIK</th>
      <td>30.516</td>
      <td>30.019</td>
      <td>29.921</td>
      <td>30.723</td>
      <td>30.192</td>
      <td>30.066</td>
      <td>30.054</td>
    </tr>
    <tr>
      <th>SSEHINEGETAMLVCK</th>
      <td>29.820</td>
      <td>29.273</td>
      <td>28.994</td>
      <td>29.760</td>
      <td>29.393</td>
      <td>29.319</td>
      <td>29.157</td>
    </tr>
  </tbody>
</table>
<p>4923 rows × 7 columns</p>
</div>



### Add plot of latent space

- 2 dimensional latent space: just plot
- more than 2 dimensional: PCA, etc


```python
latent_space = []
for b in dls.valid:
    model_input = b[0]
    b_mu, b_std = model.get_mu_and_logvar(model_input, detach=True)
    latent_space.append(b_mu)


df_vae_latent = build_df_from_pred_batches(latent_space,
                                           index=_df_valid.items.index,
                                           columns=[f'latent dimension {i+1}' for i in range(latent_dim)])
df_vae_latent.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latent dimension 1</th>
      <th>latent dimension 2</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <td>-0.041</td>
      <td>0.082</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.085</td>
      <td>0.096</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.057</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.071</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.079</td>
      <td>0.087</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15, 15))
analyzers.plot_date_map(df=df_vae_latent, fig=fig, ax=ax,
                        dates=analysis.df_meta.date.loc[df_vae_latent.index])
vaep.io_images._savefig(fig, name='vae_latent_by_date',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\vae_latent_by_date
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_146_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))

meta_col = 'ms_instrument'

analyzers.seaborn_scatter(df=df_vae_latent,
                          fig=fig,
                          ax=ax,
                          meta=analysis.df_meta[meta_col].loc[df_vae_latent.index],
                          title='by MS instrument')

vaep.io_images._savefig(
    fig, name=f'vae_latent_by_{meta_col}', folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\vae_latent_by_ms_instrument
    


    
![png](latent_2D_50_10_files/latent_2D_50_10_147_1.png)
    


## Compare the 3 models

- replicates: replace NAs with neighbouring ("close") values
- train average, median: Replace NA with average or median from training data


```python
import sklearn.metrics as sklm
pred_columns = df_pred.columns[1:]
scoring = [('MSE', sklm.mean_squared_error),
           ('MAE', sklm.mean_absolute_error)]

y_true = df_pred['intensity']

metrics = {}
for col in pred_columns:
    _y_pred = df_pred[col].dropna()
    if len(df_pred[col]) > len(_y_pred):
        logger.info(
            f"Drop indices for {col}: {[(idx[0], idx[1]) for idx in df_pred[col].index.difference(_y_pred.index)]}")

    metrics[col] = dict(
        [(k, f(y_true=y_true.loc[_y_pred.index], y_pred=_y_pred))
         for k, f in scoring]
    )

metrics = pd.DataFrame(metrics)
metrics.to_csv(folder / f'exp_02_metrics.csv',
               float_format='{:.3f}'.format)
metrics.sort_values(by=[k for k, f in scoring], axis=1)
```

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03', 'IIAPPERK'), ('20190111_QE8_nLC1_ASD_QC_HeLa_01', 'SSEHINEGETAMLVCK'), ('20190115_QE2_NLC10_TW_QC_MNT_HeLa_01', 'IWHHTFYNELR'), ('20190118_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'TATPQQAQEVHEK'), ('20190122_QE6_nLC6_SIS_QC_MNT_HeLa_01', 'FGYVDFESAEDLEK'), ('20190126_QE6_nLC6_SIS_QC_MNT_HeLa_05', 'KTEAPAAPAAQETK'), ('20190204_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01', 'MALIGLGVSHPVLK'), ('20190219_QE10_nLC14_FaCo_QC_HeLa_50cm_20190221093339', 'VLSAPPHFHFGQTNR'), ('20190219_QE2_NLC1_GP_QC_MNT_HELA_01', 'VLSAPPHFHFGQTNR'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_6', 'DHENIVIAK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_8', 'EQISDIDDAVR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_30', 'EAAENSLVAYK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_25', 'VLSAPPHFHFGQTNR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_27', 'IIAPPERK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_18_30', 'IIAPPERK'), ('20190228_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'TVLMNPNIASVQTNEVGLK'), ('20190318_QE2_NLC1_AB_MNT_HELA_01', 'ASNGDAWVEAHGK'), ('20190318_QE7_nLC5_MJ_HeLa_MNT_01', 'LMDVGLIAIR'), ('20190324_QE7_nLC3_RJC_WIMS_QC_MNT_HeLa_02', 'HLAGLGLTEAIDK'), ('20190325_QE9_nLC0_JM_MNT_Hela_50cm_01', 'GANDFMCDEMER'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01_20190326190317', 'EMNDAAMFYTNR'), ('20190331_QE10_nLC13_LiNi_QC_45cm_HeLa_01', 'FQSSHHPTDITSLDQYVER'), ('20190408_QE4_LC12_IAH_QC_MNT_HeLa_04', 'TYFSCTSAHTSTGDGTAMITR'), ('20190415_QE10_nLC9_LiNi_QC_MNT_45cm_HeLa_01', 'LLLGAGAVAYGVR'), ('20190416_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-002', 'IIAPPERK'), ('20190416_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-003', 'IIAPPERK'), ('20190417_QX4_JoSw_MA_HeLa_500ng_BR14_new', 'SEIDLFNIRK'), ('20190420_QE8_nLC14_RG_QC_HeLa_02', 'EMNDAAMFYTNR'), ('20190422_QE4_LC12_JE-IAH_QC_MNT_HeLa_01', 'EMNDAAMFYTNR'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'PLRLPLQDVYK'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'VLSAPPHFHFGQTNR'), ('20190425_QX8_JuSc_MA_HeLa_500ng_1', 'NYIQGINLVQAK'), ('20190428_QE9_nLC0_LiNi_QC_45cm_HeLa_ending', 'IIAPPERK'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336', 'IIAPPERK'), ('20190429_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'AAVPSGASTGIYEALELRDNDK'), ('20190429_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'PLRLPLQDVYK'), ('20190506_QX7_ChDe_MA_HeLa_500ng', 'DSYVGDEAQSK'), ('20190509_QE4_LC12_AS_QC_MNT_HeLa_02', 'DSYVGDEAQSK'), ('20190511_QX0_ChDe_MA_HeLa_500ng_LC07_1_BR14', 'SLEDQVEMLR'), ('20190513_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'TLTAVHDAILEDLVFPSEIVGK'), ('20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01', 'FEELNMDLFR'), ('20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01', 'SHTILLVQPTK'), ('20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01', 'YNILGTNTIMDK'), ('20190514_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'AIVAIENPADVSVISSR'), ('20190514_QX4_JiYu_MA_HeLa_500ng', 'AIVAIENPADVSVISSR'), ('20190515_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'AIVAIENPADVSVISSR'), ('20190521_QX4_JoSw_MA_HeLa_500ng', 'HLAGLGLTEAIDK'), ('20190521_QX6_AsJa_MA_HeLa_Br14_500ng_LC09', 'ALLFVPR'), ('20190524_QE4_LC12_IAH_QC_MNT_HeLa_02', 'PLRLPLQDVYK'), ('20190527_QX1_PhGe_MA_HeLa_500ng_LC10', 'EAAENSLVAYK'), ('20190527_QX3_LiSc_MA_Hela_500ng_LC15_190527171650', 'NYIQGINLVQAK'), ('20190603_QX3_AnSe_MA_Hela_500ng_LC15_190603172414', 'PLRLPLQDVYK'), ('20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_02', 'EAAENSLVAYK'), ('20190604_QX8_MiWi_MA_HeLa_BR14_500ng', 'EAAENSLVAYK'), ('20190605_QX0_MePh_MA_HeLa_500ng_LC07_1_BR14', 'EAAENSLVAYK'), ('20190615_QE9_nLC0_FaCo_QC_MNT_Hela_50cm', 'TVTAMDVVYALK'), ('20190617_QE_LC_UHG_QC_MNT_HELA_03', 'ILLTEPPMNPTK'), ('20190623_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_MUC_01', 'FGYVDFESAEDLEK'), ('20190624_QE6_LC4_AS_QC_MNT_HeLa_02', 'DSYVGDEAQSK'), ('20190624_QX3_MaMu_MA_Hela_500ng_LC15', 'ALTSEIALLQSR'), ('20190625_QE1_nLC2_GP_QC_MNT_HELA_03', 'IIAPPERK'), ('20190626_QE2_NLC1_JM_QC_MNT_HELA_01', 'HEQNIDCGGGYVK'), ('20190626_QE2_NLC1_JM_QC_MNT_HELA_02', 'HEQNIDCGGGYVK'), ('20190626_QX8_ChDe_MA_HeLa_BR14_500ng', 'TVLMNPNIASVQTNEVGLK'), ('20190626_QX8_ChDe_MA_HeLa_BR14_500ng_190626194235', 'ESEPQAAAEPAEAK'), ('20190701_QE7_nLC7_BKH_MNT_QC_HeLa_01', 'HEQNIDCGGGYVK'), ('20190702_QE10_nLC0_FaCo_QC_MNT_HeLa_MUC', 'PLRLPLQDVYK'), ('20190703_QX4_MaTa_MA_HeLa_500ng_MAX_ALLOWED', 'LMDVGLIAIR'), ('20190709_QE3_nLC5_GF_QC_MNT_Hela_02', 'DSYVGDEAQSK'), ('20190710_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'EQISDIDDAVR'), ('20190712_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'TLTAVHDAILEDLVFPSEIVGK'), ('20190712_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'TLTAVHDAILEDLVFPSEIVGK'), ('20190722_QX3_MiWi_MA_Hela_500ng_LC15', 'EAAENSLVAYK'), ('20190723_QE4_LC12_IAH_QC_MNT_HeLa_02', 'SSEHINEGETAMLVCK'), ('20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'IWHHTFYNELR'), ('20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'NLDIERPTYTNLNR'), ('20190728_QX3_MiWi_MA_Hela_500ng_LC15', 'MALIGLGVSHPVLK'), ('20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02', 'KQELEEICHDLEAR'), ('20190731_QE8_nLC14_ASD_QC_MNT_HeLa_03', 'FGYVDFESAEDLEK'), ('20190731_QX8_ChSc_MA_HeLa_500ng', 'EMNDAAMFYTNR'), ('20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_01_20190801145136', 'NLDIERPTYTNLNR'), ('20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05', 'DLEEDHACIPIK'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_300ng', 'KQELEEICHDLEAR')]
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intensity_pred_collab</th>
      <th>intensity_pred_dae</th>
      <th>replicates</th>
      <th>intensity_pred_vae</th>
      <th>train_average</th>
      <th>train_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSE</th>
      <td>0.598</td>
      <td>0.672</td>
      <td>1.634</td>
      <td>1.925</td>
      <td>1.931</td>
      <td>1.985</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.474</td>
      <td>0.504</td>
      <td>0.871</td>
      <td>1.004</td>
      <td>1.016</td>
      <td>0.993</td>
    </tr>
  </tbody>
</table>
</div>



Save final prediction values of validation data for later comparison.


```python
df_pred.to_csv(folder /
               f"{config.FOLDER_DATA}_valid_pred.csv")
```

## PCA plot for imputed and denoised data

two setups:
 - impute missing values
 - additinally change observed values


```python

```
