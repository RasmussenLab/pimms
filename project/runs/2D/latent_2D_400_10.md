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
n_feat = 400
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
    DFTVSAMHGDMDQK                     997
    LTGMAFR                            975
    AAVPSGASTGIYEALELR                 955
    TGVHHYSGNNIELGTACGK                998
    FEDENFILK                          995
    MDATANDVPSPYEVR                    998
    TATESFASDPILYRPVAVALDTK            999
    SEIDLFNIRK                         951
    ELSDIAHR                           995
    TTHFVEGGDAGNREDQINR                996
    TYDATTHFETTCDDIK                   996
    SNFAEALAAHK                        997
    AHGPGLEGGLVGKPAEFTIDTK             964
    ACANPAAGSVILLENLR                  957
    GADFLVTEVENGGSLGSK                 997
    LSVLGAITSVQQR                      959
    FIIPQIVK                           995
    IQLINNMLDK                         986
    TLQIFNIEMK                         989
    ATQALVLAPTR                        965
    YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK   1,000
    VLITTDLLAR                         997
    VNFAMNVGK                          987
    TITLEVEPSDTIENVK                 1,000
    FVPAEMGTHTVSVK                     992
    TAFQEALDAAGDK                    1,000
    VLTVINQTQK                         990
    LGDVYVNDAFGTAHR                    983
    VAYVSFGPHAGK                       980
    SSILLDVKPWDDETDMAK                 990
    DVQIGDIVTVGECRPLSK                 980
    IAGYVTHLMK                         994
    LALVTGGEIASTFDHPELVK               997
    THLPGFVEQAEALK                   1,000
    EHALLAYTLGVK                     1,000
    NTGIICTIGPASR                      954
    TAFDEAIAELDTLNEDSYK                974
    FWEVISDEHGIDPTGTYHGDSDLQLDR        971
    AQLLQPTLEINPR                      998
    LNSVQSSERPLFLVHPIEGSTTVFHSLASR     952
    LVGQGASAVLLDLPNSGGEAQAK            975
    KFDQLLAEEK                         995
    LGGSAVISLEGKPL                     993
    HLPTLDHPIIPADYVAIK                 997
    HLEINPDHSIIETLR                    998
    ATAVVDGAFK                         974
    LMDVGLIAIR                         996
    ILDQGEDFPASEMTR                    993
    TATPQQAQEVHEK                      997
    ETNLDSLPLVDTHSK                    975
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
      <th>DFTVSAMHGDMDQK</th>
      <td>28.586</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>31.017</td>
    </tr>
    <tr>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
    </tr>
    <tr>
      <th>TGVHHYSGNNIELGTACGK</th>
      <td>29.271</td>
    </tr>
    <tr>
      <th>FEDENFILK</th>
      <td>31.761</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ATAVVDGAFK</th>
      <td>30.326</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
    </tr>
    <tr>
      <th>ILDQGEDFPASEMTR</th>
      <td>26.447</td>
    </tr>
    <tr>
      <th>TATPQQAQEVHEK</th>
      <td>32.062</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.479</td>
    </tr>
  </tbody>
</table>
<p>49296 rows × 1 columns</p>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_24_1.png)
    



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
      <td>0.986</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.024</td>
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
      <th>DFTVSAMHGDMDQK</th>
      <td>28.586</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>31.017</td>
    </tr>
    <tr>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
    </tr>
    <tr>
      <th>TGVHHYSGNNIELGTACGK</th>
      <td>29.271</td>
    </tr>
    <tr>
      <th>FEDENFILK</th>
      <td>31.761</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ATAVVDGAFK</th>
      <td>30.326</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
    </tr>
    <tr>
      <th>ILDQGEDFPASEMTR</th>
      <td>26.447</td>
    </tr>
    <tr>
      <th>TATPQQAQEVHEK</th>
      <td>32.062</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.479</td>
    </tr>
  </tbody>
</table>
<p>49296 rows × 1 columns</p>
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
      <th>20190415_QE10_nLC9_LiNi_QC_45cm_HeLa_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>29.747</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>29.046</td>
    </tr>
    <tr>
      <th>20190625_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>28.139</td>
    </tr>
    <tr>
      <th>20190528_QX8_MiWi_MA_HeLa_BR14_500ng</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>34.117</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_4</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>29.334</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190426_QX1_JoMu_MA_HeLa_500ng_LC11</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
      <td>32.196</td>
    </tr>
    <tr>
      <th>20190715_QE2_NLC1_ANHO_MNT_HELA_03</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
      <td>28.914</td>
    </tr>
    <tr>
      <th>20190513_QE6_LC4_IAH_QC_MNT_HeLa_01</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
      <td>29.678</td>
    </tr>
    <tr>
      <th>20190506_QX6_ChDe_MA_HeLa_Br13_500ng_LC09</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
      <td>30.733</td>
    </tr>
    <tr>
      <th>20190531_QE4_nLC12_MM_QC_MNT_HELA_01_20190604231757</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
      <td>26.679</td>
    </tr>
  </tbody>
</table>
<p>44368 rows × 1 columns</p>
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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
      <td>29.355</td>
      <td>30.376</td>
      <td>30.189</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.200</td>
      <td>30.004</td>
      <td>30.017</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>29.572</td>
      <td>29.123</td>
      <td>29.743</td>
      <td>29.458</td>
    </tr>
    <tr>
      <th>TLQIFNIEMK</th>
      <td>27.540</td>
      <td>28.825</td>
      <td>28.591</td>
      <td>27.546</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>30.412</td>
      <td>31.210</td>
      <td>31.017</td>
      <td>30.844</td>
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
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ELSDIAHR</th>
      <td>31.582</td>
      <td>31.429</td>
      <td>31.309</td>
      <td>31.294</td>
    </tr>
    <tr>
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.776</td>
      <td>30.108</td>
      <td>27.612</td>
    </tr>
    <tr>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
      <td>30.491</td>
      <td>30.395</td>
      <td>31.195</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.438</td>
      <td>29.526</td>
      <td>27.558</td>
    </tr>
    <tr>
      <th>SSILLDVKPWDDETDMAK</th>
      <td>30.523</td>
      <td>30.107</td>
      <td>29.931</td>
      <td>28.145</td>
    </tr>
  </tbody>
</table>
<p>4928 rows × 4 columns</p>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <th>ATAVVDGAFK</th>
      <td>30.067</td>
      <td>30.200</td>
      <td>30.004</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03</th>
      <th>LALVTGGEIASTFDHPELVK</th>
      <td>30.674</td>
      <td>30.482</td>
      <td>30.409</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181228_QE6_nLC6_CSC_QC_MNT_HeLa_01</th>
      <th>FIIPQIVK</th>
      <td>32.131</td>
      <td>31.775</td>
      <td>31.722</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE2_NLC10_ANHO_QC_MNT_HELA_02</th>
      <th>FVPAEMGTHTVSVK</th>
      <td>31.047</td>
      <td>30.146</td>
      <td>29.735</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_01</th>
      <th>LNSVQSSERPLFLVHPIEGSTTVFHSLASR</th>
      <td>31.253</td>
      <td>30.577</td>
      <td>30.381</td>
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
      <th>20190718_QE8_nLC14_RG_QC_HeLa_MUC_50cm_1</th>
      <th>AQLLQPTLEINPR</th>
      <td>28.579</td>
      <td>28.363</td>
      <td>28.551</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190722_QX8_ChSc_MA_HeLa_500ng_190722174431</th>
      <th>LALVTGGEIASTFDHPELVK</th>
      <td>32.400</td>
      <td>30.482</td>
      <td>30.409</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>34.012</td>
      <td>29.355</td>
      <td>30.376</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01</th>
      <th>TTHFVEGGDAGNREDQINR</th>
      <td>31.605</td>
      <td>30.137</td>
      <td>30.539</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05</th>
      <th>TYDATTHFETTCDDIK</th>
      <td>30.868</td>
      <td>29.145</td>
      <td>29.257</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 4 columns</p>
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
      <td>ACANPAAGSVILLENLR</td>
      <td>30.430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AHGPGLEGGLVGKPAEFTIDTK</td>
      <td>27.649</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AQLLQPTLEINPR</td>
      <td>27.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ATQALVLAPTR</td>
      <td>29.220</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DFTVSAMHGDMDQK</td>
      <td>28.586</td>
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
      <td>AAVPSGASTGIYEALELR</td>
      <td>30.134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ATAVVDGAFK</td>
      <td>29.988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ETNLDSLPLVDTHSK</td>
      <td>29.572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TLQIFNIEMK</td>
      <td>27.540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>ACANPAAGSVILLENLR</td>
      <td>30.412</td>
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
      <td>20190510_QE1_nLC2_ANHO_QC_MNT_HELA_07</td>
      <td>FVPAEMGTHTVSVK</td>
      <td>31.550</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190204_QE2_NLC10_ANHO_QC_MNT_HELA_02</td>
      <td>TLQIFNIEMK</td>
      <td>29.439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190625_QX7_IgPa_MA_HeLa_Br14_500ng</td>
      <td>LGGSAVISLEGKPL</td>
      <td>32.048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190606_QE4_LC12_JE_QC_MNT_HeLa_02</td>
      <td>NTGIICTIGPASR</td>
      <td>31.064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190429_QX0_ChDe_MA_HeLa_500ng_LC07_1_BR13_190507121913</td>
      <td>SSILLDVKPWDDETDMAK</td>
      <td>31.890</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190731_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05</td>
      <td>TATPQQAQEVHEK</td>
      <td>33.247</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03</td>
      <td>ELSDIAHR</td>
      <td>31.028</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190411_QE3_nLC5_DS_QC_MNT_HeLa_01</td>
      <td>NTGIICTIGPASR</td>
      <td>31.296</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_1</td>
      <td>VLITTDLLAR</td>
      <td>28.087</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190104_QE6_nLC6_MM_QC_MNT_HELA_02_190107214303</td>
      <td>VNFAMNVGK</td>
      <td>29.516</td>
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
      <td>20190604_QX8_MiWi_MA_HeLa_BR14_500ng</td>
      <td>TGVHHYSGNNIELGTACGK</td>
      <td>31.261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_02</td>
      <td>SSILLDVKPWDDETDMAK</td>
      <td>30.419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190415_QE10_nLC9_LiNi_QC_45cm_HeLa_02</td>
      <td>LTGMAFR</td>
      <td>29.840</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190111_QE8_nLC1_ASD_QC_HeLa_02</td>
      <td>DFTVSAMHGDMDQK</td>
      <td>29.884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190716_QE8_nLC14_RG_QC_MNT_HeLa_MUC_50cm_2</td>
      <td>KFDQLLAEEK</td>
      <td>28.047</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190404_QE7_nLC3_AL_QC_MNT_HeLa_02</td>
      <td>DFTVSAMHGDMDQK</td>
      <td>28.989</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190204_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>VLITTDLLAR</td>
      <td>29.661</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190220_QE3_nLC7_TSB_QC_MNT_HELA_02</td>
      <td>LMDVGLIAIR</td>
      <td>25.651</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190521_QE7_nLC5_TSB_QC_MNT_HeLa_01</td>
      <td>SEIDLFNIRK</td>
      <td>31.161</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190509_QX6_ChDe_MA_HeLa_Br14_500ng_LC09_20190509120700</td>
      <td>LNSVQSSERPLFLVHPIEGSTTVFHSLASR</td>
      <td>30.473</td>
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




    (1378, 154)



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
    
    Optimizer used: <function Adam at 0x00000198F1837040>
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
      <td>1.981202</td>
      <td>1.803697</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.786131</td>
      <td>0.760438</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.638046</td>
      <td>0.632467</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.633735</td>
      <td>0.598015</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.506649</td>
      <td>0.586428</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.510225</td>
      <td>0.570241</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.612370</td>
      <td>0.558717</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.496692</td>
      <td>0.553455</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.523756</td>
      <td>0.551424</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.484973</td>
      <td>0.550786</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_58_1.png)
    


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
      <th>3,085</th>
      <td>629</td>
      <td>40</td>
      <td>31.261</td>
    </tr>
    <tr>
      <th>621</th>
      <td>125</td>
      <td>35</td>
      <td>30.419</td>
    </tr>
    <tr>
      <th>1,926</th>
      <td>382</td>
      <td>29</td>
      <td>29.840</td>
    </tr>
    <tr>
      <th>210</th>
      <td>45</td>
      <td>7</td>
      <td>29.884</td>
    </tr>
    <tr>
      <th>4,291</th>
      <td>871</td>
      <td>22</td>
      <td>28.047</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2,134</th>
      <td>425</td>
      <td>6</td>
      <td>31.057</td>
    </tr>
    <tr>
      <th>4,509</th>
      <td>913</td>
      <td>43</td>
      <td>28.322</td>
    </tr>
    <tr>
      <th>2,685</th>
      <td>544</td>
      <td>1</td>
      <td>28.728</td>
    </tr>
    <tr>
      <th>1,995</th>
      <td>393</td>
      <td>10</td>
      <td>30.882</td>
    </tr>
    <tr>
      <th>4,215</th>
      <td>857</td>
      <td>4</td>
      <td>28.884</td>
    </tr>
  </tbody>
</table>
<p>4928 rows × 3 columns</p>
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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
      <td>29.355</td>
      <td>30.376</td>
      <td>30.189</td>
      <td>29.939</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.200</td>
      <td>30.004</td>
      <td>30.017</td>
      <td>28.952</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>29.572</td>
      <td>29.123</td>
      <td>29.743</td>
      <td>29.458</td>
      <td>29.119</td>
    </tr>
    <tr>
      <th>TLQIFNIEMK</th>
      <td>27.540</td>
      <td>28.825</td>
      <td>28.591</td>
      <td>27.546</td>
      <td>27.401</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>30.412</td>
      <td>31.210</td>
      <td>31.017</td>
      <td>30.844</td>
      <td>30.301</td>
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
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ELSDIAHR</th>
      <td>31.582</td>
      <td>31.429</td>
      <td>31.309</td>
      <td>31.294</td>
      <td>31.099</td>
    </tr>
    <tr>
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.776</td>
      <td>30.108</td>
      <td>27.612</td>
      <td>29.417</td>
    </tr>
    <tr>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
      <td>30.491</td>
      <td>30.395</td>
      <td>31.195</td>
      <td>30.347</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.438</td>
      <td>29.526</td>
      <td>27.558</td>
      <td>29.229</td>
    </tr>
    <tr>
      <th>SSILLDVKPWDDETDMAK</th>
      <td>30.523</td>
      <td>30.107</td>
      <td>29.931</td>
      <td>28.145</td>
      <td>30.000</td>
    </tr>
  </tbody>
</table>
<p>4928 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.075
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.123
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.250
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.185
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.254
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_66_1.png)
    



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
      <td>0.301</td>
      <td>0.186</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.155</td>
      <td>0.225</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.065</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.112</td>
      <td>0.280</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.184</td>
      <td>0.138</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_68_1.png)
    



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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_69_1.png)
    


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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK</th>
      <th>TITLEVEPSDTIENVK</th>
      <th>TLQIFNIEMK</th>
      <th>TTHFVEGGDAGNREDQINR</th>
      <th>TYDATTHFETTCDDIK</th>
      <th>VAYVSFGPHAGK</th>
      <th>VLITTDLLAR</th>
      <th>VLTVINQTQK</th>
      <th>VNFAMNVGK</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
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
      <td>30.134</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.988</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.540</td>
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
      <td>30.412</td>
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
      <td>27.201</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>28.291</td>
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
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.990</td>
      <td>30.036</td>
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
      <td>30.004</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.067</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.808</td>
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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_na</th>
      <th>TITLEVEPSDTIENVK_na</th>
      <th>TLQIFNIEMK_na</th>
      <th>TTHFVEGGDAGNREDQINR_na</th>
      <th>TYDATTHFETTCDDIK_na</th>
      <th>VAYVSFGPHAGK_na</th>
      <th>VLITTDLLAR_na</th>
      <th>VLTVINQTQK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_na</th>
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
      <td>-0.394</td>
      <td>-0.655</td>
      <td>-0.853</td>
      <td>-0.933</td>
      <td>0.137</td>
      <td>0.026</td>
      <td>-1.308</td>
      <td>-0.091</td>
      <td>-0.055</td>
      <td>-0.709</td>
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
      <td>-0.019</td>
      <td>0.178</td>
      <td>-0.787</td>
      <td>-1.052</td>
      <td>0.015</td>
      <td>0.037</td>
      <td>-1.329</td>
      <td>-0.159</td>
      <td>-0.172</td>
      <td>-0.199</td>
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
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.394</td>
      <td>0.228</td>
      <td>-0.185</td>
      <td>-0.331</td>
      <td>-0.033</td>
      <td>-0.636</td>
      <td>-0.782</td>
      <td>-0.514</td>
      <td>-1.805</td>
      <td>-0.699</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.164</td>
      <td>-0.453</td>
      <td>-1.056</td>
      <td>-0.125</td>
      <td>0.137</td>
      <td>0.065</td>
      <td>-1.178</td>
      <td>0.208</td>
      <td>-0.031</td>
      <td>-0.336</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.698</td>
      <td>0.798</td>
      <td>-0.794</td>
      <td>-0.607</td>
      <td>0.137</td>
      <td>0.116</td>
      <td>-1.240</td>
      <td>0.133</td>
      <td>-0.036</td>
      <td>-1.020</td>
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
      <td>1.547</td>
      <td>0.920</td>
      <td>-0.598</td>
      <td>1.006</td>
      <td>0.413</td>
      <td>1.046</td>
      <td>0.579</td>
      <td>1.288</td>
      <td>0.667</td>
      <td>0.769</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.643</td>
      <td>0.241</td>
      <td>0.211</td>
      <td>-0.236</td>
      <td>0.412</td>
      <td>-0.707</td>
      <td>0.567</td>
      <td>-0.079</td>
      <td>-0.572</td>
      <td>0.021</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.614</td>
      <td>0.178</td>
      <td>0.425</td>
      <td>-0.532</td>
      <td>0.268</td>
      <td>-0.818</td>
      <td>0.450</td>
      <td>-0.084</td>
      <td>0.456</td>
      <td>-0.019</td>
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
      <td>-0.394</td>
      <td>-1.270</td>
      <td>0.510</td>
      <td>-0.591</td>
      <td>0.240</td>
      <td>-0.969</td>
      <td>0.311</td>
      <td>-2.197</td>
      <td>-0.419</td>
      <td>-0.023</td>
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
      <td>-0.696</td>
      <td>0.699</td>
      <td>0.828</td>
      <td>-0.125</td>
      <td>0.237</td>
      <td>-0.858</td>
      <td>0.038</td>
      <td>-0.369</td>
      <td>-0.748</td>
      <td>0.082</td>
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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_na</th>
      <th>TITLEVEPSDTIENVK_na</th>
      <th>TLQIFNIEMK_na</th>
      <th>TTHFVEGGDAGNREDQINR_na</th>
      <th>TYDATTHFETTCDDIK_na</th>
      <th>VAYVSFGPHAGK_na</th>
      <th>VLITTDLLAR_na</th>
      <th>VLTVINQTQK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_na</th>
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
      <td>-0.431</td>
      <td>-0.582</td>
      <td>-0.767</td>
      <td>-0.898</td>
      <td>0.147</td>
      <td>-0.014</td>
      <td>-1.235</td>
      <td>-0.096</td>
      <td>-0.056</td>
      <td>-0.662</td>
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
      <td>-0.079</td>
      <td>0.192</td>
      <td>-0.706</td>
      <td>-1.011</td>
      <td>0.032</td>
      <td>-0.004</td>
      <td>-1.254</td>
      <td>-0.159</td>
      <td>-0.167</td>
      <td>-0.179</td>
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
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.431</td>
      <td>0.238</td>
      <td>-0.143</td>
      <td>-0.327</td>
      <td>-0.013</td>
      <td>-0.634</td>
      <td>-0.736</td>
      <td>-0.493</td>
      <td>-1.716</td>
      <td>-0.652</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.093</td>
      <td>-0.394</td>
      <td>-0.957</td>
      <td>-0.132</td>
      <td>0.147</td>
      <td>0.022</td>
      <td>-1.111</td>
      <td>0.186</td>
      <td>-0.033</td>
      <td>-0.309</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.595</td>
      <td>0.769</td>
      <td>-0.712</td>
      <td>-0.589</td>
      <td>0.147</td>
      <td>0.071</td>
      <td>-1.170</td>
      <td>0.115</td>
      <td>-0.038</td>
      <td>-0.956</td>
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
      <td>1.392</td>
      <td>0.883</td>
      <td>-0.529</td>
      <td>0.940</td>
      <td>0.405</td>
      <td>0.941</td>
      <td>0.553</td>
      <td>1.199</td>
      <td>0.629</td>
      <td>0.737</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.665</td>
      <td>0.251</td>
      <td>0.227</td>
      <td>-0.237</td>
      <td>0.404</td>
      <td>-0.701</td>
      <td>0.541</td>
      <td>-0.085</td>
      <td>-0.547</td>
      <td>0.029</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.638</td>
      <td>0.192</td>
      <td>0.427</td>
      <td>-0.518</td>
      <td>0.270</td>
      <td>-0.804</td>
      <td>0.430</td>
      <td>-0.089</td>
      <td>0.428</td>
      <td>-0.009</td>
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
      <td>-0.431</td>
      <td>-1.154</td>
      <td>0.507</td>
      <td>-0.574</td>
      <td>0.243</td>
      <td>-0.946</td>
      <td>0.298</td>
      <td>-2.073</td>
      <td>-0.401</td>
      <td>-0.013</td>
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
      <td>-0.714</td>
      <td>0.677</td>
      <td>0.804</td>
      <td>-0.132</td>
      <td>0.240</td>
      <td>-0.842</td>
      <td>0.040</td>
      <td>-0.357</td>
      <td>-0.714</td>
      <td>0.087</td>
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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
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
      <td>-0.061</td>
      <td>0.027</td>
      <td>0.030</td>
      <td>-0.014</td>
      <td>0.018</td>
      <td>-0.038</td>
      <td>0.004</td>
      <td>-0.010</td>
      <td>-0.004</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.939</td>
      <td>0.930</td>
      <td>0.935</td>
      <td>0.949</td>
      <td>0.938</td>
      <td>0.937</td>
      <td>0.947</td>
      <td>0.940</td>
      <td>0.949</td>
      <td>0.947</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.584</td>
      <td>-4.841</td>
      <td>-4.832</td>
      <td>-3.583</td>
      <td>-4.463</td>
      <td>-3.455</td>
      <td>-5.762</td>
      <td>-3.589</td>
      <td>-4.280</td>
      <td>-5.886</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.714</td>
      <td>-0.364</td>
      <td>-0.230</td>
      <td>-0.543</td>
      <td>-0.190</td>
      <td>-0.626</td>
      <td>-0.388</td>
      <td>-0.412</td>
      <td>-0.535</td>
      <td>-0.327</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.431</td>
      <td>0.192</td>
      <td>0.227</td>
      <td>-0.132</td>
      <td>0.147</td>
      <td>-0.288</td>
      <td>0.040</td>
      <td>-0.085</td>
      <td>-0.038</td>
      <td>0.087</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.914</td>
      <td>0.623</td>
      <td>0.608</td>
      <td>0.475</td>
      <td>0.500</td>
      <td>0.712</td>
      <td>0.490</td>
      <td>0.497</td>
      <td>0.637</td>
      <td>0.549</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.911</td>
      <td>1.894</td>
      <td>2.201</td>
      <td>2.357</td>
      <td>1.719</td>
      <td>2.122</td>
      <td>2.062</td>
      <td>2.222</td>
      <td>1.857</td>
      <td>1.810</td>
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




    ((#50) ['AAVPSGASTGIYEALELR','ACANPAAGSVILLENLR','AHGPGLEGGLVGKPAEFTIDTK','AQLLQPTLEINPR','ATAVVDGAFK','ATQALVLAPTR','DFTVSAMHGDMDQK','DVQIGDIVTVGECRPLSK','EHALLAYTLGVK','ELSDIAHR'...],
     (#50) ['AAVPSGASTGIYEALELR_na','ACANPAAGSVILLENLR_na','AHGPGLEGGLVGKPAEFTIDTK_na','AQLLQPTLEINPR_na','ATAVVDGAFK_na','ATQALVLAPTR_na','DFTVSAMHGDMDQK_na','DVQIGDIVTVGECRPLSK_na','EHALLAYTLGVK_na','ELSDIAHR_na'...])




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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>96.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>97.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.103</td>
      <td>0.200</td>
      <td>-0.072</td>
      <td>0.053</td>
      <td>0.058</td>
      <td>-0.037</td>
      <td>0.134</td>
      <td>0.072</td>
      <td>-0.079</td>
      <td>-0.167</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.934</td>
      <td>0.998</td>
      <td>1.009</td>
      <td>0.979</td>
      <td>1.055</td>
      <td>0.934</td>
      <td>1.046</td>
      <td>0.995</td>
      <td>1.038</td>
      <td>1.140</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.973</td>
      <td>-3.288</td>
      <td>-3.022</td>
      <td>-2.810</td>
      <td>-3.459</td>
      <td>-1.840</td>
      <td>-4.763</td>
      <td>-3.069</td>
      <td>-3.687</td>
      <td>-5.201</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.746</td>
      <td>-0.131</td>
      <td>-0.623</td>
      <td>-0.567</td>
      <td>-0.255</td>
      <td>-0.723</td>
      <td>-0.270</td>
      <td>-0.544</td>
      <td>-0.519</td>
      <td>-0.568</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.493</td>
      <td>0.344</td>
      <td>0.212</td>
      <td>-0.080</td>
      <td>0.246</td>
      <td>-0.302</td>
      <td>0.118</td>
      <td>-0.026</td>
      <td>-0.112</td>
      <td>-0.124</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.808</td>
      <td>0.891</td>
      <td>0.625</td>
      <td>0.960</td>
      <td>0.718</td>
      <td>0.704</td>
      <td>0.680</td>
      <td>0.967</td>
      <td>0.602</td>
      <td>0.457</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.674</td>
      <td>2.101</td>
      <td>1.518</td>
      <td>1.947</td>
      <td>1.636</td>
      <td>2.235</td>
      <td>2.016</td>
      <td>1.790</td>
      <td>1.814</td>
      <td>1.437</td>
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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_val</th>
      <th>TITLEVEPSDTIENVK_val</th>
      <th>TLQIFNIEMK_val</th>
      <th>TTHFVEGGDAGNREDQINR_val</th>
      <th>TYDATTHFETTCDDIK_val</th>
      <th>VAYVSFGPHAGK_val</th>
      <th>VLITTDLLAR_val</th>
      <th>VLTVINQTQK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_val</th>
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
      <td>-0.431</td>
      <td>-0.582</td>
      <td>-0.767</td>
      <td>-0.898</td>
      <td>0.147</td>
      <td>-0.014</td>
      <td>-1.235</td>
      <td>-0.096</td>
      <td>-0.056</td>
      <td>-0.662</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.909</td>
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
      <td>-0.079</td>
      <td>0.192</td>
      <td>-0.706</td>
      <td>-1.011</td>
      <td>0.032</td>
      <td>-0.004</td>
      <td>-1.254</td>
      <td>-0.159</td>
      <td>-0.167</td>
      <td>-0.179</td>
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
      <td>-1.456</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.431</td>
      <td>0.238</td>
      <td>-0.143</td>
      <td>-0.327</td>
      <td>-0.013</td>
      <td>-0.634</td>
      <td>-0.736</td>
      <td>-0.493</td>
      <td>-1.716</td>
      <td>-0.652</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.093</td>
      <td>-0.394</td>
      <td>-0.957</td>
      <td>-0.132</td>
      <td>0.147</td>
      <td>0.022</td>
      <td>-1.111</td>
      <td>0.186</td>
      <td>-0.033</td>
      <td>-0.309</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.649</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.595</td>
      <td>0.769</td>
      <td>-0.712</td>
      <td>-0.589</td>
      <td>0.147</td>
      <td>0.071</td>
      <td>-1.170</td>
      <td>0.115</td>
      <td>-0.038</td>
      <td>-0.956</td>
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
      <td>1.392</td>
      <td>0.883</td>
      <td>-0.529</td>
      <td>0.940</td>
      <td>0.405</td>
      <td>0.941</td>
      <td>0.553</td>
      <td>1.199</td>
      <td>0.629</td>
      <td>0.737</td>
      <td>...</td>
      <td>0.985</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.665</td>
      <td>0.251</td>
      <td>0.227</td>
      <td>-0.237</td>
      <td>0.404</td>
      <td>-0.701</td>
      <td>0.541</td>
      <td>-0.085</td>
      <td>-0.547</td>
      <td>0.029</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.381</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.154</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.638</td>
      <td>0.192</td>
      <td>0.427</td>
      <td>-0.518</td>
      <td>0.270</td>
      <td>-0.804</td>
      <td>0.430</td>
      <td>-0.089</td>
      <td>0.428</td>
      <td>-0.009</td>
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
      <td>-0.431</td>
      <td>-1.154</td>
      <td>0.507</td>
      <td>-0.574</td>
      <td>0.243</td>
      <td>-0.946</td>
      <td>0.298</td>
      <td>-2.073</td>
      <td>-0.401</td>
      <td>-0.013</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.236</td>
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
      <td>-0.714</td>
      <td>0.677</td>
      <td>0.804</td>
      <td>-0.132</td>
      <td>0.240</td>
      <td>-0.842</td>
      <td>0.040</td>
      <td>-0.357</td>
      <td>-0.714</td>
      <td>0.087</td>
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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_val</th>
      <th>TITLEVEPSDTIENVK_val</th>
      <th>TLQIFNIEMK_val</th>
      <th>TTHFVEGGDAGNREDQINR_val</th>
      <th>TYDATTHFETTCDDIK_val</th>
      <th>VAYVSFGPHAGK_val</th>
      <th>VLITTDLLAR_val</th>
      <th>VLTVINQTQK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_val</th>
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
      <td>-0.431</td>
      <td>-0.582</td>
      <td>-0.767</td>
      <td>-0.898</td>
      <td>0.147</td>
      <td>-0.014</td>
      <td>-1.235</td>
      <td>-0.096</td>
      <td>-0.056</td>
      <td>-0.662</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.909</td>
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
      <td>-0.079</td>
      <td>0.192</td>
      <td>-0.706</td>
      <td>-1.011</td>
      <td>0.032</td>
      <td>-0.004</td>
      <td>-1.254</td>
      <td>-0.159</td>
      <td>-0.167</td>
      <td>-0.179</td>
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
      <td>-1.456</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.431</td>
      <td>0.238</td>
      <td>-0.143</td>
      <td>-0.327</td>
      <td>-0.013</td>
      <td>-0.634</td>
      <td>-0.736</td>
      <td>-0.493</td>
      <td>-1.716</td>
      <td>-0.652</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.093</td>
      <td>-0.394</td>
      <td>-0.957</td>
      <td>-0.132</td>
      <td>0.147</td>
      <td>0.022</td>
      <td>-1.111</td>
      <td>0.186</td>
      <td>-0.033</td>
      <td>-0.309</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.649</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.595</td>
      <td>0.769</td>
      <td>-0.712</td>
      <td>-0.589</td>
      <td>0.147</td>
      <td>0.071</td>
      <td>-1.170</td>
      <td>0.115</td>
      <td>-0.038</td>
      <td>-0.956</td>
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
      <td>1.392</td>
      <td>0.883</td>
      <td>-0.529</td>
      <td>0.940</td>
      <td>0.405</td>
      <td>0.941</td>
      <td>0.553</td>
      <td>1.199</td>
      <td>0.629</td>
      <td>0.737</td>
      <td>...</td>
      <td>0.985</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.665</td>
      <td>0.251</td>
      <td>0.227</td>
      <td>-0.237</td>
      <td>0.404</td>
      <td>-0.701</td>
      <td>0.541</td>
      <td>-0.085</td>
      <td>-0.547</td>
      <td>0.029</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.381</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.154</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.638</td>
      <td>0.192</td>
      <td>0.427</td>
      <td>-0.518</td>
      <td>0.270</td>
      <td>-0.804</td>
      <td>0.430</td>
      <td>-0.089</td>
      <td>0.428</td>
      <td>-0.009</td>
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
      <td>-0.431</td>
      <td>-1.154</td>
      <td>0.507</td>
      <td>-0.574</td>
      <td>0.243</td>
      <td>-0.946</td>
      <td>0.298</td>
      <td>-2.073</td>
      <td>-0.401</td>
      <td>-0.013</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.236</td>
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
      <td>-0.714</td>
      <td>0.677</td>
      <td>0.804</td>
      <td>-0.132</td>
      <td>0.240</td>
      <td>-0.842</td>
      <td>0.040</td>
      <td>-0.357</td>
      <td>-0.714</td>
      <td>0.087</td>
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
      <th>AAVPSGASTGIYEALELR_val</th>
      <th>ACANPAAGSVILLENLR_val</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK_val</th>
      <th>AQLLQPTLEINPR_val</th>
      <th>ATAVVDGAFK_val</th>
      <th>ATQALVLAPTR_val</th>
      <th>DFTVSAMHGDMDQK_val</th>
      <th>DVQIGDIVTVGECRPLSK_val</th>
      <th>EHALLAYTLGVK_val</th>
      <th>ELSDIAHR_val</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_val</th>
      <th>TITLEVEPSDTIENVK_val</th>
      <th>TLQIFNIEMK_val</th>
      <th>TTHFVEGGDAGNREDQINR_val</th>
      <th>TYDATTHFETTCDDIK_val</th>
      <th>VAYVSFGPHAGK_val</th>
      <th>VLITTDLLAR_val</th>
      <th>VLTVINQTQK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>96.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>97.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>...</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.103</td>
      <td>0.200</td>
      <td>-0.072</td>
      <td>0.053</td>
      <td>0.058</td>
      <td>-0.037</td>
      <td>0.134</td>
      <td>0.072</td>
      <td>-0.079</td>
      <td>-0.167</td>
      <td>...</td>
      <td>-0.096</td>
      <td>0.031</td>
      <td>-0.047</td>
      <td>-0.095</td>
      <td>0.002</td>
      <td>-0.036</td>
      <td>-0.065</td>
      <td>0.001</td>
      <td>-0.091</td>
      <td>-0.025</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.934</td>
      <td>0.998</td>
      <td>1.009</td>
      <td>0.979</td>
      <td>1.055</td>
      <td>0.934</td>
      <td>1.046</td>
      <td>0.995</td>
      <td>1.038</td>
      <td>1.140</td>
      <td>...</td>
      <td>0.978</td>
      <td>0.864</td>
      <td>1.112</td>
      <td>1.200</td>
      <td>0.818</td>
      <td>1.011</td>
      <td>0.856</td>
      <td>1.014</td>
      <td>0.927</td>
      <td>1.111</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.973</td>
      <td>-3.288</td>
      <td>-3.022</td>
      <td>-2.810</td>
      <td>-3.459</td>
      <td>-1.840</td>
      <td>-4.763</td>
      <td>-3.069</td>
      <td>-3.687</td>
      <td>-5.201</td>
      <td>...</td>
      <td>-2.599</td>
      <td>-3.113</td>
      <td>-4.150</td>
      <td>-4.475</td>
      <td>-2.566</td>
      <td>-3.952</td>
      <td>-2.831</td>
      <td>-3.588</td>
      <td>-2.387</td>
      <td>-3.512</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.746</td>
      <td>-0.131</td>
      <td>-0.623</td>
      <td>-0.567</td>
      <td>-0.255</td>
      <td>-0.723</td>
      <td>-0.270</td>
      <td>-0.544</td>
      <td>-0.519</td>
      <td>-0.568</td>
      <td>...</td>
      <td>-0.627</td>
      <td>-0.426</td>
      <td>-0.232</td>
      <td>-0.852</td>
      <td>-0.521</td>
      <td>-0.418</td>
      <td>-0.566</td>
      <td>-0.448</td>
      <td>-0.607</td>
      <td>-0.534</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.493</td>
      <td>0.344</td>
      <td>0.212</td>
      <td>-0.080</td>
      <td>0.246</td>
      <td>-0.302</td>
      <td>0.118</td>
      <td>-0.026</td>
      <td>-0.112</td>
      <td>-0.124</td>
      <td>...</td>
      <td>-0.137</td>
      <td>0.139</td>
      <td>0.241</td>
      <td>-0.309</td>
      <td>-0.003</td>
      <td>0.039</td>
      <td>-0.041</td>
      <td>0.035</td>
      <td>-0.338</td>
      <td>0.089</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.808</td>
      <td>0.891</td>
      <td>0.625</td>
      <td>0.960</td>
      <td>0.718</td>
      <td>0.704</td>
      <td>0.680</td>
      <td>0.967</td>
      <td>0.602</td>
      <td>0.457</td>
      <td>...</td>
      <td>0.413</td>
      <td>0.598</td>
      <td>0.610</td>
      <td>1.036</td>
      <td>0.293</td>
      <td>0.539</td>
      <td>0.397</td>
      <td>0.760</td>
      <td>0.637</td>
      <td>0.608</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.674</td>
      <td>2.101</td>
      <td>1.518</td>
      <td>1.947</td>
      <td>1.636</td>
      <td>2.235</td>
      <td>2.016</td>
      <td>1.790</td>
      <td>1.814</td>
      <td>1.437</td>
      <td>...</td>
      <td>2.032</td>
      <td>1.806</td>
      <td>1.480</td>
      <td>2.127</td>
      <td>1.864</td>
      <td>1.925</td>
      <td>1.620</td>
      <td>1.588</td>
      <td>1.870</td>
      <td>2.410</td>
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
      <th>AAVPSGASTGIYEALELR_na</th>
      <th>ACANPAAGSVILLENLR_na</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK_na</th>
      <th>AQLLQPTLEINPR_na</th>
      <th>ATAVVDGAFK_na</th>
      <th>ATQALVLAPTR_na</th>
      <th>DFTVSAMHGDMDQK_na</th>
      <th>DVQIGDIVTVGECRPLSK_na</th>
      <th>EHALLAYTLGVK_na</th>
      <th>ELSDIAHR_na</th>
      <th>...</th>
      <th>THLPGFVEQAEALK_na</th>
      <th>TITLEVEPSDTIENVK_na</th>
      <th>TLQIFNIEMK_na</th>
      <th>TTHFVEGGDAGNREDQINR_na</th>
      <th>TYDATTHFETTCDDIK_na</th>
      <th>VAYVSFGPHAGK_na</th>
      <th>VLITTDLLAR_na</th>
      <th>VLTVINQTQK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK_na</th>
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
      <td>False</td>
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
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
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
      <td>True</td>
      <td>True</td>
      <td>True</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
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
      <td>False</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
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
      <td>True</td>
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
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
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
    
    Optimizer used: <function Adam at 0x00000198F1837040>
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








    SuggestedLRs(valley=0.0063095735386013985)




    
![png](latent_2D_400_10_files/latent_2D_400_10_108_2.png)
    


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
      <td>0.960389</td>
      <td>0.800686</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.694432</td>
      <td>0.406642</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.509491</td>
      <td>0.344599</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.416974</td>
      <td>0.323897</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.369971</td>
      <td>0.312906</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.340372</td>
      <td>0.306111</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.323770</td>
      <td>0.304895</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.313772</td>
      <td>0.302952</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.305704</td>
      <td>0.300414</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.302792</td>
      <td>0.301964</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4928, 4928)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3005)




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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
      <td>29.355</td>
      <td>30.376</td>
      <td>30.189</td>
      <td>29.939</td>
      <td>29.234</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.200</td>
      <td>30.004</td>
      <td>30.017</td>
      <td>28.952</td>
      <td>28.625</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>29.572</td>
      <td>29.123</td>
      <td>29.743</td>
      <td>29.458</td>
      <td>29.119</td>
      <td>28.638</td>
    </tr>
    <tr>
      <th>TLQIFNIEMK</th>
      <td>27.540</td>
      <td>28.825</td>
      <td>28.591</td>
      <td>27.546</td>
      <td>27.401</td>
      <td>26.856</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>30.412</td>
      <td>31.210</td>
      <td>31.017</td>
      <td>30.844</td>
      <td>30.301</td>
      <td>29.997</td>
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
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ELSDIAHR</th>
      <td>31.582</td>
      <td>31.429</td>
      <td>31.309</td>
      <td>31.294</td>
      <td>31.099</td>
      <td>31.016</td>
    </tr>
    <tr>
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.776</td>
      <td>30.108</td>
      <td>27.612</td>
      <td>29.417</td>
      <td>29.425</td>
    </tr>
    <tr>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
      <td>30.491</td>
      <td>30.395</td>
      <td>31.195</td>
      <td>30.347</td>
      <td>30.274</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.438</td>
      <td>29.526</td>
      <td>27.558</td>
      <td>29.229</td>
      <td>29.237</td>
    </tr>
    <tr>
      <th>SSILLDVKPWDDETDMAK</th>
      <td>30.523</td>
      <td>30.107</td>
      <td>29.931</td>
      <td>28.145</td>
      <td>30.000</td>
      <td>29.821</td>
    </tr>
  </tbody>
</table>
<p>4928 rows × 6 columns</p>
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
      <td>-0.087</td>
      <td>-0.357</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.002</td>
      <td>-0.282</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.617</td>
      <td>-0.705</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.220</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.323</td>
      <td>-0.280</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_122_1.png)
    



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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_123_1.png)
    


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
      <th>AAVPSGASTGIYEALELR</th>
      <th>ACANPAAGSVILLENLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AQLLQPTLEINPR</th>
      <th>ATAVVDGAFK</th>
      <th>ATQALVLAPTR</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DVQIGDIVTVGECRPLSK</th>
      <th>EHALLAYTLGVK</th>
      <th>ELSDIAHR</th>
      <th>...</th>
      <th>THLPGFVEQAEALK</th>
      <th>TITLEVEPSDTIENVK</th>
      <th>TLQIFNIEMK</th>
      <th>TTHFVEGGDAGNREDQINR</th>
      <th>TYDATTHFETTCDDIK</th>
      <th>VAYVSFGPHAGK</th>
      <th>VLITTDLLAR</th>
      <th>VLTVINQTQK</th>
      <th>VNFAMNVGK</th>
      <th>YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK</th>
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
      <td>0.552</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.637</td>
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
      <td>0.630</td>
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
      <td>0.350</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.379</td>
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
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.537</td>
      <td>0.726</td>
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
      <td>0.635</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.730</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.703</td>
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
    
    Optimizer used: <function Adam at 0x00000198F1837040>
    Loss function: <function loss_fct_vae at 0x00000198F1854940>
    
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








    SuggestedLRs(valley=0.005248074419796467)




    
![png](latent_2D_400_10_files/latent_2D_400_10_136_2.png)
    



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
      <td>1978.838623</td>
      <td>220.671448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1927.004883</td>
      <td>211.857315</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1859.598633</td>
      <td>200.139099</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1816.534546</td>
      <td>197.365402</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1787.527100</td>
      <td>197.981628</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1768.001831</td>
      <td>203.443756</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1753.424316</td>
      <td>204.681076</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1742.513550</td>
      <td>205.166031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1735.343018</td>
      <td>205.929565</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1730.480957</td>
      <td>205.916885</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4928)




```python
len(pred[0])
```




    4928




```python
learn.loss_func(pred, target)
```




    tensor(3249.8103)




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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAVPSGASTGIYEALELR</th>
      <td>30.134</td>
      <td>29.355</td>
      <td>30.376</td>
      <td>30.189</td>
      <td>29.939</td>
      <td>29.234</td>
      <td>30.015</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.200</td>
      <td>30.004</td>
      <td>30.017</td>
      <td>28.952</td>
      <td>28.625</td>
      <td>30.135</td>
    </tr>
    <tr>
      <th>ETNLDSLPLVDTHSK</th>
      <td>29.572</td>
      <td>29.123</td>
      <td>29.743</td>
      <td>29.458</td>
      <td>29.119</td>
      <td>28.638</td>
      <td>29.373</td>
    </tr>
    <tr>
      <th>TLQIFNIEMK</th>
      <td>27.540</td>
      <td>28.825</td>
      <td>28.591</td>
      <td>27.546</td>
      <td>27.401</td>
      <td>26.856</td>
      <td>28.716</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>30.412</td>
      <td>31.210</td>
      <td>31.017</td>
      <td>30.844</td>
      <td>30.301</td>
      <td>29.997</td>
      <td>31.122</td>
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
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>ELSDIAHR</th>
      <td>31.582</td>
      <td>31.429</td>
      <td>31.309</td>
      <td>31.294</td>
      <td>31.099</td>
      <td>31.016</td>
      <td>31.042</td>
    </tr>
    <tr>
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.776</td>
      <td>30.108</td>
      <td>27.612</td>
      <td>29.417</td>
      <td>29.425</td>
      <td>29.431</td>
    </tr>
    <tr>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
      <td>30.491</td>
      <td>30.395</td>
      <td>31.195</td>
      <td>30.347</td>
      <td>30.274</td>
      <td>30.143</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.438</td>
      <td>29.526</td>
      <td>27.558</td>
      <td>29.229</td>
      <td>29.237</td>
      <td>29.235</td>
    </tr>
    <tr>
      <th>SSILLDVKPWDDETDMAK</th>
      <td>30.523</td>
      <td>30.107</td>
      <td>29.931</td>
      <td>28.145</td>
      <td>30.000</td>
      <td>29.821</td>
      <td>29.608</td>
    </tr>
  </tbody>
</table>
<p>4928 rows × 7 columns</p>
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
      <td>0.152</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.057</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.510</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.314</td>
      <td>-0.048</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.153</td>
      <td>0.041</td>
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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_146_1.png)
    



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
    


    
![png](latent_2D_400_10_files/latent_2D_400_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'ATAVVDGAFK'), ('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03', 'LALVTGGEIASTFDHPELVK'), ('20181228_QE6_nLC6_CSC_QC_MNT_HeLa_01', 'FIIPQIVK'), ('20190111_QE2_NLC10_ANHO_QC_MNT_HELA_02', 'FVPAEMGTHTVSVK'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_01', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_02', 'KFDQLLAEEK'), ('20190126_QE6_nLC6_SIS_QC_MNT_HeLa_04', 'LSVLGAITSVQQR'), ('20190129_QE10_nLC0_FM_QC_MNT_HeLa_50cm_01_20190130121912', 'VNFAMNVGK'), ('20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01', 'IAGYVTHLMK'), ('20190203_QE3_nLC3_KBE_QC_MNT_HeLa_02', 'NTGIICTIGPASR'), ('20190205_QE7_nLC7_MEM_QC_MNT_HeLa_01', 'LVGQGASAVLLDLPNSGGEAQAK'), ('20190208_QE2_NLC1_AB_QC_MNT_HELA_4', 'LTGMAFR'), ('20190211_QE6_LC6_AS_QC_MNT_HeLa_03', 'THLPGFVEQAEALK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_01', 'TAFDEAIAELDTLNEDSYK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_02', 'TAFDEAIAELDTLNEDSYK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_03', 'LALVTGGEIASTFDHPELVK'), ('20190219_QE10_nLC14_FaCo_QC_HeLa_50cm_20190219185517', 'TAFQEALDAAGDK'), ('20190219_QE5_Evo1_UHG_QC_MNT_HELA_01_190219173213', 'FIIPQIVK'), ('20190220_QE2_NLC1_GP_QC_MNT_HELA_01', 'FVPAEMGTHTVSVK'), ('20190225_QE9_nLC0_RS_MNT_Hela_02', 'AAVPSGASTGIYEALELR'), ('20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_1_20190227115340', 'AAVPSGASTGIYEALELR'), ('20190228_QE4_LC12_JE_QC_MNT_HeLa_01', 'YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK'), ('20190305_QE2_NLC1_AB_QC_MNT_HELA_02', 'AQLLQPTLEINPR'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_02', 'HLEINPDHSIIETLR'), ('20190306_QE2_NLC1_AB_MNT_HELA_02', 'NTGIICTIGPASR'), ('20190311_QE9_nLC0_JM_MNT_Hela_01_20190311212727', 'FEDENFILK'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_01_20190317211403', 'LVGQGASAVLLDLPNSGGEAQAK'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_02', 'TYDATTHFETTCDDIK'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'THLPGFVEQAEALK'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01_20190326190317', 'FEDENFILK'), ('20190328_QE1_nLC2_GP_MNT_QC_hela_01', 'ATQALVLAPTR'), ('20190413_QE10_nLC13_RG_QC_45cm_HeLa_01', 'LVGQGASAVLLDLPNSGGEAQAK'), ('20190424_QE2_NLC1_ANHO_MNT_HELA_01', 'SEIDLFNIRK'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336', 'SEIDLFNIRK'), ('20190429_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'LSVLGAITSVQQR'), ('20190506_QX7_ChDe_MA_HeLaBr14_500ng', 'VLTVINQTQK'), ('20190506_QX7_ChDe_MA_HeLa_500ng', 'VLTVINQTQK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'EHALLAYTLGVK'), ('20190520_QX3_LiSc_MA_Hela_500ng_LC15', 'FVPAEMGTHTVSVK'), ('20190521_QE1_nLC2_GP_QC_MNT_HELA_01', 'SEIDLFNIRK'), ('20190523_QX8_MiWi_MA_HeLa_BR14_500ng', 'DVQIGDIVTVGECRPLSK'), ('20190524_QE4_LC12_IAH_QC_MNT_HeLa_02', 'FVPAEMGTHTVSVK'), ('20190526_QX8_IgPa_MA_HeLa_BR14_500ng', 'KFDQLLAEEK'), ('20190528_QX1_PhGe_MA_HeLa_500ng_LC10', 'ATAVVDGAFK'), ('20190606_QE4_LC12_JE_QC_MNT_HeLa_02b', 'EHALLAYTLGVK'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_1', 'ACANPAAGSVILLENLR'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_03', 'ACANPAAGSVILLENLR'), ('20190617_QE_LC_UHG_QC_MNT_HELA_03', 'IQLINNMLDK'), ('20190621_QX3_MePh_MA_Hela_500ng_LC15_190621150413', 'LSVLGAITSVQQR'), ('20190623_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_MUC_02', 'NTGIICTIGPASR'), ('20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01', 'NTGIICTIGPASR'), ('20190625_QE1_nLC2_GP_QC_MNT_HELA_02', 'ETNLDSLPLVDTHSK'), ('20190625_QE6_LC4_AS_QC_MNT_HeLa_02', 'MDATANDVPSPYEVR'), ('20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_02', 'TGVHHYSGNNIELGTACGK'), ('20190625_QX0_MaPe_MA_HeLa_500ng_LC07_1', 'AHGPGLEGGLVGKPAEFTIDTK'), ('20190627_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'FEDENFILK'), ('20190628_QE2_NLC1_TL_QC_MNT_HELA_05', 'SEIDLFNIRK'), ('20190630_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'TGVHHYSGNNIELGTACGK'), ('20190630_QX3_ChMa_MA_Hela_500ng_LC15', 'ATQALVLAPTR'), ('20190701_QE4_LC12_IAH_QC_MNT_HeLa_02', 'YTLPPGVDPTQVSSSLSPEGTLTVEAPMPK'), ('20190702_QE3_nLC5_GF_QC_MNT_Hela_02', 'KFDQLLAEEK'), ('20190702_QE8_nLC14_FM_QC_MNT_50cm_Hela_01_20190705211303', 'SNFAEALAAHK'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'DFTVSAMHGDMDQK'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05_190709143552', 'TAFDEAIAELDTLNEDSYK'), ('20190715_QE8_nLC14_RG_QC_MNT_50cm_Hela_02', 'NTGIICTIGPASR'), ('20190718_QE8_nLC14_RG_QC_HeLa_MUC_50cm_1', 'AQLLQPTLEINPR'), ('20190722_QX8_ChSc_MA_HeLa_500ng_190722174431', 'LALVTGGEIASTFDHPELVK'), ('20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar', 'AAVPSGASTGIYEALELR'), ('20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'TTHFVEGGDAGNREDQINR'), ('20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05', 'TYDATTHFETTCDDIK')]
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intensity_pred_collab</th>
      <th>intensity_pred_dae</th>
      <th>intensity_pred_vae</th>
      <th>replicates</th>
      <th>train_average</th>
      <th>train_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSE</th>
      <td>0.551</td>
      <td>0.583</td>
      <td>1.337</td>
      <td>1.569</td>
      <td>2.048</td>
      <td>2.099</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.446</td>
      <td>0.472</td>
      <td>0.815</td>
      <td>0.854</td>
      <td>1.061</td>
      <td>1.032</td>
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
