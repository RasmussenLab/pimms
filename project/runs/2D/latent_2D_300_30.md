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
n_feat = 300
n_epochs = 30
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
    ATQALVLAPTR                          965
    ALTVPELTQQVFDAK                      968
    KTEAPAAPAAQETK                       994
    NRPTSISWDGLDSGK                      994
    IMDPNIVGSEHYDVAR                     999
    NLDIERPTYTNLNR                       998
    GYSLVSGGTDNHLVLVDLRPK                990
    LQMEAPHIIVGTPGR                    1,000
    IAGYVTHLMK                           994
    GLTPSQIGVILR                         986
    FAQPGSFEYEYAMR                       961
    TAFDEAIAELDTLNEDSYK                  974
    HVFGESDELIGQK                      1,000
    SLEDQVEMLR                           982
    TIGTGLVTNTLAMTEEEK                   989
    EQVANSAFVER                          998
    QAFTDVATGSLGQGLGAACGMAYTGK           999
    RLIDLHSPSEIVK                        913
    DDEFTHLYTLIVRPDNTYEVK                954
    AVFPSIVGRPR                          959
    FWEVISDEHGIDPTGTYHGDSDLQLER          976
    EAYPGDVFYLHSR                        996
    IYVDDGLISLQVK                        995
    EGMNIVEAMER                          998
    LTGMAFR                              975
    TVFAEHISDECK                         998
    VQASLAANTFTITGHAETK                  987
    HLPTLDHPIIPADYVAIK                   997
    LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR     968
    GIPHLVTHDAR                        1,000
    IHPVSTMVK                            989
    VNFAMNVGK                            987
    LMDVGLIAIR                           996
    VVLAYEPVWAIGTGK                      943
    MDDREDLVYQAK                         998
    LQLWDTAGQER                          974
    SAEFLLHMLK                           999
    LQVTNVLSQPLTQATVK                    994
    DNSTMGYMMAK                          937
    TVAGGAWTYNTTSAVTVK                   992
    VICILSHPIK                           995
    EIGNIISDAMK                          997
    TNHIGHTGYLNTVTVSPDGSLCASGGK        1,000
    IVLLDSSLEYK                          987
    DSYVGDEAQSK                          999
    FGGEHVPNSPFQVTALAGDQPSVQPPLR         972
    EMNDAAMFYTNR                         990
    NLQEAEEWYK                           985
    SKPGAAMVEMADGYAVDR                   990
    MIVDPVEPHGEMK                        989
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
      <th>ATQALVLAPTR</th>
      <td>29.220</td>
    </tr>
    <tr>
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>30.450</td>
    </tr>
    <tr>
      <th>NRPTSISWDGLDSGK</th>
      <td>28.121</td>
    </tr>
    <tr>
      <th>IMDPNIVGSEHYDVAR</th>
      <td>30.207</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>FGGEHVPNSPFQVTALAGDQPSVQPPLR</th>
      <td>26.881</td>
    </tr>
    <tr>
      <th>EMNDAAMFYTNR</th>
      <td>27.354</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
    </tr>
    <tr>
      <th>SKPGAAMVEMADGYAVDR</th>
      <td>29.472</td>
    </tr>
    <tr>
      <th>MIVDPVEPHGEMK</th>
      <td>27.597</td>
    </tr>
  </tbody>
</table>
<p>49220 rows × 1 columns</p>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_24_1.png)
    



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
      <td>0.984</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.840</td>
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
      <th>ATQALVLAPTR</th>
      <td>29.220</td>
    </tr>
    <tr>
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
    </tr>
    <tr>
      <th>KTEAPAAPAAQETK</th>
      <td>30.450</td>
    </tr>
    <tr>
      <th>NRPTSISWDGLDSGK</th>
      <td>28.121</td>
    </tr>
    <tr>
      <th>IMDPNIVGSEHYDVAR</th>
      <td>30.207</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>FGGEHVPNSPFQVTALAGDQPSVQPPLR</th>
      <td>26.881</td>
    </tr>
    <tr>
      <th>EMNDAAMFYTNR</th>
      <td>27.354</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
    </tr>
    <tr>
      <th>SKPGAAMVEMADGYAVDR</th>
      <td>29.472</td>
    </tr>
    <tr>
      <th>MIVDPVEPHGEMK</th>
      <td>27.597</td>
    </tr>
  </tbody>
</table>
<p>49220 rows × 1 columns</p>
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
      <th>ALTVPELTQQVFDAK</th>
      <td>32.860</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>ALTVPELTQQVFDAK</th>
      <td>32.211</td>
    </tr>
    <tr>
      <th>20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01</th>
      <th>ALTVPELTQQVFDAK</th>
      <td>32.823</td>
    </tr>
    <tr>
      <th>20190527_QE3_nLC3_DS_QC_MNT_HeLa_01</th>
      <th>ALTVPELTQQVFDAK</th>
      <td>31.338</td>
    </tr>
    <tr>
      <th>20190208_QE1_nLC2_GP_QC_MNT_HELA_01</th>
      <th>ALTVPELTQQVFDAK</th>
      <td>32.604</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190719_QX1_JoMu_MA_HeLa_500ng_LC10</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>32.518</td>
    </tr>
    <tr>
      <th>20190122_QE6_nLC6_SIS_QC_MNT_HeLa_01</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>32.464</td>
    </tr>
    <tr>
      <th>20190506_QE3_nLC3_DBJ_QC_MNT_HeLa_01</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>30.656</td>
    </tr>
    <tr>
      <th>20190429_QX6_ChDe_MA_HeLa_Br13_500ng_LC09</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>31.356</td>
    </tr>
    <tr>
      <th>20190420_QE8_nLC14_RG_QC_HeLa_01</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>31.866</td>
    </tr>
  </tbody>
</table>
<p>44297 rows × 1 columns</p>
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
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
      <td>32.125</td>
      <td>32.139</td>
      <td>30.720</td>
    </tr>
    <tr>
      <th>DNSTMGYMMAK</th>
      <td>27.949</td>
      <td>27.487</td>
      <td>28.607</td>
      <td>27.479</td>
    </tr>
    <tr>
      <th>EQVANSAFVER</th>
      <td>30.598</td>
      <td>31.644</td>
      <td>31.765</td>
      <td>30.934</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>29.024</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.756</td>
    </tr>
    <tr>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <td>28.383</td>
      <td>28.558</td>
      <td>28.657</td>
      <td>28.265</td>
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
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.775</td>
      <td>30.120</td>
      <td>27.612</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.472</td>
      <td>29.527</td>
      <td>27.558</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>29.881</td>
      <td>29.632</td>
      <td>30.506</td>
      <td>29.422</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.941</td>
    </tr>
    <tr>
      <th>RLIDLHSPSEIVK</th>
      <td>30.407</td>
      <td>30.277</td>
      <td>29.368</td>
      <td>31.066</td>
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
      <th>20190107_QE10_nLC0_KS_QC_MNT_HeLa_01</th>
      <th>NLQEAEEWYK</th>
      <td>30.496</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02</th>
      <th>LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR</th>
      <td>29.637</td>
      <td>28.450</td>
      <td>28.832</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>VICILSHPIK</th>
      <td>28.663</td>
      <td>28.993</td>
      <td>28.974</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03</th>
      <th>LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR</th>
      <td>29.510</td>
      <td>28.450</td>
      <td>28.832</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE7_nLC7_AL_QC_MNT_HeLa_01</th>
      <th>LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR</th>
      <td>28.319</td>
      <td>28.450</td>
      <td>28.832</td>
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
      <th>20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2</th>
      <th>IYVDDGLISLQVK</th>
      <td>29.130</td>
      <td>31.131</td>
      <td>30.853</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QX6_AsJa_MA_HeLa_500ng_LC09</th>
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>31.236</td>
      <td>29.775</td>
      <td>30.120</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05</th>
      <th>TVAGGAWTYNTTSAVTVK</th>
      <td>27.918</td>
      <td>27.509</td>
      <td>27.651</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>NRPTSISWDGLDSGK</th>
      <td>31.624</td>
      <td>29.140</td>
      <td>29.186</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190804_QX0_AsJa_MA_HeLa_500ng_LC07_01</th>
      <th>NRPTSISWDGLDSGK</th>
      <td>31.928</td>
      <td>29.140</td>
      <td>29.186</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 4 columns</p>
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
      <td>ATQALVLAPTR</td>
      <td>29.220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVFPSIVGRPR</td>
      <td>32.920</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DDEFTHLYTLIVRPDNTYEVK</td>
      <td>31.332</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DSYVGDEAQSK</td>
      <td>32.783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EAYPGDVFYLHSR</td>
      <td>29.054</td>
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
      <td>ALTVPELTQQVFDAK</td>
      <td>30.611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DNSTMGYMMAK</td>
      <td>27.949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EQVANSAFVER</td>
      <td>30.598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>NLQEAEEWYK</td>
      <td>29.024</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TIGTGLVTNTLAMTEEEK</td>
      <td>28.383</td>
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
      <td>20190709_QE3_nLC5_GF_QC_MNT_Hela_02</td>
      <td>FGGEHVPNSPFQVTALAGDQPSVQPPLR</td>
      <td>27.338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard_190501203657</td>
      <td>IYVDDGLISLQVK</td>
      <td>32.412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_01</td>
      <td>LQVTNVLSQPLTQATVK</td>
      <td>27.715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190129_QE10_nLC0_FM_QC_MNT_HeLa_50cm_01_20190130121912</td>
      <td>NLQEAEEWYK</td>
      <td>30.771</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190611_QX7_IgPa_MA_HeLa_Br14_500ng_190618134442</td>
      <td>LQVTNVLSQPLTQATVK</td>
      <td>27.957</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190526_QX4_LiSc_MA_HeLa_500ng</td>
      <td>DNSTMGYMMAK</td>
      <td>30.829</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01</td>
      <td>NRPTSISWDGLDSGK</td>
      <td>28.875</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190310_QE2_NLC1_GP_MNT_HELA_01</td>
      <td>VICILSHPIK</td>
      <td>29.028</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190129_QE10_nLC0_FM_QC_MNT_HeLa_50cm_01_20190130121912</td>
      <td>DNSTMGYMMAK</td>
      <td>28.063</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190701_QE1_nLC13_ANHO_QC_MNT_HELA_02</td>
      <td>SLEDQVEMLR</td>
      <td>28.473</td>
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
      <td>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</td>
      <td>EIGNIISDAMK</td>
      <td>31.682</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190615_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>RLIDLHSPSEIVK</td>
      <td>30.375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2</td>
      <td>VICILSHPIK</td>
      <td>28.379</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190513_QX8_MiWi_MA_HeLa_BR14_500ng</td>
      <td>TIGTGLVTNTLAMTEEEK</td>
      <td>30.183</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard</td>
      <td>EIGNIISDAMK</td>
      <td>32.849</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190717_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05_190719190656</td>
      <td>NLDIERPTYTNLNR</td>
      <td>33.262</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_02</td>
      <td>NLQEAEEWYK</td>
      <td>29.873</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190611_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>LTGMAFR</td>
      <td>28.894</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190424_QE2_NLC1_ANHO_MNT_HELA_01</td>
      <td>SLEDQVEMLR</td>
      <td>28.897</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190731_QX1_LiSc_MA_HeLa_500ng_LC10</td>
      <td>MDDREDLVYQAK</td>
      <td>32.105</td>
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




    (1376, 154)



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
    
    Optimizer used: <function Adam at 0x000001EB9BB65040>
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
      <td>2.216376</td>
      <td>2.000522</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.861284</td>
      <td>0.857614</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.671642</td>
      <td>0.628072</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.594637</td>
      <td>0.591081</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.643259</td>
      <td>0.571392</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.578705</td>
      <td>0.562102</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.569903</td>
      <td>0.548814</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.496955</td>
      <td>0.541330</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.512298</td>
      <td>0.539772</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.492611</td>
      <td>0.540228</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_58_1.png)
    


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
      <th>4,899</th>
      <td>990</td>
      <td>9</td>
      <td>31.682</td>
    </tr>
    <tr>
      <th>3,314</th>
      <td>681</td>
      <td>38</td>
      <td>30.375</td>
    </tr>
    <tr>
      <th>4,717</th>
      <td>957</td>
      <td>47</td>
      <td>28.379</td>
    </tr>
    <tr>
      <th>2,564</th>
      <td>519</td>
      <td>43</td>
      <td>30.183</td>
    </tr>
    <tr>
      <th>2,259</th>
      <td>456</td>
      <td>9</td>
      <td>32.849</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4,336</th>
      <td>883</td>
      <td>17</td>
      <td>30.704</td>
    </tr>
    <tr>
      <th>590</th>
      <td>124</td>
      <td>20</td>
      <td>31.094</td>
    </tr>
    <tr>
      <th>1,801</th>
      <td>365</td>
      <td>21</td>
      <td>31.032</td>
    </tr>
    <tr>
      <th>4,738</th>
      <td>961</td>
      <td>12</td>
      <td>28.770</td>
    </tr>
    <tr>
      <th>2,778</th>
      <td>565</td>
      <td>8</td>
      <td>34.497</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
      <td>32.125</td>
      <td>32.139</td>
      <td>30.720</td>
      <td>31.638</td>
    </tr>
    <tr>
      <th>DNSTMGYMMAK</th>
      <td>27.949</td>
      <td>27.487</td>
      <td>28.607</td>
      <td>27.479</td>
      <td>27.272</td>
    </tr>
    <tr>
      <th>EQVANSAFVER</th>
      <td>30.598</td>
      <td>31.644</td>
      <td>31.765</td>
      <td>30.934</td>
      <td>31.171</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>29.024</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.756</td>
      <td>29.490</td>
    </tr>
    <tr>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <td>28.383</td>
      <td>28.558</td>
      <td>28.657</td>
      <td>28.265</td>
      <td>27.538</td>
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
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.775</td>
      <td>30.120</td>
      <td>27.612</td>
      <td>29.407</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.472</td>
      <td>29.527</td>
      <td>27.558</td>
      <td>29.184</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>29.881</td>
      <td>29.632</td>
      <td>30.506</td>
      <td>29.422</td>
      <td>29.163</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.941</td>
      <td>30.316</td>
    </tr>
    <tr>
      <th>RLIDLHSPSEIVK</th>
      <td>30.407</td>
      <td>30.277</td>
      <td>29.368</td>
      <td>31.066</td>
      <td>30.753</td>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.134
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.141
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.228
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.212
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.194
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_66_1.png)
    



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
      <td>-0.068</td>
      <td>0.094</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.141</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.084</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.169</td>
      <td>-0.093</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.104</td>
      <td>0.150</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_68_1.png)
    



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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_69_1.png)
    


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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR</th>
      <th>TAFDEAIAELDTLNEDSYK</th>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK</th>
      <th>TVAGGAWTYNTTSAVTVK</th>
      <th>TVFAEHISDECK</th>
      <th>VICILSHPIK</th>
      <th>VNFAMNVGK</th>
      <th>VQASLAANTFTITGHAETK</th>
      <th>VVLAYEPVWAIGTGK</th>
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
      <td>30.611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.949</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.383</td>
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
      <td>29.237</td>
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
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>32.164</td>
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
      <td>31.765</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.614</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.708</td>
      <td>28.649</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.064</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>TAFDEAIAELDTLNEDSYK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>TVFAEHISDECK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VVLAYEPVWAIGTGK_na</th>
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
      <td>-0.010</td>
      <td>0.037</td>
      <td>0.152</td>
      <td>0.403</td>
      <td>-0.380</td>
      <td>-0.532</td>
      <td>0.480</td>
      <td>-0.546</td>
      <td>-0.213</td>
      <td>-0.770</td>
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
      <td>-1.203</td>
      <td>-0.275</td>
      <td>0.201</td>
      <td>0.349</td>
      <td>0.117</td>
      <td>-0.441</td>
      <td>0.349</td>
      <td>-0.592</td>
      <td>-0.154</td>
      <td>-0.457</td>
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
      <td>-0.010</td>
      <td>-0.636</td>
      <td>0.878</td>
      <td>-0.440</td>
      <td>-0.882</td>
      <td>0.157</td>
      <td>-0.743</td>
      <td>-0.538</td>
      <td>-0.835</td>
      <td>-0.776</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.010</td>
      <td>0.076</td>
      <td>0.466</td>
      <td>0.239</td>
      <td>-0.380</td>
      <td>-0.034</td>
      <td>0.715</td>
      <td>-0.234</td>
      <td>0.187</td>
      <td>-0.341</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.062</td>
      <td>0.129</td>
      <td>0.145</td>
      <td>0.176</td>
      <td>-0.380</td>
      <td>-0.368</td>
      <td>0.272</td>
      <td>-0.468</td>
      <td>-0.056</td>
      <td>-0.581</td>
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
      <td>1.318</td>
      <td>1.074</td>
      <td>-1.733</td>
      <td>0.450</td>
      <td>1.224</td>
      <td>0.218</td>
      <td>1.244</td>
      <td>0.517</td>
      <td>0.691</td>
      <td>0.266</td>
      <td>...</td>
      <td>True</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.480</td>
      <td>-0.709</td>
      <td>0.443</td>
      <td>0.575</td>
      <td>-0.458</td>
      <td>0.262</td>
      <td>-0.581</td>
      <td>0.045</td>
      <td>-2.084</td>
      <td>-0.064</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.170</td>
      <td>-0.821</td>
      <td>0.475</td>
      <td>0.897</td>
      <td>-0.488</td>
      <td>0.469</td>
      <td>-0.192</td>
      <td>0.070</td>
      <td>-1.788</td>
      <td>-0.200</td>
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
      <td>-0.180</td>
      <td>-0.975</td>
      <td>0.565</td>
      <td>0.991</td>
      <td>-0.526</td>
      <td>0.394</td>
      <td>-0.395</td>
      <td>0.190</td>
      <td>-1.437</td>
      <td>-0.429</td>
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
      <td>-0.346</td>
      <td>-0.862</td>
      <td>0.744</td>
      <td>0.176</td>
      <td>-0.889</td>
      <td>0.171</td>
      <td>-0.250</td>
      <td>0.166</td>
      <td>-0.033</td>
      <td>-0.029</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>TAFDEAIAELDTLNEDSYK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>TVFAEHISDECK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VVLAYEPVWAIGTGK_na</th>
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
      <td>-0.011</td>
      <td>-0.005</td>
      <td>0.202</td>
      <td>0.401</td>
      <td>-0.420</td>
      <td>-0.485</td>
      <td>0.428</td>
      <td>-0.510</td>
      <td>-0.208</td>
      <td>-0.730</td>
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
      <td>-1.124</td>
      <td>-0.297</td>
      <td>0.248</td>
      <td>0.351</td>
      <td>0.042</td>
      <td>-0.398</td>
      <td>0.304</td>
      <td>-0.554</td>
      <td>-0.152</td>
      <td>-0.435</td>
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
      <td>-0.011</td>
      <td>-0.636</td>
      <td>0.885</td>
      <td>-0.382</td>
      <td>-0.887</td>
      <td>0.170</td>
      <td>-0.733</td>
      <td>-0.503</td>
      <td>-0.797</td>
      <td>-0.736</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.011</td>
      <td>0.032</td>
      <td>0.497</td>
      <td>0.249</td>
      <td>-0.420</td>
      <td>-0.011</td>
      <td>0.651</td>
      <td>-0.214</td>
      <td>0.171</td>
      <td>-0.326</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.060</td>
      <td>0.081</td>
      <td>0.196</td>
      <td>0.191</td>
      <td>-0.420</td>
      <td>-0.329</td>
      <td>0.231</td>
      <td>-0.436</td>
      <td>-0.059</td>
      <td>-0.552</td>
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
      <td>1.229</td>
      <td>0.967</td>
      <td>-1.571</td>
      <td>0.445</td>
      <td>1.073</td>
      <td>0.228</td>
      <td>1.154</td>
      <td>0.498</td>
      <td>0.649</td>
      <td>0.248</td>
      <td>...</td>
      <td>True</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.450</td>
      <td>-0.703</td>
      <td>0.476</td>
      <td>0.561</td>
      <td>-0.492</td>
      <td>0.270</td>
      <td>-0.580</td>
      <td>0.050</td>
      <td>-1.979</td>
      <td>-0.064</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>-0.160</td>
      <td>-0.808</td>
      <td>0.506</td>
      <td>0.860</td>
      <td>-0.520</td>
      <td>0.467</td>
      <td>-0.210</td>
      <td>0.074</td>
      <td>-1.700</td>
      <td>-0.192</td>
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
      <td>-0.169</td>
      <td>-0.953</td>
      <td>0.591</td>
      <td>0.948</td>
      <td>-0.555</td>
      <td>0.395</td>
      <td>-0.403</td>
      <td>0.187</td>
      <td>-1.366</td>
      <td>-0.408</td>
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
      <td>-0.324</td>
      <td>-0.847</td>
      <td>0.759</td>
      <td>0.191</td>
      <td>-0.893</td>
      <td>0.183</td>
      <td>-0.266</td>
      <td>0.165</td>
      <td>-0.038</td>
      <td>-0.030</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
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
      <td>-0.001</td>
      <td>-0.039</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>-0.066</td>
      <td>0.021</td>
      <td>-0.028</td>
      <td>0.008</td>
      <td>-0.006</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.934</td>
      <td>0.937</td>
      <td>0.941</td>
      <td>0.929</td>
      <td>0.931</td>
      <td>0.950</td>
      <td>0.950</td>
      <td>0.948</td>
      <td>0.947</td>
      <td>0.944</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.883</td>
      <td>-3.505</td>
      <td>-4.043</td>
      <td>-6.189</td>
      <td>-2.545</td>
      <td>-6.723</td>
      <td>-3.814</td>
      <td>-7.845</td>
      <td>-4.385</td>
      <td>-4.218</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.389</td>
      <td>-0.638</td>
      <td>-0.658</td>
      <td>-0.288</td>
      <td>-0.651</td>
      <td>-0.191</td>
      <td>-0.624</td>
      <td>-0.370</td>
      <td>-0.420</td>
      <td>-0.476</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.011</td>
      <td>-0.297</td>
      <td>0.431</td>
      <td>0.191</td>
      <td>-0.420</td>
      <td>0.205</td>
      <td>-0.266</td>
      <td>0.074</td>
      <td>-0.059</td>
      <td>-0.030</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.534</td>
      <td>0.679</td>
      <td>0.723</td>
      <td>0.576</td>
      <td>0.659</td>
      <td>0.537</td>
      <td>0.741</td>
      <td>0.528</td>
      <td>0.499</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.678</td>
      <td>2.168</td>
      <td>1.728</td>
      <td>1.880</td>
      <td>1.923</td>
      <td>1.729</td>
      <td>2.009</td>
      <td>1.918</td>
      <td>2.005</td>
      <td>2.531</td>
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




    ((#50) ['ALTVPELTQQVFDAK','ATQALVLAPTR','AVFPSIVGRPR','DDEFTHLYTLIVRPDNTYEVK','DNSTMGYMMAK','DSYVGDEAQSK','EAYPGDVFYLHSR','EGMNIVEAMER','EIGNIISDAMK','EMNDAAMFYTNR'...],
     (#50) ['ALTVPELTQQVFDAK_na','ATQALVLAPTR_na','AVFPSIVGRPR_na','DDEFTHLYTLIVRPDNTYEVK_na','DNSTMGYMMAK_na','DSYVGDEAQSK_na','EAYPGDVFYLHSR_na','EGMNIVEAMER_na','EIGNIISDAMK_na','EMNDAAMFYTNR_na'...])




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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97.000</td>
      <td>97.000</td>
      <td>96.000</td>
      <td>95.000</td>
      <td>94.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.124</td>
      <td>0.074</td>
      <td>-0.182</td>
      <td>0.097</td>
      <td>0.014</td>
      <td>0.029</td>
      <td>0.228</td>
      <td>0.226</td>
      <td>-0.084</td>
      <td>-0.112</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.078</td>
      <td>1.117</td>
      <td>1.122</td>
      <td>0.992</td>
      <td>1.077</td>
      <td>1.034</td>
      <td>1.078</td>
      <td>0.723</td>
      <td>1.047</td>
      <td>0.919</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.330</td>
      <td>-2.746</td>
      <td>-2.932</td>
      <td>-5.020</td>
      <td>-1.848</td>
      <td>-5.270</td>
      <td>-2.437</td>
      <td>-1.956</td>
      <td>-3.147</td>
      <td>-3.184</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.636</td>
      <td>-0.644</td>
      <td>-1.039</td>
      <td>-0.290</td>
      <td>-0.835</td>
      <td>-0.140</td>
      <td>-0.583</td>
      <td>-0.190</td>
      <td>-0.592</td>
      <td>-0.659</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.007</td>
      <td>-0.265</td>
      <td>0.124</td>
      <td>0.306</td>
      <td>-0.356</td>
      <td>0.249</td>
      <td>-0.040</td>
      <td>0.171</td>
      <td>-0.041</td>
      <td>-0.150</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.492</td>
      <td>1.156</td>
      <td>0.715</td>
      <td>0.628</td>
      <td>1.218</td>
      <td>0.638</td>
      <td>1.393</td>
      <td>0.610</td>
      <td>0.665</td>
      <td>0.416</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.786</td>
      <td>2.283</td>
      <td>1.505</td>
      <td>1.681</td>
      <td>1.728</td>
      <td>1.708</td>
      <td>1.847</td>
      <td>1.685</td>
      <td>1.713</td>
      <td>2.086</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>TAFDEAIAELDTLNEDSYK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>TVFAEHISDECK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VVLAYEPVWAIGTGK_val</th>
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
      <td>-0.011</td>
      <td>-0.005</td>
      <td>0.202</td>
      <td>0.401</td>
      <td>-0.420</td>
      <td>-0.485</td>
      <td>0.428</td>
      <td>-0.510</td>
      <td>-0.208</td>
      <td>-0.730</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.159</td>
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
      <td>-1.124</td>
      <td>-0.297</td>
      <td>0.248</td>
      <td>0.351</td>
      <td>0.042</td>
      <td>-0.398</td>
      <td>0.304</td>
      <td>-0.554</td>
      <td>-0.152</td>
      <td>-0.435</td>
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
      <td>-0.011</td>
      <td>-0.636</td>
      <td>0.885</td>
      <td>-0.382</td>
      <td>-0.887</td>
      <td>0.170</td>
      <td>-0.733</td>
      <td>-0.503</td>
      <td>-0.797</td>
      <td>-0.736</td>
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
      <td>-0.011</td>
      <td>0.032</td>
      <td>0.497</td>
      <td>0.249</td>
      <td>-0.420</td>
      <td>-0.011</td>
      <td>0.651</td>
      <td>-0.214</td>
      <td>0.171</td>
      <td>-0.326</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.060</td>
      <td>0.081</td>
      <td>0.196</td>
      <td>0.191</td>
      <td>-0.420</td>
      <td>-0.329</td>
      <td>0.231</td>
      <td>-0.436</td>
      <td>-0.059</td>
      <td>-0.552</td>
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
      <td>1.229</td>
      <td>0.967</td>
      <td>-1.571</td>
      <td>0.445</td>
      <td>1.073</td>
      <td>0.228</td>
      <td>1.154</td>
      <td>0.498</td>
      <td>0.649</td>
      <td>0.248</td>
      <td>...</td>
      <td>0.686</td>
      <td>0.746</td>
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
      <td>-0.450</td>
      <td>-0.703</td>
      <td>0.476</td>
      <td>0.561</td>
      <td>-0.492</td>
      <td>0.270</td>
      <td>-0.580</td>
      <td>0.050</td>
      <td>-1.979</td>
      <td>-0.064</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.031</td>
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
      <td>-0.160</td>
      <td>-0.808</td>
      <td>0.506</td>
      <td>0.860</td>
      <td>-0.520</td>
      <td>0.467</td>
      <td>-0.210</td>
      <td>0.074</td>
      <td>-1.700</td>
      <td>-0.192</td>
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
      <td>-0.169</td>
      <td>-0.953</td>
      <td>0.591</td>
      <td>0.948</td>
      <td>-0.555</td>
      <td>0.395</td>
      <td>-0.403</td>
      <td>0.187</td>
      <td>-1.366</td>
      <td>-0.408</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.337</td>
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
      <td>-0.324</td>
      <td>-0.847</td>
      <td>0.759</td>
      <td>0.191</td>
      <td>-0.893</td>
      <td>0.183</td>
      <td>-0.266</td>
      <td>0.165</td>
      <td>-0.038</td>
      <td>-0.030</td>
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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>TAFDEAIAELDTLNEDSYK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>TVFAEHISDECK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VVLAYEPVWAIGTGK_val</th>
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
      <td>-0.011</td>
      <td>-0.005</td>
      <td>0.202</td>
      <td>0.401</td>
      <td>-0.420</td>
      <td>-0.485</td>
      <td>0.428</td>
      <td>-0.510</td>
      <td>-0.208</td>
      <td>-0.730</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.159</td>
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
      <td>-1.124</td>
      <td>-0.297</td>
      <td>0.248</td>
      <td>0.351</td>
      <td>0.042</td>
      <td>-0.398</td>
      <td>0.304</td>
      <td>-0.554</td>
      <td>-0.152</td>
      <td>-0.435</td>
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
      <td>-0.011</td>
      <td>-0.636</td>
      <td>0.885</td>
      <td>-0.382</td>
      <td>-0.887</td>
      <td>0.170</td>
      <td>-0.733</td>
      <td>-0.503</td>
      <td>-0.797</td>
      <td>-0.736</td>
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
      <td>-0.011</td>
      <td>0.032</td>
      <td>0.497</td>
      <td>0.249</td>
      <td>-0.420</td>
      <td>-0.011</td>
      <td>0.651</td>
      <td>-0.214</td>
      <td>0.171</td>
      <td>-0.326</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.060</td>
      <td>0.081</td>
      <td>0.196</td>
      <td>0.191</td>
      <td>-0.420</td>
      <td>-0.329</td>
      <td>0.231</td>
      <td>-0.436</td>
      <td>-0.059</td>
      <td>-0.552</td>
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
      <td>1.229</td>
      <td>0.967</td>
      <td>-1.571</td>
      <td>0.445</td>
      <td>1.073</td>
      <td>0.228</td>
      <td>1.154</td>
      <td>0.498</td>
      <td>0.649</td>
      <td>0.248</td>
      <td>...</td>
      <td>0.686</td>
      <td>0.746</td>
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
      <td>-0.450</td>
      <td>-0.703</td>
      <td>0.476</td>
      <td>0.561</td>
      <td>-0.492</td>
      <td>0.270</td>
      <td>-0.580</td>
      <td>0.050</td>
      <td>-1.979</td>
      <td>-0.064</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.031</td>
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
      <td>-0.160</td>
      <td>-0.808</td>
      <td>0.506</td>
      <td>0.860</td>
      <td>-0.520</td>
      <td>0.467</td>
      <td>-0.210</td>
      <td>0.074</td>
      <td>-1.700</td>
      <td>-0.192</td>
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
      <td>-0.169</td>
      <td>-0.953</td>
      <td>0.591</td>
      <td>0.948</td>
      <td>-0.555</td>
      <td>0.395</td>
      <td>-0.403</td>
      <td>0.187</td>
      <td>-1.366</td>
      <td>-0.408</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.337</td>
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
      <td>-0.324</td>
      <td>-0.847</td>
      <td>0.759</td>
      <td>0.191</td>
      <td>-0.893</td>
      <td>0.183</td>
      <td>-0.266</td>
      <td>0.165</td>
      <td>-0.038</td>
      <td>-0.030</td>
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
      <th>ALTVPELTQQVFDAK_val</th>
      <th>ATQALVLAPTR_val</th>
      <th>AVFPSIVGRPR_val</th>
      <th>DDEFTHLYTLIVRPDNTYEVK_val</th>
      <th>DNSTMGYMMAK_val</th>
      <th>DSYVGDEAQSK_val</th>
      <th>EAYPGDVFYLHSR_val</th>
      <th>EGMNIVEAMER_val</th>
      <th>EIGNIISDAMK_val</th>
      <th>EMNDAAMFYTNR_val</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>TAFDEAIAELDTLNEDSYK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>TVFAEHISDECK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VNFAMNVGK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VVLAYEPVWAIGTGK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97.000</td>
      <td>97.000</td>
      <td>96.000</td>
      <td>95.000</td>
      <td>94.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>...</td>
      <td>98.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>94.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.124</td>
      <td>0.074</td>
      <td>-0.182</td>
      <td>0.097</td>
      <td>0.014</td>
      <td>0.029</td>
      <td>0.228</td>
      <td>0.226</td>
      <td>-0.084</td>
      <td>-0.112</td>
      <td>...</td>
      <td>0.018</td>
      <td>0.031</td>
      <td>0.107</td>
      <td>-0.080</td>
      <td>-0.173</td>
      <td>-0.149</td>
      <td>-0.340</td>
      <td>-0.018</td>
      <td>0.020</td>
      <td>-0.011</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.078</td>
      <td>1.117</td>
      <td>1.122</td>
      <td>0.992</td>
      <td>1.077</td>
      <td>1.034</td>
      <td>1.078</td>
      <td>0.723</td>
      <td>1.047</td>
      <td>0.919</td>
      <td>...</td>
      <td>0.747</td>
      <td>1.019</td>
      <td>1.093</td>
      <td>1.088</td>
      <td>0.920</td>
      <td>1.246</td>
      <td>1.129</td>
      <td>0.896</td>
      <td>0.931</td>
      <td>1.039</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.330</td>
      <td>-2.746</td>
      <td>-2.932</td>
      <td>-5.020</td>
      <td>-1.848</td>
      <td>-5.270</td>
      <td>-2.437</td>
      <td>-1.956</td>
      <td>-3.147</td>
      <td>-3.184</td>
      <td>...</td>
      <td>-2.357</td>
      <td>-4.056</td>
      <td>-3.247</td>
      <td>-3.570</td>
      <td>-2.496</td>
      <td>-4.027</td>
      <td>-4.160</td>
      <td>-2.901</td>
      <td>-3.542</td>
      <td>-3.225</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.636</td>
      <td>-0.644</td>
      <td>-1.039</td>
      <td>-0.290</td>
      <td>-0.835</td>
      <td>-0.140</td>
      <td>-0.583</td>
      <td>-0.190</td>
      <td>-0.592</td>
      <td>-0.659</td>
      <td>...</td>
      <td>-0.463</td>
      <td>-0.369</td>
      <td>-0.409</td>
      <td>-0.644</td>
      <td>-0.897</td>
      <td>-0.770</td>
      <td>-0.657</td>
      <td>-0.588</td>
      <td>-0.394</td>
      <td>-0.250</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.007</td>
      <td>-0.265</td>
      <td>0.124</td>
      <td>0.306</td>
      <td>-0.356</td>
      <td>0.249</td>
      <td>-0.040</td>
      <td>0.171</td>
      <td>-0.041</td>
      <td>-0.150</td>
      <td>...</td>
      <td>0.073</td>
      <td>0.190</td>
      <td>-0.036</td>
      <td>-0.126</td>
      <td>-0.200</td>
      <td>-0.274</td>
      <td>-0.208</td>
      <td>-0.252</td>
      <td>0.103</td>
      <td>0.249</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.492</td>
      <td>1.156</td>
      <td>0.715</td>
      <td>0.628</td>
      <td>1.218</td>
      <td>0.638</td>
      <td>1.393</td>
      <td>0.610</td>
      <td>0.665</td>
      <td>0.416</td>
      <td>...</td>
      <td>0.467</td>
      <td>0.750</td>
      <td>0.793</td>
      <td>0.564</td>
      <td>0.466</td>
      <td>0.900</td>
      <td>0.346</td>
      <td>0.786</td>
      <td>0.345</td>
      <td>0.659</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.786</td>
      <td>2.283</td>
      <td>1.505</td>
      <td>1.681</td>
      <td>1.728</td>
      <td>1.708</td>
      <td>1.847</td>
      <td>1.685</td>
      <td>1.713</td>
      <td>2.086</td>
      <td>...</td>
      <td>1.539</td>
      <td>2.001</td>
      <td>1.984</td>
      <td>2.165</td>
      <td>1.814</td>
      <td>2.007</td>
      <td>1.646</td>
      <td>1.802</td>
      <td>1.914</td>
      <td>1.634</td>
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
      <th>ALTVPELTQQVFDAK_na</th>
      <th>ATQALVLAPTR_na</th>
      <th>AVFPSIVGRPR_na</th>
      <th>DDEFTHLYTLIVRPDNTYEVK_na</th>
      <th>DNSTMGYMMAK_na</th>
      <th>DSYVGDEAQSK_na</th>
      <th>EAYPGDVFYLHSR_na</th>
      <th>EGMNIVEAMER_na</th>
      <th>EIGNIISDAMK_na</th>
      <th>EMNDAAMFYTNR_na</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>TAFDEAIAELDTLNEDSYK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>TVFAEHISDECK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VNFAMNVGK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VVLAYEPVWAIGTGK_na</th>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
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
    
    Optimizer used: <function Adam at 0x000001EB9BB65040>
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








    SuggestedLRs(valley=0.015848932787775993)




    
![png](latent_2D_300_30_files/latent_2D_300_30_108_2.png)
    


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
      <td>0.960499</td>
      <td>0.731012</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.624207</td>
      <td>0.351730</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.465063</td>
      <td>0.341939</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.391728</td>
      <td>0.328446</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.352718</td>
      <td>0.322034</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.332563</td>
      <td>0.315798</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.317923</td>
      <td>0.304021</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.305752</td>
      <td>0.299434</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.298427</td>
      <td>0.296606</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.292984</td>
      <td>0.297310</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_112_1.png)
    



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




    TensorBase(0.2948)




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
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
      <td>32.125</td>
      <td>32.139</td>
      <td>30.720</td>
      <td>31.638</td>
      <td>31.266</td>
    </tr>
    <tr>
      <th>DNSTMGYMMAK</th>
      <td>27.949</td>
      <td>27.487</td>
      <td>28.607</td>
      <td>27.479</td>
      <td>27.272</td>
      <td>27.265</td>
    </tr>
    <tr>
      <th>EQVANSAFVER</th>
      <td>30.598</td>
      <td>31.644</td>
      <td>31.765</td>
      <td>30.934</td>
      <td>31.171</td>
      <td>30.852</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>29.024</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.756</td>
      <td>29.490</td>
      <td>29.206</td>
    </tr>
    <tr>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <td>28.383</td>
      <td>28.558</td>
      <td>28.657</td>
      <td>28.265</td>
      <td>27.538</td>
      <td>27.433</td>
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
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.775</td>
      <td>30.120</td>
      <td>27.612</td>
      <td>29.407</td>
      <td>29.467</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.472</td>
      <td>29.527</td>
      <td>27.558</td>
      <td>29.184</td>
      <td>29.243</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>29.881</td>
      <td>29.632</td>
      <td>30.506</td>
      <td>29.422</td>
      <td>29.163</td>
      <td>29.211</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.941</td>
      <td>30.316</td>
      <td>30.399</td>
    </tr>
    <tr>
      <th>RLIDLHSPSEIVK</th>
      <td>30.407</td>
      <td>30.277</td>
      <td>29.368</td>
      <td>31.066</td>
      <td>30.753</td>
      <td>30.903</td>
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
      <td>-0.372</td>
      <td>-0.140</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.392</td>
      <td>-0.338</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.965</td>
      <td>-0.604</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.229</td>
      <td>0.049</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.396</td>
      <td>-0.362</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_122_1.png)
    



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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_123_1.png)
    


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
      <th>ALTVPELTQQVFDAK</th>
      <th>ATQALVLAPTR</th>
      <th>AVFPSIVGRPR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DNSTMGYMMAK</th>
      <th>DSYVGDEAQSK</th>
      <th>EAYPGDVFYLHSR</th>
      <th>EGMNIVEAMER</th>
      <th>EIGNIISDAMK</th>
      <th>EMNDAAMFYTNR</th>
      <th>...</th>
      <th>SLEDQVEMLR</th>
      <th>TAFDEAIAELDTLNEDSYK</th>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <th>TNHIGHTGYLNTVTVSPDGSLCASGGK</th>
      <th>TVAGGAWTYNTTSAVTVK</th>
      <th>TVFAEHISDECK</th>
      <th>VICILSHPIK</th>
      <th>VNFAMNVGK</th>
      <th>VQASLAANTFTITGHAETK</th>
      <th>VVLAYEPVWAIGTGK</th>
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
      <td>0.593</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.592</td>
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
      <td>0.619</td>
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
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.722</td>
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
      <td>0.689</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.570</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.942</td>
      <td>0.573</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.683</td>
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
    
    Optimizer used: <function Adam at 0x000001EB9BB65040>
    Loss function: <function loss_fct_vae at 0x000001EB9BB8C940>
    
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




    
![png](latent_2D_300_30_files/latent_2D_300_30_136_2.png)
    



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
      <td>1995.070312</td>
      <td>219.995560</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1953.674194</td>
      <td>207.048996</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1891.621094</td>
      <td>197.193390</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1841.435181</td>
      <td>194.323563</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1808.689575</td>
      <td>194.896866</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1786.820557</td>
      <td>195.844528</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1770.896606</td>
      <td>196.715576</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1761.256226</td>
      <td>197.231766</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1751.897949</td>
      <td>197.542435</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1747.370850</td>
      <td>197.617249</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_138_1.png)
    


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




    tensor(3117.7485)




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
      <th>ALTVPELTQQVFDAK</th>
      <td>30.611</td>
      <td>32.125</td>
      <td>32.139</td>
      <td>30.720</td>
      <td>31.638</td>
      <td>31.266</td>
      <td>32.290</td>
    </tr>
    <tr>
      <th>DNSTMGYMMAK</th>
      <td>27.949</td>
      <td>27.487</td>
      <td>28.607</td>
      <td>27.479</td>
      <td>27.272</td>
      <td>27.265</td>
      <td>28.526</td>
    </tr>
    <tr>
      <th>EQVANSAFVER</th>
      <td>30.598</td>
      <td>31.644</td>
      <td>31.765</td>
      <td>30.934</td>
      <td>31.171</td>
      <td>30.852</td>
      <td>31.873</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>29.024</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.756</td>
      <td>29.490</td>
      <td>29.206</td>
      <td>30.353</td>
    </tr>
    <tr>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <td>28.383</td>
      <td>28.558</td>
      <td>28.657</td>
      <td>28.265</td>
      <td>27.538</td>
      <td>27.433</td>
      <td>28.721</td>
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
      <th>HLPTLDHPIIPADYVAIK</th>
      <td>27.738</td>
      <td>29.775</td>
      <td>30.120</td>
      <td>27.612</td>
      <td>29.407</td>
      <td>29.467</td>
      <td>30.103</td>
    </tr>
    <tr>
      <th>LMDVGLIAIR</th>
      <td>29.522</td>
      <td>29.472</td>
      <td>29.527</td>
      <td>27.558</td>
      <td>29.184</td>
      <td>29.243</td>
      <td>29.603</td>
    </tr>
    <tr>
      <th>LTGMAFR</th>
      <td>29.881</td>
      <td>29.632</td>
      <td>30.506</td>
      <td>29.422</td>
      <td>29.163</td>
      <td>29.211</td>
      <td>30.425</td>
    </tr>
    <tr>
      <th>NLQEAEEWYK</th>
      <td>30.113</td>
      <td>30.340</td>
      <td>30.230</td>
      <td>29.941</td>
      <td>30.316</td>
      <td>30.399</td>
      <td>30.336</td>
    </tr>
    <tr>
      <th>RLIDLHSPSEIVK</th>
      <td>30.407</td>
      <td>30.277</td>
      <td>29.368</td>
      <td>31.066</td>
      <td>30.753</td>
      <td>30.903</td>
      <td>29.653</td>
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
      <td>0.033</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.037</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.193</td>
      <td>-0.093</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.015</td>
      <td>-0.032</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.115</td>
      <td>-0.103</td>
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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_146_1.png)
    



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
    


    
![png](latent_2D_300_30_files/latent_2D_300_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20190107_QE10_nLC0_KS_QC_MNT_HeLa_01', 'NLQEAEEWYK'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02', 'LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02', 'VICILSHPIK'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR'), ('20190114_QE7_nLC7_AL_QC_MNT_HeLa_01', 'LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR'), ('20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03', 'IMDPNIVGSEHYDVAR'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'SLEDQVEMLR'), ('20190128_QE3_nLC3_MJ_MNT_HeLa_01', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190129_QE8_nLC14_FaCo_QC_MNT_50cm_Hela_20190129205246', 'EMNDAAMFYTNR'), ('20190131_QE10_nLC0_NHS_MNT_HELA_50cm_02', 'LQLWDTAGQER'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_01', 'IVLLDSSLEYK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_03', 'NRPTSISWDGLDSGK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_15_24', 'LTGMAFR'), ('20190228_QE4_LC12_JE_QC_MNT_HeLa_01', 'TVAGGAWTYNTTSAVTVK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_01', 'ATQALVLAPTR'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_01_20190317211403', 'ALTVPELTQQVFDAK'), ('20190403_QE10_nLC13_LiNi_QC_45cm_HeLa_01', 'IAGYVTHLMK'), ('20190403_QE1_nLC2_GP_MNT_QC_hela_01', 'DNSTMGYMMAK'), ('20190408_QE1_nLC2_GP_MNT_QC_hela_02_20190408131505', 'VVLAYEPVWAIGTGK'), ('20190408_QE6_LC6_AS_QC_MNT_HeLa_01', 'TIGTGLVTNTLAMTEEEK'), ('20190417_QX0_MaTa_MA_HeLa_500ng_LC07_1_too_much_for_a_cleaned_MS', 'LTGMAFR'), ('20190417_QX4_JoSw_MA_HeLa_500ng_BR14_new', 'AVFPSIVGRPR'), ('20190426_QX1_JoMu_MA_HeLa_500ng_LC11', 'GIPHLVTHDAR'), ('20190430_QX6_ChDe_MA_HeLa_Br13_500ng_LC09', 'ALTVPELTQQVFDAK'), ('20190506_QX7_ChDe_MA_HeLaBr14_500ng', 'RLIDLHSPSEIVK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_04', 'EIGNIISDAMK'), ('20190511_QX0_ChDe_MA_HeLa_500ng_LC07_1_BR14', 'IHPVSTMVK'), ('20190513_QX3_ChDe_MA_Hela_500ng_LC15', 'TAFDEAIAELDTLNEDSYK'), ('20190513_QX8_MiWi_MA_HeLa_BR14_500ng', 'EIGNIISDAMK'), ('20190522_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'ATQALVLAPTR'), ('20190523_QX8_MiWi_MA_HeLa_BR14_500ng', 'ALTVPELTQQVFDAK'), ('20190530_QE2_NLC1_GP_QC_MNT_HELA_01', 'VICILSHPIK'), ('20190606_QE4_LC12_JE_QC_MNT_HeLa_02b', 'VVLAYEPVWAIGTGK'), ('20190606_QX8_MiWi_MA_HeLa_BR14_500ng', 'GIPHLVTHDAR'), ('20190606_QX8_MiWi_MA_HeLa_BR14_500ng', 'VICILSHPIK'), ('20190611_QE3_nLC3_DS_QC_MNT_HeLa_02', 'NLDIERPTYTNLNR'), ('20190611_QX0_MaTa_MA_HeLa_500ng_LC07_1', 'DDEFTHLYTLIVRPDNTYEVK'), ('20190611_QX4_JiYu_MA_HeLa_500ng', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190611_QX7_IgPa_MA_HeLa_Br14_500ng', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_03', 'GIPHLVTHDAR'), ('20190620_QE1_nLC2_GP_QC_MNT_HELA_01', 'LQLWDTAGQER'), ('20190624_QE4_nLC12_MM_QC_MNT_HELA_02', 'IVLLDSSLEYK'), ('20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'HVFGESDELIGQK'), ('20190625_QX7_IgPa_MA_HeLa_Br14_500ng', 'ATQALVLAPTR'), ('20190626_QE2_NLC1_JM_QC_MNT_HELA_01', 'MDDREDLVYQAK'), ('20190627_QE6_LC4_AS_QC_MNT_HeLa_01', 'GLTPSQIGVILR'), ('20190627_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'FAQPGSFEYEYAMR'), ('20190627_QX6_JoMu_MA_HeLa_500ng_LC09', 'LQVTNVLSQPLTQATVK'), ('20190628_QX0_AnBr_MA_HeLa_500ng_LC07_02', 'ALTVPELTQQVFDAK'), ('20190630_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'SKPGAAMVEMADGYAVDR'), ('20190702_QE8_nLC14_FM_QC_MNT_50cm_Hela_01_20190705211303', 'ATQALVLAPTR'), ('20190707_QX3_MaTa_MA_Hela_500ng_LC15', 'VNFAMNVGK'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'EAYPGDVFYLHSR'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'EAYPGDVFYLHSR'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'EMNDAAMFYTNR'), ('20190712_QE8_nLC14_AnMu_QC_MNT_50cm_Hela_01', 'TVAGGAWTYNTTSAVTVK'), ('20190712_QE9_nLC9_NHS_MNT_HELA_50cm_MUC_01', 'TVAGGAWTYNTTSAVTVK'), ('20190717_QE6_LC4_SCL_QC_MNT_Hela_04', 'DDEFTHLYTLIVRPDNTYEVK'), ('20190722_QX4_StEb_MA_HeLa_500ng', 'VVLAYEPVWAIGTGK'), ('20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_1', 'IYVDDGLISLQVK'), ('20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2', 'IYVDDGLISLQVK'), ('20190730_QX6_AsJa_MA_HeLa_500ng_LC09', 'HLPTLDHPIIPADYVAIK'), ('20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05', 'TVAGGAWTYNTTSAVTVK'), ('20190803_QX8_AnPi_MA_HeLa_BR14_500ng', 'NRPTSISWDGLDSGK'), ('20190804_QX0_AsJa_MA_HeLa_500ng_LC07_01', 'NRPTSISWDGLDSGK')]
    




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
      <td>0.540</td>
      <td>0.576</td>
      <td>1.730</td>
      <td>1.884</td>
      <td>2.240</td>
      <td>2.371</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.465</td>
      <td>0.488</td>
      <td>0.898</td>
      <td>1.014</td>
      <td>1.122</td>
      <td>1.096</td>
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
