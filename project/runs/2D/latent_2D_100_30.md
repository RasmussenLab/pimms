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
n_feat = 100
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
    PMFIVNTNVPR                1,000
    IIALDGDTK                    996
    SGAQASSTPLSPTR               950
    HSGPNSADSANDGFVR             990
    EMEAELEDERK                  981
    AVLVDLEPGTMDSVR              999
    STGGAPTFNVTVTK               976
    SLHDALCVLAQTVK               999
    LATQSNEITIPVTFESR          1,000
    GCITIIGGGDTATCCAK            998
    FDTGNLCMVTGGANLGR            999
    AVLFCLSEDK                   992
    TFVNITPAEVGVLVGK             957
    GLTSVINQK                    994
    VAPEEHPVLLTEAPLNPK         1,000
    YHTSQSGDEMTSLSEYVSR        1,000
    VIHDNFGIVEGLMTTVHAITATQK     998
    ATAVVDGAFK                   974
    SPYQEFTDHLVK                 998
    AGVNTVTTLVENKK               966
    GQYISPFHDIPIYADK             999
    SLAGSSGPGASSGTSGDHGELVVR     997
    VLNNMEIGTSLFDEEGAK           999
    AFGPGLQGGSAGSPAR             948
    HLPTLDHPIIPADYVAIK           997
    LLQDFFNGK                    982
    FIQENIFGICPHMTEDNK           985
    VANVSLLALYK                  987
    NPDDITNEEYGEFYK              970
    QEMQEVQSSR                   986
    FYPEDVSEELIQDITQR            949
    IKDPDASKPEDWDER              981
    VHVIFNYK                     998
    LGDVYVNDAFGTAHR              983
    DLLLTSSYLSDSGSTGEHTK       1,000
    GITINAAHVEYSTAAR             997
    SVTEQGAELSNEER               998
    LIALLEVLSQK                  989
    LSSLIILMPHHVEPLER            997
    YADLTEDQLPSCESLK             998
    LVAIVDVIDQNR               1,000
    GLFIIDDK                     995
    ALDTMNFDVIK                  979
    VVVLMGSTSDLGHCEK             996
    VNVPVIGGHAGK                 990
    LNFSHGTHEYHAETIK             999
    TICSHVQNMIK                  992
    IFAPNHVVAK                   994
    AAEAAAAPAESAAPAAGEEPSK       993
    DSLLQDGEFSMDLR               978
    Name: intensity, dtype: int64




```python
analysis.df = analysis.df[freq_per_pepitde.index]
# ToDo: clean-up other attributes needs to be integrated
del analysis._df_long  # , analysis._df_wide
analysis.df_long
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>PMFIVNTNVPR</th>
      <td>31.099</td>
    </tr>
    <tr>
      <th>IIALDGDTK</th>
      <td>31.266</td>
    </tr>
    <tr>
      <th>SGAQASSTPLSPTR</th>
      <td>28.051</td>
    </tr>
    <tr>
      <th>HSGPNSADSANDGFVR</th>
      <td>27.176</td>
    </tr>
    <tr>
      <th>EMEAELEDERK</th>
      <td>29.215</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
    </tr>
    <tr>
      <th>TICSHVQNMIK</th>
      <td>30.735</td>
    </tr>
    <tr>
      <th>IFAPNHVVAK</th>
      <td>30.048</td>
    </tr>
    <tr>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>29.296</td>
    </tr>
    <tr>
      <th>DSLLQDGEFSMDLR</th>
      <td>27.783</td>
    </tr>
  </tbody>
</table>
<p>49423 rows × 1 columns</p>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_24_1.png)
    



```python
# ToDo add df_meta property
analysis.df_meta.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>0.988</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022</td>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>PMFIVNTNVPR</th>
      <td>31.099</td>
    </tr>
    <tr>
      <th>IIALDGDTK</th>
      <td>31.266</td>
    </tr>
    <tr>
      <th>SGAQASSTPLSPTR</th>
      <td>28.051</td>
    </tr>
    <tr>
      <th>HSGPNSADSANDGFVR</th>
      <td>27.176</td>
    </tr>
    <tr>
      <th>EMEAELEDERK</th>
      <td>29.215</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
    </tr>
    <tr>
      <th>TICSHVQNMIK</th>
      <td>30.735</td>
    </tr>
    <tr>
      <th>IFAPNHVVAK</th>
      <td>30.048</td>
    </tr>
    <tr>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>29.296</td>
    </tr>
    <tr>
      <th>DSLLQDGEFSMDLR</th>
      <td>27.783</td>
    </tr>
  </tbody>
</table>
<p>49423 rows × 1 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>20190411_QE3_nLC5_DS_QC_MNT_HeLa_01</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>28.643</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>28.777</td>
    </tr>
    <tr>
      <th>20190624_QE4_nLC12_MM_QC_MNT_HELA_01_20190625144904</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>28.355</td>
    </tr>
    <tr>
      <th>20190527_QX4_IgPa_MA_HeLa_500ng</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>29.058</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_2</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>29.053</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190408_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-005</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <td>31.234</td>
    </tr>
    <tr>
      <th>20190327_QE6_LC6_SCL_QC_MNT_Hela_04</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <td>31.625</td>
    </tr>
    <tr>
      <th>20190628_QE2_NLC1_TL_QC_MNT_HELA_05</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <td>31.308</td>
    </tr>
    <tr>
      <th>20190514_QE10_Evosep_LiNi_QC_EVOMNT_Hela_44min_1_20190514114256</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <td>29.612</td>
    </tr>
    <tr>
      <th>20190212_QE9_nLC9_JM_QC_HeLa_50cm_01_20190212183741</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <td>31.754</td>
    </tr>
  </tbody>
</table>
<p>44477 rows × 1 columns</p>
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
    Shape in validation: (993, 50)
    




    ((993, 50), (993, 50))



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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>30.289</td>
      <td>28.539</td>
      <td>28.400</td>
      <td>29.274</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.218</td>
      <td>30.022</td>
      <td>30.017</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <td>29.280</td>
      <td>29.345</td>
      <td>29.171</td>
      <td>29.406</td>
    </tr>
    <tr>
      <th>GLFIIDDK</th>
      <td>31.055</td>
      <td>31.974</td>
      <td>31.874</td>
      <td>31.178</td>
    </tr>
    <tr>
      <th>LGDVYVNDAFGTAHR</th>
      <td>31.049</td>
      <td>30.839</td>
      <td>30.823</td>
      <td>30.325</td>
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
      <td>29.808</td>
      <td>30.127</td>
      <td>27.612</td>
    </tr>
    <tr>
      <th>LIALLEVLSQK</th>
      <td>26.275</td>
      <td>30.076</td>
      <td>29.830</td>
      <td>28.351</td>
    </tr>
    <tr>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
      <td>32.689</td>
      <td>32.561</td>
      <td>32.071</td>
    </tr>
    <tr>
      <th>STGGAPTFNVTVTK</th>
      <td>32.294</td>
      <td>32.374</td>
      <td>32.187</td>
      <td>32.074</td>
    </tr>
    <tr>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>29.156</td>
      <td>34.246</td>
      <td>33.500</td>
      <td>29.025</td>
    </tr>
  </tbody>
</table>
<p>4946 rows × 4 columns</p>
</div>




```python
if any(df_pred.isna()):
    print("Consecutive NaNs are not imputed using replicates.")
    display(df_pred.loc[df_pred.isna().any(axis=1)])
```

    Consecutive NaNs are not imputed using replicates.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>30.218</td>
      <td>30.022</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-052</th>
      <th>SLAGSSGPGASSGTSGDHGELVVR</th>
      <td>30.218</td>
      <td>30.604</td>
      <td>30.756</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE7_nLC7_AL_QC_MNT_HeLa_01</th>
      <th>LLQDFFNGK</th>
      <td>30.769</td>
      <td>30.949</td>
      <td>31.084</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE7_nLC7_AL_QC_MNT_HeLa_02</th>
      <th>LLQDFFNGK</th>
      <td>30.545</td>
      <td>30.949</td>
      <td>31.084</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE9_nLC9_NHS_MNT_HELA_01</th>
      <th>LLQDFFNGK</th>
      <td>30.290</td>
      <td>30.949</td>
      <td>31.084</td>
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
      <th>20190702_QE3_nLC5_TSB_QC_MNT_HELA_02</th>
      <th>VVVLMGSTSDLGHCEK</th>
      <td>29.631</td>
      <td>29.489</td>
      <td>29.410</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190709_QX2_JoMu_MA_HeLa_500ng_LC05_190709143552</th>
      <th>HSGPNSADSANDGFVR</th>
      <td>29.296</td>
      <td>28.682</td>
      <td>28.822</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01</th>
      <th>VNVPVIGGHAGK</th>
      <td>30.808</td>
      <td>27.406</td>
      <td>28.252</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190715_QE8_nLC14_RG_QC_MNT_50cm_Hela_02</th>
      <th>IIALDGDTK</th>
      <td>31.106</td>
      <td>31.082</td>
      <td>31.142</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_1</th>
      <th>FIQENIFGICPHMTEDNK</th>
      <td>30.801</td>
      <td>29.192</td>
      <td>29.232</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 4 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>AFGPGLQGGSAGSPAR</td>
      <td>28.974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AGVNTVTTLVENKK</td>
      <td>28.355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALDTMNFDVIK</td>
      <td>28.452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVLFCLSEDK</td>
      <td>30.558</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVLVDLEPGTMDSVR</td>
      <td>30.155</td>
    </tr>
  </tbody>
</table>
</div>




```python
collab.df_valid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>AAEAAAAPAESAAPAAGEEPSK</td>
      <td>30.289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ATAVVDGAFK</td>
      <td>29.988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>AFGPGLQGGSAGSPAR</td>
      <td>29.280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>GLFIIDDK</td>
      <td>31.055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>LGDVYVNDAFGTAHR</td>
      <td>31.049</td>
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
      <td>20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_02_20190326215828</td>
      <td>SPYQEFTDHLVK</td>
      <td>30.337</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190705_QX6_ChDe_MA_HeLa_500ng_LC09</td>
      <td>VIHDNFGIVEGLMTTVHAITATQK</td>
      <td>35.840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190201_QE9_nLC9_NHS_MNT_HELA_45cm_04</td>
      <td>GCITIIGGGDTATCCAK</td>
      <td>28.878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190712_QE9_nLC9_NHS_MNT_HELA_50cm_MUC_01</td>
      <td>ALDTMNFDVIK</td>
      <td>29.639</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190715_QE2_NLC1_ANHO_MNT_HELA_01</td>
      <td>IFAPNHVVAK</td>
      <td>30.018</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED</td>
      <td>TICSHVQNMIK</td>
      <td>31.425</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190502_QX7_ChDe_MA_HeLa_500ng</td>
      <td>NPDDITNEEYGEFYK</td>
      <td>33.049</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190709_QE3_nLC5_GF_QC_MNT_Hela_01</td>
      <td>VNVPVIGGHAGK</td>
      <td>24.883</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190515_QE4_LC12_AS_QC_MNT_HeLa_01_20190515230141</td>
      <td>AAEAAAAPAESAAPAAGEEPSK</td>
      <td>29.065</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_02_20190326215828</td>
      <td>LNFSHGTHEYHAETIK</td>
      <td>33.122</td>
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
      <td>20190410_QE1_nLC2_ANHO_MNT_QC_hela_01</td>
      <td>GITINAAHVEYSTAAR</td>
      <td>27.830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190207_QE8_nLC0_ASD_QC_HeLa_43cm1_20190207172050</td>
      <td>LNFSHGTHEYHAETIK</td>
      <td>34.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard_190501203657</td>
      <td>STGGAPTFNVTVTK</td>
      <td>33.289</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181228_QE6_nLC6_CSC_QC_MNT_HeLa_01</td>
      <td>SPYQEFTDHLVK</td>
      <td>30.712</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190219_QE2_NLC1_GP_QC_MNT_HELA_01</td>
      <td>YHTSQSGDEMTSLSEYVSR</td>
      <td>30.862</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_1_20190207182540</td>
      <td>LGDVYVNDAFGTAHR</td>
      <td>30.045</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190630_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</td>
      <td>AAEAAAAPAESAAPAAGEEPSK</td>
      <td>28.984</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190423_QX7_JuSc_MA_HeLaBr14_500ng_LC02</td>
      <td>NPDDITNEEYGEFYK</td>
      <td>33.006</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190519_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>SVTEQGAELSNEER</td>
      <td>30.039</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190604_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02</td>
      <td>LVAIVDVIDQNR</td>
      <td>28.083</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (994, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1379, 155)



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
     'n_samples': 994,
     'y_range': (21, 38)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1988       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 994        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,135
    Total trainable params: 3,135
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x00000227D5115040>
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
      <td>1.573657</td>
      <td>1.480536</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.668847</td>
      <td>0.737137</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.522409</td>
      <td>0.679105</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.649089</td>
      <td>0.662055</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.498343</td>
      <td>0.658082</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.508846</td>
      <td>0.642804</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.551109</td>
      <td>0.631480</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.465316</td>
      <td>0.625472</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.460377</td>
      <td>0.622706</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.460110</td>
      <td>0.622252</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_58_1.png)
    


### Evaluation


```python
collab.dls.valid_ds.items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>1,856</th>
      <td>366</td>
      <td>15</td>
      <td>27.830</td>
    </tr>
    <tr>
      <th>747</th>
      <td>147</td>
      <td>28</td>
      <td>34.001</td>
    </tr>
    <tr>
      <th>2,326</th>
      <td>456</td>
      <td>38</td>
      <td>33.289</td>
    </tr>
    <tr>
      <th>73</th>
      <td>14</td>
      <td>37</td>
      <td>30.712</td>
    </tr>
    <tr>
      <th>962</th>
      <td>187</td>
      <td>50</td>
      <td>30.862</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3,931</th>
      <td>784</td>
      <td>42</td>
      <td>29.590</td>
    </tr>
    <tr>
      <th>541</th>
      <td>110</td>
      <td>37</td>
      <td>29.373</td>
    </tr>
    <tr>
      <th>4,249</th>
      <td>850</td>
      <td>28</td>
      <td>28.283</td>
    </tr>
    <tr>
      <th>1,311</th>
      <td>252</td>
      <td>23</td>
      <td>29.704</td>
    </tr>
    <tr>
      <th>1,868</th>
      <td>368</td>
      <td>42</td>
      <td>29.687</td>
    </tr>
  </tbody>
</table>
<p>4946 rows × 3 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>30.289</td>
      <td>28.539</td>
      <td>28.400</td>
      <td>29.274</td>
      <td>27.786</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.218</td>
      <td>30.022</td>
      <td>30.017</td>
      <td>29.096</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <td>29.280</td>
      <td>29.345</td>
      <td>29.171</td>
      <td>29.406</td>
      <td>28.683</td>
    </tr>
    <tr>
      <th>GLFIIDDK</th>
      <td>31.055</td>
      <td>31.974</td>
      <td>31.874</td>
      <td>31.178</td>
      <td>31.164</td>
    </tr>
    <tr>
      <th>LGDVYVNDAFGTAHR</th>
      <td>31.049</td>
      <td>30.839</td>
      <td>30.823</td>
      <td>30.325</td>
      <td>30.038</td>
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
      <td>29.808</td>
      <td>30.127</td>
      <td>27.612</td>
      <td>29.411</td>
    </tr>
    <tr>
      <th>LIALLEVLSQK</th>
      <td>26.275</td>
      <td>30.076</td>
      <td>29.830</td>
      <td>28.351</td>
      <td>30.575</td>
    </tr>
    <tr>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
      <td>32.689</td>
      <td>32.561</td>
      <td>32.071</td>
      <td>32.572</td>
    </tr>
    <tr>
      <th>STGGAPTFNVTVTK</th>
      <td>32.294</td>
      <td>32.374</td>
      <td>32.187</td>
      <td>32.074</td>
      <td>32.638</td>
    </tr>
    <tr>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>29.156</td>
      <td>34.246</td>
      <td>33.500</td>
      <td>29.025</td>
      <td>32.983</td>
    </tr>
  </tbody>
</table>
<p>4946 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.068
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.081
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.161
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.114
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.152
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_66_1.png)
    



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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>0.169</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.102</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.009</td>
      <td>-0.104</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.208</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.375</td>
      <td>-0.058</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_68_1.png)
    



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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_69_1.png)
    


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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK</th>
      <th>VANVSLLALYK</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <th>VHVIFNYK</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK</th>
      <th>VLNNMEIGTSLFDEEGAK</th>
      <th>VNVPVIGGHAGK</th>
      <th>VVVLMGSTSDLGHCEK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
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
      <td>30.289</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>29.280</td>
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
      <td>34.309</td>
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
      <td>28.677</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>30.736</td>
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
      <td>29.963</td>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK_na</th>
      <th>VANVSLLALYK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_na</th>
      <th>VLNNMEIGTSLFDEEGAK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VVVLMGSTSDLGHCEK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
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
      <td>0.105</td>
      <td>-0.226</td>
      <td>-0.984</td>
      <td>-0.866</td>
      <td>0.137</td>
      <td>-0.122</td>
      <td>-1.032</td>
      <td>-1.232</td>
      <td>0.011</td>
      <td>-0.116</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>1.743</td>
      <td>0.151</td>
      <td>-1.077</td>
      <td>-0.830</td>
      <td>0.001</td>
      <td>-0.317</td>
      <td>-1.325</td>
      <td>-1.245</td>
      <td>-0.024</td>
      <td>-0.543</td>
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
      <td>-0.293</td>
      <td>0.652</td>
      <td>0.188</td>
      <td>-0.843</td>
      <td>-0.048</td>
      <td>-0.737</td>
      <td>-0.413</td>
      <td>-0.485</td>
      <td>-0.872</td>
      <td>-0.446</td>
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
      <td>2.243</td>
      <td>0.529</td>
      <td>-0.973</td>
      <td>-0.006</td>
      <td>0.137</td>
      <td>-0.162</td>
      <td>-0.574</td>
      <td>-0.591</td>
      <td>0.159</td>
      <td>-1.864</td>
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
      <td>2.007</td>
      <td>0.177</td>
      <td>-0.973</td>
      <td>-0.396</td>
      <td>0.137</td>
      <td>0.400</td>
      <td>-0.577</td>
      <td>-1.042</td>
      <td>-0.272</td>
      <td>-0.757</td>
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
      <td>-0.409</td>
      <td>-1.313</td>
      <td>-3.049</td>
      <td>0.637</td>
      <td>0.400</td>
      <td>0.685</td>
      <td>0.800</td>
      <td>0.423</td>
      <td>1.651</td>
      <td>-0.359</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
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
      <td>0.861</td>
      <td>0.355</td>
      <td>0.316</td>
      <td>-0.448</td>
      <td>0.399</td>
      <td>-0.174</td>
      <td>-0.392</td>
      <td>0.059</td>
      <td>-0.289</td>
      <td>0.777</td>
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
      <td>1.041</td>
      <td>0.209</td>
      <td>-0.070</td>
      <td>-0.342</td>
      <td>0.255</td>
      <td>0.378</td>
      <td>-0.306</td>
      <td>-0.271</td>
      <td>-0.320</td>
      <td>1.093</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.894</td>
      <td>0.740</td>
      <td>0.713</td>
      <td>0.060</td>
      <td>0.226</td>
      <td>0.003</td>
      <td>-0.042</td>
      <td>-0.044</td>
      <td>-0.435</td>
      <td>1.108</td>
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
      <td>0.744</td>
      <td>0.344</td>
      <td>0.443</td>
      <td>-0.006</td>
      <td>0.224</td>
      <td>-0.037</td>
      <td>-0.482</td>
      <td>-0.026</td>
      <td>-0.649</td>
      <td>0.918</td>
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
<p>993 rows × 100 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK_na</th>
      <th>VANVSLLALYK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_na</th>
      <th>VLNNMEIGTSLFDEEGAK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VVVLMGSTSDLGHCEK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
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
      <td>0.111</td>
      <td>-0.185</td>
      <td>-0.885</td>
      <td>-0.813</td>
      <td>0.147</td>
      <td>-0.125</td>
      <td>-0.983</td>
      <td>-1.163</td>
      <td>-0.025</td>
      <td>-0.081</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>1.660</td>
      <td>0.164</td>
      <td>-0.971</td>
      <td>-0.780</td>
      <td>0.019</td>
      <td>-0.309</td>
      <td>-1.261</td>
      <td>-1.175</td>
      <td>-0.058</td>
      <td>-0.484</td>
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
      <td>-0.265</td>
      <td>0.627</td>
      <td>0.213</td>
      <td>-0.792</td>
      <td>-0.027</td>
      <td>-0.707</td>
      <td>-0.396</td>
      <td>-0.454</td>
      <td>-0.857</td>
      <td>-0.392</td>
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
      <td>2.133</td>
      <td>0.514</td>
      <td>-0.874</td>
      <td>-0.006</td>
      <td>0.147</td>
      <td>-0.163</td>
      <td>-0.548</td>
      <td>-0.554</td>
      <td>0.115</td>
      <td>-1.730</td>
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
      <td>1.910</td>
      <td>0.188</td>
      <td>-0.874</td>
      <td>-0.373</td>
      <td>0.147</td>
      <td>0.368</td>
      <td>-0.551</td>
      <td>-0.982</td>
      <td>-0.292</td>
      <td>-0.686</td>
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
      <td>-0.375</td>
      <td>-1.191</td>
      <td>-2.819</td>
      <td>0.597</td>
      <td>0.393</td>
      <td>0.637</td>
      <td>0.753</td>
      <td>0.407</td>
      <td>1.521</td>
      <td>-0.310</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
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
      <td>0.826</td>
      <td>0.352</td>
      <td>0.333</td>
      <td>-0.421</td>
      <td>0.392</td>
      <td>-0.175</td>
      <td>-0.376</td>
      <td>0.062</td>
      <td>-0.308</td>
      <td>0.761</td>
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
      <td>0.996</td>
      <td>0.217</td>
      <td>-0.028</td>
      <td>-0.322</td>
      <td>0.257</td>
      <td>0.347</td>
      <td>-0.295</td>
      <td>-0.251</td>
      <td>-0.337</td>
      <td>1.059</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.858</td>
      <td>0.709</td>
      <td>0.705</td>
      <td>0.056</td>
      <td>0.230</td>
      <td>-0.007</td>
      <td>-0.045</td>
      <td>-0.036</td>
      <td>-0.445</td>
      <td>1.073</td>
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
      <td>0.716</td>
      <td>0.342</td>
      <td>0.452</td>
      <td>-0.006</td>
      <td>0.228</td>
      <td>-0.045</td>
      <td>-0.462</td>
      <td>-0.018</td>
      <td>-0.647</td>
      <td>0.894</td>
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
<p>993 rows × 100 columns</p>
</div>



Check mean and standard deviation after normalization


```python
to.items.iloc[:, :10].describe()  # not perferct anymore as expected
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
      <td>993.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.012</td>
      <td>0.024</td>
      <td>0.037</td>
      <td>-0.001</td>
      <td>0.018</td>
      <td>-0.010</td>
      <td>-0.005</td>
      <td>0.006</td>
      <td>-0.035</td>
      <td>0.028</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.946</td>
      <td>0.926</td>
      <td>0.937</td>
      <td>0.939</td>
      <td>0.938</td>
      <td>0.946</td>
      <td>0.948</td>
      <td>0.949</td>
      <td>0.943</td>
      <td>0.943</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.952</td>
      <td>-4.785</td>
      <td>-4.904</td>
      <td>-3.938</td>
      <td>-4.491</td>
      <td>-4.787</td>
      <td>-4.270</td>
      <td>-4.608</td>
      <td>-2.625</td>
      <td>-4.744</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.353</td>
      <td>-0.331</td>
      <td>-0.100</td>
      <td>-0.434</td>
      <td>-0.201</td>
      <td>-0.501</td>
      <td>-0.434</td>
      <td>-0.446</td>
      <td>-0.594</td>
      <td>-0.333</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.111</td>
      <td>0.164</td>
      <td>0.283</td>
      <td>-0.006</td>
      <td>0.147</td>
      <td>-0.094</td>
      <td>-0.045</td>
      <td>0.062</td>
      <td>-0.292</td>
      <td>0.242</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.409</td>
      <td>0.610</td>
      <td>0.574</td>
      <td>0.428</td>
      <td>0.517</td>
      <td>0.566</td>
      <td>0.486</td>
      <td>0.546</td>
      <td>0.583</td>
      <td>0.644</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.463</td>
      <td>1.978</td>
      <td>1.623</td>
      <td>1.989</td>
      <td>1.712</td>
      <td>2.295</td>
      <td>2.059</td>
      <td>2.205</td>
      <td>2.270</td>
      <td>1.562</td>
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




    ((#50) ['AAEAAAAPAESAAPAAGEEPSK','AFGPGLQGGSAGSPAR','AGVNTVTTLVENKK','ALDTMNFDVIK','ATAVVDGAFK','AVLFCLSEDK','AVLVDLEPGTMDSVR','DLLLTSSYLSDSGSTGEHTK','DSLLQDGEFSMDLR','EMEAELEDERK'...],
     (#50) ['AAEAAAAPAESAAPAAGEEPSK_na','AFGPGLQGGSAGSPAR_na','AGVNTVTTLVENKK_na','ALDTMNFDVIK_na','ATAVVDGAFK_na','AVLFCLSEDK_na','AVLVDLEPGTMDSVR_na','DLLLTSSYLSDSGSTGEHTK_na','DSLLQDGEFSMDLR_na','EMEAELEDERK_na'...])




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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>95.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.085</td>
      <td>-0.050</td>
      <td>0.057</td>
      <td>0.013</td>
      <td>-0.067</td>
      <td>-0.204</td>
      <td>0.111</td>
      <td>0.021</td>
      <td>0.073</td>
      <td>-0.072</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.848</td>
      <td>0.996</td>
      <td>0.816</td>
      <td>1.152</td>
      <td>1.015</td>
      <td>1.003</td>
      <td>1.192</td>
      <td>0.937</td>
      <td>1.061</td>
      <td>1.136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.065</td>
      <td>-3.943</td>
      <td>-2.842</td>
      <td>-4.703</td>
      <td>-3.603</td>
      <td>-3.326</td>
      <td>-5.405</td>
      <td>-2.359</td>
      <td>-3.511</td>
      <td>-4.415</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.399</td>
      <td>-0.509</td>
      <td>-0.107</td>
      <td>-0.622</td>
      <td>-0.271</td>
      <td>-0.765</td>
      <td>-0.477</td>
      <td>-0.550</td>
      <td>-0.599</td>
      <td>-0.429</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.129</td>
      <td>0.075</td>
      <td>0.306</td>
      <td>0.075</td>
      <td>0.044</td>
      <td>-0.243</td>
      <td>0.054</td>
      <td>-0.031</td>
      <td>-0.205</td>
      <td>0.290</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.434</td>
      <td>0.649</td>
      <td>0.573</td>
      <td>0.717</td>
      <td>0.374</td>
      <td>0.308</td>
      <td>1.249</td>
      <td>0.653</td>
      <td>1.050</td>
      <td>0.662</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.246</td>
      <td>1.419</td>
      <td>1.078</td>
      <td>1.750</td>
      <td>1.627</td>
      <td>2.096</td>
      <td>1.963</td>
      <td>1.860</td>
      <td>1.981</td>
      <td>1.291</td>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK_val</th>
      <th>VANVSLLALYK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_val</th>
      <th>VLNNMEIGTSLFDEEGAK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VVVLMGSTSDLGHCEK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
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
      <td>0.111</td>
      <td>-0.185</td>
      <td>-0.885</td>
      <td>-0.813</td>
      <td>0.147</td>
      <td>-0.125</td>
      <td>-0.983</td>
      <td>-1.163</td>
      <td>-0.025</td>
      <td>-0.081</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>1.660</td>
      <td>0.164</td>
      <td>-0.971</td>
      <td>-0.780</td>
      <td>0.019</td>
      <td>-0.309</td>
      <td>-1.261</td>
      <td>-1.175</td>
      <td>-0.058</td>
      <td>-0.484</td>
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
      <td>-0.265</td>
      <td>0.627</td>
      <td>0.213</td>
      <td>-0.792</td>
      <td>-0.027</td>
      <td>-0.707</td>
      <td>-0.396</td>
      <td>-0.454</td>
      <td>-0.857</td>
      <td>-0.392</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.402</td>
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
      <td>2.133</td>
      <td>0.514</td>
      <td>-0.874</td>
      <td>-0.006</td>
      <td>0.147</td>
      <td>-0.163</td>
      <td>-0.548</td>
      <td>-0.554</td>
      <td>0.115</td>
      <td>-1.730</td>
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
      <td>-0.472</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>1.910</td>
      <td>0.188</td>
      <td>-0.874</td>
      <td>-0.373</td>
      <td>0.147</td>
      <td>0.368</td>
      <td>-0.551</td>
      <td>-0.982</td>
      <td>-0.292</td>
      <td>-0.686</td>
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
      <td>-0.375</td>
      <td>-1.191</td>
      <td>-2.819</td>
      <td>0.597</td>
      <td>0.393</td>
      <td>0.637</td>
      <td>0.753</td>
      <td>0.407</td>
      <td>1.521</td>
      <td>-0.310</td>
      <td>...</td>
      <td>0.494</td>
      <td>0.258</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.826</td>
      <td>0.352</td>
      <td>0.333</td>
      <td>-0.421</td>
      <td>0.392</td>
      <td>-0.175</td>
      <td>-0.376</td>
      <td>0.062</td>
      <td>-0.308</td>
      <td>0.761</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.641</td>
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
      <td>0.996</td>
      <td>0.217</td>
      <td>-0.028</td>
      <td>-0.322</td>
      <td>0.257</td>
      <td>0.347</td>
      <td>-0.295</td>
      <td>-0.251</td>
      <td>-0.337</td>
      <td>1.059</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.297</td>
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
      <td>0.858</td>
      <td>0.709</td>
      <td>0.705</td>
      <td>0.056</td>
      <td>0.230</td>
      <td>-0.007</td>
      <td>-0.045</td>
      <td>-0.036</td>
      <td>-0.445</td>
      <td>1.073</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.489</td>
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
      <td>0.716</td>
      <td>0.342</td>
      <td>0.452</td>
      <td>-0.006</td>
      <td>0.228</td>
      <td>-0.045</td>
      <td>-0.462</td>
      <td>-0.018</td>
      <td>-0.647</td>
      <td>0.894</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.160</td>
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
<p>993 rows × 150 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK_val</th>
      <th>VANVSLLALYK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_val</th>
      <th>VLNNMEIGTSLFDEEGAK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VVVLMGSTSDLGHCEK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
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
      <td>0.111</td>
      <td>-0.185</td>
      <td>-0.885</td>
      <td>-0.813</td>
      <td>0.147</td>
      <td>-0.125</td>
      <td>-0.983</td>
      <td>-1.163</td>
      <td>-0.025</td>
      <td>-0.081</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>1.660</td>
      <td>0.164</td>
      <td>-0.971</td>
      <td>-0.780</td>
      <td>0.019</td>
      <td>-0.309</td>
      <td>-1.261</td>
      <td>-1.175</td>
      <td>-0.058</td>
      <td>-0.484</td>
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
      <td>-0.265</td>
      <td>0.627</td>
      <td>0.213</td>
      <td>-0.792</td>
      <td>-0.027</td>
      <td>-0.707</td>
      <td>-0.396</td>
      <td>-0.454</td>
      <td>-0.857</td>
      <td>-0.392</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.402</td>
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
      <td>2.133</td>
      <td>0.514</td>
      <td>-0.874</td>
      <td>-0.006</td>
      <td>0.147</td>
      <td>-0.163</td>
      <td>-0.548</td>
      <td>-0.554</td>
      <td>0.115</td>
      <td>-1.730</td>
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
      <td>-0.472</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>1.910</td>
      <td>0.188</td>
      <td>-0.874</td>
      <td>-0.373</td>
      <td>0.147</td>
      <td>0.368</td>
      <td>-0.551</td>
      <td>-0.982</td>
      <td>-0.292</td>
      <td>-0.686</td>
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
      <td>-0.375</td>
      <td>-1.191</td>
      <td>-2.819</td>
      <td>0.597</td>
      <td>0.393</td>
      <td>0.637</td>
      <td>0.753</td>
      <td>0.407</td>
      <td>1.521</td>
      <td>-0.310</td>
      <td>...</td>
      <td>0.494</td>
      <td>0.258</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.826</td>
      <td>0.352</td>
      <td>0.333</td>
      <td>-0.421</td>
      <td>0.392</td>
      <td>-0.175</td>
      <td>-0.376</td>
      <td>0.062</td>
      <td>-0.308</td>
      <td>0.761</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.641</td>
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
      <td>0.996</td>
      <td>0.217</td>
      <td>-0.028</td>
      <td>-0.322</td>
      <td>0.257</td>
      <td>0.347</td>
      <td>-0.295</td>
      <td>-0.251</td>
      <td>-0.337</td>
      <td>1.059</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.297</td>
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
      <td>0.858</td>
      <td>0.709</td>
      <td>0.705</td>
      <td>0.056</td>
      <td>0.230</td>
      <td>-0.007</td>
      <td>-0.045</td>
      <td>-0.036</td>
      <td>-0.445</td>
      <td>1.073</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.489</td>
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
      <td>0.716</td>
      <td>0.342</td>
      <td>0.452</td>
      <td>-0.006</td>
      <td>0.228</td>
      <td>-0.045</td>
      <td>-0.462</td>
      <td>-0.018</td>
      <td>-0.647</td>
      <td>0.894</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.160</td>
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
<p>993 rows × 150 columns</p>
</div>




```python
stats_valid = to_valid.targ.iloc[:, :100].describe()
stats_valid
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK_val</th>
      <th>AFGPGLQGGSAGSPAR_val</th>
      <th>AGVNTVTTLVENKK_val</th>
      <th>ALDTMNFDVIK_val</th>
      <th>ATAVVDGAFK_val</th>
      <th>AVLFCLSEDK_val</th>
      <th>AVLVDLEPGTMDSVR_val</th>
      <th>DLLLTSSYLSDSGSTGEHTK_val</th>
      <th>DSLLQDGEFSMDLR_val</th>
      <th>EMEAELEDERK_val</th>
      <th>...</th>
      <th>TICSHVQNMIK_val</th>
      <th>VANVSLLALYK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_val</th>
      <th>VLNNMEIGTSLFDEEGAK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VVVLMGSTSDLGHCEK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>95.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.085</td>
      <td>-0.050</td>
      <td>0.057</td>
      <td>0.013</td>
      <td>-0.067</td>
      <td>-0.204</td>
      <td>0.111</td>
      <td>0.021</td>
      <td>0.073</td>
      <td>-0.072</td>
      <td>...</td>
      <td>0.063</td>
      <td>0.177</td>
      <td>-0.191</td>
      <td>-0.032</td>
      <td>-0.044</td>
      <td>0.171</td>
      <td>0.211</td>
      <td>0.031</td>
      <td>-0.188</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.848</td>
      <td>0.996</td>
      <td>0.816</td>
      <td>1.152</td>
      <td>1.015</td>
      <td>1.003</td>
      <td>1.192</td>
      <td>0.937</td>
      <td>1.061</td>
      <td>1.136</td>
      <td>...</td>
      <td>1.073</td>
      <td>0.800</td>
      <td>1.193</td>
      <td>1.153</td>
      <td>0.856</td>
      <td>0.696</td>
      <td>1.031</td>
      <td>0.830</td>
      <td>1.115</td>
      <td>0.931</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.065</td>
      <td>-3.943</td>
      <td>-2.842</td>
      <td>-4.703</td>
      <td>-3.603</td>
      <td>-3.326</td>
      <td>-5.405</td>
      <td>-2.359</td>
      <td>-3.511</td>
      <td>-4.415</td>
      <td>...</td>
      <td>-4.288</td>
      <td>-2.682</td>
      <td>-3.723</td>
      <td>-4.525</td>
      <td>-2.599</td>
      <td>-2.064</td>
      <td>-1.876</td>
      <td>-2.863</td>
      <td>-3.710</td>
      <td>-2.378</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.399</td>
      <td>-0.509</td>
      <td>-0.107</td>
      <td>-0.622</td>
      <td>-0.271</td>
      <td>-0.765</td>
      <td>-0.477</td>
      <td>-0.550</td>
      <td>-0.599</td>
      <td>-0.429</td>
      <td>...</td>
      <td>-0.316</td>
      <td>-0.180</td>
      <td>-0.753</td>
      <td>-0.542</td>
      <td>-0.439</td>
      <td>-0.165</td>
      <td>-0.600</td>
      <td>-0.349</td>
      <td>-0.729</td>
      <td>-0.456</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.129</td>
      <td>0.075</td>
      <td>0.306</td>
      <td>0.075</td>
      <td>0.044</td>
      <td>-0.243</td>
      <td>0.054</td>
      <td>-0.031</td>
      <td>-0.205</td>
      <td>0.290</td>
      <td>...</td>
      <td>0.207</td>
      <td>0.287</td>
      <td>0.344</td>
      <td>-0.136</td>
      <td>-0.033</td>
      <td>0.181</td>
      <td>-0.125</td>
      <td>0.035</td>
      <td>-0.211</td>
      <td>-0.132</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.434</td>
      <td>0.649</td>
      <td>0.573</td>
      <td>0.717</td>
      <td>0.374</td>
      <td>0.308</td>
      <td>1.249</td>
      <td>0.653</td>
      <td>1.050</td>
      <td>0.662</td>
      <td>...</td>
      <td>0.702</td>
      <td>0.645</td>
      <td>0.556</td>
      <td>0.833</td>
      <td>0.614</td>
      <td>0.636</td>
      <td>1.291</td>
      <td>0.523</td>
      <td>0.281</td>
      <td>0.586</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.246</td>
      <td>1.419</td>
      <td>1.078</td>
      <td>1.750</td>
      <td>1.627</td>
      <td>2.096</td>
      <td>1.963</td>
      <td>1.860</td>
      <td>1.981</td>
      <td>1.291</td>
      <td>...</td>
      <td>1.850</td>
      <td>1.619</td>
      <td>1.096</td>
      <td>1.983</td>
      <td>1.501</td>
      <td>1.566</td>
      <td>2.071</td>
      <td>1.985</td>
      <td>1.886</td>
      <td>1.928</td>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK_na</th>
      <th>AFGPGLQGGSAGSPAR_na</th>
      <th>AGVNTVTTLVENKK_na</th>
      <th>ALDTMNFDVIK_na</th>
      <th>ATAVVDGAFK_na</th>
      <th>AVLFCLSEDK_na</th>
      <th>AVLVDLEPGTMDSVR_na</th>
      <th>DLLLTSSYLSDSGSTGEHTK_na</th>
      <th>DSLLQDGEFSMDLR_na</th>
      <th>EMEAELEDERK_na</th>
      <th>...</th>
      <th>TICSHVQNMIK_na</th>
      <th>VANVSLLALYK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK_na</th>
      <th>VLNNMEIGTSLFDEEGAK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VVVLMGSTSDLGHCEK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
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
<p>993 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x00000227D5115040>
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








    SuggestedLRs(valley=0.007585775572806597)




    
![png](latent_2D_100_30_files/latent_2D_100_30_108_2.png)
    


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
      <td>0.984246</td>
      <td>0.877290</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.703910</td>
      <td>0.446222</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.499312</td>
      <td>0.385190</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.396913</td>
      <td>0.351688</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.340926</td>
      <td>0.336645</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.312803</td>
      <td>0.335878</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.295363</td>
      <td>0.327199</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.282332</td>
      <td>0.324755</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.275781</td>
      <td>0.321707</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.271010</td>
      <td>0.323776</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4946, 4946)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3195)




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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>30.289</td>
      <td>28.539</td>
      <td>28.400</td>
      <td>29.274</td>
      <td>27.786</td>
      <td>27.917</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.218</td>
      <td>30.022</td>
      <td>30.017</td>
      <td>29.096</td>
      <td>29.360</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <td>29.280</td>
      <td>29.345</td>
      <td>29.171</td>
      <td>29.406</td>
      <td>28.683</td>
      <td>29.302</td>
    </tr>
    <tr>
      <th>GLFIIDDK</th>
      <td>31.055</td>
      <td>31.974</td>
      <td>31.874</td>
      <td>31.178</td>
      <td>31.164</td>
      <td>31.159</td>
    </tr>
    <tr>
      <th>LGDVYVNDAFGTAHR</th>
      <td>31.049</td>
      <td>30.839</td>
      <td>30.823</td>
      <td>30.325</td>
      <td>30.038</td>
      <td>29.928</td>
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
      <td>29.808</td>
      <td>30.127</td>
      <td>27.612</td>
      <td>29.411</td>
      <td>29.510</td>
    </tr>
    <tr>
      <th>LIALLEVLSQK</th>
      <td>26.275</td>
      <td>30.076</td>
      <td>29.830</td>
      <td>28.351</td>
      <td>30.575</td>
      <td>30.741</td>
    </tr>
    <tr>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
      <td>32.689</td>
      <td>32.561</td>
      <td>32.071</td>
      <td>32.572</td>
      <td>32.536</td>
    </tr>
    <tr>
      <th>STGGAPTFNVTVTK</th>
      <td>32.294</td>
      <td>32.374</td>
      <td>32.187</td>
      <td>32.074</td>
      <td>32.638</td>
      <td>32.594</td>
    </tr>
    <tr>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>29.156</td>
      <td>34.246</td>
      <td>33.500</td>
      <td>29.025</td>
      <td>32.983</td>
      <td>33.092</td>
    </tr>
  </tbody>
</table>
<p>4946 rows × 6 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>-0.600</td>
      <td>0.492</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.605</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.523</td>
      <td>0.589</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.311</td>
      <td>0.411</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.488</td>
      <td>0.658</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_122_1.png)
    



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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_123_1.png)
    


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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>ATAVVDGAFK</th>
      <th>AVLFCLSEDK</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DLLLTSSYLSDSGSTGEHTK</th>
      <th>DSLLQDGEFSMDLR</th>
      <th>EMEAELEDERK</th>
      <th>...</th>
      <th>TICSHVQNMIK</th>
      <th>VANVSLLALYK</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <th>VHVIFNYK</th>
      <th>VIHDNFGIVEGLMTTVHAITATQK</th>
      <th>VLNNMEIGTSLFDEEGAK</th>
      <th>VNVPVIGGHAGK</th>
      <th>VVVLMGSTSDLGHCEK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
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
      <td>0.768</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>0.723</td>
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
      <td>0.846</td>
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
      <td>0.555</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>0.545</td>
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
      <td>0.617</td>
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
    
    Optimizer used: <function Adam at 0x00000227D5115040>
    Loss function: <function loss_fct_vae at 0x00000227D513D940>
    
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








    SuggestedLRs(valley=0.0030199517495930195)




    
![png](latent_2D_100_30_files/latent_2D_100_30_136_2.png)
    



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
      <td>2020.807739</td>
      <td>221.711914</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1982.927124</td>
      <td>216.146133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1920.371582</td>
      <td>210.258453</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1865.234009</td>
      <td>201.004623</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1827.205444</td>
      <td>200.791077</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1800.115723</td>
      <td>201.149750</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1782.311890</td>
      <td>200.850006</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1768.448730</td>
      <td>199.686691</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1758.092041</td>
      <td>200.488861</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1750.121704</td>
      <td>200.611832</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4946)




```python
len(pred[0])
```




    4946




```python
learn.loss_func(pred, target)
```




    tensor(3158.8127)




```python
_pred = pd.Series(pred[0], index=analysis.df_valid.stack().index).unstack()
_pred = scaler.inverse_transform(_pred).stack()

df_pred['intensity_pred_vae'] = _pred
df_pred
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AAEAAAAPAESAAPAAGEEPSK</th>
      <td>30.289</td>
      <td>28.539</td>
      <td>28.400</td>
      <td>29.274</td>
      <td>27.786</td>
      <td>27.917</td>
      <td>28.429</td>
    </tr>
    <tr>
      <th>ATAVVDGAFK</th>
      <td>29.988</td>
      <td>30.218</td>
      <td>30.022</td>
      <td>30.017</td>
      <td>29.096</td>
      <td>29.360</td>
      <td>30.139</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFGPGLQGGSAGSPAR</th>
      <td>29.280</td>
      <td>29.345</td>
      <td>29.171</td>
      <td>29.406</td>
      <td>28.683</td>
      <td>29.302</td>
      <td>29.276</td>
    </tr>
    <tr>
      <th>GLFIIDDK</th>
      <td>31.055</td>
      <td>31.974</td>
      <td>31.874</td>
      <td>31.178</td>
      <td>31.164</td>
      <td>31.159</td>
      <td>31.977</td>
    </tr>
    <tr>
      <th>LGDVYVNDAFGTAHR</th>
      <td>31.049</td>
      <td>30.839</td>
      <td>30.823</td>
      <td>30.325</td>
      <td>30.038</td>
      <td>29.928</td>
      <td>31.062</td>
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
      <td>29.808</td>
      <td>30.127</td>
      <td>27.612</td>
      <td>29.411</td>
      <td>29.510</td>
      <td>30.180</td>
    </tr>
    <tr>
      <th>LIALLEVLSQK</th>
      <td>26.275</td>
      <td>30.076</td>
      <td>29.830</td>
      <td>28.351</td>
      <td>30.575</td>
      <td>30.741</td>
      <td>30.009</td>
    </tr>
    <tr>
      <th>LNFSHGTHEYHAETIK</th>
      <td>33.024</td>
      <td>32.689</td>
      <td>32.561</td>
      <td>32.071</td>
      <td>32.572</td>
      <td>32.536</td>
      <td>32.670</td>
    </tr>
    <tr>
      <th>STGGAPTFNVTVTK</th>
      <td>32.294</td>
      <td>32.374</td>
      <td>32.187</td>
      <td>32.074</td>
      <td>32.638</td>
      <td>32.594</td>
      <td>32.244</td>
    </tr>
    <tr>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>29.156</td>
      <td>34.246</td>
      <td>33.500</td>
      <td>29.025</td>
      <td>32.983</td>
      <td>33.092</td>
      <td>33.628</td>
    </tr>
  </tbody>
</table>
<p>4946 rows × 7 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>0.016</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.000</td>
      <td>0.217</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.039</td>
      <td>-0.316</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.005</td>
      <td>-0.053</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.013</td>
      <td>-0.056</td>
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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_146_1.png)
    



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
    


    
![png](latent_2D_100_30_files/latent_2D_100_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'ATAVVDGAFK'), ('20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-052', 'SLAGSSGPGASSGTSGDHGELVVR'), ('20190114_QE7_nLC7_AL_QC_MNT_HeLa_01', 'LLQDFFNGK'), ('20190114_QE7_nLC7_AL_QC_MNT_HeLa_02', 'LLQDFFNGK'), ('20190114_QE9_nLC9_NHS_MNT_HELA_01', 'LLQDFFNGK'), ('20190115_QE2_NLC10_TW_QC_MNT_HeLa_01', 'LGDVYVNDAFGTAHR'), ('20190121_QE1_nLC2_GP_QC_MNT_HELA_01', 'ALDTMNFDVIK'), ('20190121_QE2_NLC1_GP_QC_MNT_HELA_01', 'ALDTMNFDVIK'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02', 'SLHDALCVLAQTVK'), ('20190128_QE3_nLC3_MJ_MNT_HeLa_03', 'LNFSHGTHEYHAETIK'), ('20190129_QE1_nLC2_GP_QC_MNT_HELA_02', 'IKDPDASKPEDWDER'), ('20190202_QE7_nLC7_MEM_QC_MNT_HeLa_02', 'HLPTLDHPIIPADYVAIK'), ('20190205_QE7_nLC7_MEM_QC_MNT_HeLa_02', 'LATQSNEITIPVTFESR'), ('20190207_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01', 'GCITIIGGGDTATCCAK'), ('20190208_QE2_NLC1_AB_QC_MNT_HELA_2', 'LSSLIILMPHHVEPLER'), ('20190208_QE2_NLC1_AB_QC_MNT_HELA_3', 'LSSLIILMPHHVEPLER'), ('20190211_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'VIHDNFGIVEGLMTTVHAITATQK'), ('20190211_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'VIHDNFGIVEGLMTTVHAITATQK'), ('20190211_QE4_nLC12_SIS_QC_MNT_Hela_1', 'VANVSLLALYK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_02', 'IKDPDASKPEDWDER'), ('20190215_QE7_nLC3_CK_QC_MNT_HeLa_01', 'FYPEDVSEELIQDITQR'), ('20190220_QE2_NLC1_GP_QC_MNT_HELA_01', 'GITINAAHVEYSTAAR'), ('20190224_QE2_NLC1_GP_MNT_HeLa_1', 'IFAPNHVVAK'), ('20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_3', 'EMEAELEDERK'), ('20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_3', 'NPDDITNEEYGEFYK'), ('20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_3', 'QEMQEVQSSR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_1', 'NPDDITNEEYGEFYK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_1', 'QEMQEVQSSR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_2', 'GQYISPFHDIPIYADK'), ('20190305_QE2_NLC1_AB_QC_MNT_HELA_02', 'FIQENIFGICPHMTEDNK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_02', 'LGDVYVNDAFGTAHR'), ('20190310_QE2_NLC1_GP_MNT_HELA_01', 'DLLLTSSYLSDSGSTGEHTK'), ('20190325_QE3_nLC5_DS_QC_MNT_HeLa_02', 'LNFSHGTHEYHAETIK'), ('20190325_QE9_nLC0_JM_MNT_Hela_50cm_01', 'FIQENIFGICPHMTEDNK'), ('20190329_QE8_nLC14_RS_QC_MNT_Hela_50cm_02', 'SGAQASSTPLSPTR'), ('20190404_QE10_nLC13_FaCo_QC_45cm_HeLa_01', 'GLTSVINQK'), ('20190408_QE1_nLC2_GP_MNT_QC_hela_02_20190408131505', 'FDTGNLCMVTGGANLGR'), ('20190416_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-002', 'LVAIVDVIDQNR'), ('20190423_QX0_MaPe_MA_HeLa_500ng_LC07_1_high', 'TICSHVQNMIK'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'SGAQASSTPLSPTR'), ('20190429_QX0_ChDe_MA_HeLa_500ng_LC07_1_BR13_190507121913', 'AFGPGLQGGSAGSPAR'), ('20190506_QX7_ChDe_MA_HeLaBr14_500ng', 'AGVNTVTTLVENKK'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'AFGPGLQGGSAGSPAR'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'SGAQASSTPLSPTR'), ('20190508_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff_1', 'AFGPGLQGGSAGSPAR'), ('20190513_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'TFVNITPAEVGVLVGK'), ('20190513_QX7_ChDe_MA_HeLaBr14_500ng', 'DSLLQDGEFSMDLR'), ('20190513_QX8_MiWi_MA_HeLa_BR14_500ng', 'DSLLQDGEFSMDLR'), ('20190515_QX3_AsJa_MA_Hela_500ng_LC15', 'IFAPNHVVAK'), ('20190515_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'SVTEQGAELSNEER'), ('20190519_QE1_nLC2_GP_QC_MNT_HELA_01', 'LLQDFFNGK'), ('20190522_QE3_nLC3_AP_QC_MNT_HeLa_03', 'IIALDGDTK'), ('20190523_QE2_NLC1_GP_MNT_HELA_01', 'GQYISPFHDIPIYADK'), ('20190530_QE2_NLC1_GP_QC_MNT_HELA_01', 'DSLLQDGEFSMDLR'), ('20190531_QE4_nLC12_MM_QC_MNT_HELA_02_20190605020529', 'VNVPVIGGHAGK'), ('20190606_QX4_JiYu_MA_HeLa_500ng', 'HSGPNSADSANDGFVR'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_4', 'LLQDFFNGK'), ('20190612_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'PMFIVNTNVPR'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02', 'LATQSNEITIPVTFESR'), ('20190614_QX3_JoSw_MA_Hela_500ng_LC15', 'DSLLQDGEFSMDLR'), ('20190619_QE2_NLC1_GP_QC_MNT_HELA_01', 'IFAPNHVVAK'), ('20190621_QX4_JoMu_MA_HeLa_500ng_190621161214', 'DSLLQDGEFSMDLR'), ('20190628_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'FYPEDVSEELIQDITQR'), ('20190629_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'STGGAPTFNVTVTK'), ('20190701_QX2_LiSc_MA_HeLa_500ng_LC05_without_columnoven', 'TFVNITPAEVGVLVGK'), ('20190702_QE3_nLC5_TSB_QC_MNT_HELA_02', 'VVVLMGSTSDLGHCEK'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05_190709143552', 'HSGPNSADSANDGFVR'), ('20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01', 'VNVPVIGGHAGK'), ('20190715_QE8_nLC14_RG_QC_MNT_50cm_Hela_02', 'IIALDGDTK'), ('20190802_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_1', 'FIQENIFGICPHMTEDNK')]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>0.622</td>
      <td>0.649</td>
      <td>1.616</td>
      <td>1.699</td>
      <td>2.139</td>
      <td>2.244</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.455</td>
      <td>0.476</td>
      <td>0.842</td>
      <td>0.932</td>
      <td>1.064</td>
      <td>1.041</td>
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
