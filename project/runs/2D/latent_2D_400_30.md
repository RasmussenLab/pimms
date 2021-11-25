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
    HRPELIEYDK                                 959
    LASVPAGGAVAVSAAPGSAAPAAGSAPAAAEEK          999
    TPAQYDASELK                                982
    VNIIPLIAK                                  993
    GNDISSGTVLSDYVGSGPPK                       996
    IVSQLLTLMDGLK                              990
    EEEIAALVIDNGSGMCK                          988
    EIGNIISDAMK                                997
    GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK     993
    LLDFGSLSNLQVTQPTVGMNFK                     960
    LITPAVVSER                                 975
    TATESFASDPILYRPVAVALDTK                    999
    IKGEHPGLSIGDVAK                            951
    MTNGFSGADLTEICQR                           993
    FWEVISDEHGIDPTGTYHGDSDLQLDR                971
    DPFAHLPK                                   990
    SYSPYDMLESIRK                              954
    NFGSYVTHETK                                971
    VFITDDFHDMMPK                              994
    TCTTVAFTQVNSEDK                          1,000
    ISSLLEEQFQQGK                              990
    LVLVGDGGTGK                                996
    FDTGNLCMVTGGANLGR                          999
    AYFHLLNQIAPK                               999
    FWEVISDEHGIDPTGTYHGDSDLQLER                976
    SEHPGLSIGDTAK                              975
    EAAENSLVAYK                                952
    GHYTEGAELVDSVLDVVR                         959
    GLVLGPIHK                                  986
    NYIQGINLVQAK                               982
    ADRDESSPYAAMLAAQDVAQR                      998
    TANDMIHAENMR                               990
    HVLVTLGEK                                  994
    TKPYIQVDIGGGQTK                            994
    VVLAYEPVWAIGTGK                            943
    NMMAACDPR                                  997
    NIIHGSDSVK                                 988
    VLAQNSGFDLQETLVK                           991
    MLISILTER                                  980
    IIPGFMCQGGDFTR                           1,000
    AIAELGIYPAVDPLDSTSR                        949
    TIAPALVSK                                  997
    PLRLPLQDVYK                                947
    GQHVPGSPFQFTVGPLGEGGAHK                    999
    DLSHIGDAVVISCAK                            993
    LNSNTQVVLLSATMPSDVLEVTK                    999
    HEQILVLDPPTDLK                             981
    STGEAFVQFASQEIAEK                          992
    ILPTLEAVAALGNK                             999
    GLNSESMTEETLK                              987
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
      <th>HRPELIEYDK</th>
      <td>29.967</td>
    </tr>
    <tr>
      <th>LASVPAGGAVAVSAAPGSAAPAAGSAPAAAEEK</th>
      <td>30.288</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>30.852</td>
    </tr>
    <tr>
      <th>VNIIPLIAK</th>
      <td>27.776</td>
    </tr>
    <tr>
      <th>GNDISSGTVLSDYVGSGPPK</th>
      <td>27.077</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>LNSNTQVVLLSATMPSDVLEVTK</th>
      <td>29.019</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
    </tr>
    <tr>
      <th>GLNSESMTEETLK</th>
      <td>28.795</td>
    </tr>
  </tbody>
</table>
<p>49187 rows × 1 columns</p>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_24_1.png)
    



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
      <td>0.984</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.025</td>
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
      <th>HRPELIEYDK</th>
      <td>29.967</td>
    </tr>
    <tr>
      <th>LASVPAGGAVAVSAAPGSAAPAAGSAPAAAEEK</th>
      <td>30.288</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>30.852</td>
    </tr>
    <tr>
      <th>VNIIPLIAK</th>
      <td>27.776</td>
    </tr>
    <tr>
      <th>GNDISSGTVLSDYVGSGPPK</th>
      <td>27.077</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>LNSNTQVVLLSATMPSDVLEVTK</th>
      <td>29.019</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
    </tr>
    <tr>
      <th>GLNSESMTEETLK</th>
      <td>28.795</td>
    </tr>
  </tbody>
</table>
<p>49187 rows × 1 columns</p>
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
      <th>20190411_QE6_LC6_AS_QC_MNT_HeLa_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>28.899</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.632</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_01</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>25.435</td>
    </tr>
    <tr>
      <th>20190528_QX1_PhGe_MA_HeLa_DMSO_500ng_LC14_190528164924</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>32.218</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.291</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190606_QE4_LC12_JE_QC_MNT_HeLa_02b</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>29.808</td>
    </tr>
    <tr>
      <th>20190429_QX4_ChDe_MA_HeLa_500ng_BR14_standard</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>32.363</td>
    </tr>
    <tr>
      <th>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_18_30_</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>32.128</td>
    </tr>
    <tr>
      <th>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_15_25</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>31.994</td>
    </tr>
    <tr>
      <th>20190115_QE2_NLC10_TW_QC_MNT_HeLa_02</th>
      <th>VVLAYEPVWAIGTGK</th>
      <td>30.897</td>
    </tr>
  </tbody>
</table>
<p>44269 rows × 1 columns</p>
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
    Shape in validation: (996, 50)
    




    ((996, 50), (996, 50))



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
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.282</td>
      <td>29.554</td>
      <td>29.678</td>
      <td>28.917</td>
    </tr>
    <tr>
      <th>DPFAHLPK</th>
      <td>29.130</td>
      <td>27.868</td>
      <td>29.035</td>
      <td>28.298</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>27.325</td>
      <td>30.109</td>
      <td>30.061</td>
      <td>28.220</td>
    </tr>
    <tr>
      <th>NYIQGINLVQAK</th>
      <td>27.138</td>
      <td>28.461</td>
      <td>28.505</td>
      <td>27.550</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>28.366</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>28.875</td>
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
      <th>DLSHIGDAVVISCAK</th>
      <td>28.656</td>
      <td>28.166</td>
      <td>28.427</td>
      <td>28.326</td>
    </tr>
    <tr>
      <th>EEEIAALVIDNGSGMCK</th>
      <td>32.530</td>
      <td>32.191</td>
      <td>31.751</td>
      <td>32.697</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
      <td>27.794</td>
      <td>27.688</td>
      <td>28.165</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
      <td>29.571</td>
      <td>29.450</td>
      <td>29.802</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>29.789</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>30.121</td>
    </tr>
  </tbody>
</table>
<p>4918 rows × 4 columns</p>
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
      <th>DPFAHLPK</th>
      <td>28.357</td>
      <td>27.868</td>
      <td>29.035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</th>
      <th>GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK</th>
      <td>31.252</td>
      <td>31.834</td>
      <td>31.924</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190204_QE4_LC12_SCL_QC_MNT_HeLa_02</th>
      <th>GHYTEGAELVDSVLDVVR</th>
      <td>29.947</td>
      <td>29.645</td>
      <td>30.026</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190204_QE8_nLC14_RG_QC_HeLa_15cm_02</th>
      <th>MTNGFSGADLTEICQR</th>
      <td>29.014</td>
      <td>27.926</td>
      <td>27.898</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_0</th>
      <th>MTNGFSGADLTEICQR</th>
      <td>27.732</td>
      <td>27.926</td>
      <td>27.898</td>
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
      <th>20190726_QE7_nLC7_MEM_QC_MNT_HeLa_01</th>
      <th>TKPYIQVDIGGGQTK</th>
      <td>29.281</td>
      <td>29.211</td>
      <td>29.205</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2</th>
      <th>GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK</th>
      <td>28.156</td>
      <td>31.834</td>
      <td>31.924</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_02</th>
      <th>IKGEHPGLSIGDVAK</th>
      <td>28.400</td>
      <td>30.626</td>
      <td>30.394</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SEHPGLSIGDTAK</th>
      <td>28.356</td>
      <td>28.826</td>
      <td>28.654</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>HEQILVLDPPTDLK</th>
      <td>27.762</td>
      <td>27.794</td>
      <td>27.688</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>73 rows × 4 columns</p>
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
      <td>AIAELGIYPAVDPLDSTSR</td>
      <td>29.355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AYFHLLNQIAPK</td>
      <td>27.749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DLSHIGDAVVISCAK</td>
      <td>28.155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EAAENSLVAYK</td>
      <td>29.716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EEEIAALVIDNGSGMCK</td>
      <td>31.237</td>
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
      <td>ADRDESSPYAAMLAAQDVAQR</td>
      <td>29.282</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DPFAHLPK</td>
      <td>29.130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>FDTGNLCMVTGGANLGR</td>
      <td>27.325</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>NYIQGINLVQAK</td>
      <td>27.138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TCTTVAFTQVNSEDK</td>
      <td>28.366</td>
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
      <td>20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01_20190326190317</td>
      <td>GLNSESMTEETLK</td>
      <td>27.470</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190802_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2</td>
      <td>NMMAACDPR</td>
      <td>32.741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190211_QE4_nLC12_SIS_QC_MNT_Hela_2</td>
      <td>GNDISSGTVLSDYVGSGPPK</td>
      <td>28.904</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190515_QX3_AsJa_MA_Hela_500ng_LC15</td>
      <td>SEHPGLSIGDTAK</td>
      <td>29.512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190121_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>GHYTEGAELVDSVLDVVR</td>
      <td>26.341</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190408_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-007</td>
      <td>VVLAYEPVWAIGTGK</td>
      <td>32.125</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190708_QE4_LC12_IAH_QC_MNT_HeLa_02</td>
      <td>NYIQGINLVQAK</td>
      <td>28.068</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190701_QX8_AnPi_MA_HeLa_BR14_500ng</td>
      <td>STGEAFVQFASQEIAEK</td>
      <td>32.474</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190617_QE_LC_UHG_QC_MNT_HELA_01_20190617212753</td>
      <td>EAAENSLVAYK</td>
      <td>29.876</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190606_QE4_LC12_JE_QC_MNT_HeLa_03</td>
      <td>PLRLPLQDVYK</td>
      <td>28.026</td>
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
      <td>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_02</td>
      <td>TATESFASDPILYRPVAVALDTK</td>
      <td>31.397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190731_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05</td>
      <td>EEEIAALVIDNGSGMCK</td>
      <td>29.698</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190626_QX1_JoMu_MA_HeLa_500ng_LC10_190626135146</td>
      <td>FWEVISDEHGIDPTGTYHGDSDLQLER</td>
      <td>32.836</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190526_QX4_LiSc_MA_HeLa_500ng</td>
      <td>IIPGFMCQGGDFTR</td>
      <td>34.506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190118_QE9_nLC9_NHS_MNT_HELA_50cm_05</td>
      <td>NMMAACDPR</td>
      <td>30.795</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190708_QX8_AnPi_MA_HeLa_BR14_500ng</td>
      <td>ADRDESSPYAAMLAAQDVAQR</td>
      <td>30.188</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190221_QE8_nLC9_JM_QC_MNT_HeLa_01</td>
      <td>MLISILTER</td>
      <td>28.724</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190729_QX0_AsJa_MA_HeLa_500ng_LC07_01</td>
      <td>TKPYIQVDIGGGQTK</td>
      <td>29.719</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190301_QE6_nLC6_KBE_QC_MNT_Hela_01</td>
      <td>HEQILVLDPPTDLK</td>
      <td>27.055</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190305_QE2_NLC1_AB_QC_MNT_HELA_02</td>
      <td>PLRLPLQDVYK</td>
      <td>28.599</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (997, 51)




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
     'n_samples': 997,
     'y_range': (20, 37)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1994       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 997        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,144
    Total trainable params: 3,144
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x00000292EBB96040>
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
      <td>1.637323</td>
      <td>1.546816</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.853444</td>
      <td>0.818353</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.693131</td>
      <td>0.703823</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.722875</td>
      <td>0.674954</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.659389</td>
      <td>0.648734</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.619238</td>
      <td>0.642107</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.570030</td>
      <td>0.627260</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.551716</td>
      <td>0.621810</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.557485</td>
      <td>0.619420</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.601056</td>
      <td>0.619685</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_58_1.png)
    


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
      <th>369</th>
      <td>77</td>
      <td>42</td>
      <td>31.397</td>
    </tr>
    <tr>
      <th>4,739</th>
      <td>962</td>
      <td>7</td>
      <td>29.698</td>
    </tr>
    <tr>
      <th>3,696</th>
      <td>753</td>
      <td>11</td>
      <td>32.836</td>
    </tr>
    <tr>
      <th>2,856</th>
      <td>580</td>
      <td>21</td>
      <td>34.506</td>
    </tr>
    <tr>
      <th>332</th>
      <td>69</td>
      <td>35</td>
      <td>30.795</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2,213</th>
      <td>446</td>
      <td>20</td>
      <td>29.999</td>
    </tr>
    <tr>
      <th>3,295</th>
      <td>673</td>
      <td>11</td>
      <td>32.681</td>
    </tr>
    <tr>
      <th>1,968</th>
      <td>395</td>
      <td>33</td>
      <td>30.014</td>
    </tr>
    <tr>
      <th>2,993</th>
      <td>611</td>
      <td>32</td>
      <td>27.391</td>
    </tr>
    <tr>
      <th>1,709</th>
      <td>345</td>
      <td>20</td>
      <td>29.856</td>
    </tr>
  </tbody>
</table>
<p>4918 rows × 3 columns</p>
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
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.282</td>
      <td>29.554</td>
      <td>29.678</td>
      <td>28.917</td>
      <td>28.946</td>
    </tr>
    <tr>
      <th>DPFAHLPK</th>
      <td>29.130</td>
      <td>27.868</td>
      <td>29.035</td>
      <td>28.298</td>
      <td>27.962</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>27.325</td>
      <td>30.109</td>
      <td>30.061</td>
      <td>28.220</td>
      <td>29.329</td>
    </tr>
    <tr>
      <th>NYIQGINLVQAK</th>
      <td>27.138</td>
      <td>28.461</td>
      <td>28.505</td>
      <td>27.550</td>
      <td>27.632</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>28.366</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>28.875</td>
      <td>28.875</td>
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
      <th>DLSHIGDAVVISCAK</th>
      <td>28.656</td>
      <td>28.166</td>
      <td>28.427</td>
      <td>28.326</td>
      <td>27.687</td>
    </tr>
    <tr>
      <th>EEEIAALVIDNGSGMCK</th>
      <td>32.530</td>
      <td>32.191</td>
      <td>31.751</td>
      <td>32.697</td>
      <td>32.452</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
      <td>27.794</td>
      <td>27.688</td>
      <td>28.165</td>
      <td>27.793</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
      <td>29.571</td>
      <td>29.450</td>
      <td>29.802</td>
      <td>29.568</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>29.789</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>30.121</td>
      <td>29.879</td>
    </tr>
  </tbody>
</table>
<p>4918 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.135
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.147
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.224
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.112
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.230
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_66_1.png)
    



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
      <td>-0.195</td>
      <td>-0.033</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.178</td>
      <td>-0.054</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.151</td>
      <td>-0.062</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.149</td>
      <td>-0.159</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.248</td>
      <td>-0.059</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_68_1.png)
    



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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_69_1.png)
    


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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR</th>
      <th>TATESFASDPILYRPVAVALDTK</th>
      <th>TCTTVAFTQVNSEDK</th>
      <th>TIAPALVSK</th>
      <th>TKPYIQVDIGGGQTK</th>
      <th>TPAQYDASELK</th>
      <th>VFITDDFHDMMPK</th>
      <th>VLAQNSGFDLQETLVK</th>
      <th>VNIIPLIAK</th>
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
      <td>29.282</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.130</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.325</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.309</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
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
      <td>30.608</td>
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
      <td>28.593</td>
      <td>28.911</td>
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
      <td>NaN</td>
      <td>28.357</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.192</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.924</td>
      <td>NaN</td>
      <td>31.301</td>
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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR_na</th>
      <th>TATESFASDPILYRPVAVALDTK_na</th>
      <th>TCTTVAFTQVNSEDK_na</th>
      <th>TIAPALVSK_na</th>
      <th>TKPYIQVDIGGGQTK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAQNSGFDLQETLVK_na</th>
      <th>VNIIPLIAK_na</th>
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
      <td>-0.078</td>
      <td>-0.548</td>
      <td>-1.060</td>
      <td>-0.128</td>
      <td>-0.415</td>
      <td>-1.111</td>
      <td>-0.383</td>
      <td>-0.187</td>
      <td>0.037</td>
      <td>-0.351</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.555</td>
      <td>-0.448</td>
      <td>-1.044</td>
      <td>-0.085</td>
      <td>0.113</td>
      <td>-0.840</td>
      <td>-0.804</td>
      <td>-0.127</td>
      <td>-2.667</td>
      <td>-0.539</td>
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
      <td>-0.494</td>
      <td>0.005</td>
      <td>-0.852</td>
      <td>-0.470</td>
      <td>-0.600</td>
      <td>0.171</td>
      <td>0.260</td>
      <td>-0.814</td>
      <td>-0.510</td>
      <td>-0.845</td>
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
      <td>-0.207</td>
      <td>-0.623</td>
      <td>-0.843</td>
      <td>-0.122</td>
      <td>-0.415</td>
      <td>-0.754</td>
      <td>-0.733</td>
      <td>0.217</td>
      <td>-0.883</td>
      <td>-0.609</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.131</td>
      <td>0.776</td>
      <td>-1.345</td>
      <td>-0.068</td>
      <td>-0.415</td>
      <td>-1.016</td>
      <td>0.295</td>
      <td>0.014</td>
      <td>0.037</td>
      <td>0.320</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
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
      <td>0.945</td>
      <td>1.200</td>
      <td>0.745</td>
      <td>0.933</td>
      <td>1.242</td>
      <td>0.498</td>
      <td>0.588</td>
      <td>0.725</td>
      <td>0.864</td>
      <td>0.836</td>
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
      <td>-0.840</td>
      <td>-0.222</td>
      <td>0.134</td>
      <td>-0.059</td>
      <td>-0.371</td>
      <td>0.481</td>
      <td>0.489</td>
      <td>-2.072</td>
      <td>-0.037</td>
      <td>0.103</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.109</td>
      <td>0.020</td>
      <td>0.603</td>
      <td>0.004</td>
      <td>-0.342</td>
      <td>0.235</td>
      <td>0.610</td>
      <td>-1.774</td>
      <td>0.203</td>
      <td>-1.803</td>
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
      <td>-0.153</td>
      <td>0.194</td>
      <td>-1.751</td>
      <td>-0.080</td>
      <td>-0.439</td>
      <td>0.109</td>
      <td>0.266</td>
      <td>-1.420</td>
      <td>-0.125</td>
      <td>0.028</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-0.127</td>
      <td>0.007</td>
      <td>0.353</td>
      <td>-0.122</td>
      <td>-0.362</td>
      <td>0.302</td>
      <td>0.266</td>
      <td>-0.006</td>
      <td>0.062</td>
      <td>-0.003</td>
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
<p>996 rows × 100 columns</p>
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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR_na</th>
      <th>TATESFASDPILYRPVAVALDTK_na</th>
      <th>TCTTVAFTQVNSEDK_na</th>
      <th>TIAPALVSK_na</th>
      <th>TKPYIQVDIGGGQTK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAQNSGFDLQETLVK_na</th>
      <th>VNIIPLIAK_na</th>
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
      <td>-0.082</td>
      <td>-0.483</td>
      <td>-1.001</td>
      <td>-0.135</td>
      <td>-0.445</td>
      <td>-1.019</td>
      <td>-0.331</td>
      <td>-0.181</td>
      <td>0.039</td>
      <td>-0.313</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.534</td>
      <td>-0.391</td>
      <td>-0.985</td>
      <td>-0.094</td>
      <td>0.059</td>
      <td>-0.768</td>
      <td>-0.730</td>
      <td>-0.124</td>
      <td>-2.525</td>
      <td>-0.488</td>
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
      <td>-0.477</td>
      <td>0.029</td>
      <td>-0.804</td>
      <td>-0.458</td>
      <td>-0.621</td>
      <td>0.167</td>
      <td>0.278</td>
      <td>-0.775</td>
      <td>-0.480</td>
      <td>-0.774</td>
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
      <td>-0.205</td>
      <td>-0.553</td>
      <td>-0.795</td>
      <td>-0.129</td>
      <td>-0.445</td>
      <td>-0.689</td>
      <td>-0.663</td>
      <td>0.201</td>
      <td>-0.833</td>
      <td>-0.554</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.133</td>
      <td>0.743</td>
      <td>-1.271</td>
      <td>-0.078</td>
      <td>-0.445</td>
      <td>-0.932</td>
      <td>0.311</td>
      <td>0.009</td>
      <td>0.039</td>
      <td>0.315</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
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
      <td>0.887</td>
      <td>1.135</td>
      <td>0.711</td>
      <td>0.869</td>
      <td>1.136</td>
      <td>0.470</td>
      <td>0.589</td>
      <td>0.682</td>
      <td>0.823</td>
      <td>0.798</td>
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
      <td>-0.805</td>
      <td>-0.182</td>
      <td>0.132</td>
      <td>-0.070</td>
      <td>-0.403</td>
      <td>0.454</td>
      <td>0.494</td>
      <td>-1.966</td>
      <td>-0.031</td>
      <td>0.113</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.095</td>
      <td>0.043</td>
      <td>0.576</td>
      <td>-0.010</td>
      <td>-0.374</td>
      <td>0.227</td>
      <td>0.609</td>
      <td>-1.684</td>
      <td>0.196</td>
      <td>-1.671</td>
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
      <td>-0.153</td>
      <td>0.203</td>
      <td>-1.655</td>
      <td>-0.090</td>
      <td>-0.468</td>
      <td>0.110</td>
      <td>0.283</td>
      <td>-1.348</td>
      <td>-0.115</td>
      <td>0.042</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-0.129</td>
      <td>0.030</td>
      <td>0.339</td>
      <td>-0.129</td>
      <td>-0.394</td>
      <td>0.288</td>
      <td>0.283</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.014</td>
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
<p>996 rows × 100 columns</p>
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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
      <td>996.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.008</td>
      <td>0.024</td>
      <td>0.004</td>
      <td>-0.014</td>
      <td>-0.049</td>
      <td>0.009</td>
      <td>0.032</td>
      <td>-0.004</td>
      <td>0.004</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.948</td>
      <td>0.926</td>
      <td>0.949</td>
      <td>0.947</td>
      <td>0.954</td>
      <td>0.926</td>
      <td>0.947</td>
      <td>0.947</td>
      <td>0.948</td>
      <td>0.936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.866</td>
      <td>-4.347</td>
      <td>-4.697</td>
      <td>-2.971</td>
      <td>-2.650</td>
      <td>-7.635</td>
      <td>-6.309</td>
      <td>-4.067</td>
      <td>-4.317</td>
      <td>-5.259</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.542</td>
      <td>-0.385</td>
      <td>-0.493</td>
      <td>-0.539</td>
      <td>-0.705</td>
      <td>-0.384</td>
      <td>-0.384</td>
      <td>-0.417</td>
      <td>-0.446</td>
      <td>-0.317</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.082</td>
      <td>0.165</td>
      <td>0.043</td>
      <td>-0.129</td>
      <td>-0.445</td>
      <td>0.063</td>
      <td>0.283</td>
      <td>-0.040</td>
      <td>0.039</td>
      <td>0.126</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.534</td>
      <td>0.615</td>
      <td>0.556</td>
      <td>0.668</td>
      <td>1.039</td>
      <td>0.480</td>
      <td>0.632</td>
      <td>0.519</td>
      <td>0.556</td>
      <td>0.514</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.311</td>
      <td>2.086</td>
      <td>2.070</td>
      <td>2.051</td>
      <td>1.813</td>
      <td>2.118</td>
      <td>1.634</td>
      <td>1.975</td>
      <td>2.445</td>
      <td>1.929</td>
    </tr>
  </tbody>
</table>
</div>



Mask is added as type bool


```python
to.items.dtypes.value_counts()
```




    bool      50
    float64   50
    dtype: int64



with the suffix `_na` where `True` is indicating a missing value replaced by the `FillMissing` transformation


```python
to.cont_names, to.cat_names
```




    ((#50) ['ADRDESSPYAAMLAAQDVAQR','AIAELGIYPAVDPLDSTSR','AYFHLLNQIAPK','DLSHIGDAVVISCAK','DPFAHLPK','EAAENSLVAYK','EEEIAALVIDNGSGMCK','EIGNIISDAMK','FDTGNLCMVTGGANLGR','FWEVISDEHGIDPTGTYHGDSDLQLDR'...],
     (#50) ['ADRDESSPYAAMLAAQDVAQR_na','AIAELGIYPAVDPLDSTSR_na','AYFHLLNQIAPK_na','DLSHIGDAVVISCAK_na','DPFAHLPK_na','EAAENSLVAYK_na','EEEIAALVIDNGSGMCK_na','EIGNIISDAMK_na','FDTGNLCMVTGGANLGR_na','FWEVISDEHGIDPTGTYHGDSDLQLDR_na'...])




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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.114</td>
      <td>-0.010</td>
      <td>-0.060</td>
      <td>0.032</td>
      <td>-0.126</td>
      <td>-0.195</td>
      <td>-0.078</td>
      <td>0.221</td>
      <td>0.049</td>
      <td>-0.002</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.007</td>
      <td>1.027</td>
      <td>0.978</td>
      <td>0.962</td>
      <td>0.969</td>
      <td>1.194</td>
      <td>1.181</td>
      <td>1.108</td>
      <td>1.116</td>
      <td>1.086</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.576</td>
      <td>-3.272</td>
      <td>-4.692</td>
      <td>-2.545</td>
      <td>-2.169</td>
      <td>-4.759</td>
      <td>-5.892</td>
      <td>-4.390</td>
      <td>-3.480</td>
      <td>-3.987</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.404</td>
      <td>-0.432</td>
      <td>-0.587</td>
      <td>-0.580</td>
      <td>-0.724</td>
      <td>-0.512</td>
      <td>-0.568</td>
      <td>-0.338</td>
      <td>-0.558</td>
      <td>-0.358</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.010</td>
      <td>0.178</td>
      <td>-0.017</td>
      <td>-0.070</td>
      <td>-0.514</td>
      <td>-0.006</td>
      <td>0.264</td>
      <td>0.102</td>
      <td>0.105</td>
      <td>0.143</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.968</td>
      <td>0.685</td>
      <td>0.424</td>
      <td>0.775</td>
      <td>0.974</td>
      <td>0.520</td>
      <td>0.704</td>
      <td>1.127</td>
      <td>0.807</td>
      <td>0.565</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.910</td>
      <td>1.728</td>
      <td>1.652</td>
      <td>1.740</td>
      <td>1.612</td>
      <td>1.548</td>
      <td>1.300</td>
      <td>2.050</td>
      <td>1.980</td>
      <td>1.973</td>
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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR_val</th>
      <th>TATESFASDPILYRPVAVALDTK_val</th>
      <th>TCTTVAFTQVNSEDK_val</th>
      <th>TIAPALVSK_val</th>
      <th>TKPYIQVDIGGGQTK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAQNSGFDLQETLVK_val</th>
      <th>VNIIPLIAK_val</th>
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
      <td>-0.082</td>
      <td>-0.483</td>
      <td>-1.001</td>
      <td>-0.135</td>
      <td>-0.445</td>
      <td>-1.019</td>
      <td>-0.331</td>
      <td>-0.181</td>
      <td>0.039</td>
      <td>-0.313</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.249</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.263</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.534</td>
      <td>-0.391</td>
      <td>-0.985</td>
      <td>-0.094</td>
      <td>0.059</td>
      <td>-0.768</td>
      <td>-0.730</td>
      <td>-0.124</td>
      <td>-2.525</td>
      <td>-0.488</td>
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
      <td>-0.747</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.477</td>
      <td>0.029</td>
      <td>-0.804</td>
      <td>-0.458</td>
      <td>-0.621</td>
      <td>0.167</td>
      <td>0.278</td>
      <td>-0.775</td>
      <td>-0.480</td>
      <td>-0.774</td>
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
      <td>-0.205</td>
      <td>-0.553</td>
      <td>-0.795</td>
      <td>-0.129</td>
      <td>-0.445</td>
      <td>-0.689</td>
      <td>-0.663</td>
      <td>0.201</td>
      <td>-0.833</td>
      <td>-0.554</td>
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
      <td>-0.133</td>
      <td>0.743</td>
      <td>-1.271</td>
      <td>-0.078</td>
      <td>-0.445</td>
      <td>-0.932</td>
      <td>0.311</td>
      <td>0.009</td>
      <td>0.039</td>
      <td>0.315</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.734</td>
      <td>NaN</td>
      <td>-0.455</td>
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
      <td>0.887</td>
      <td>1.135</td>
      <td>0.711</td>
      <td>0.869</td>
      <td>1.136</td>
      <td>0.470</td>
      <td>0.589</td>
      <td>0.682</td>
      <td>0.823</td>
      <td>0.798</td>
      <td>...</td>
      <td>0.232</td>
      <td>1.211</td>
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
      <td>-0.805</td>
      <td>-0.182</td>
      <td>0.132</td>
      <td>-0.070</td>
      <td>-0.403</td>
      <td>0.454</td>
      <td>0.494</td>
      <td>-1.966</td>
      <td>-0.031</td>
      <td>0.113</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.095</td>
      <td>0.043</td>
      <td>0.576</td>
      <td>-0.010</td>
      <td>-0.374</td>
      <td>0.227</td>
      <td>0.609</td>
      <td>-1.684</td>
      <td>0.196</td>
      <td>-1.671</td>
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
      <td>-0.153</td>
      <td>0.203</td>
      <td>-1.655</td>
      <td>-0.090</td>
      <td>-0.468</td>
      <td>0.110</td>
      <td>0.283</td>
      <td>-1.348</td>
      <td>-0.115</td>
      <td>0.042</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-0.129</td>
      <td>0.030</td>
      <td>0.339</td>
      <td>-0.129</td>
      <td>-0.394</td>
      <td>0.288</td>
      <td>0.283</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.153</td>
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
<p>996 rows × 150 columns</p>
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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR_val</th>
      <th>TATESFASDPILYRPVAVALDTK_val</th>
      <th>TCTTVAFTQVNSEDK_val</th>
      <th>TIAPALVSK_val</th>
      <th>TKPYIQVDIGGGQTK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAQNSGFDLQETLVK_val</th>
      <th>VNIIPLIAK_val</th>
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
      <td>-0.082</td>
      <td>-0.483</td>
      <td>-1.001</td>
      <td>-0.135</td>
      <td>-0.445</td>
      <td>-1.019</td>
      <td>-0.331</td>
      <td>-0.181</td>
      <td>0.039</td>
      <td>-0.313</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.249</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.263</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.534</td>
      <td>-0.391</td>
      <td>-0.985</td>
      <td>-0.094</td>
      <td>0.059</td>
      <td>-0.768</td>
      <td>-0.730</td>
      <td>-0.124</td>
      <td>-2.525</td>
      <td>-0.488</td>
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
      <td>-0.747</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.477</td>
      <td>0.029</td>
      <td>-0.804</td>
      <td>-0.458</td>
      <td>-0.621</td>
      <td>0.167</td>
      <td>0.278</td>
      <td>-0.775</td>
      <td>-0.480</td>
      <td>-0.774</td>
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
      <td>-0.205</td>
      <td>-0.553</td>
      <td>-0.795</td>
      <td>-0.129</td>
      <td>-0.445</td>
      <td>-0.689</td>
      <td>-0.663</td>
      <td>0.201</td>
      <td>-0.833</td>
      <td>-0.554</td>
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
      <td>-0.133</td>
      <td>0.743</td>
      <td>-1.271</td>
      <td>-0.078</td>
      <td>-0.445</td>
      <td>-0.932</td>
      <td>0.311</td>
      <td>0.009</td>
      <td>0.039</td>
      <td>0.315</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.734</td>
      <td>NaN</td>
      <td>-0.455</td>
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
      <td>0.887</td>
      <td>1.135</td>
      <td>0.711</td>
      <td>0.869</td>
      <td>1.136</td>
      <td>0.470</td>
      <td>0.589</td>
      <td>0.682</td>
      <td>0.823</td>
      <td>0.798</td>
      <td>...</td>
      <td>0.232</td>
      <td>1.211</td>
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
      <td>-0.805</td>
      <td>-0.182</td>
      <td>0.132</td>
      <td>-0.070</td>
      <td>-0.403</td>
      <td>0.454</td>
      <td>0.494</td>
      <td>-1.966</td>
      <td>-0.031</td>
      <td>0.113</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_02</th>
      <td>0.095</td>
      <td>0.043</td>
      <td>0.576</td>
      <td>-0.010</td>
      <td>-0.374</td>
      <td>0.227</td>
      <td>0.609</td>
      <td>-1.684</td>
      <td>0.196</td>
      <td>-1.671</td>
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
      <td>-0.153</td>
      <td>0.203</td>
      <td>-1.655</td>
      <td>-0.090</td>
      <td>-0.468</td>
      <td>0.110</td>
      <td>0.283</td>
      <td>-1.348</td>
      <td>-0.115</td>
      <td>0.042</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-0.129</td>
      <td>0.030</td>
      <td>0.339</td>
      <td>-0.129</td>
      <td>-0.394</td>
      <td>0.288</td>
      <td>0.283</td>
      <td>-0.009</td>
      <td>0.063</td>
      <td>0.014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.153</td>
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
<p>996 rows × 150 columns</p>
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
      <th>ADRDESSPYAAMLAAQDVAQR_val</th>
      <th>AIAELGIYPAVDPLDSTSR_val</th>
      <th>AYFHLLNQIAPK_val</th>
      <th>DLSHIGDAVVISCAK_val</th>
      <th>DPFAHLPK_val</th>
      <th>EAAENSLVAYK_val</th>
      <th>EEEIAALVIDNGSGMCK_val</th>
      <th>EIGNIISDAMK_val</th>
      <th>FDTGNLCMVTGGANLGR_val</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR_val</th>
      <th>...</th>
      <th>TANDMIHAENMR_val</th>
      <th>TATESFASDPILYRPVAVALDTK_val</th>
      <th>TCTTVAFTQVNSEDK_val</th>
      <th>TIAPALVSK_val</th>
      <th>TKPYIQVDIGGGQTK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAQNSGFDLQETLVK_val</th>
      <th>VNIIPLIAK_val</th>
      <th>VVLAYEPVWAIGTGK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>94.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.114</td>
      <td>-0.010</td>
      <td>-0.060</td>
      <td>0.032</td>
      <td>-0.126</td>
      <td>-0.195</td>
      <td>-0.078</td>
      <td>0.221</td>
      <td>0.049</td>
      <td>-0.002</td>
      <td>...</td>
      <td>0.004</td>
      <td>0.088</td>
      <td>0.049</td>
      <td>0.030</td>
      <td>0.063</td>
      <td>-0.337</td>
      <td>-0.071</td>
      <td>-0.064</td>
      <td>0.024</td>
      <td>-0.004</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.007</td>
      <td>1.027</td>
      <td>0.978</td>
      <td>0.962</td>
      <td>0.969</td>
      <td>1.194</td>
      <td>1.181</td>
      <td>1.108</td>
      <td>1.116</td>
      <td>1.086</td>
      <td>...</td>
      <td>0.968</td>
      <td>0.912</td>
      <td>0.902</td>
      <td>0.927</td>
      <td>1.030</td>
      <td>1.474</td>
      <td>1.041</td>
      <td>0.948</td>
      <td>0.842</td>
      <td>0.916</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.576</td>
      <td>-3.272</td>
      <td>-4.692</td>
      <td>-2.545</td>
      <td>-2.169</td>
      <td>-4.759</td>
      <td>-5.892</td>
      <td>-4.390</td>
      <td>-3.480</td>
      <td>-3.987</td>
      <td>...</td>
      <td>-4.403</td>
      <td>-2.728</td>
      <td>-2.286</td>
      <td>-4.008</td>
      <td>-3.636</td>
      <td>-7.416</td>
      <td>-2.643</td>
      <td>-2.687</td>
      <td>-3.610</td>
      <td>-2.933</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.404</td>
      <td>-0.432</td>
      <td>-0.587</td>
      <td>-0.580</td>
      <td>-0.724</td>
      <td>-0.512</td>
      <td>-0.568</td>
      <td>-0.338</td>
      <td>-0.558</td>
      <td>-0.358</td>
      <td>...</td>
      <td>-0.386</td>
      <td>-0.331</td>
      <td>-0.570</td>
      <td>-0.268</td>
      <td>-0.343</td>
      <td>-0.854</td>
      <td>-0.596</td>
      <td>-0.657</td>
      <td>-0.549</td>
      <td>-0.408</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.010</td>
      <td>0.178</td>
      <td>-0.017</td>
      <td>-0.070</td>
      <td>-0.514</td>
      <td>-0.006</td>
      <td>0.264</td>
      <td>0.102</td>
      <td>0.105</td>
      <td>0.143</td>
      <td>...</td>
      <td>0.232</td>
      <td>0.147</td>
      <td>0.121</td>
      <td>0.064</td>
      <td>0.099</td>
      <td>0.119</td>
      <td>-0.234</td>
      <td>-0.163</td>
      <td>0.226</td>
      <td>0.191</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.968</td>
      <td>0.685</td>
      <td>0.424</td>
      <td>0.775</td>
      <td>0.974</td>
      <td>0.520</td>
      <td>0.704</td>
      <td>1.127</td>
      <td>0.807</td>
      <td>0.565</td>
      <td>...</td>
      <td>0.586</td>
      <td>0.680</td>
      <td>0.726</td>
      <td>0.632</td>
      <td>0.756</td>
      <td>0.619</td>
      <td>0.554</td>
      <td>0.473</td>
      <td>0.624</td>
      <td>0.622</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.910</td>
      <td>1.728</td>
      <td>1.652</td>
      <td>1.740</td>
      <td>1.612</td>
      <td>1.548</td>
      <td>1.300</td>
      <td>2.050</td>
      <td>1.980</td>
      <td>1.973</td>
      <td>...</td>
      <td>1.633</td>
      <td>1.867</td>
      <td>1.815</td>
      <td>1.640</td>
      <td>1.713</td>
      <td>1.544</td>
      <td>2.105</td>
      <td>2.147</td>
      <td>1.440</td>
      <td>1.503</td>
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
      <th>ADRDESSPYAAMLAAQDVAQR_na</th>
      <th>AIAELGIYPAVDPLDSTSR_na</th>
      <th>AYFHLLNQIAPK_na</th>
      <th>DLSHIGDAVVISCAK_na</th>
      <th>DPFAHLPK_na</th>
      <th>EAAENSLVAYK_na</th>
      <th>EEEIAALVIDNGSGMCK_na</th>
      <th>EIGNIISDAMK_na</th>
      <th>FDTGNLCMVTGGANLGR_na</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR_na</th>
      <th>...</th>
      <th>TANDMIHAENMR_na</th>
      <th>TATESFASDPILYRPVAVALDTK_na</th>
      <th>TCTTVAFTQVNSEDK_na</th>
      <th>TIAPALVSK_na</th>
      <th>TKPYIQVDIGGGQTK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAQNSGFDLQETLVK_na</th>
      <th>VNIIPLIAK_na</th>
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
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
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
      <td>False</td>
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
      <td>True</td>
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
      <td>False</td>
      <td>True</td>
      <td>False</td>
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
<p>996 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x00000292EBB96040>
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








    SuggestedLRs(valley=0.013182567432522774)




    
![png](latent_2D_400_30_files/latent_2D_400_30_108_2.png)
    


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
      <td>0.963075</td>
      <td>0.778328</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.679780</td>
      <td>0.429657</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.513754</td>
      <td>0.391700</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.435630</td>
      <td>0.367198</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.393968</td>
      <td>0.353080</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.369720</td>
      <td>0.353809</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.356257</td>
      <td>0.351991</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.347538</td>
      <td>0.339241</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.338719</td>
      <td>0.340313</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.331593</td>
      <td>0.336444</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4918, 4918)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3387)




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
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.282</td>
      <td>29.554</td>
      <td>29.678</td>
      <td>28.917</td>
      <td>28.946</td>
      <td>28.498</td>
    </tr>
    <tr>
      <th>DPFAHLPK</th>
      <td>29.130</td>
      <td>27.868</td>
      <td>29.035</td>
      <td>28.298</td>
      <td>27.962</td>
      <td>27.654</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>27.325</td>
      <td>30.109</td>
      <td>30.061</td>
      <td>28.220</td>
      <td>29.329</td>
      <td>29.005</td>
    </tr>
    <tr>
      <th>NYIQGINLVQAK</th>
      <td>27.138</td>
      <td>28.461</td>
      <td>28.505</td>
      <td>27.550</td>
      <td>27.632</td>
      <td>27.512</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>28.366</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>28.875</td>
      <td>28.875</td>
      <td>28.781</td>
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
      <th>DLSHIGDAVVISCAK</th>
      <td>28.656</td>
      <td>28.166</td>
      <td>28.427</td>
      <td>28.326</td>
      <td>27.687</td>
      <td>27.556</td>
    </tr>
    <tr>
      <th>EEEIAALVIDNGSGMCK</th>
      <td>32.530</td>
      <td>32.191</td>
      <td>31.751</td>
      <td>32.697</td>
      <td>32.452</td>
      <td>32.576</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
      <td>27.794</td>
      <td>27.688</td>
      <td>28.165</td>
      <td>27.793</td>
      <td>27.834</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
      <td>29.571</td>
      <td>29.450</td>
      <td>29.802</td>
      <td>29.568</td>
      <td>29.566</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>29.789</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>30.121</td>
      <td>29.879</td>
      <td>29.877</td>
    </tr>
  </tbody>
</table>
<p>4918 rows × 6 columns</p>
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
      <td>-0.470</td>
      <td>-0.219</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.669</td>
      <td>-0.349</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.823</td>
      <td>-0.654</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.500</td>
      <td>-0.247</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.269</td>
      <td>-0.416</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_122_1.png)
    



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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_123_1.png)
    


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
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>AYFHLLNQIAPK</th>
      <th>DLSHIGDAVVISCAK</th>
      <th>DPFAHLPK</th>
      <th>EAAENSLVAYK</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <th>EIGNIISDAMK</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <th>...</th>
      <th>TANDMIHAENMR</th>
      <th>TATESFASDPILYRPVAVALDTK</th>
      <th>TCTTVAFTQVNSEDK</th>
      <th>TIAPALVSK</th>
      <th>TKPYIQVDIGGGQTK</th>
      <th>TPAQYDASELK</th>
      <th>VFITDDFHDMMPK</th>
      <th>VLAQNSGFDLQETLVK</th>
      <th>VNIIPLIAK</th>
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
      <td>0.583</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.602</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.308</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.478</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.394</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
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
      <td>0.618</td>
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
      <td>0.608</td>
      <td>0.583</td>
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
      <td>NaN</td>
      <td>0.536</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.534</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.768</td>
      <td>NaN</td>
      <td>0.742</td>
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
    
    Optimizer used: <function Adam at 0x00000292EBB96040>
    Loss function: <function loss_fct_vae at 0x00000292EBBB3940>
    
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








    SuggestedLRs(valley=0.004365158267319202)




    
![png](latent_2D_400_30_files/latent_2D_400_30_136_2.png)
    



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
      <td>2001.712891</td>
      <td>221.076401</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1944.176758</td>
      <td>214.032684</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1866.610107</td>
      <td>201.981491</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1812.121338</td>
      <td>192.563644</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1778.323242</td>
      <td>191.358627</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1755.488892</td>
      <td>189.850403</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1740.204834</td>
      <td>190.314911</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1731.015503</td>
      <td>191.307755</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1722.716187</td>
      <td>190.542557</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1716.395020</td>
      <td>190.509201</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4918)




```python
len(pred[0])
```




    4918




```python
learn.loss_func(pred, target)
```




    tensor(3002.7756)




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
      <th rowspan="5" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ADRDESSPYAAMLAAQDVAQR</th>
      <td>29.282</td>
      <td>29.554</td>
      <td>29.678</td>
      <td>28.917</td>
      <td>28.946</td>
      <td>28.498</td>
      <td>29.643</td>
    </tr>
    <tr>
      <th>DPFAHLPK</th>
      <td>29.130</td>
      <td>27.868</td>
      <td>29.035</td>
      <td>28.298</td>
      <td>27.962</td>
      <td>27.654</td>
      <td>28.963</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>27.325</td>
      <td>30.109</td>
      <td>30.061</td>
      <td>28.220</td>
      <td>29.329</td>
      <td>29.005</td>
      <td>30.044</td>
    </tr>
    <tr>
      <th>NYIQGINLVQAK</th>
      <td>27.138</td>
      <td>28.461</td>
      <td>28.505</td>
      <td>27.550</td>
      <td>27.632</td>
      <td>27.512</td>
      <td>28.555</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>28.366</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>28.875</td>
      <td>28.875</td>
      <td>28.781</td>
      <td>29.636</td>
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
      <th>DLSHIGDAVVISCAK</th>
      <td>28.656</td>
      <td>28.166</td>
      <td>28.427</td>
      <td>28.326</td>
      <td>27.687</td>
      <td>27.556</td>
      <td>28.550</td>
    </tr>
    <tr>
      <th>EEEIAALVIDNGSGMCK</th>
      <td>32.530</td>
      <td>32.191</td>
      <td>31.751</td>
      <td>32.697</td>
      <td>32.452</td>
      <td>32.576</td>
      <td>31.890</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
      <td>27.794</td>
      <td>27.688</td>
      <td>28.165</td>
      <td>27.793</td>
      <td>27.834</td>
      <td>27.896</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
      <td>29.571</td>
      <td>29.450</td>
      <td>29.802</td>
      <td>29.568</td>
      <td>29.566</td>
      <td>29.656</td>
    </tr>
    <tr>
      <th>TCTTVAFTQVNSEDK</th>
      <td>29.789</td>
      <td>29.704</td>
      <td>29.633</td>
      <td>30.121</td>
      <td>29.879</td>
      <td>29.877</td>
      <td>29.700</td>
    </tr>
  </tbody>
</table>
<p>4918 rows × 7 columns</p>
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
      <td>-0.219</td>
      <td>-0.274</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.069</td>
      <td>-0.085</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.283</td>
      <td>-0.243</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.099</td>
      <td>-0.075</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.074</td>
      <td>-0.010</td>
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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_146_1.png)
    



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
    


    
![png](latent_2D_400_30_files/latent_2D_400_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'DPFAHLPK'), ('20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190204_QE4_LC12_SCL_QC_MNT_HeLa_02', 'GHYTEGAELVDSVLDVVR'), ('20190204_QE8_nLC14_RG_QC_HeLa_15cm_02', 'MTNGFSGADLTEICQR'), ('20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_0', 'MTNGFSGADLTEICQR'), ('20190207_QE7_nLC7_TSB_QC_MNT_HeLa_01', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm1_20190207172050', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190207_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01', 'VLAQNSGFDLQETLVK'), ('20190211_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_01', 'LASVPAGGAVAVSAAPGSAAPAAGSAPAAAEEK'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_01', 'TIAPALVSK'), ('20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_04', 'IIPGFMCQGGDFTR'), ('20190219_QE3_Evo2_UHG_QC_MNT_HELA_01_190219173213', 'PLRLPLQDVYK'), ('20190224_QE1_nLC2_GP_QC_MNT_HELA_01', 'DPFAHLPK'), ('20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_1_20190227115340', 'GHYTEGAELVDSVLDVVR'), ('20190308_QE9_nLC0_FaCo_QC_MNT_Hela_50cm', 'EIGNIISDAMK'), ('20190325_QE3_nLC5_DS_QC_MNT_HeLa_02', 'FWEVISDEHGIDPTGTYHGDSDLQLDR'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_02_20190326215828', 'TCTTVAFTQVNSEDK'), ('20190331_QE10_nLC13_LiNi_QC_45cm_HeLa_01', 'HRPELIEYDK'), ('20190401_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'TIAPALVSK'), ('20190402_QE1_nLC2_GP_MNT_QC_hela_01', 'TIAPALVSK'), ('20190402_QE6_LC6_AS_QC_MNT_HeLa_01', 'TIAPALVSK'), ('20190405_QE1_nLC2_GP_MNT_QC_hela_02', 'SYSPYDMLESIRK'), ('20190415_QE10_nLC9_LiNi_QC_MNT_45cm_HeLa_01', 'GQHVPGSPFQFTVGPLGEGGAHK'), ('20190417_QE4_LC12_JE-IAH_QC_MNT_HeLa_02', 'ISSLLEEQFQQGK'), ('20190423_QE3_nLC5_DS_QC_MNT_HeLa_02', 'MLISILTER'), ('20190423_QX6_MaTa_MA_HeLa_Br14_500ng_DIA_LC09', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190425_QX4_JoSw_MA_HeLa_500ng_BR13_standard_190426221220', 'VFITDDFHDMMPK'), ('20190429_QX2_ChDe_MA_HeLa_500ng_LC05_CTCDoff_newcol', 'ISSLLEEQFQQGK'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15', 'ISSLLEEQFQQGK'), ('20190429_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'SYSPYDMLESIRK'), ('20190502_QX7_ChDe_MA_HeLa_500ng', 'FDTGNLCMVTGGANLGR'), ('20190506_QE3_nLC3_DBJ_QC_MNT_HeLa_02', 'GLVLGPIHK'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'SYSPYDMLESIRK'), ('20190509_QE8_nLC14_AGF_QC_MNT_HeLa_01', 'SEHPGLSIGDTAK'), ('20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01', 'FDTGNLCMVTGGANLGR'), ('20190514_QX4_JiYu_MA_HeLa_500ng_BR14', 'IKGEHPGLSIGDVAK'), ('20190515_QX8_MiWi_MA_HeLa_BR14_500ng_190516123056', 'NFGSYVTHETK'), ('20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02', 'NYIQGINLVQAK'), ('20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02', 'STGEAFVQFASQEIAEK'), ('20190520_QX3_LiSc_MA_Hela_500ng_LC15', 'LITPAVVSER'), ('20190521_QE1_nLC2_GP_QC_MNT_HELA_01', 'TATESFASDPILYRPVAVALDTK'), ('20190523_QX3_LiSc_MA_Hela_500ng_LC15', 'MLISILTER'), ('20190530_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'ISSLLEEQFQQGK'), ('20190603_QE9_nLC0_FaCo_QC_MNT_Hela_50cm_20190603142002', 'VFITDDFHDMMPK'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_5', 'GHYTEGAELVDSVLDVVR'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02', 'VVLAYEPVWAIGTGK'), ('20190615_QX4_JiYu_MA_HeLa_500ng', 'FWEVISDEHGIDPTGTYHGDSDLQLDR'), ('20190618_QX3_LiSc_MA_Hela_500ng_LC15_190619053902', 'EAAENSLVAYK'), ('20190621_QE2_NLC1_GP_QC_MNT_HELA_01', 'ADRDESSPYAAMLAAQDVAQR'), ('20190621_QX2_SeVW_MA_HeLa_500ng_LC05', 'MTNGFSGADLTEICQR'), ('20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01', 'DPFAHLPK'), ('20190624_QE6_LC4_AS_QC_MNT_HeLa_01', 'EAAENSLVAYK'), ('20190625_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'NFGSYVTHETK'), ('20190626_QX1_JoMu_MA_HeLa_500ng_LC10_190626135146', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190626_QX6_ChDe_MA_HeLa_500ng_LC09', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190627_QX3_MaMu_MA_Hela_500ng_LC15', 'FWEVISDEHGIDPTGTYHGDSDLQLDR'), ('20190628_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'AIAELGIYPAVDPLDSTSR'), ('20190628_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'IVSQLLTLMDGLK'), ('20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03', 'DLSHIGDAVVISCAK'), ('20190702_QX0_AnBr_MA_HeLa_500ng_LC07_01_190702180001', 'IKGEHPGLSIGDVAK'), ('20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01_20190713004325', 'STGEAFVQFASQEIAEK'), ('20190715_QE4_LC12_IAH_QC_MNT_HeLa_03', 'NYIQGINLVQAK'), ('20190717_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'GHYTEGAELVDSVLDVVR'), ('20190719_QX8_ChSc_MA_HeLa_500ng', 'AIAELGIYPAVDPLDSTSR'), ('20190719_QX8_ChSc_MA_HeLa_500ng', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190722_QX4_StEb_MA_HeLa_500ng', 'VVLAYEPVWAIGTGK'), ('20190726_QE7_nLC7_MEM_QC_MNT_HeLa_01', 'ILPTLEAVAALGNK'), ('20190726_QE7_nLC7_MEM_QC_MNT_HeLa_01', 'TKPYIQVDIGGGQTK'), ('20190730_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_02', 'IKGEHPGLSIGDVAK'), ('20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_02', 'SEHPGLSIGDTAK'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'HEQILVLDPPTDLK')]
    




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
      <td>0.620</td>
      <td>0.651</td>
      <td>1.675</td>
      <td>1.846</td>
      <td>1.985</td>
      <td>2.045</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.488</td>
      <td>0.512</td>
      <td>0.874</td>
      <td>0.990</td>
      <td>1.037</td>
      <td>1.016</td>
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
