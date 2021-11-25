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
n_feat = 200
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
    IAILTCPFEPPKPK                           1,000
    DDDIAALVVDNGSGMCK                          991
    GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK     993
    HGVVPLATYMR                                999
    AIAELGIYPAVDPLDSTSR                        949
    QLFHPEQLITGK                               996
    LIALLEVLSQK                                989
    ATCIGNNSAAAVSMLK                           977
    DGLAFNALIHR                              1,000
    LQVTNVLSQPLTQATVK                          994
    KYEDICPSTHNMDVPNIK                       1,000
    AGVNTVTTLVENKK                             966
    LVSIGAEEIVDGNAK                            997
    ACANPAAGSVILLENLR                          957
    EEEIAALVIDNGSGMCK                          988
    IGGVQQDTILAEGLHFR                          951
    YSLEPVAVELK                                972
    AVAEQIPLLVQGVR                             981
    FLSQPFQVAEVFTGHMGK                         983
    SYSPYDMLESIRK                              954
    ISSLLEEQFQQGK                              990
    LFIGGLNTETNEK                              993
    LSLEGDHSTPPSAYGSVK                       1,000
    STGEAFVQFASQEIAEK                          992
    HVVFIAQR                                   960
    EMEENFAVEAANYQDTIGR                        979
    DLISHDEMFSDIYK                             989
    EMNDAAMFYTNR                               990
    DHENIVIAK                                  986
    DATNVGDEGGFAPNILENK                        999
    VLALPEPSPAAPTLR                            983
    GQAAVQQLQAEGLSPR                           995
    GYSFTTTAER                                 995
    IGEEQSAEDAEDGPPELLFIHGGHTAK                979
    LATLLGLQAPPTR                              986
    LNSVQSSERPLFLVHPIEGSTTVFHSLASR             952
    INISEGNCPER                                983
    VMLGETNPADSKPGTIR                        1,000
    GVNLPGAAVDLPAVSEK                        1,000
    YGINTTDIFQTVDLWEGK                         973
    TISHVIIGLK                                 993
    LSSLIILMPHHVEPLER                          997
    VFQFLNAK                                   984
    EGPYDVVVLPGGNLGAQNLSESAAVK                 999
    DYAFVHFEDR                                 987
    DPVQEAWAEDVDLR                             986
    MALIGLGVSHPVLK                             999
    FWEVISDEHGIDPTGTYHGDSDLQLDR                971
    LYSVSYLLK                                  961
    LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK         982
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
      <th>IAILTCPFEPPKPK</th>
      <td>27.348</td>
    </tr>
    <tr>
      <th>DDDIAALVVDNGSGMCK</th>
      <td>31.112</td>
    </tr>
    <tr>
      <th>GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK</th>
      <td>31.778</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>28.979</td>
    </tr>
    <tr>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <td>29.355</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>DPVQEAWAEDVDLR</th>
      <td>27.497</td>
    </tr>
    <tr>
      <th>MALIGLGVSHPVLK</th>
      <td>28.230</td>
    </tr>
    <tr>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
    </tr>
    <tr>
      <th>LYSVSYLLK</th>
      <td>29.982</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_24_1.png)
    



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
      <th>IAILTCPFEPPKPK</th>
      <td>27.348</td>
    </tr>
    <tr>
      <th>DDDIAALVVDNGSGMCK</th>
      <td>31.112</td>
    </tr>
    <tr>
      <th>GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK</th>
      <td>31.778</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>28.979</td>
    </tr>
    <tr>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <td>29.355</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>DPVQEAWAEDVDLR</th>
      <td>27.497</td>
    </tr>
    <tr>
      <th>MALIGLGVSHPVLK</th>
      <td>28.230</td>
    </tr>
    <tr>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
    </tr>
    <tr>
      <th>LYSVSYLLK</th>
      <td>29.982</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
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
      <th>20190408_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-005</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>31.577</td>
    </tr>
    <tr>
      <th>20190729_QX4_PhGe_MA_Hela_500ng_LC15</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>30.970</td>
    </tr>
    <tr>
      <th>20190621_QX4_JoMu_MA_HeLa_500ng_190621161214</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>32.129</td>
    </tr>
    <tr>
      <th>20190526_QX8_IgPa_MA_HeLa_BR14_500ng</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>31.984</td>
    </tr>
    <tr>
      <th>20190207_QE8_nLC0_ASD_QC_HeLa_43cm4</th>
      <th>ACANPAAGSVILLENLR</th>
      <td>31.804</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190415_QE3_nLC5_DS_QC_MNT_HeLa_02</th>
      <th>YSLEPVAVELK</th>
      <td>29.700</td>
    </tr>
    <tr>
      <th>20190606_QE4_LC12_JE_QC_MNT_HeLa_01</th>
      <th>YSLEPVAVELK</th>
      <td>29.661</td>
    </tr>
    <tr>
      <th>20190723_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>YSLEPVAVELK</th>
      <td>29.752</td>
    </tr>
    <tr>
      <th>20190604_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02</th>
      <th>YSLEPVAVELK</th>
      <td>29.855</td>
    </tr>
    <tr>
      <th>20190131_QE6_LC6_AS_MNT_HeLa_01</th>
      <th>YSLEPVAVELK</th>
      <td>30.851</td>
    </tr>
  </tbody>
</table>
<p>44299 rows × 1 columns</p>
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
    Shape in validation: (992, 50)
    




    ((992, 50), (992, 50))



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
      <th>ACANPAAGSVILLENLR</th>
      <td>30.430</td>
      <td>31.219</td>
      <td>31.026</td>
      <td>30.412</td>
    </tr>
    <tr>
      <th>AVAEQIPLLVQGVR</th>
      <td>26.531</td>
      <td>26.585</td>
      <td>26.580</td>
      <td>26.424</td>
    </tr>
    <tr>
      <th>DPVQEAWAEDVDLR</th>
      <td>29.418</td>
      <td>28.676</td>
      <td>29.250</td>
      <td>28.626</td>
    </tr>
    <tr>
      <th>LSLEGDHSTPPSAYGSVK</th>
      <td>30.405</td>
      <td>31.974</td>
      <td>31.798</td>
      <td>31.367</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>29.171</td>
      <td>30.291</td>
      <td>30.241</td>
      <td>29.274</td>
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
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
      <td>31.512</td>
      <td>31.286</td>
      <td>30.006</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>29.586</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>29.493</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
      <td>28.735</td>
      <td>28.754</td>
      <td>28.618</td>
    </tr>
    <tr>
      <th>LQVTNVLSQPLTQATVK</th>
      <td>28.521</td>
      <td>27.796</td>
      <td>27.631</td>
      <td>28.129</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
      <td>30.620</td>
      <td>30.400</td>
      <td>30.242</td>
    </tr>
  </tbody>
</table>
<p>4921 rows × 4 columns</p>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03</th>
      <th>HGVVPLATYMR</th>
      <td>28.772</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181227_QE6_nLC6_CSC_QC_MNT_HeLa_01</th>
      <th>GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK</th>
      <td>31.643</td>
      <td>31.835</td>
      <td>31.925</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</th>
      <th>EEEIAALVIDNGSGMCK</th>
      <td>32.324</td>
      <td>32.174</td>
      <td>31.701</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190104_QE6_nLC6_MM_QC_MNT_HELA_01_190108181708</th>
      <th>VLALPEPSPAAPTLR</th>
      <td>27.715</td>
      <td>27.816</td>
      <td>27.811</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE2_NLC1_GP_QC_MNT_HELA_02</th>
      <th>EMEENFAVEAANYQDTIGR</th>
      <td>26.771</td>
      <td>26.976</td>
      <td>27.818</td>
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
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>VFQFLNAK</th>
      <td>30.738</td>
      <td>30.188</td>
      <td>30.101</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02</th>
      <th>VFQFLNAK</th>
      <td>30.592</td>
      <td>30.188</td>
      <td>30.101</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QE8_nLC14_ASD_QC_MNT_HeLa_03</th>
      <th>LYSVSYLLK</th>
      <td>25.411</td>
      <td>29.189</td>
      <td>28.664</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>30.957</td>
      <td>31.512</td>
      <td>31.286</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>29.304</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 4 columns</p>
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
      <td>AGVNTVTTLVENKK</td>
      <td>28.355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AIAELGIYPAVDPLDSTSR</td>
      <td>29.355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ATCIGNNSAAAVSMLK</td>
      <td>26.757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DATNVGDEGGFAPNILENK</td>
      <td>31.756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DDDIAALVVDNGSGMCK</td>
      <td>31.112</td>
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
      <td>ACANPAAGSVILLENLR</td>
      <td>30.430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVAEQIPLLVQGVR</td>
      <td>26.531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DPVQEAWAEDVDLR</td>
      <td>29.418</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>LSLEGDHSTPPSAYGSVK</td>
      <td>30.405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>STGEAFVQFASQEIAEK</td>
      <td>29.171</td>
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
      <td>20190527_QX1_PhGe_MA_HeLa_500ng_LC10</td>
      <td>SYSPYDMLESIRK</td>
      <td>30.037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_1_20190207182540</td>
      <td>IAILTCPFEPPKPK</td>
      <td>28.219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190503_QX1_LiSc_MA_HeLa_500ng_LC10</td>
      <td>EMEENFAVEAANYQDTIGR</td>
      <td>26.675</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03</td>
      <td>YSLEPVAVELK</td>
      <td>30.498</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190729_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_5thcolumn_1</td>
      <td>EGPYDVVVLPGGNLGAQNLSESAAVK</td>
      <td>31.443</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190422_QE4_LC12_JE-IAH_QC_MNT_HeLa_02</td>
      <td>AGVNTVTTLVENKK</td>
      <td>29.116</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190613_QX0_MePh_MA_HeLa_500ng_LC07_1</td>
      <td>FLSQPFQVAEVFTGHMGK</td>
      <td>33.355</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03</td>
      <td>LSSLIILMPHHVEPLER</td>
      <td>30.496</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03</td>
      <td>YGINTTDIFQTVDLWEGK</td>
      <td>29.656</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190326_QE9_nLC0_JM_MNT_Hela_50cm_01</td>
      <td>ISSLLEEQFQQGK</td>
      <td>28.715</td>
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
      <td>20190611_QE7_nLC5_MEM_QC_MNT_HeLa_04</td>
      <td>DDDIAALVVDNGSGMCK</td>
      <td>34.328</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190319_QE4_LC12_IAH-JE_QC_MNT_HeLa_01</td>
      <td>LQVTNVLSQPLTQATVK</td>
      <td>27.974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190206_QE8_nLC0_ASD_QC_HeLa_50cm_20190206192638</td>
      <td>LYSVSYLLK</td>
      <td>29.621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190719_QE2_NLC1_ANHO_MNT_HELA_01</td>
      <td>ATCIGNNSAAAVSMLK</td>
      <td>28.141</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190423_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff</td>
      <td>LVSIGAEEIVDGNAK</td>
      <td>28.424</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190225_QE10_PhGe_Evosep_88min_HeLa_1_20190225173940</td>
      <td>IAILTCPFEPPKPK</td>
      <td>28.717</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190107_QE10_nLC0_KS_QC_MNT_HeLa_01</td>
      <td>HGVVPLATYMR</td>
      <td>29.026</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190429_QX4_ChDe_MA_HeLa_500ng_BR14_standard</td>
      <td>HVVFIAQR</td>
      <td>31.062</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2</td>
      <td>VLALPEPSPAAPTLR</td>
      <td>26.917</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190719_QE8_nLC14_RG_QC_HeLa_MUC_50cm_1</td>
      <td>SYSPYDMLESIRK</td>
      <td>28.523</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (993, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1372, 154)



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
     'n_samples': 993,
     'y_range': (20, 37)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1986       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 993        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,132
    Total trainable params: 3,132
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000002C2B2677040>
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
      <td>1.746172</td>
      <td>1.669874</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.674881</td>
      <td>0.615793</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.616020</td>
      <td>0.585901</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.545843</td>
      <td>0.559943</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.650328</td>
      <td>0.544262</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.553523</td>
      <td>0.525358</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.542064</td>
      <td>0.513932</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.478438</td>
      <td>0.509272</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.457587</td>
      <td>0.507915</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.490250</td>
      <td>0.507682</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_58_1.png)
    


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
      <th>3,197</th>
      <td>651</td>
      <td>7</td>
      <td>34.328</td>
    </tr>
    <tr>
      <th>1,523</th>
      <td>303</td>
      <td>35</td>
      <td>27.974</td>
    </tr>
    <tr>
      <th>677</th>
      <td>139</td>
      <td>40</td>
      <td>29.621</td>
    </tr>
    <tr>
      <th>4,394</th>
      <td>890</td>
      <td>4</td>
      <td>28.141</td>
    </tr>
    <tr>
      <th>2,118</th>
      <td>420</td>
      <td>39</td>
      <td>28.424</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4,343</th>
      <td>879</td>
      <td>27</td>
      <td>30.724</td>
    </tr>
    <tr>
      <th>3,277</th>
      <td>666</td>
      <td>33</td>
      <td>31.139</td>
    </tr>
    <tr>
      <th>3,927</th>
      <td>796</td>
      <td>3</td>
      <td>27.998</td>
    </tr>
    <tr>
      <th>1,825</th>
      <td>364</td>
      <td>21</td>
      <td>30.553</td>
    </tr>
    <tr>
      <th>3,329</th>
      <td>677</td>
      <td>43</td>
      <td>31.850</td>
    </tr>
  </tbody>
</table>
<p>4921 rows × 3 columns</p>
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
      <th>ACANPAAGSVILLENLR</th>
      <td>30.430</td>
      <td>31.219</td>
      <td>31.026</td>
      <td>30.412</td>
      <td>30.477</td>
    </tr>
    <tr>
      <th>AVAEQIPLLVQGVR</th>
      <td>26.531</td>
      <td>26.585</td>
      <td>26.580</td>
      <td>26.424</td>
      <td>25.873</td>
    </tr>
    <tr>
      <th>DPVQEAWAEDVDLR</th>
      <td>29.418</td>
      <td>28.676</td>
      <td>29.250</td>
      <td>28.626</td>
      <td>28.516</td>
    </tr>
    <tr>
      <th>LSLEGDHSTPPSAYGSVK</th>
      <td>30.405</td>
      <td>31.974</td>
      <td>31.798</td>
      <td>31.367</td>
      <td>31.203</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>29.171</td>
      <td>30.291</td>
      <td>30.241</td>
      <td>29.274</td>
      <td>29.657</td>
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
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
      <td>31.512</td>
      <td>31.286</td>
      <td>30.006</td>
      <td>31.169</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>29.586</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>29.493</td>
      <td>29.737</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
      <td>28.735</td>
      <td>28.754</td>
      <td>28.618</td>
      <td>28.304</td>
    </tr>
    <tr>
      <th>LQVTNVLSQPLTQATVK</th>
      <td>28.521</td>
      <td>27.796</td>
      <td>27.631</td>
      <td>28.129</td>
      <td>27.726</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
      <td>30.620</td>
      <td>30.400</td>
      <td>30.242</td>
      <td>30.339</td>
    </tr>
  </tbody>
</table>
<p>4921 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.067
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.054
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.147
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.107
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.132
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_66_1.png)
    



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
      <td>-0.250</td>
      <td>-0.182</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.195</td>
      <td>-0.147</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.266</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.143</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.347</td>
      <td>-0.318</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_68_1.png)
    



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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_69_1.png)
    


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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK</th>
      <th>QLFHPEQLITGK</th>
      <th>STGEAFVQFASQEIAEK</th>
      <th>SYSPYDMLESIRK</th>
      <th>TISHVIIGLK</th>
      <th>VFQFLNAK</th>
      <th>VLALPEPSPAAPTLR</th>
      <th>VMLGETNPADSKPGTIR</th>
      <th>YGINTTDIFQTVDLWEGK</th>
      <th>YSLEPVAVELK</th>
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
      <td>30.430</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26.531</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.171</td>
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
      <td>28.238</td>
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
      <td>27.316</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>31.257</td>
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
      <td>NaN</td>
      <td>27.602</td>
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
      <td>25.928</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.339</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.243</td>
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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_na</th>
      <th>QLFHPEQLITGK_na</th>
      <th>STGEAFVQFASQEIAEK_na</th>
      <th>SYSPYDMLESIRK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>VFQFLNAK_na</th>
      <th>VLALPEPSPAAPTLR_na</th>
      <th>VMLGETNPADSKPGTIR_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YSLEPVAVELK_na</th>
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
      <td>0.175</td>
      <td>-0.964</td>
      <td>-0.542</td>
      <td>-1.277</td>
      <td>0.004</td>
      <td>-0.758</td>
      <td>-0.431</td>
      <td>-0.399</td>
      <td>-0.474</td>
      <td>-0.130</td>
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
      <td>-0.674</td>
      <td>0.269</td>
      <td>-0.443</td>
      <td>-1.064</td>
      <td>-0.372</td>
      <td>-0.951</td>
      <td>-0.504</td>
      <td>-0.542</td>
      <td>-0.283</td>
      <td>-0.230</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.175</td>
      <td>0.196</td>
      <td>0.009</td>
      <td>-0.230</td>
      <td>0.059</td>
      <td>0.057</td>
      <td>0.501</td>
      <td>-0.253</td>
      <td>-0.851</td>
      <td>-0.910</td>
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
      <td>-0.456</td>
      <td>-0.953</td>
      <td>-0.617</td>
      <td>-0.752</td>
      <td>0.004</td>
      <td>-0.263</td>
      <td>-0.135</td>
      <td>-0.615</td>
      <td>-2.224</td>
      <td>-0.470</td>
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
      <td>0.777</td>
      <td>-0.953</td>
      <td>0.777</td>
      <td>-0.502</td>
      <td>0.004</td>
      <td>-0.166</td>
      <td>0.377</td>
      <td>-0.171</td>
      <td>-0.027</td>
      <td>-0.168</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>0.897</td>
      <td>-3.007</td>
      <td>1.198</td>
      <td>0.308</td>
      <td>1.053</td>
      <td>0.452</td>
      <td>-0.120</td>
      <td>-0.748</td>
      <td>-0.027</td>
      <td>1.358</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.228</td>
      <td>0.322</td>
      <td>-0.218</td>
      <td>-0.579</td>
      <td>-1.004</td>
      <td>-0.271</td>
      <td>0.526</td>
      <td>0.660</td>
      <td>-0.030</td>
      <td>-1.235</td>
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
      <td>0.650</td>
      <td>0.269</td>
      <td>0.023</td>
      <td>-0.139</td>
      <td>-0.532</td>
      <td>-0.006</td>
      <td>0.545</td>
      <td>0.822</td>
      <td>0.342</td>
      <td>-0.143</td>
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
      <td>-1.261</td>
      <td>0.715</td>
      <td>0.196</td>
      <td>-0.023</td>
      <td>-0.946</td>
      <td>-0.023</td>
      <td>0.637</td>
      <td>-0.061</td>
      <td>0.628</td>
      <td>-0.190</td>
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
      <td>0.679</td>
      <td>0.448</td>
      <td>0.010</td>
      <td>0.071</td>
      <td>-0.661</td>
      <td>-0.095</td>
      <td>0.258</td>
      <td>0.479</td>
      <td>0.694</td>
      <td>-0.054</td>
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
<p>992 rows × 100 columns</p>
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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_na</th>
      <th>QLFHPEQLITGK_na</th>
      <th>STGEAFVQFASQEIAEK_na</th>
      <th>SYSPYDMLESIRK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>VFQFLNAK_na</th>
      <th>VLALPEPSPAAPTLR_na</th>
      <th>VMLGETNPADSKPGTIR_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YSLEPVAVELK_na</th>
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
      <td>0.189</td>
      <td>-0.866</td>
      <td>-0.476</td>
      <td>-1.188</td>
      <td>0.005</td>
      <td>-0.701</td>
      <td>-0.378</td>
      <td>-0.359</td>
      <td>-0.450</td>
      <td>-0.129</td>
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
      <td>-0.600</td>
      <td>0.291</td>
      <td>-0.385</td>
      <td>-0.989</td>
      <td>-0.349</td>
      <td>-0.884</td>
      <td>-0.448</td>
      <td>-0.495</td>
      <td>-0.270</td>
      <td>-0.223</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.189</td>
      <td>0.221</td>
      <td>0.033</td>
      <td>-0.206</td>
      <td>0.056</td>
      <td>0.072</td>
      <td>0.505</td>
      <td>-0.220</td>
      <td>-0.804</td>
      <td>-0.865</td>
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
      <td>-0.397</td>
      <td>-0.855</td>
      <td>-0.546</td>
      <td>-0.695</td>
      <td>0.005</td>
      <td>-0.231</td>
      <td>-0.098</td>
      <td>-0.564</td>
      <td>-2.097</td>
      <td>-0.450</td>
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
      <td>0.749</td>
      <td>-0.855</td>
      <td>0.744</td>
      <td>-0.461</td>
      <td>0.005</td>
      <td>-0.139</td>
      <td>0.387</td>
      <td>-0.142</td>
      <td>-0.028</td>
      <td>-0.164</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>0.860</td>
      <td>-2.781</td>
      <td>1.135</td>
      <td>0.298</td>
      <td>0.990</td>
      <td>0.448</td>
      <td>-0.084</td>
      <td>-0.691</td>
      <td>-0.028</td>
      <td>1.274</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.238</td>
      <td>0.340</td>
      <td>-0.176</td>
      <td>-0.534</td>
      <td>-0.942</td>
      <td>-0.238</td>
      <td>0.528</td>
      <td>0.647</td>
      <td>-0.032</td>
      <td>-1.171</td>
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
      <td>0.631</td>
      <td>0.291</td>
      <td>0.047</td>
      <td>-0.121</td>
      <td>-0.499</td>
      <td>0.013</td>
      <td>0.547</td>
      <td>0.801</td>
      <td>0.319</td>
      <td>-0.142</td>
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
      <td>-1.146</td>
      <td>0.708</td>
      <td>0.207</td>
      <td>-0.012</td>
      <td>-0.888</td>
      <td>-0.003</td>
      <td>0.634</td>
      <td>-0.038</td>
      <td>0.588</td>
      <td>-0.186</td>
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
      <td>0.657</td>
      <td>0.458</td>
      <td>0.035</td>
      <td>0.075</td>
      <td>-0.621</td>
      <td>-0.072</td>
      <td>0.275</td>
      <td>0.475</td>
      <td>0.650</td>
      <td>-0.058</td>
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
<p>992 rows × 100 columns</p>
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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
      <td>992.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.026</td>
      <td>0.038</td>
      <td>0.025</td>
      <td>0.009</td>
      <td>0.001</td>
      <td>0.019</td>
      <td>0.030</td>
      <td>0.020</td>
      <td>-0.003</td>
      <td>-0.006</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.930</td>
      <td>0.938</td>
      <td>0.926</td>
      <td>0.938</td>
      <td>0.940</td>
      <td>0.950</td>
      <td>0.948</td>
      <td>0.951</td>
      <td>0.942</td>
      <td>0.944</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.776</td>
      <td>-4.846</td>
      <td>-4.307</td>
      <td>-4.716</td>
      <td>-5.075</td>
      <td>-6.818</td>
      <td>-6.395</td>
      <td>-7.041</td>
      <td>-4.493</td>
      <td>-3.515</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.356</td>
      <td>-0.089</td>
      <td>-0.388</td>
      <td>-0.363</td>
      <td>-0.419</td>
      <td>-0.248</td>
      <td>-0.270</td>
      <td>-0.245</td>
      <td>-0.419</td>
      <td>-0.470</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.189</td>
      <td>0.291</td>
      <td>0.172</td>
      <td>0.075</td>
      <td>0.005</td>
      <td>0.185</td>
      <td>0.275</td>
      <td>0.200</td>
      <td>-0.028</td>
      <td>-0.058</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.620</td>
      <td>0.579</td>
      <td>0.611</td>
      <td>0.519</td>
      <td>0.573</td>
      <td>0.520</td>
      <td>0.655</td>
      <td>0.564</td>
      <td>0.533</td>
      <td>0.550</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.060</td>
      <td>1.618</td>
      <td>2.082</td>
      <td>2.310</td>
      <td>2.213</td>
      <td>1.882</td>
      <td>1.455</td>
      <td>2.030</td>
      <td>2.420</td>
      <td>2.228</td>
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




    ((#50) ['ACANPAAGSVILLENLR','AGVNTVTTLVENKK','AIAELGIYPAVDPLDSTSR','ATCIGNNSAAAVSMLK','AVAEQIPLLVQGVR','DATNVGDEGGFAPNILENK','DDDIAALVVDNGSGMCK','DGLAFNALIHR','DHENIVIAK','DLISHDEMFSDIYK'...],
     (#50) ['ACANPAAGSVILLENLR_na','AGVNTVTTLVENKK_na','AIAELGIYPAVDPLDSTSR_na','ATCIGNNSAAAVSMLK_na','AVAEQIPLLVQGVR_na','DATNVGDEGGFAPNILENK_na','DDDIAALVVDNGSGMCK_na','DGLAFNALIHR_na','DHENIVIAK_na','DLISHDEMFSDIYK_na'...])




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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>96.000</td>
      <td>97.000</td>
      <td>95.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.022</td>
      <td>0.111</td>
      <td>0.014</td>
      <td>-0.184</td>
      <td>0.094</td>
      <td>0.088</td>
      <td>-0.107</td>
      <td>-0.048</td>
      <td>0.045</td>
      <td>-0.048</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.900</td>
      <td>0.747</td>
      <td>0.988</td>
      <td>0.863</td>
      <td>1.127</td>
      <td>0.857</td>
      <td>0.955</td>
      <td>0.918</td>
      <td>0.962</td>
      <td>1.042</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.829</td>
      <td>-3.168</td>
      <td>-4.325</td>
      <td>-2.880</td>
      <td>-3.799</td>
      <td>-2.657</td>
      <td>-4.373</td>
      <td>-3.940</td>
      <td>-3.294</td>
      <td>-2.520</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.395</td>
      <td>-0.193</td>
      <td>-0.393</td>
      <td>-0.550</td>
      <td>-0.598</td>
      <td>-0.326</td>
      <td>-0.507</td>
      <td>-0.333</td>
      <td>-0.389</td>
      <td>-0.586</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.099</td>
      <td>0.291</td>
      <td>0.117</td>
      <td>-0.178</td>
      <td>0.266</td>
      <td>0.235</td>
      <td>0.068</td>
      <td>0.122</td>
      <td>-0.019</td>
      <td>-0.104</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.723</td>
      <td>0.538</td>
      <td>0.718</td>
      <td>0.368</td>
      <td>0.997</td>
      <td>0.687</td>
      <td>0.594</td>
      <td>0.625</td>
      <td>0.876</td>
      <td>0.774</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.508</td>
      <td>1.423</td>
      <td>1.418</td>
      <td>1.454</td>
      <td>1.995</td>
      <td>1.623</td>
      <td>1.455</td>
      <td>1.381</td>
      <td>2.089</td>
      <td>1.935</td>
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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_val</th>
      <th>QLFHPEQLITGK_val</th>
      <th>STGEAFVQFASQEIAEK_val</th>
      <th>SYSPYDMLESIRK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>VFQFLNAK_val</th>
      <th>VLALPEPSPAAPTLR_val</th>
      <th>VMLGETNPADSKPGTIR_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YSLEPVAVELK_val</th>
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
      <td>0.189</td>
      <td>-0.866</td>
      <td>-0.476</td>
      <td>-1.188</td>
      <td>0.005</td>
      <td>-0.701</td>
      <td>-0.378</td>
      <td>-0.359</td>
      <td>-0.450</td>
      <td>-0.129</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.855</td>
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
      <td>-0.600</td>
      <td>0.291</td>
      <td>-0.385</td>
      <td>-0.989</td>
      <td>-0.349</td>
      <td>-0.884</td>
      <td>-0.448</td>
      <td>-0.495</td>
      <td>-0.270</td>
      <td>-0.223</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.189</td>
      <td>0.221</td>
      <td>0.033</td>
      <td>-0.206</td>
      <td>0.056</td>
      <td>0.072</td>
      <td>0.505</td>
      <td>-0.220</td>
      <td>-0.804</td>
      <td>-0.865</td>
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
      <td>-0.397</td>
      <td>-0.855</td>
      <td>-0.546</td>
      <td>-0.695</td>
      <td>0.005</td>
      <td>-0.231</td>
      <td>-0.098</td>
      <td>-0.564</td>
      <td>-2.097</td>
      <td>-0.450</td>
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
      <td>0.749</td>
      <td>-0.855</td>
      <td>0.744</td>
      <td>-0.461</td>
      <td>0.005</td>
      <td>-0.139</td>
      <td>0.387</td>
      <td>-0.142</td>
      <td>-0.028</td>
      <td>-0.164</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.828</td>
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
      <td>0.860</td>
      <td>-2.781</td>
      <td>1.135</td>
      <td>0.298</td>
      <td>0.990</td>
      <td>0.448</td>
      <td>-0.084</td>
      <td>-0.691</td>
      <td>-0.028</td>
      <td>1.274</td>
      <td>...</td>
      <td>-0.425</td>
      <td>1.127</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.205</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.238</td>
      <td>0.340</td>
      <td>-0.176</td>
      <td>-0.534</td>
      <td>-0.942</td>
      <td>-0.238</td>
      <td>0.528</td>
      <td>0.647</td>
      <td>-0.032</td>
      <td>-1.171</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.277</td>
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
      <td>0.631</td>
      <td>0.291</td>
      <td>0.047</td>
      <td>-0.121</td>
      <td>-0.499</td>
      <td>0.013</td>
      <td>0.547</td>
      <td>0.801</td>
      <td>0.319</td>
      <td>-0.142</td>
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
      <td>-1.146</td>
      <td>0.708</td>
      <td>0.207</td>
      <td>-0.012</td>
      <td>-0.888</td>
      <td>-0.003</td>
      <td>0.634</td>
      <td>-0.038</td>
      <td>0.588</td>
      <td>-0.186</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.375</td>
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
      <td>0.657</td>
      <td>0.458</td>
      <td>0.035</td>
      <td>0.075</td>
      <td>-0.621</td>
      <td>-0.072</td>
      <td>0.275</td>
      <td>0.475</td>
      <td>0.650</td>
      <td>-0.058</td>
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
<p>992 rows × 150 columns</p>
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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_val</th>
      <th>QLFHPEQLITGK_val</th>
      <th>STGEAFVQFASQEIAEK_val</th>
      <th>SYSPYDMLESIRK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>VFQFLNAK_val</th>
      <th>VLALPEPSPAAPTLR_val</th>
      <th>VMLGETNPADSKPGTIR_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YSLEPVAVELK_val</th>
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
      <td>0.189</td>
      <td>-0.866</td>
      <td>-0.476</td>
      <td>-1.188</td>
      <td>0.005</td>
      <td>-0.701</td>
      <td>-0.378</td>
      <td>-0.359</td>
      <td>-0.450</td>
      <td>-0.129</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.855</td>
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
      <td>-0.600</td>
      <td>0.291</td>
      <td>-0.385</td>
      <td>-0.989</td>
      <td>-0.349</td>
      <td>-0.884</td>
      <td>-0.448</td>
      <td>-0.495</td>
      <td>-0.270</td>
      <td>-0.223</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.189</td>
      <td>0.221</td>
      <td>0.033</td>
      <td>-0.206</td>
      <td>0.056</td>
      <td>0.072</td>
      <td>0.505</td>
      <td>-0.220</td>
      <td>-0.804</td>
      <td>-0.865</td>
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
      <td>-0.397</td>
      <td>-0.855</td>
      <td>-0.546</td>
      <td>-0.695</td>
      <td>0.005</td>
      <td>-0.231</td>
      <td>-0.098</td>
      <td>-0.564</td>
      <td>-2.097</td>
      <td>-0.450</td>
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
      <td>0.749</td>
      <td>-0.855</td>
      <td>0.744</td>
      <td>-0.461</td>
      <td>0.005</td>
      <td>-0.139</td>
      <td>0.387</td>
      <td>-0.142</td>
      <td>-0.028</td>
      <td>-0.164</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.828</td>
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
      <td>0.860</td>
      <td>-2.781</td>
      <td>1.135</td>
      <td>0.298</td>
      <td>0.990</td>
      <td>0.448</td>
      <td>-0.084</td>
      <td>-0.691</td>
      <td>-0.028</td>
      <td>1.274</td>
      <td>...</td>
      <td>-0.425</td>
      <td>1.127</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.205</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.238</td>
      <td>0.340</td>
      <td>-0.176</td>
      <td>-0.534</td>
      <td>-0.942</td>
      <td>-0.238</td>
      <td>0.528</td>
      <td>0.647</td>
      <td>-0.032</td>
      <td>-1.171</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.277</td>
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
      <td>0.631</td>
      <td>0.291</td>
      <td>0.047</td>
      <td>-0.121</td>
      <td>-0.499</td>
      <td>0.013</td>
      <td>0.547</td>
      <td>0.801</td>
      <td>0.319</td>
      <td>-0.142</td>
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
      <td>-1.146</td>
      <td>0.708</td>
      <td>0.207</td>
      <td>-0.012</td>
      <td>-0.888</td>
      <td>-0.003</td>
      <td>0.634</td>
      <td>-0.038</td>
      <td>0.588</td>
      <td>-0.186</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.375</td>
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
      <td>0.657</td>
      <td>0.458</td>
      <td>0.035</td>
      <td>0.075</td>
      <td>-0.621</td>
      <td>-0.072</td>
      <td>0.275</td>
      <td>0.475</td>
      <td>0.650</td>
      <td>-0.058</td>
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
<p>992 rows × 150 columns</p>
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
      <th>ACANPAAGSVILLENLR_val</th>
      <th>AGVNTVTTLVENKK_val</th>
      <th>AIAELGIYPAVDPLDSTSR_val</th>
      <th>ATCIGNNSAAAVSMLK_val</th>
      <th>AVAEQIPLLVQGVR_val</th>
      <th>DATNVGDEGGFAPNILENK_val</th>
      <th>DDDIAALVVDNGSGMCK_val</th>
      <th>DGLAFNALIHR_val</th>
      <th>DHENIVIAK_val</th>
      <th>DLISHDEMFSDIYK_val</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_val</th>
      <th>QLFHPEQLITGK_val</th>
      <th>STGEAFVQFASQEIAEK_val</th>
      <th>SYSPYDMLESIRK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>VFQFLNAK_val</th>
      <th>VLALPEPSPAAPTLR_val</th>
      <th>VMLGETNPADSKPGTIR_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YSLEPVAVELK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>96.000</td>
      <td>97.000</td>
      <td>95.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>...</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>97.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.022</td>
      <td>0.111</td>
      <td>0.014</td>
      <td>-0.184</td>
      <td>0.094</td>
      <td>0.088</td>
      <td>-0.107</td>
      <td>-0.048</td>
      <td>0.045</td>
      <td>-0.048</td>
      <td>...</td>
      <td>-0.082</td>
      <td>-0.031</td>
      <td>-0.215</td>
      <td>-0.183</td>
      <td>-0.060</td>
      <td>-0.017</td>
      <td>-0.109</td>
      <td>-0.000</td>
      <td>-0.046</td>
      <td>0.048</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.900</td>
      <td>0.747</td>
      <td>0.988</td>
      <td>0.863</td>
      <td>1.127</td>
      <td>0.857</td>
      <td>0.955</td>
      <td>0.918</td>
      <td>0.962</td>
      <td>1.042</td>
      <td>...</td>
      <td>1.152</td>
      <td>0.975</td>
      <td>1.284</td>
      <td>1.345</td>
      <td>1.142</td>
      <td>1.099</td>
      <td>1.224</td>
      <td>0.892</td>
      <td>1.051</td>
      <td>0.784</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.829</td>
      <td>-3.168</td>
      <td>-4.325</td>
      <td>-2.880</td>
      <td>-3.799</td>
      <td>-2.657</td>
      <td>-4.373</td>
      <td>-3.940</td>
      <td>-3.294</td>
      <td>-2.520</td>
      <td>...</td>
      <td>-4.403</td>
      <td>-2.288</td>
      <td>-4.889</td>
      <td>-5.966</td>
      <td>-4.030</td>
      <td>-4.086</td>
      <td>-4.032</td>
      <td>-2.758</td>
      <td>-4.540</td>
      <td>-2.503</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.395</td>
      <td>-0.193</td>
      <td>-0.393</td>
      <td>-0.550</td>
      <td>-0.598</td>
      <td>-0.326</td>
      <td>-0.507</td>
      <td>-0.333</td>
      <td>-0.389</td>
      <td>-0.586</td>
      <td>...</td>
      <td>-0.582</td>
      <td>-0.655</td>
      <td>-0.579</td>
      <td>-0.612</td>
      <td>-0.753</td>
      <td>-0.561</td>
      <td>-0.686</td>
      <td>-0.592</td>
      <td>-0.328</td>
      <td>-0.365</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.099</td>
      <td>0.291</td>
      <td>0.117</td>
      <td>-0.178</td>
      <td>0.266</td>
      <td>0.235</td>
      <td>0.068</td>
      <td>0.122</td>
      <td>-0.019</td>
      <td>-0.104</td>
      <td>...</td>
      <td>0.245</td>
      <td>-0.362</td>
      <td>-0.062</td>
      <td>0.086</td>
      <td>-0.084</td>
      <td>0.046</td>
      <td>-0.094</td>
      <td>0.023</td>
      <td>0.233</td>
      <td>0.136</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.723</td>
      <td>0.538</td>
      <td>0.718</td>
      <td>0.368</td>
      <td>0.997</td>
      <td>0.687</td>
      <td>0.594</td>
      <td>0.625</td>
      <td>0.876</td>
      <td>0.774</td>
      <td>...</td>
      <td>0.620</td>
      <td>1.071</td>
      <td>0.488</td>
      <td>0.735</td>
      <td>0.650</td>
      <td>0.682</td>
      <td>0.617</td>
      <td>0.584</td>
      <td>0.589</td>
      <td>0.521</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.508</td>
      <td>1.423</td>
      <td>1.418</td>
      <td>1.454</td>
      <td>1.995</td>
      <td>1.623</td>
      <td>1.455</td>
      <td>1.381</td>
      <td>2.089</td>
      <td>1.935</td>
      <td>...</td>
      <td>1.632</td>
      <td>1.662</td>
      <td>1.830</td>
      <td>1.303</td>
      <td>2.548</td>
      <td>1.861</td>
      <td>1.896</td>
      <td>2.023</td>
      <td>1.510</td>
      <td>1.598</td>
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
      <th>ACANPAAGSVILLENLR_na</th>
      <th>AGVNTVTTLVENKK_na</th>
      <th>AIAELGIYPAVDPLDSTSR_na</th>
      <th>ATCIGNNSAAAVSMLK_na</th>
      <th>AVAEQIPLLVQGVR_na</th>
      <th>DATNVGDEGGFAPNILENK_na</th>
      <th>DDDIAALVVDNGSGMCK_na</th>
      <th>DGLAFNALIHR_na</th>
      <th>DHENIVIAK_na</th>
      <th>DLISHDEMFSDIYK_na</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK_na</th>
      <th>QLFHPEQLITGK_na</th>
      <th>STGEAFVQFASQEIAEK_na</th>
      <th>SYSPYDMLESIRK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>VFQFLNAK_na</th>
      <th>VLALPEPSPAAPTLR_na</th>
      <th>VMLGETNPADSKPGTIR_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YSLEPVAVELK_na</th>
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
      <td>False</td>
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
      <td>True</td>
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
      <td>False</td>
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
      <td>False</td>
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
<p>992 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x000002C2B2677040>
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




    
![png](latent_2D_200_10_files/latent_2D_200_10_108_2.png)
    


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
      <td>0.985683</td>
      <td>0.844692</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.699860</td>
      <td>0.389946</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.504913</td>
      <td>0.341548</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.409964</td>
      <td>0.324390</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.360650</td>
      <td>0.307544</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.333663</td>
      <td>0.309682</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.315587</td>
      <td>0.300084</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.305307</td>
      <td>0.299010</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.297704</td>
      <td>0.298051</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.291551</td>
      <td>0.294273</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4921, 4921)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.2960)




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
      <th>ACANPAAGSVILLENLR</th>
      <td>30.430</td>
      <td>31.219</td>
      <td>31.026</td>
      <td>30.412</td>
      <td>30.477</td>
      <td>30.463</td>
    </tr>
    <tr>
      <th>AVAEQIPLLVQGVR</th>
      <td>26.531</td>
      <td>26.585</td>
      <td>26.580</td>
      <td>26.424</td>
      <td>25.873</td>
      <td>25.896</td>
    </tr>
    <tr>
      <th>DPVQEAWAEDVDLR</th>
      <td>29.418</td>
      <td>28.676</td>
      <td>29.250</td>
      <td>28.626</td>
      <td>28.516</td>
      <td>28.201</td>
    </tr>
    <tr>
      <th>LSLEGDHSTPPSAYGSVK</th>
      <td>30.405</td>
      <td>31.974</td>
      <td>31.798</td>
      <td>31.367</td>
      <td>31.203</td>
      <td>31.161</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>29.171</td>
      <td>30.291</td>
      <td>30.241</td>
      <td>29.274</td>
      <td>29.657</td>
      <td>29.428</td>
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
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
      <td>31.512</td>
      <td>31.286</td>
      <td>30.006</td>
      <td>31.169</td>
      <td>31.312</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>29.586</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>29.493</td>
      <td>29.737</td>
      <td>29.810</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
      <td>28.735</td>
      <td>28.754</td>
      <td>28.618</td>
      <td>28.304</td>
      <td>28.488</td>
    </tr>
    <tr>
      <th>LQVTNVLSQPLTQATVK</th>
      <td>28.521</td>
      <td>27.796</td>
      <td>27.631</td>
      <td>28.129</td>
      <td>27.726</td>
      <td>27.827</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
      <td>30.620</td>
      <td>30.400</td>
      <td>30.242</td>
      <td>30.339</td>
      <td>30.454</td>
    </tr>
  </tbody>
</table>
<p>4921 rows × 6 columns</p>
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
      <td>-0.301</td>
      <td>-0.451</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.228</td>
      <td>-0.446</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.553</td>
      <td>-0.718</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.212</td>
      <td>-0.180</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.521</td>
      <td>-0.478</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_122_1.png)
    



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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_123_1.png)
    


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
      <th>ACANPAAGSVILLENLR</th>
      <th>AGVNTVTTLVENKK</th>
      <th>AIAELGIYPAVDPLDSTSR</th>
      <th>ATCIGNNSAAAVSMLK</th>
      <th>AVAEQIPLLVQGVR</th>
      <th>DATNVGDEGGFAPNILENK</th>
      <th>DDDIAALVVDNGSGMCK</th>
      <th>DGLAFNALIHR</th>
      <th>DHENIVIAK</th>
      <th>DLISHDEMFSDIYK</th>
      <th>...</th>
      <th>MALIGLGVSHPVLK</th>
      <th>QLFHPEQLITGK</th>
      <th>STGEAFVQFASQEIAEK</th>
      <th>SYSPYDMLESIRK</th>
      <th>TISHVIIGLK</th>
      <th>VFQFLNAK</th>
      <th>VLALPEPSPAAPTLR</th>
      <th>VMLGETNPADSKPGTIR</th>
      <th>YGINTTDIFQTVDLWEGK</th>
      <th>YSLEPVAVELK</th>
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
      <td>0.613</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.690</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.567</td>
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
      <td>0.603</td>
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
      <td>0.583</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.732</td>
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
      <td>NaN</td>
      <td>0.828</td>
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
      <td>0.612</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.576</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.680</td>
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
    
    Optimizer used: <function Adam at 0x000002C2B2677040>
    Loss function: <function loss_fct_vae at 0x000002C2B2695940>
    
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








    SuggestedLRs(valley=0.0020892962347716093)




    
![png](latent_2D_200_10_files/latent_2D_200_10_136_2.png)
    



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
      <td>1991.794800</td>
      <td>219.898117</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1964.891968</td>
      <td>215.665436</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1915.432129</td>
      <td>209.767761</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1864.668823</td>
      <td>205.532684</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1827.672119</td>
      <td>202.502304</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1801.396729</td>
      <td>199.460007</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1782.294922</td>
      <td>197.347260</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1768.872559</td>
      <td>196.689468</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1759.480713</td>
      <td>196.966324</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1753.105835</td>
      <td>196.922256</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4921)




```python
len(pred[0])
```




    4921




```python
learn.loss_func(pred, target)
```




    tensor(3098.6965)




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
      <th>ACANPAAGSVILLENLR</th>
      <td>30.430</td>
      <td>31.219</td>
      <td>31.026</td>
      <td>30.412</td>
      <td>30.477</td>
      <td>30.463</td>
      <td>31.061</td>
    </tr>
    <tr>
      <th>AVAEQIPLLVQGVR</th>
      <td>26.531</td>
      <td>26.585</td>
      <td>26.580</td>
      <td>26.424</td>
      <td>25.873</td>
      <td>25.896</td>
      <td>26.661</td>
    </tr>
    <tr>
      <th>DPVQEAWAEDVDLR</th>
      <td>29.418</td>
      <td>28.676</td>
      <td>29.250</td>
      <td>28.626</td>
      <td>28.516</td>
      <td>28.201</td>
      <td>29.247</td>
    </tr>
    <tr>
      <th>LSLEGDHSTPPSAYGSVK</th>
      <td>30.405</td>
      <td>31.974</td>
      <td>31.798</td>
      <td>31.367</td>
      <td>31.203</td>
      <td>31.161</td>
      <td>31.856</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>29.171</td>
      <td>30.291</td>
      <td>30.241</td>
      <td>29.274</td>
      <td>29.657</td>
      <td>29.428</td>
      <td>30.385</td>
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
      <th>FWEVISDEHGIDPTGTYHGDSDLQLDR</th>
      <td>31.312</td>
      <td>31.512</td>
      <td>31.286</td>
      <td>30.006</td>
      <td>31.169</td>
      <td>31.312</td>
      <td>31.282</td>
    </tr>
    <tr>
      <th>HGVVPLATYMR</th>
      <td>29.586</td>
      <td>29.904</td>
      <td>29.809</td>
      <td>29.493</td>
      <td>29.737</td>
      <td>29.810</td>
      <td>29.678</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
      <td>28.735</td>
      <td>28.754</td>
      <td>28.618</td>
      <td>28.304</td>
      <td>28.488</td>
      <td>28.787</td>
    </tr>
    <tr>
      <th>LQVTNVLSQPLTQATVK</th>
      <td>28.521</td>
      <td>27.796</td>
      <td>27.631</td>
      <td>28.129</td>
      <td>27.726</td>
      <td>27.827</td>
      <td>27.763</td>
    </tr>
    <tr>
      <th>LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK</th>
      <td>30.274</td>
      <td>30.620</td>
      <td>30.400</td>
      <td>30.242</td>
      <td>30.339</td>
      <td>30.454</td>
      <td>30.506</td>
    </tr>
  </tbody>
</table>
<p>4921 rows × 7 columns</p>
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
      <td>-0.120</td>
      <td>-0.002</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.114</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.149</td>
      <td>-0.013</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.071</td>
      <td>-0.058</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.245</td>
      <td>-0.017</td>
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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_146_1.png)
    



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
    


    
![png](latent_2D_200_10_files/latent_2D_200_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_03', 'HGVVPLATYMR'), ('20181227_QE6_nLC6_CSC_QC_MNT_HeLa_01', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007', 'EEEIAALVIDNGSGMCK'), ('20190104_QE6_nLC6_MM_QC_MNT_HELA_01_190108181708', 'VLALPEPSPAAPTLR'), ('20190121_QE2_NLC1_GP_QC_MNT_HELA_02', 'EMEENFAVEAANYQDTIGR'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'LSLEGDHSTPPSAYGSVK'), ('20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01', 'VFQFLNAK'), ('20190204_QE1_nLC2_GP_QC_MNT_HELA_01', 'SYSPYDMLESIRK'), ('20190204_QE2_NLC10_ANHO_QC_MNT_HELA_02', 'KYEDICPSTHNMDVPNIK'), ('20190206_QE8_nLC0_ASD_QC_HeLa_50cm_20190206192638', 'IAILTCPFEPPKPK'), ('20190207_QE7_nLC7_TSB_QC_MNT_HeLa_01', 'EEEIAALVIDNGSGMCK'), ('20190208_QE2_NLC1_AB_QC_MNT_HELA_3', 'ISSLLEEQFQQGK'), ('20190219_QE7_nLC7_OOE_QC_MNT_HeLa_250ng_RO-005', 'EMEENFAVEAANYQDTIGR'), ('20190221_QE4_LC12_IAH_QC_MNT_HeLa_02', 'KYEDICPSTHNMDVPNIK'), ('20190225_QE9_nLC0_RS_MNT_Hela_02', 'DYAFVHFEDR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_23', 'ISSLLEEQFQQGK'), ('20190226_QE3_nLC3_MR_QC_MNT_HELA_Easy_03', 'SYSPYDMLESIRK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_01', 'DHENIVIAK'), ('20190311_QE9_nLC0_JM_MNT_Hela_01_20190311212727', 'DDDIAALVVDNGSGMCK'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_01', 'DYAFVHFEDR'), ('20190324_QE7_nLC3_RJC_WIMS_QC_MNT_HeLa_01', 'DHENIVIAK'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'MALIGLGVSHPVLK'), ('20190327_QE6_LC6_SCL_QC_MNT_Hela_02', 'SYSPYDMLESIRK'), ('20190327_QE6_LC6_SCL_QC_MNT_Hela_03', 'SYSPYDMLESIRK'), ('20190401_QE4_LC12_IAH-JE_QC_MNT_HeLa_02', 'YSLEPVAVELK'), ('20190403_QE10_nLC13_LiNi_QC_45cm_HeLa_02', 'EMEENFAVEAANYQDTIGR'), ('20190411_QE3_nLC5_DS_QC_MNT_HeLa_01', 'YSLEPVAVELK'), ('20190423_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff', 'AGVNTVTTLVENKK'), ('20190423_QX7_JuSc_MA_HeLa_500ng_LC01', 'INISEGNCPER'), ('20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard', 'HVVFIAQR'), ('20190429_QX4_ChDe_MA_HeLa_500ng_BR14_standard', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190430_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'ACANPAAGSVILLENLR'), ('20190506_QX6_ChDe_MA_HeLa_Br13_500ng_LC09', 'MALIGLGVSHPVLK'), ('20190506_QX7_ChDe_MA_HeLaBr14_500ng', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'SYSPYDMLESIRK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_06', 'STGEAFVQFASQEIAEK'), ('20190510_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'AIAELGIYPAVDPLDSTSR'), ('20190513_QX3_ChDe_MA_Hela_500ng_LC15', 'DHENIVIAK'), ('20190513_QX7_ChDe_MA_HeLaBr14_500ng', 'ATCIGNNSAAAVSMLK'), ('20190513_QX7_ChDe_MA_HeLaBr14_500ng', 'DHENIVIAK'), ('20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190514_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'QLFHPEQLITGK'), ('20190515_QX8_MiWi_MA_HeLa_BR14_500ng', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190517_QX0_AlRe_MA_HeLa_500ng_LC07_1_BR14', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190523_QX8_MiWi_MA_HeLa_BR14_500ng_08isolation', 'LATLLGLQAPPTR'), ('20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625163359', 'STGEAFVQFASQEIAEK'), ('20190617_QE_LC_UHG_QC_MNT_HELA_04', 'ATCIGNNSAAAVSMLK'), ('20190618_QX3_LiSc_MA_Hela_500ng_LC15_190619053902', 'LATLLGLQAPPTR'), ('20190618_QX4_JiYu_MA_HeLa_500ng', 'DLISHDEMFSDIYK'), ('20190618_QX4_JiYu_MA_HeLa_500ng', 'LATLLGLQAPPTR'), ('20190618_QX4_JiYu_MA_HeLa_500ng_190619010035', 'LVQAFQYTDEHGEVCPAGWKPGSDTIKPNVDDSK'), ('20190618_QX4_JiYu_MA_HeLa_500ng_centroid', 'GILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVK'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'GVNLPGAAVDLPAVSEK'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'LFIGGLNTETNEK'), ('20190621_QX4_JoMu_MA_HeLa_500ng_190621161214', 'DDDIAALVVDNGSGMCK'), ('20190624_QE4_nLC12_MM_QC_MNT_HELA_02', 'IGGVQQDTILAEGLHFR'), ('20190624_QX3_MaMu_MA_Hela_500ng_LC15', 'DPVQEAWAEDVDLR'), ('20190624_QX4_JiYu_MA_HeLa_500ng', 'DPVQEAWAEDVDLR'), ('20190625_QE6_LC4_AS_QC_MNT_HeLa_02', 'EEEIAALVIDNGSGMCK'), ('20190627_QE6_LC4_AS_QC_MNT_HeLa_02', 'LVSIGAEEIVDGNAK'), ('20190627_QX0_AnBr_MA_HeLa_500ng_LC07_01', 'GYSFTTTAER'), ('20190701_QE4_LC12_IAH_QC_MNT_HeLa_01', 'VFQFLNAK'), ('20190708_QX7_MaMu_MA_HeLa_Br14_500ng', 'IGEEQSAEDAEDGPPELLFIHGGHTAK'), ('20190712_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'VLALPEPSPAAPTLR'), ('20190715_QE4_LC12_IAH_QC_MNT_HeLa_03', 'LFIGGLNTETNEK'), ('20190717_QX3_OzKa_MA_Hela_500ng_LC15_190720214645', 'QLFHPEQLITGK'), ('20190717_QX3_OzKa_MA_Hela_500ng_LC15_190721144939', 'QLFHPEQLITGK'), ('20190720_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_3rdcolumn_3', 'VFQFLNAK'), ('20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar_3', 'LNSVQSSERPLFLVHPIEGSTTVFHSLASR'), ('20190725_QX2_MePh_MA_HeLa_500ng', 'SYSPYDMLESIRK'), ('20190726_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_4thcolumn_1', 'SYSPYDMLESIRK'), ('20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01', 'VFQFLNAK'), ('20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02', 'VFQFLNAK'), ('20190731_QE8_nLC14_ASD_QC_MNT_HeLa_03', 'LYSVSYLLK'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'FWEVISDEHGIDPTGTYHGDSDLQLDR'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'HGVVPLATYMR')]
    




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
      <td>0.508</td>
      <td>0.563</td>
      <td>1.530</td>
      <td>1.888</td>
      <td>2.028</td>
      <td>2.132</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.464</td>
      <td>0.497</td>
      <td>0.846</td>
      <td>1.023</td>
      <td>1.060</td>
      <td>1.043</td>
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
