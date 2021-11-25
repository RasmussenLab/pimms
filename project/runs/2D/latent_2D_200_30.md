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
    MLDAEDIVNTARPDEK                     992
    IIGLDQVAGMSETALPGAFK                 992
    FNADEFEDMVAEK                        991
    DHENIVIAK                            986
    SLTNDWEDHLAVK                      1,000
    VHIGQVIMSIR                          950
    GNPTVEVDLFTSK                        976
    LYSVSYLLK                            961
    STNGDTFLGGEDFDQALLR                  988
    LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR     968
    MVNHFIAEFK                         1,000
    AVTEQGAELSNEER                       999
    LNFSHGTHEYHAETIK                     999
    TTFDEAMADLHTLSEDSYK                  997
    AGGAAVVITEPEHTK                      989
    SGDAAIVDMVPGKPMCVESFSDYPPLGR         969
    HGESAWNLENR                          986
    YYVTIIDAPGHR                         982
    DNLAEDIMR                            951
    SLHDALCVIR                           995
    LLETTDRPDGHQNNLR                     999
    GLVEPVDVVDNADGTQTVNYVPSR             948
    FDLMYAK                              983
    VQASLAANTFTITGHAETK                  987
    AVFVDLEPTVIDEVR                      953
    VGLQVVAVK                            969
    HTNYTMEHIR                           987
    TICSHVQNMIK                          992
    NNQFQALLQYADPVSAQHAK                 989
    TCTTVAFTQVNSEDK                    1,000
    GPDGLTAFEATDNQAIK                    985
    SGGMSNELNNIISR                       981
    LALVTGGEIASTFDHPELVK                 997
    STTTGHLIYK                           994
    ILNIFGVIK                            965
    DALSDLALHFLNK                        991
    FQSSHHPTDITSLDQYVER                  998
    VIVVGNPANTNCLTASK                    966
    MLVLDEADEMLNK                        983
    GAEAANVTGPGGVPVQGSK                  988
    PSQMEHAMETMMFTFHK                    990
    LVIITAGAR                            998
    VSVHVIEGDHR                          996
    TSRPENAIIYNNNEDFQVGQAK             1,000
    MDATANDVPSPYEVR                      998
    VSHLLGINVTDFTR                       998
    KYEQGFITDPVVLSPK                     998
    IGEEQSAEDAEDGPPELLFIHGGHTAK          979
    IFAPNHVVAK                           994
    YMACCLLYR                            958
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
      <th>MLDAEDIVNTARPDEK</th>
      <td>29.573</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>28.800</td>
    </tr>
    <tr>
      <th>FNADEFEDMVAEK</th>
      <td>28.614</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>VSHLLGINVTDFTR</th>
      <td>31.104</td>
    </tr>
    <tr>
      <th>KYEQGFITDPVVLSPK</th>
      <td>29.023</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
    </tr>
    <tr>
      <th>IFAPNHVVAK</th>
      <td>30.048</td>
    </tr>
    <tr>
      <th>YMACCLLYR</th>
      <td>28.618</td>
    </tr>
  </tbody>
</table>
<p>49235 rows × 1 columns</p>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_24_1.png)
    



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
      <td>0.985</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.023</td>
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
      <th>MLDAEDIVNTARPDEK</th>
      <td>29.573</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>28.800</td>
    </tr>
    <tr>
      <th>FNADEFEDMVAEK</th>
      <td>28.614</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>VSHLLGINVTDFTR</th>
      <td>31.104</td>
    </tr>
    <tr>
      <th>KYEQGFITDPVVLSPK</th>
      <td>29.023</td>
    </tr>
    <tr>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>28.798</td>
    </tr>
    <tr>
      <th>IFAPNHVVAK</th>
      <td>30.048</td>
    </tr>
    <tr>
      <th>YMACCLLYR</th>
      <td>28.618</td>
    </tr>
  </tbody>
</table>
<p>49235 rows × 1 columns</p>
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
      <th>20190411_QE6_LC6_AS_QC_MNT_HeLa_03</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>28.687</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>29.714</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_03</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>25.457</td>
    </tr>
    <tr>
      <th>20190528_QX8_MiWi_MA_HeLa_BR14_500ng</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>28.428</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>28.913</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190306_QE2_NLC1_AB_MNT_HELA_03</th>
      <th>YYVTIIDAPGHR</th>
      <td>26.896</td>
    </tr>
    <tr>
      <th>20190717_QX3_OzKa_MA_Hela_500ng_LC15_190720214645</th>
      <th>YYVTIIDAPGHR</th>
      <td>33.408</td>
    </tr>
    <tr>
      <th>20190524_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <th>YYVTIIDAPGHR</th>
      <td>29.869</td>
    </tr>
    <tr>
      <th>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04</th>
      <th>YYVTIIDAPGHR</th>
      <td>28.439</td>
    </tr>
    <tr>
      <th>20190611_QE4_LC12_JE_QC_MNT_HeLa_02</th>
      <th>YYVTIIDAPGHR</th>
      <td>26.098</td>
    </tr>
  </tbody>
</table>
<p>44308 rows × 1 columns</p>
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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>27.754</td>
      <td>29.058</td>
      <td>28.822</td>
      <td>28.604</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.989</td>
      <td>30.039</td>
      <td>29.301</td>
    </tr>
    <tr>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>28.313</td>
      <td>27.177</td>
      <td>27.190</td>
      <td>28.079</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
      <td>32.283</td>
      <td>32.237</td>
      <td>31.237</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>IFAPNHVVAK</th>
      <td>30.090</td>
      <td>30.048</td>
      <td>30.259</td>
      <td>29.849</td>
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
      <th>FDLMYAK</th>
      <td>31.384</td>
      <td>30.820</td>
      <td>30.909</td>
      <td>31.288</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>29.540</td>
      <td>29.706</td>
      <td>29.416</td>
      <td>27.829</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>29.466</td>
      <td>29.640</td>
      <td>29.810</td>
      <td>29.743</td>
    </tr>
    <tr>
      <th>STTTGHLIYK</th>
      <td>31.948</td>
      <td>32.791</td>
      <td>32.792</td>
      <td>32.030</td>
    </tr>
    <tr>
      <th>VGLQVVAVK</th>
      <td>31.861</td>
      <td>31.736</td>
      <td>31.697</td>
      <td>31.722</td>
    </tr>
  </tbody>
</table>
<p>4927 rows × 4 columns</p>
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
      <th>20190104_QE6_nLC6_MM_QC_MNT_HELA_02_190107214303</th>
      <th>VSVHVIEGDHR</th>
      <td>28.828</td>
      <td>29.512</td>
      <td>29.425</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE2_NLC10_ANHO_QC_MNT_HELA_03</th>
      <th>VSHLLGINVTDFTR</th>
      <td>30.128</td>
      <td>30.099</td>
      <td>29.895</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190118_QE1_nLC2_ANHO_QC_MNT_HELA_03</th>
      <th>LLETTDRPDGHQNNLR</th>
      <td>27.720</td>
      <td>29.883</td>
      <td>29.976</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03</th>
      <th>LLETTDRPDGHQNNLR</th>
      <td>29.030</td>
      <td>29.883</td>
      <td>29.976</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190129_QE1_nLC2_GP_QC_MNT_HELA_02</th>
      <th>HGESAWNLENR</th>
      <td>28.317</td>
      <td>28.279</td>
      <td>28.662</td>
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
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>27.036</td>
      <td>27.177</td>
      <td>27.190</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02</th>
      <th>VQASLAANTFTITGHAETK</th>
      <td>30.215</td>
      <td>29.910</td>
      <td>29.939</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05</th>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>28.323</td>
      <td>29.706</td>
      <td>29.416</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2</th>
      <th>TICSHVQNMIK</th>
      <td>29.297</td>
      <td>30.312</td>
      <td>30.202</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_04</th>
      <th>MVNHFIAEFK</th>
      <td>33.057</td>
      <td>31.176</td>
      <td>31.337</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 4 columns</p>
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
      <td>AVFVDLEPTVIDEVR</td>
      <td>32.479</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVTEQGAELSNEER</td>
      <td>28.774</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DALSDLALHFLNK</td>
      <td>29.888</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DNLAEDIMR</td>
      <td>28.755</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>FDLMYAK</td>
      <td>31.085</td>
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
      <td>AGGAAVVITEPEHTK</td>
      <td>27.754</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DHENIVIAK</td>
      <td>29.417</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>GLVEPVDVVDNADGTQTVNYVPSR</td>
      <td>28.313</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>SLTNDWEDHLAVK</td>
      <td>30.814</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>IFAPNHVVAK</td>
      <td>30.090</td>
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
      <td>20190108_QE1_nLC2_MB_QC_MNT_HELA_old_01</td>
      <td>AVFVDLEPTVIDEVR</td>
      <td>31.767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190624_QE4_nLC12_MM_QC_MNT_HELA_01</td>
      <td>MLVLDEADEMLNK</td>
      <td>28.477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190527_QX1_PhGe_MA_HeLa_500ng_LC10</td>
      <td>VSHLLGINVTDFTR</td>
      <td>30.441</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190409_QE1_nLC2_ANHO_MNT_QC_hela_01</td>
      <td>HGESAWNLENR</td>
      <td>27.646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190225_QE10_PhGe_Evosep_88min_HeLa_5</td>
      <td>MVNHFIAEFK</td>
      <td>30.821</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190717_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05_190719190656</td>
      <td>VSHLLGINVTDFTR</td>
      <td>29.782</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190104_QE6_nLC6_MM_QC_MNT_HELA_01_190107120859</td>
      <td>SGGMSNELNNIISR</td>
      <td>29.976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190208_QE2_NLC1_AB_QC_MNT_HELA_1</td>
      <td>TTFDEAMADLHTLSEDSYK</td>
      <td>29.649</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_01</td>
      <td>LALVTGGEIASTFDHPELVK</td>
      <td>30.791</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190710_QE1_nLC13_ANHO_QC_MNT_HELA_01</td>
      <td>VQASLAANTFTITGHAETK</td>
      <td>28.333</td>
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
      <td>20190510_QE1_nLC2_ANHO_QC_MNT_HELA_06</td>
      <td>DHENIVIAK</td>
      <td>29.504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190201_QE9_nLC9_NHS_MNT_HELA_45cm_04</td>
      <td>VSHLLGINVTDFTR</td>
      <td>29.656</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190802_QX6_MaTa_MA_HeLa_500ng_LC09_20190803134200</td>
      <td>LNFSHGTHEYHAETIK</td>
      <td>33.078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190128_QE7_nLC7_MEM_QC_MNT_HeLa_03</td>
      <td>MDATANDVPSPYEVR</td>
      <td>27.289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190501_QX8_MiWi_MA_HeLa_500ng_new</td>
      <td>HGESAWNLENR</td>
      <td>30.340</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190709_QE1_nLC13_ANHO_QC_MNT_HELA_02</td>
      <td>NNQFQALLQYADPVSAQHAK</td>
      <td>30.232</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190709_QX2_JoMu_MA_HeLa_500ng_LC05</td>
      <td>VSVHVIEGDHR</td>
      <td>30.335</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190111_QE8_nLC1_ASD_QC_HeLa_02</td>
      <td>IGEEQSAEDAEDGPPELLFIHGGHTAK</td>
      <td>25.401</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190224_QE2_NLC1_GP_MNT_HeLa_1</td>
      <td>LVIITAGAR</td>
      <td>31.318</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190104_QE6_nLC6_MM_QC_MNT_HELA_01_190107120859</td>
      <td>SLHDALCVIR</td>
      <td>29.878</td>
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
    
    Optimizer used: <function Adam at 0x000002CC64326040>
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
      <td>1.849704</td>
      <td>1.757563</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.854544</td>
      <td>0.818184</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.670191</td>
      <td>0.706633</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.758262</td>
      <td>0.663512</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.736809</td>
      <td>0.637151</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.611611</td>
      <td>0.631592</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.617459</td>
      <td>0.618518</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.536608</td>
      <td>0.610046</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.572971</td>
      <td>0.609424</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.496442</td>
      <td>0.609505</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_58_1.png)
    


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
      <th>2,495</th>
      <td>500</td>
      <td>5</td>
      <td>29.504</td>
    </tr>
    <tr>
      <th>524</th>
      <td>114</td>
      <td>47</td>
      <td>29.656</td>
    </tr>
    <tr>
      <th>4,839</th>
      <td>978</td>
      <td>23</td>
      <td>33.078</td>
    </tr>
    <tr>
      <th>421</th>
      <td>91</td>
      <td>27</td>
      <td>27.289</td>
    </tr>
    <tr>
      <th>2,295</th>
      <td>463</td>
      <td>14</td>
      <td>30.340</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4,607</th>
      <td>934</td>
      <td>5</td>
      <td>29.167</td>
    </tr>
    <tr>
      <th>572</th>
      <td>124</td>
      <td>10</td>
      <td>29.917</td>
    </tr>
    <tr>
      <th>2,892</th>
      <td>587</td>
      <td>24</td>
      <td>26.404</td>
    </tr>
    <tr>
      <th>4,501</th>
      <td>912</td>
      <td>7</td>
      <td>29.542</td>
    </tr>
    <tr>
      <th>2,150</th>
      <td>432</td>
      <td>10</td>
      <td>29.208</td>
    </tr>
  </tbody>
</table>
<p>4927 rows × 3 columns</p>
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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>27.754</td>
      <td>29.058</td>
      <td>28.822</td>
      <td>28.604</td>
      <td>27.712</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.989</td>
      <td>30.039</td>
      <td>29.301</td>
      <td>29.030</td>
    </tr>
    <tr>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>28.313</td>
      <td>27.177</td>
      <td>27.190</td>
      <td>28.079</td>
      <td>26.017</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
      <td>32.283</td>
      <td>32.237</td>
      <td>31.237</td>
      <td>31.594</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>IFAPNHVVAK</th>
      <td>30.090</td>
      <td>30.048</td>
      <td>30.259</td>
      <td>29.849</td>
      <td>29.412</td>
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
      <th>FDLMYAK</th>
      <td>31.384</td>
      <td>30.820</td>
      <td>30.909</td>
      <td>31.288</td>
      <td>30.566</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>29.540</td>
      <td>29.706</td>
      <td>29.416</td>
      <td>27.829</td>
      <td>29.882</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>29.466</td>
      <td>29.640</td>
      <td>29.810</td>
      <td>29.743</td>
      <td>29.312</td>
    </tr>
    <tr>
      <th>STTTGHLIYK</th>
      <td>31.948</td>
      <td>32.791</td>
      <td>32.792</td>
      <td>32.030</td>
      <td>32.499</td>
    </tr>
    <tr>
      <th>VGLQVVAVK</th>
      <td>31.861</td>
      <td>31.736</td>
      <td>31.697</td>
      <td>31.722</td>
      <td>31.689</td>
    </tr>
  </tbody>
</table>
<p>4927 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.083
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.080
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.198
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.115
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.212
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_66_1.png)
    



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
      <td>-0.072</td>
      <td>-0.345</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.134</td>
      <td>-0.337</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.062</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.212</td>
      <td>-0.280</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.235</td>
      <td>0.028</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_68_1.png)
    



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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_69_1.png)
    


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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK</th>
      <th>TTFDEAMADLHTLSEDSYK</th>
      <th>VGLQVVAVK</th>
      <th>VHIGQVIMSIR</th>
      <th>VIVVGNPANTNCLTASK</th>
      <th>VQASLAANTFTITGHAETK</th>
      <th>VSHLLGINVTDFTR</th>
      <th>VSVHVIEGDHR</th>
      <th>YMACCLLYR</th>
      <th>YYVTIIDAPGHR</th>
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
      <td>27.754</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.417</td>
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
      <td>29.115</td>
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
      <td>29.951</td>
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
      <td>29.432</td>
      <td>27.219</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.704</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.200</td>
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
      <td>29.732</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>29.436</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.394</td>
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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_na</th>
      <th>TTFDEAMADLHTLSEDSYK_na</th>
      <th>VGLQVVAVK_na</th>
      <th>VHIGQVIMSIR_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VSHLLGINVTDFTR_na</th>
      <th>VSVHVIEGDHR_na</th>
      <th>YMACCLLYR_na</th>
      <th>YYVTIIDAPGHR_na</th>
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
      <td>0.165</td>
      <td>-0.051</td>
      <td>-0.772</td>
      <td>-0.394</td>
      <td>-0.036</td>
      <td>-0.015</td>
      <td>0.142</td>
      <td>-0.884</td>
      <td>-0.496</td>
      <td>-1.605</td>
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
      <td>-0.449</td>
      <td>-0.056</td>
      <td>-0.653</td>
      <td>-0.655</td>
      <td>-0.305</td>
      <td>-0.080</td>
      <td>0.364</td>
      <td>-1.097</td>
      <td>-0.508</td>
      <td>-1.556</td>
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
      <td>0.067</td>
      <td>-0.091</td>
      <td>-0.778</td>
      <td>-0.777</td>
      <td>-0.882</td>
      <td>-0.047</td>
      <td>-0.627</td>
      <td>-0.478</td>
      <td>-0.692</td>
      <td>-0.069</td>
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
      <td>-0.388</td>
      <td>-0.585</td>
      <td>-0.350</td>
      <td>0.044</td>
      <td>-0.036</td>
      <td>0.271</td>
      <td>-0.223</td>
      <td>-0.984</td>
      <td>-0.116</td>
      <td>-0.677</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-1.217</td>
      <td>0.975</td>
      <td>-0.543</td>
      <td>0.291</td>
      <td>-0.562</td>
      <td>0.346</td>
      <td>0.236</td>
      <td>-0.624</td>
      <td>0.037</td>
      <td>-1.461</td>
      <td>...</td>
      <td>False</td>
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
      <td>0.300</td>
      <td>1.246</td>
      <td>0.650</td>
      <td>0.717</td>
      <td>0.533</td>
      <td>0.812</td>
      <td>0.890</td>
      <td>0.814</td>
      <td>0.776</td>
      <td>0.500</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.154</td>
      <td>-0.330</td>
      <td>0.078</td>
      <td>-0.324</td>
      <td>-0.047</td>
      <td>-0.259</td>
      <td>0.289</td>
      <td>0.199</td>
      <td>-0.086</td>
      <td>0.093</td>
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
      <td>0.307</td>
      <td>0.086</td>
      <td>-0.012</td>
      <td>-0.021</td>
      <td>0.332</td>
      <td>-0.100</td>
      <td>0.290</td>
      <td>0.616</td>
      <td>-0.125</td>
      <td>-0.003</td>
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
      <td>0.535</td>
      <td>-0.023</td>
      <td>-0.036</td>
      <td>0.044</td>
      <td>0.622</td>
      <td>-0.213</td>
      <td>0.304</td>
      <td>0.631</td>
      <td>-0.156</td>
      <td>-0.105</td>
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
      <td>0.867</td>
      <td>-0.029</td>
      <td>-0.031</td>
      <td>0.044</td>
      <td>0.690</td>
      <td>-0.270</td>
      <td>-0.060</td>
      <td>0.658</td>
      <td>0.037</td>
      <td>-0.311</td>
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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_na</th>
      <th>TTFDEAMADLHTLSEDSYK_na</th>
      <th>VGLQVVAVK_na</th>
      <th>VHIGQVIMSIR_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VSHLLGINVTDFTR_na</th>
      <th>VSVHVIEGDHR_na</th>
      <th>YMACCLLYR_na</th>
      <th>YYVTIIDAPGHR_na</th>
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
      <td>-0.021</td>
      <td>-0.742</td>
      <td>-0.367</td>
      <td>-0.038</td>
      <td>-0.022</td>
      <td>0.126</td>
      <td>-0.826</td>
      <td>-0.466</td>
      <td>-1.522</td>
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
      <td>-0.405</td>
      <td>-0.026</td>
      <td>-0.629</td>
      <td>-0.614</td>
      <td>-0.291</td>
      <td>-0.082</td>
      <td>0.335</td>
      <td>-1.026</td>
      <td>-0.477</td>
      <td>-1.475</td>
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
      <td>0.083</td>
      <td>-0.058</td>
      <td>-0.748</td>
      <td>-0.729</td>
      <td>-0.835</td>
      <td>-0.051</td>
      <td>-0.597</td>
      <td>-0.442</td>
      <td>-0.652</td>
      <td>-0.073</td>
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
      <td>-0.347</td>
      <td>-0.516</td>
      <td>-0.342</td>
      <td>0.047</td>
      <td>-0.038</td>
      <td>0.243</td>
      <td>-0.217</td>
      <td>-0.920</td>
      <td>-0.106</td>
      <td>-0.647</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-1.131</td>
      <td>0.931</td>
      <td>-0.525</td>
      <td>0.280</td>
      <td>-0.534</td>
      <td>0.313</td>
      <td>0.215</td>
      <td>-0.579</td>
      <td>0.039</td>
      <td>-1.386</td>
      <td>...</td>
      <td>False</td>
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
      <td>0.303</td>
      <td>1.183</td>
      <td>0.607</td>
      <td>0.682</td>
      <td>0.498</td>
      <td>0.744</td>
      <td>0.829</td>
      <td>0.778</td>
      <td>0.739</td>
      <td>0.464</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.165</td>
      <td>-0.280</td>
      <td>0.064</td>
      <td>-0.301</td>
      <td>-0.049</td>
      <td>-0.247</td>
      <td>0.264</td>
      <td>0.197</td>
      <td>-0.077</td>
      <td>0.079</td>
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
      <td>0.310</td>
      <td>0.107</td>
      <td>-0.022</td>
      <td>-0.015</td>
      <td>0.308</td>
      <td>-0.100</td>
      <td>0.265</td>
      <td>0.591</td>
      <td>-0.114</td>
      <td>-0.011</td>
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
      <td>0.525</td>
      <td>0.005</td>
      <td>-0.044</td>
      <td>0.047</td>
      <td>0.582</td>
      <td>-0.205</td>
      <td>0.279</td>
      <td>0.606</td>
      <td>-0.144</td>
      <td>-0.107</td>
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
      <td>0.839</td>
      <td>-0.000</td>
      <td>-0.039</td>
      <td>0.047</td>
      <td>0.645</td>
      <td>-0.257</td>
      <td>-0.064</td>
      <td>0.631</td>
      <td>0.039</td>
      <td>-0.302</td>
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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
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
      <td>0.019</td>
      <td>0.026</td>
      <td>-0.010</td>
      <td>0.005</td>
      <td>-0.004</td>
      <td>-0.007</td>
      <td>-0.007</td>
      <td>0.010</td>
      <td>0.004</td>
      <td>-0.008</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.945</td>
      <td>0.929</td>
      <td>0.949</td>
      <td>0.945</td>
      <td>0.942</td>
      <td>0.925</td>
      <td>0.941</td>
      <td>0.945</td>
      <td>0.948</td>
      <td>0.943</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.244</td>
      <td>-5.201</td>
      <td>-4.126</td>
      <td>-4.185</td>
      <td>-3.990</td>
      <td>-3.636</td>
      <td>-4.195</td>
      <td>-4.640</td>
      <td>-5.399</td>
      <td>-3.676</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.304</td>
      <td>-0.284</td>
      <td>-0.521</td>
      <td>-0.440</td>
      <td>-0.416</td>
      <td>-0.378</td>
      <td>-0.491</td>
      <td>-0.417</td>
      <td>-0.452</td>
      <td>-0.508</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.175</td>
      <td>0.184</td>
      <td>-0.099</td>
      <td>0.047</td>
      <td>-0.038</td>
      <td>-0.051</td>
      <td>-0.064</td>
      <td>0.089</td>
      <td>0.039</td>
      <td>-0.073</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.564</td>
      <td>0.538</td>
      <td>0.533</td>
      <td>0.638</td>
      <td>0.548</td>
      <td>0.441</td>
      <td>0.514</td>
      <td>0.553</td>
      <td>0.483</td>
      <td>0.497</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.854</td>
      <td>1.772</td>
      <td>2.257</td>
      <td>2.011</td>
      <td>2.446</td>
      <td>2.177</td>
      <td>1.985</td>
      <td>2.058</td>
      <td>2.115</td>
      <td>2.571</td>
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




    ((#50) ['AGGAAVVITEPEHTK','AVFVDLEPTVIDEVR','AVTEQGAELSNEER','DALSDLALHFLNK','DHENIVIAK','DNLAEDIMR','FDLMYAK','FNADEFEDMVAEK','FQSSHHPTDITSLDQYVER','GAEAANVTGPGGVPVQGSK'...],
     (#50) ['AGGAAVVITEPEHTK_na','AVFVDLEPTVIDEVR_na','AVTEQGAELSNEER_na','DALSDLALHFLNK_na','DHENIVIAK_na','DNLAEDIMR_na','FDLMYAK_na','FNADEFEDMVAEK_na','FQSSHHPTDITSLDQYVER_na','GAEAANVTGPGGVPVQGSK_na'...])




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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.094</td>
      <td>-0.039</td>
      <td>-0.014</td>
      <td>-0.075</td>
      <td>-0.141</td>
      <td>0.212</td>
      <td>0.160</td>
      <td>-0.005</td>
      <td>0.040</td>
      <td>-0.020</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.987</td>
      <td>0.912</td>
      <td>0.990</td>
      <td>1.020</td>
      <td>1.091</td>
      <td>0.997</td>
      <td>1.065</td>
      <td>1.071</td>
      <td>1.027</td>
      <td>1.051</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.244</td>
      <td>-3.852</td>
      <td>-3.315</td>
      <td>-2.591</td>
      <td>-4.589</td>
      <td>-3.109</td>
      <td>-4.211</td>
      <td>-3.127</td>
      <td>-4.934</td>
      <td>-2.768</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.650</td>
      <td>-0.457</td>
      <td>-0.647</td>
      <td>-0.643</td>
      <td>-0.613</td>
      <td>-0.262</td>
      <td>-0.447</td>
      <td>-0.455</td>
      <td>-0.392</td>
      <td>-0.656</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.107</td>
      <td>0.069</td>
      <td>-0.142</td>
      <td>0.049</td>
      <td>-0.114</td>
      <td>0.284</td>
      <td>0.056</td>
      <td>-0.024</td>
      <td>0.089</td>
      <td>-0.020</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.683</td>
      <td>0.529</td>
      <td>0.668</td>
      <td>0.448</td>
      <td>0.412</td>
      <td>1.006</td>
      <td>0.997</td>
      <td>0.764</td>
      <td>0.734</td>
      <td>0.702</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.750</td>
      <td>1.663</td>
      <td>2.132</td>
      <td>1.986</td>
      <td>1.945</td>
      <td>1.665</td>
      <td>1.897</td>
      <td>1.867</td>
      <td>1.788</td>
      <td>2.344</td>
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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_val</th>
      <th>TTFDEAMADLHTLSEDSYK_val</th>
      <th>VGLQVVAVK_val</th>
      <th>VHIGQVIMSIR_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VSHLLGINVTDFTR_val</th>
      <th>VSVHVIEGDHR_val</th>
      <th>YMACCLLYR_val</th>
      <th>YYVTIIDAPGHR_val</th>
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
      <td>-0.021</td>
      <td>-0.742</td>
      <td>-0.367</td>
      <td>-0.038</td>
      <td>-0.022</td>
      <td>0.126</td>
      <td>-0.826</td>
      <td>-0.466</td>
      <td>-1.522</td>
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
      <td>-0.405</td>
      <td>-0.026</td>
      <td>-0.629</td>
      <td>-0.614</td>
      <td>-0.291</td>
      <td>-0.082</td>
      <td>0.335</td>
      <td>-1.026</td>
      <td>-0.477</td>
      <td>-1.475</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.083</td>
      <td>-0.058</td>
      <td>-0.748</td>
      <td>-0.729</td>
      <td>-0.835</td>
      <td>-0.051</td>
      <td>-0.597</td>
      <td>-0.442</td>
      <td>-0.652</td>
      <td>-0.073</td>
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
      <td>-0.347</td>
      <td>-0.516</td>
      <td>-0.342</td>
      <td>0.047</td>
      <td>-0.038</td>
      <td>0.243</td>
      <td>-0.217</td>
      <td>-0.920</td>
      <td>-0.106</td>
      <td>-0.647</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.352</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-1.131</td>
      <td>0.931</td>
      <td>-0.525</td>
      <td>0.280</td>
      <td>-0.534</td>
      <td>0.313</td>
      <td>0.215</td>
      <td>-0.579</td>
      <td>0.039</td>
      <td>-1.386</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.292</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.732</td>
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
      <td>0.303</td>
      <td>1.183</td>
      <td>0.607</td>
      <td>0.682</td>
      <td>0.498</td>
      <td>0.744</td>
      <td>0.829</td>
      <td>0.778</td>
      <td>0.739</td>
      <td>0.464</td>
      <td>...</td>
      <td>0.803</td>
      <td>0.892</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.035</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.165</td>
      <td>-0.280</td>
      <td>0.064</td>
      <td>-0.301</td>
      <td>-0.049</td>
      <td>-0.247</td>
      <td>0.264</td>
      <td>0.197</td>
      <td>-0.077</td>
      <td>0.079</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.015</td>
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
      <td>0.310</td>
      <td>0.107</td>
      <td>-0.022</td>
      <td>-0.015</td>
      <td>0.308</td>
      <td>-0.100</td>
      <td>0.265</td>
      <td>0.591</td>
      <td>-0.114</td>
      <td>-0.011</td>
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
      <td>0.525</td>
      <td>0.005</td>
      <td>-0.044</td>
      <td>0.047</td>
      <td>0.582</td>
      <td>-0.205</td>
      <td>0.279</td>
      <td>0.606</td>
      <td>-0.144</td>
      <td>-0.107</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.992</td>
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
      <td>0.839</td>
      <td>-0.000</td>
      <td>-0.039</td>
      <td>0.047</td>
      <td>0.645</td>
      <td>-0.257</td>
      <td>-0.064</td>
      <td>0.631</td>
      <td>0.039</td>
      <td>-0.302</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.144</td>
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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_val</th>
      <th>TTFDEAMADLHTLSEDSYK_val</th>
      <th>VGLQVVAVK_val</th>
      <th>VHIGQVIMSIR_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VSHLLGINVTDFTR_val</th>
      <th>VSVHVIEGDHR_val</th>
      <th>YMACCLLYR_val</th>
      <th>YYVTIIDAPGHR_val</th>
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
      <td>-0.021</td>
      <td>-0.742</td>
      <td>-0.367</td>
      <td>-0.038</td>
      <td>-0.022</td>
      <td>0.126</td>
      <td>-0.826</td>
      <td>-0.466</td>
      <td>-1.522</td>
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
      <td>-0.405</td>
      <td>-0.026</td>
      <td>-0.629</td>
      <td>-0.614</td>
      <td>-0.291</td>
      <td>-0.082</td>
      <td>0.335</td>
      <td>-1.026</td>
      <td>-0.477</td>
      <td>-1.475</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.083</td>
      <td>-0.058</td>
      <td>-0.748</td>
      <td>-0.729</td>
      <td>-0.835</td>
      <td>-0.051</td>
      <td>-0.597</td>
      <td>-0.442</td>
      <td>-0.652</td>
      <td>-0.073</td>
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
      <td>-0.347</td>
      <td>-0.516</td>
      <td>-0.342</td>
      <td>0.047</td>
      <td>-0.038</td>
      <td>0.243</td>
      <td>-0.217</td>
      <td>-0.920</td>
      <td>-0.106</td>
      <td>-0.647</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.352</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-1.131</td>
      <td>0.931</td>
      <td>-0.525</td>
      <td>0.280</td>
      <td>-0.534</td>
      <td>0.313</td>
      <td>0.215</td>
      <td>-0.579</td>
      <td>0.039</td>
      <td>-1.386</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.292</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.732</td>
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
      <td>0.303</td>
      <td>1.183</td>
      <td>0.607</td>
      <td>0.682</td>
      <td>0.498</td>
      <td>0.744</td>
      <td>0.829</td>
      <td>0.778</td>
      <td>0.739</td>
      <td>0.464</td>
      <td>...</td>
      <td>0.803</td>
      <td>0.892</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.035</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.165</td>
      <td>-0.280</td>
      <td>0.064</td>
      <td>-0.301</td>
      <td>-0.049</td>
      <td>-0.247</td>
      <td>0.264</td>
      <td>0.197</td>
      <td>-0.077</td>
      <td>0.079</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.015</td>
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
      <td>0.310</td>
      <td>0.107</td>
      <td>-0.022</td>
      <td>-0.015</td>
      <td>0.308</td>
      <td>-0.100</td>
      <td>0.265</td>
      <td>0.591</td>
      <td>-0.114</td>
      <td>-0.011</td>
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
      <td>0.525</td>
      <td>0.005</td>
      <td>-0.044</td>
      <td>0.047</td>
      <td>0.582</td>
      <td>-0.205</td>
      <td>0.279</td>
      <td>0.606</td>
      <td>-0.144</td>
      <td>-0.107</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.992</td>
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
      <td>0.839</td>
      <td>-0.000</td>
      <td>-0.039</td>
      <td>0.047</td>
      <td>0.645</td>
      <td>-0.257</td>
      <td>-0.064</td>
      <td>0.631</td>
      <td>0.039</td>
      <td>-0.302</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.144</td>
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
      <th>AGGAAVVITEPEHTK_val</th>
      <th>AVFVDLEPTVIDEVR_val</th>
      <th>AVTEQGAELSNEER_val</th>
      <th>DALSDLALHFLNK_val</th>
      <th>DHENIVIAK_val</th>
      <th>DNLAEDIMR_val</th>
      <th>FDLMYAK_val</th>
      <th>FNADEFEDMVAEK_val</th>
      <th>FQSSHHPTDITSLDQYVER_val</th>
      <th>GAEAANVTGPGGVPVQGSK_val</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_val</th>
      <th>TTFDEAMADLHTLSEDSYK_val</th>
      <th>VGLQVVAVK_val</th>
      <th>VHIGQVIMSIR_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VQASLAANTFTITGHAETK_val</th>
      <th>VSHLLGINVTDFTR_val</th>
      <th>VSVHVIEGDHR_val</th>
      <th>YMACCLLYR_val</th>
      <th>YYVTIIDAPGHR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>...</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>95.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>96.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.094</td>
      <td>-0.039</td>
      <td>-0.014</td>
      <td>-0.075</td>
      <td>-0.141</td>
      <td>0.212</td>
      <td>0.160</td>
      <td>-0.005</td>
      <td>0.040</td>
      <td>-0.020</td>
      <td>...</td>
      <td>0.110</td>
      <td>-0.082</td>
      <td>0.111</td>
      <td>-0.049</td>
      <td>0.085</td>
      <td>-0.101</td>
      <td>-0.030</td>
      <td>-0.040</td>
      <td>-0.043</td>
      <td>-0.152</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.987</td>
      <td>0.912</td>
      <td>0.990</td>
      <td>1.020</td>
      <td>1.091</td>
      <td>0.997</td>
      <td>1.065</td>
      <td>1.071</td>
      <td>1.027</td>
      <td>1.051</td>
      <td>...</td>
      <td>0.942</td>
      <td>1.118</td>
      <td>1.008</td>
      <td>1.079</td>
      <td>1.000</td>
      <td>1.101</td>
      <td>1.153</td>
      <td>0.950</td>
      <td>1.135</td>
      <td>0.926</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.244</td>
      <td>-3.852</td>
      <td>-3.315</td>
      <td>-2.591</td>
      <td>-4.589</td>
      <td>-3.109</td>
      <td>-4.211</td>
      <td>-3.127</td>
      <td>-4.934</td>
      <td>-2.768</td>
      <td>...</td>
      <td>-3.822</td>
      <td>-4.695</td>
      <td>-5.664</td>
      <td>-3.616</td>
      <td>-3.882</td>
      <td>-3.759</td>
      <td>-6.194</td>
      <td>-3.030</td>
      <td>-4.200</td>
      <td>-1.827</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.650</td>
      <td>-0.457</td>
      <td>-0.647</td>
      <td>-0.643</td>
      <td>-0.613</td>
      <td>-0.262</td>
      <td>-0.447</td>
      <td>-0.455</td>
      <td>-0.392</td>
      <td>-0.656</td>
      <td>...</td>
      <td>-0.345</td>
      <td>-0.473</td>
      <td>-0.369</td>
      <td>-0.354</td>
      <td>-0.353</td>
      <td>-0.531</td>
      <td>-0.499</td>
      <td>-0.533</td>
      <td>-0.593</td>
      <td>-0.749</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.107</td>
      <td>0.069</td>
      <td>-0.142</td>
      <td>0.049</td>
      <td>-0.114</td>
      <td>0.284</td>
      <td>0.056</td>
      <td>-0.024</td>
      <td>0.089</td>
      <td>-0.020</td>
      <td>...</td>
      <td>0.242</td>
      <td>0.178</td>
      <td>0.163</td>
      <td>0.192</td>
      <td>0.183</td>
      <td>0.052</td>
      <td>0.145</td>
      <td>-0.007</td>
      <td>0.004</td>
      <td>-0.416</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.683</td>
      <td>0.529</td>
      <td>0.668</td>
      <td>0.448</td>
      <td>0.412</td>
      <td>1.006</td>
      <td>0.997</td>
      <td>0.764</td>
      <td>0.734</td>
      <td>0.702</td>
      <td>...</td>
      <td>0.785</td>
      <td>0.634</td>
      <td>0.767</td>
      <td>0.661</td>
      <td>0.813</td>
      <td>0.504</td>
      <td>0.613</td>
      <td>0.638</td>
      <td>0.654</td>
      <td>0.687</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.750</td>
      <td>1.663</td>
      <td>2.132</td>
      <td>1.986</td>
      <td>1.945</td>
      <td>1.665</td>
      <td>1.897</td>
      <td>1.867</td>
      <td>1.788</td>
      <td>2.344</td>
      <td>...</td>
      <td>1.640</td>
      <td>1.797</td>
      <td>1.588</td>
      <td>1.524</td>
      <td>1.661</td>
      <td>1.667</td>
      <td>1.827</td>
      <td>2.371</td>
      <td>2.286</td>
      <td>1.705</td>
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
      <th>AGGAAVVITEPEHTK_na</th>
      <th>AVFVDLEPTVIDEVR_na</th>
      <th>AVTEQGAELSNEER_na</th>
      <th>DALSDLALHFLNK_na</th>
      <th>DHENIVIAK_na</th>
      <th>DNLAEDIMR_na</th>
      <th>FDLMYAK_na</th>
      <th>FNADEFEDMVAEK_na</th>
      <th>FQSSHHPTDITSLDQYVER_na</th>
      <th>GAEAANVTGPGGVPVQGSK_na</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK_na</th>
      <th>TTFDEAMADLHTLSEDSYK_na</th>
      <th>VGLQVVAVK_na</th>
      <th>VHIGQVIMSIR_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VQASLAANTFTITGHAETK_na</th>
      <th>VSHLLGINVTDFTR_na</th>
      <th>VSVHVIEGDHR_na</th>
      <th>YMACCLLYR_na</th>
      <th>YYVTIIDAPGHR_na</th>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
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
      <td>False</td>
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
    
    Optimizer used: <function Adam at 0x000002CC64326040>
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








    SuggestedLRs(valley=0.02290867641568184)




    
![png](latent_2D_200_30_files/latent_2D_200_30_108_2.png)
    


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
      <td>0.941496</td>
      <td>0.676139</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.617716</td>
      <td>0.377314</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.476575</td>
      <td>0.362470</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.411938</td>
      <td>0.350412</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.378515</td>
      <td>0.342696</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.351985</td>
      <td>0.334893</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.337969</td>
      <td>0.326139</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.326436</td>
      <td>0.325405</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.316746</td>
      <td>0.324413</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.310387</td>
      <td>0.318317</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4927, 4927)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3187)




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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>27.754</td>
      <td>29.058</td>
      <td>28.822</td>
      <td>28.604</td>
      <td>27.712</td>
      <td>27.961</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.989</td>
      <td>30.039</td>
      <td>29.301</td>
      <td>29.030</td>
      <td>28.897</td>
    </tr>
    <tr>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>28.313</td>
      <td>27.177</td>
      <td>27.190</td>
      <td>28.079</td>
      <td>26.017</td>
      <td>25.912</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
      <td>32.283</td>
      <td>32.237</td>
      <td>31.237</td>
      <td>31.594</td>
      <td>31.165</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>IFAPNHVVAK</th>
      <td>30.090</td>
      <td>30.048</td>
      <td>30.259</td>
      <td>29.849</td>
      <td>29.412</td>
      <td>29.261</td>
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
      <th>FDLMYAK</th>
      <td>31.384</td>
      <td>30.820</td>
      <td>30.909</td>
      <td>31.288</td>
      <td>30.566</td>
      <td>30.602</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>29.540</td>
      <td>29.706</td>
      <td>29.416</td>
      <td>27.829</td>
      <td>29.882</td>
      <td>29.879</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>29.466</td>
      <td>29.640</td>
      <td>29.810</td>
      <td>29.743</td>
      <td>29.312</td>
      <td>29.345</td>
    </tr>
    <tr>
      <th>STTTGHLIYK</th>
      <td>31.948</td>
      <td>32.791</td>
      <td>32.792</td>
      <td>32.030</td>
      <td>32.499</td>
      <td>32.481</td>
    </tr>
    <tr>
      <th>VGLQVVAVK</th>
      <td>31.861</td>
      <td>31.736</td>
      <td>31.697</td>
      <td>31.722</td>
      <td>31.689</td>
      <td>31.688</td>
    </tr>
  </tbody>
</table>
<p>4927 rows × 6 columns</p>
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
      <td>-0.502</td>
      <td>0.796</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.625</td>
      <td>0.789</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.683</td>
      <td>0.858</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.261</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.699</td>
      <td>0.761</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_122_1.png)
    



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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_123_1.png)
    


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
      <th>AGGAAVVITEPEHTK</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVTEQGAELSNEER</th>
      <th>DALSDLALHFLNK</th>
      <th>DHENIVIAK</th>
      <th>DNLAEDIMR</th>
      <th>FDLMYAK</th>
      <th>FNADEFEDMVAEK</th>
      <th>FQSSHHPTDITSLDQYVER</th>
      <th>GAEAANVTGPGGVPVQGSK</th>
      <th>...</th>
      <th>TSRPENAIIYNNNEDFQVGQAK</th>
      <th>TTFDEAMADLHTLSEDSYK</th>
      <th>VGLQVVAVK</th>
      <th>VHIGQVIMSIR</th>
      <th>VIVVGNPANTNCLTASK</th>
      <th>VQASLAANTFTITGHAETK</th>
      <th>VSHLLGINVTDFTR</th>
      <th>VSVHVIEGDHR</th>
      <th>YMACCLLYR</th>
      <th>YYVTIIDAPGHR</th>
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
      <td>0.566</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.546</td>
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
      <td>0.642</td>
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
      <td>0.611</td>
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
      <td>0.563</td>
      <td>0.286</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.575</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.673</td>
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
      <td>0.695</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.726</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.620</td>
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
    
    Optimizer used: <function Adam at 0x000002CC64326040>
    Loss function: <function loss_fct_vae at 0x000002CC64344940>
    
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




    
![png](latent_2D_200_30_files/latent_2D_200_30_136_2.png)
    



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
      <td>1969.880005</td>
      <td>214.994476</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1944.492798</td>
      <td>209.436111</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1901.425537</td>
      <td>200.571487</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1860.161255</td>
      <td>195.453491</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1828.647583</td>
      <td>193.953552</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1805.996338</td>
      <td>193.515030</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1789.371948</td>
      <td>193.294754</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1778.342896</td>
      <td>193.148972</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1770.215942</td>
      <td>193.127670</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1764.928955</td>
      <td>193.119720</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4927)




```python
len(pred[0])
```




    4927




```python
learn.loss_func(pred, target)
```




    tensor(3041.4971)




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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>AGGAAVVITEPEHTK</th>
      <td>27.754</td>
      <td>29.058</td>
      <td>28.822</td>
      <td>28.604</td>
      <td>27.712</td>
      <td>27.961</td>
      <td>28.905</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.989</td>
      <td>30.039</td>
      <td>29.301</td>
      <td>29.030</td>
      <td>28.897</td>
      <td>30.090</td>
    </tr>
    <tr>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>28.313</td>
      <td>27.177</td>
      <td>27.190</td>
      <td>28.079</td>
      <td>26.017</td>
      <td>25.912</td>
      <td>27.285</td>
    </tr>
    <tr>
      <th>SLTNDWEDHLAVK</th>
      <td>30.814</td>
      <td>32.283</td>
      <td>32.237</td>
      <td>31.237</td>
      <td>31.594</td>
      <td>31.165</td>
      <td>32.313</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>IFAPNHVVAK</th>
      <td>30.090</td>
      <td>30.048</td>
      <td>30.259</td>
      <td>29.849</td>
      <td>29.412</td>
      <td>29.261</td>
      <td>30.300</td>
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
      <th>FDLMYAK</th>
      <td>31.384</td>
      <td>30.820</td>
      <td>30.909</td>
      <td>31.288</td>
      <td>30.566</td>
      <td>30.602</td>
      <td>30.996</td>
    </tr>
    <tr>
      <th>IIGLDQVAGMSETALPGAFK</th>
      <td>29.540</td>
      <td>29.706</td>
      <td>29.416</td>
      <td>27.829</td>
      <td>29.882</td>
      <td>29.879</td>
      <td>29.515</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>29.466</td>
      <td>29.640</td>
      <td>29.810</td>
      <td>29.743</td>
      <td>29.312</td>
      <td>29.345</td>
      <td>29.863</td>
    </tr>
    <tr>
      <th>STTTGHLIYK</th>
      <td>31.948</td>
      <td>32.791</td>
      <td>32.792</td>
      <td>32.030</td>
      <td>32.499</td>
      <td>32.481</td>
      <td>32.848</td>
    </tr>
    <tr>
      <th>VGLQVVAVK</th>
      <td>31.861</td>
      <td>31.736</td>
      <td>31.697</td>
      <td>31.722</td>
      <td>31.689</td>
      <td>31.688</td>
      <td>31.533</td>
    </tr>
  </tbody>
</table>
<p>4927 rows × 7 columns</p>
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
      <td>-0.019</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.035</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.113</td>
      <td>0.011</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.045</td>
      <td>-0.027</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.007</td>
      <td>0.009</td>
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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_146_1.png)
    



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
    


    
![png](latent_2D_200_30_files/latent_2D_200_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20190104_QE6_nLC6_MM_QC_MNT_HELA_02_190107214303', 'VSVHVIEGDHR'), ('20190111_QE2_NLC10_ANHO_QC_MNT_HELA_03', 'VSHLLGINVTDFTR'), ('20190118_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'LLETTDRPDGHQNNLR'), ('20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03', 'LLETTDRPDGHQNNLR'), ('20190129_QE1_nLC2_GP_QC_MNT_HELA_02', 'HGESAWNLENR'), ('20190131_QE6_LC6_AS_MNT_HeLa_02', 'ILNIFGVIK'), ('20190204_QE10_nLC0_NHS_MNT_HELA_45cm_newColm_01', 'STNGDTFLGGEDFDQALLR'), ('20190205_QE7_nLC7_MEM_QC_MNT_HeLa_04', 'AVFVDLEPTVIDEVR'), ('20190206_QE9_nLC9_NHS_MNT_HELA_50cm_Newcolm2_1_20190207182540', 'IGEEQSAEDAEDGPPELLFIHGGHTAK'), ('20190207_QE2_NLC10_GP_MNT_HeLa_01', 'GPDGLTAFEATDNQAIK'), ('20190211_QE6_LC6_AS_QC_MNT_HeLa_03', 'TSRPENAIIYNNNEDFQVGQAK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_01', 'LNFSHGTHEYHAETIK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_02', 'LNFSHGTHEYHAETIK'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_01', 'SLTNDWEDHLAVK'), ('20190219_QE10_nLC14_FaCo_QC_HeLa_50cm_20190219185517', 'STNGDTFLGGEDFDQALLR'), ('20190219_QE3_Evo2_UHG_QC_MNT_HELA_01_190219173213', 'SGDAAIVDMVPGKPMCVESFSDYPPLGR'), ('20190220_QE2_NLC1_GP_QC_MNT_HELA_01', 'VIVVGNPANTNCLTASK'), ('20190224_QE1_nLC2_GP_QC_MNT_HELA_01', 'LVIITAGAR'), ('20190226_QE3_nLC3_MR_QC_MNT_HELA_Easy_04', 'KYEQGFITDPVVLSPK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_01', 'LVIITAGAR'), ('20190308_QE9_nLC0_FaCo_QC_MNT_Hela_50cm', 'LYSVSYLLK'), ('20190310_QE2_NLC1_GP_MNT_HELA_01', 'HTNYTMEHIR'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_03', 'DNLAEDIMR'), ('20190327_QE6_LC6_SCL_QC_MNT_Hela_01', 'GAEAANVTGPGGVPVQGSK'), ('20190401_QE4_LC12_IAH-JE_QC_MNT_HeLa_02', 'YYVTIIDAPGHR'), ('20190401_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'YYVTIIDAPGHR'), ('20190408_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-031', 'MLVLDEADEMLNK'), ('20190410_QE1_nLC2_ANHO_MNT_QC_hela_01', 'SLTNDWEDHLAVK'), ('20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01', 'STNGDTFLGGEDFDQALLR'), ('20190423_QX0_MaPe_MA_HeLa_500ng_LC07_1_high', 'LYSVSYLLK'), ('20190424_QE2_NLC1_ANHO_MNT_HELA_01', 'YMACCLLYR'), ('20190426_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff_newcol', 'MLDAEDIVNTARPDEK'), ('20190428_QE9_nLC0_LiNi_QC_45cm_HeLa_ending', 'LLETTDRPDGHQNNLR'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336', 'LYSVSYLLK'), ('20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard_190501203657', 'LYSVSYLLK'), ('20190430_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'YMACCLLYR'), ('20190502_QE8_nLC14_AGF_QC_MNT_HeLa_01', 'PSQMEHAMETMMFTFHK'), ('20190502_QX8_MiWi_MA_HeLa_500ng_new', 'GPDGLTAFEATDNQAIK'), ('20190502_QX8_MiWi_MA_HeLa_500ng_old', 'VSVHVIEGDHR'), ('20190506_QE4_LC12_AS_QC_MNT_HeLa_02', 'MDATANDVPSPYEVR'), ('20190506_QX6_ChDe_MA_HeLa_Br13_500ng_LC09', 'LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR'), ('20190513_QE6_LC4_IAH_QC_MNT_HeLa_01', 'TSRPENAIIYNNNEDFQVGQAK'), ('20190513_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'SGGMSNELNNIISR'), ('20190513_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'ILNIFGVIK'), ('20190514_QX6_ChDe_MA_HeLa_Br14_500ng_LC09_20190515085753', 'VHIGQVIMSIR'), ('20190523_QE2_NLC1_GP_MNT_HELA_01', 'VSHLLGINVTDFTR'), ('20190524_QE9_nLC0_FaCo_QC_MNT_Hela_50cm', 'HGESAWNLENR'), ('20190526_QX8_IgPa_MA_HeLa_BR14_500ng_08isolation', 'GLVEPVDVVDNADGTQTVNYVPSR'), ('20190527_QX7_MaMu_MA_HeLa_Br14_500ng', 'YMACCLLYR'), ('20190531_QX3_AnSe_MA_Hela_500ng_LC15', 'AVFVDLEPTVIDEVR'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_5', 'GNPTVEVDLFTSK'), ('20190617_QE1_nLC2_GP_QC_MNT_HELA_01_20190617213340', 'AVTEQGAELSNEER'), ('20190617_QE6_nLC4_JE_QC_MNT_HeLa_01', 'SGGMSNELNNIISR'), ('20190618_QX4_JiYu_MA_HeLa_500ng_centroid', 'PSQMEHAMETMMFTFHK'), ('20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01', 'GPDGLTAFEATDNQAIK'), ('20190624_QE4_nLC12_MM_QC_MNT_HELA_02_20190626221327', 'PSQMEHAMETMMFTFHK'), ('20190624_QX4_JiYu_MA_HeLa_500ng', 'ILNIFGVIK'), ('20190625_QE1_nLC2_GP_QC_MNT_HELA_03', 'AVTEQGAELSNEER'), ('20190625_QE2_NLC1_GP_QC_MNT_HELA_01', 'NNQFQALLQYADPVSAQHAK'), ('20190625_QE6_LC4_AS_QC_MNT_HeLa_01', 'NNQFQALLQYADPVSAQHAK'), ('20190625_QX0_MaPe_MA_HeLa_500ng_LC07_1', 'KYEQGFITDPVVLSPK'), ('20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED', 'VIVVGNPANTNCLTASK'), ('20190701_QE4_LC12_IAH_QC_MNT_HeLa_03', 'DALSDLALHFLNK'), ('20190704_QX6_MaTa_MA_HeLa_500ng_LC09', 'MLDAEDIVNTARPDEK'), ('20190704_QX6_MaTa_MA_HeLa_500ng_LC09', 'TCTTVAFTQVNSEDK'), ('20190708_QE8_nLC14_FM_QC_MNT_50cm_Hela_01', 'LYSVSYLLK'), ('20190715_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01', 'SLTNDWEDHLAVK'), ('20190716_QE8_nLC14_RG_QC_MNT_HeLa_MUC_50cm_2', 'LLETTDRPDGHQNNLR'), ('20190722_QX8_ChSc_MA_HeLa_500ng_190722174431', 'SGDAAIVDMVPGKPMCVESFSDYPPLGR'), ('20190723_QE2_NLC1_TL_MNT_HELA_01', 'SGGMSNELNNIISR'), ('20190724_QX3_MiWi_MA_Hela_500ng_LC15', 'AVFVDLEPTVIDEVR'), ('20190729_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05', 'YMACCLLYR'), ('20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01', 'GLVEPVDVVDNADGTQTVNYVPSR'), ('20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02', 'VQASLAANTFTITGHAETK'), ('20190730_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05', 'IIGLDQVAGMSETALPGAFK'), ('20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2', 'TICSHVQNMIK'), ('20190801_QE9_nLC13_JM_MNT_MUC_Hela_15cm_04', 'MVNHFIAEFK')]
    




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
      <th>train_average</th>
      <th>intensity_pred_vae</th>
      <th>train_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSE</th>
      <td>0.610</td>
      <td>0.632</td>
      <td>1.582</td>
      <td>2.025</td>
      <td>2.039</td>
      <td>2.067</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.492</td>
      <td>0.511</td>
      <td>0.870</td>
      <td>1.066</td>
      <td>1.070</td>
      <td>1.046</td>
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
