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
n_feat = 75
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
    SGPFGQIFRPDNFVFGQSGAGNNWAK           936
    SCMLTGTPESVQSAK                      976
    VNQIGSVTESLQACK                      999
    VIMVTGDHPITAK                        985
    VFDAIMNFK                            990
    THEAQIQEMR                           986
    AMVSEFLK                             984
    LYGSAGPPPTGEEDTAEK                   985
    LMIEMDGTENK                          999
    FIQENIFGICPHMTEDNK                   985
    AGKPVICATQMLESMIK                    997
    DIISDTSGDFRK                         999
    THILLFLPK                          1,000
    LMVALAK                              979
    VALVYGQMNEPPGAR                    1,000
    SAEFLLHMLK                           999
    TFSHELSDFGLESTAGEIPVVAIR             973
    ALTGGIAHLFK                          995
    THIQDNHDGTYTVAYVPDVTGR               993
    QLFHPEQLITGK                         996
    ARFEELNADLFR                         988
    IAFAITAIK                            995
    LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR     968
    ISMPDVDLHLK                        1,000
    TAFDEAIAELDTLNEDSYK                  974
    GPLMMYISK                            999
    MDATANDVPSPYEVR                      998
    IFTSIGEDYDER                         974
    ISMPDLDLNLK                          980
    WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK     996
    ANLQIDQINTDLNLER                     977
    DALSDLALHFLNK                        991
    YNEQHVPGSPFTAR                       973
    GAGTGGLGLAVEGPSEAK                   944
    SVGGSGGGSFGDNLVTR                    962
    TVTNAVVTVPAYFNDSQR                 1,000
    ATAVMPDGQFK                        1,000
    YGINTTDIFQTVDLWEGK                   973
    DHENIVIAK                            986
    DPFAHLPK                             990
    DYGVYLEDSGHTLR                       999
    KLEEEQIILEDQNCK                      996
    ATESGAQSAPLPMEGVDISPK                999
    VHVIFNYK                             998
    TAFQEALDAAGDK                      1,000
    TLSDYNIQK                            992
    AGAGSATLSMAYAGAR                     951
    VHLVGIDIFTGK                       1,000
    VMTIAPGLFGTPLLTSLPEK                 993
    ASGPGLNTTGVPASLPVEFTIDAK             912
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
      <th>SGPFGQIFRPDNFVFGQSGAGNNWAK</th>
      <td>30.950</td>
    </tr>
    <tr>
      <th>SCMLTGTPESVQSAK</th>
      <td>26.027</td>
    </tr>
    <tr>
      <th>VNQIGSVTESLQACK</th>
      <td>31.181</td>
    </tr>
    <tr>
      <th>VIMVTGDHPITAK</th>
      <td>27.516</td>
    </tr>
    <tr>
      <th>VFDAIMNFK</th>
      <td>29.725</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>TLSDYNIQK</th>
      <td>31.625</td>
    </tr>
    <tr>
      <th>AGAGSATLSMAYAGAR</th>
      <td>29.018</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
    </tr>
    <tr>
      <th>VMTIAPGLFGTPLLTSLPEK</th>
      <td>29.863</td>
    </tr>
    <tr>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <td>29.959</td>
    </tr>
  </tbody>
</table>
<p>49264 rows × 1 columns</p>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_24_1.png)
    



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
      <th>SGPFGQIFRPDNFVFGQSGAGNNWAK</th>
      <td>30.950</td>
    </tr>
    <tr>
      <th>SCMLTGTPESVQSAK</th>
      <td>26.027</td>
    </tr>
    <tr>
      <th>VNQIGSVTESLQACK</th>
      <td>31.181</td>
    </tr>
    <tr>
      <th>VIMVTGDHPITAK</th>
      <td>27.516</td>
    </tr>
    <tr>
      <th>VFDAIMNFK</th>
      <td>29.725</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>TLSDYNIQK</th>
      <td>31.625</td>
    </tr>
    <tr>
      <th>AGAGSATLSMAYAGAR</th>
      <td>29.018</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
    </tr>
    <tr>
      <th>VMTIAPGLFGTPLLTSLPEK</th>
      <td>29.863</td>
    </tr>
    <tr>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <td>29.959</td>
    </tr>
  </tbody>
</table>
<p>49264 rows × 1 columns</p>
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
      <th>20190408_QE6_LC6_AS_QC_MNT_HeLa_01</th>
      <th>AGAGSATLSMAYAGAR</th>
      <td>29.030</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_02</th>
      <th>AGAGSATLSMAYAGAR</th>
      <td>29.550</td>
    </tr>
    <tr>
      <th>20190625_QE6_LC4_AS_QC_MNT_HeLa_01</th>
      <th>AGAGSATLSMAYAGAR</th>
      <td>28.743</td>
    </tr>
    <tr>
      <th>20190531_QE4_nLC12_MM_QC_MNT_HELA_02_20190605020529</th>
      <th>AGAGSATLSMAYAGAR</th>
      <td>27.187</td>
    </tr>
    <tr>
      <th>20190207_QE8_nLC0_ASD_QC_HeLa_43cm3</th>
      <th>AGAGSATLSMAYAGAR</th>
      <td>29.106</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625212127</th>
      <th>YNEQHVPGSPFTAR</th>
      <td>29.409</td>
    </tr>
    <tr>
      <th>20190403_QE10_nLC13_LiNi_QC_45cm_HeLa_01</th>
      <th>YNEQHVPGSPFTAR</th>
      <td>30.430</td>
    </tr>
    <tr>
      <th>20190401_QE4_LC12_IAH-JE_QC_MNT_HeLa_01</th>
      <th>YNEQHVPGSPFTAR</th>
      <td>30.872</td>
    </tr>
    <tr>
      <th>20190802_QE3_nLC3_DBJ_AMV_QC_MNT_HELA_01</th>
      <th>YNEQHVPGSPFTAR</th>
      <td>30.200</td>
    </tr>
    <tr>
      <th>20190324_QE7_nLC3_RJC_WIMS_QC_MNT_HeLa_01</th>
      <th>YNEQHVPGSPFTAR</th>
      <td>30.019</td>
    </tr>
  </tbody>
</table>
<p>44336 rows × 1 columns</p>
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
    Shape in validation: (991, 50)
    




    ((991, 50), (991, 50))



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
      <th>AGAGSATLSMAYAGAR</th>
      <td>28.868</td>
      <td>28.930</td>
      <td>28.869</td>
      <td>28.985</td>
    </tr>
    <tr>
      <th>ANLQIDQINTDLNLER</th>
      <td>27.860</td>
      <td>28.733</td>
      <td>28.535</td>
      <td>27.932</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.981</td>
      <td>30.037</td>
      <td>29.301</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>28.346</td>
      <td>29.529</td>
      <td>29.261</td>
      <td>27.899</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGKPVICATQMLESMIK</th>
      <td>30.878</td>
      <td>32.100</td>
      <td>31.672</td>
      <td>31.451</td>
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
      <th>IAFAITAIK</th>
      <td>30.131</td>
      <td>29.930</td>
      <td>29.764</td>
      <td>30.297</td>
    </tr>
    <tr>
      <th>IFTSIGEDYDER</th>
      <td>27.825</td>
      <td>28.053</td>
      <td>28.317</td>
      <td>28.100</td>
    </tr>
    <tr>
      <th>LYGSAGPPPTGEEDTAEK</th>
      <td>27.519</td>
      <td>27.649</td>
      <td>27.945</td>
      <td>27.687</td>
    </tr>
    <tr>
      <th>THIQDNHDGTYTVAYVPDVTGR</th>
      <td>28.003</td>
      <td>27.993</td>
      <td>27.812</td>
      <td>27.870</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
      <td>31.820</td>
      <td>31.680</td>
      <td>32.365</td>
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
      <th>ANLQIDQINTDLNLER</th>
      <td>28.400</td>
      <td>28.733</td>
      <td>28.535</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</th>
      <th>DPFAHLPK</th>
      <td>27.644</td>
      <td>27.841</td>
      <td>29.012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE8_nLC1_ASD_QC_HeLa_02</th>
      <th>GPLMMYISK</th>
      <td>27.262</td>
      <td>30.521</td>
      <td>30.500</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03</th>
      <th>MDATANDVPSPYEVR</th>
      <td>28.583</td>
      <td>28.684</td>
      <td>28.913</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01</th>
      <th>VALVYGQMNEPPGAR</th>
      <td>27.011</td>
      <td>28.200</td>
      <td>28.305</td>
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
      <th>20190726_QX8_ChSc_MA_HeLa_500ng</th>
      <th>GAGTGGLGLAVEGPSEAK</th>
      <td>29.017</td>
      <td>29.254</td>
      <td>29.096</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QE8_nLC14_ASD_QC_MNT_HeLa_02</th>
      <th>IAFAITAIK</th>
      <td>28.587</td>
      <td>29.930</td>
      <td>29.764</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_1</th>
      <th>TVTNAVVTVPAYFNDSQR</th>
      <td>30.946</td>
      <td>30.677</td>
      <td>30.941</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190801_QX3_StEb_MA_Hela_500ng_LC15</th>
      <th>SCMLTGTPESVQSAK</th>
      <td>28.603</td>
      <td>27.664</td>
      <td>27.740</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX7_AlRe_MA_HeLa_Br14_500ng</th>
      <th>TFSHELSDFGLESTAGEIPVVAIR</th>
      <td>32.405</td>
      <td>30.525</td>
      <td>30.054</td>
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
      <td>AGKPVICATQMLESMIK</td>
      <td>31.078</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALTGGIAHLFK</td>
      <td>27.882</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AMVSEFLK</td>
      <td>28.042</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ARFEELNADLFR</td>
      <td>30.249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ASGPGLNTTGVPASLPVEFTIDAK</td>
      <td>29.013</td>
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
      <td>AGAGSATLSMAYAGAR</td>
      <td>28.868</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ANLQIDQINTDLNLER</td>
      <td>27.860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DHENIVIAK</td>
      <td>29.417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>THEAQIQEMR</td>
      <td>28.346</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>AGKPVICATQMLESMIK</td>
      <td>30.878</td>
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
      <td>20190305_QE2_NLC1_AB_QC_MNT_HELA_01</td>
      <td>ATAVMPDGQFK</td>
      <td>32.095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190611_QE7_nLC5_MEM_QC_MNT_HeLa_03</td>
      <td>SVGGSGGGSFGDNLVTR</td>
      <td>29.412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</td>
      <td>MDATANDVPSPYEVR</td>
      <td>28.494</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190114_QE2_NLC10_ANHO_QC_MNT_HELA_01</td>
      <td>IFTSIGEDYDER</td>
      <td>27.447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190502_QX8_MiWi_MA_HeLa_500ng_old</td>
      <td>DHENIVIAK</td>
      <td>31.744</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190629_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</td>
      <td>THEAQIQEMR</td>
      <td>28.976</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190121_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>THEAQIQEMR</td>
      <td>27.150</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190405_QE1_nLC2_GP_MNT_QC_hela_02</td>
      <td>FIQENIFGICPHMTEDNK</td>
      <td>27.726</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190228_QE4_LC12_JE_QC_MNT_HeLa_03</td>
      <td>THEAQIQEMR</td>
      <td>29.678</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190718_QE9_nLC9_NHS_MNT_HELA_50cm_01</td>
      <td>VHLVGIDIFTGK</td>
      <td>30.738</td>
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
      <td>20190803_QE9_nLC13_RG_SA_HeLa_50cm_250ng</td>
      <td>YNEQHVPGSPFTAR</td>
      <td>28.820</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190513_QE7_nLC7_MEM_QC_MNT_HeLa_01</td>
      <td>GPLMMYISK</td>
      <td>28.961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190722_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_3rdcolumn_4</td>
      <td>SGPFGQIFRPDNFVFGQSGAGNNWAK</td>
      <td>31.182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190623_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_MUC_01</td>
      <td>YNEQHVPGSPFTAR</td>
      <td>29.709</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190630_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</td>
      <td>KLEEEQIILEDQNCK</td>
      <td>29.482</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</td>
      <td>THILLFLPK</td>
      <td>30.058</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190620_QE2_NLC1_GP_QC_MNT_HELA_01</td>
      <td>VHLVGIDIFTGK</td>
      <td>32.336</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190305_QE8_nLC14_ASD_QC_MNT_50cm_HELA_02</td>
      <td>ATAVMPDGQFK</td>
      <td>30.175</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190131_QE6_LC6_AS_MNT_HeLa_02</td>
      <td>THIQDNHDGTYTVAYVPDVTGR</td>
      <td>27.461</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190617_QE6_nLC4_JE_QC_MNT_HeLa_02</td>
      <td>LMVALAK</td>
      <td>31.173</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (992, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1371, 154)



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
     'n_samples': 992,
     'y_range': (20, 35)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1984       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 992        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,129
    Total trainable params: 3,129
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x0000024BFF047040>
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
      <td>2.288449</td>
      <td>2.060660</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.807624</td>
      <td>0.879812</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.612550</td>
      <td>0.639256</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.655789</td>
      <td>0.561974</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.559739</td>
      <td>0.532349</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.520791</td>
      <td>0.507277</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.455693</td>
      <td>0.495843</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.510722</td>
      <td>0.490620</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.466119</td>
      <td>0.488757</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.436170</td>
      <td>0.488634</td>
      <td>00:09</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_58_1.png)
    


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
      <th>4,872</th>
      <td>979</td>
      <td>50</td>
      <td>28.820</td>
    </tr>
    <tr>
      <th>2,581</th>
      <td>509</td>
      <td>17</td>
      <td>28.961</td>
    </tr>
    <tr>
      <th>4,458</th>
      <td>896</td>
      <td>31</td>
      <td>31.182</td>
    </tr>
    <tr>
      <th>3,544</th>
      <td>716</td>
      <td>50</td>
      <td>29.709</td>
    </tr>
    <tr>
      <th>3,848</th>
      <td>777</td>
      <td>22</td>
      <td>29.482</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1,184</th>
      <td>229</td>
      <td>11</td>
      <td>28.235</td>
    </tr>
    <tr>
      <th>563</th>
      <td>115</td>
      <td>16</td>
      <td>29.632</td>
    </tr>
    <tr>
      <th>2,262</th>
      <td>442</td>
      <td>42</td>
      <td>32.006</td>
    </tr>
    <tr>
      <th>3,783</th>
      <td>766</td>
      <td>20</td>
      <td>30.278</td>
    </tr>
    <tr>
      <th>3,378</th>
      <td>682</td>
      <td>34</td>
      <td>31.345</td>
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
      <th>AGAGSATLSMAYAGAR</th>
      <td>28.868</td>
      <td>28.930</td>
      <td>28.869</td>
      <td>28.985</td>
      <td>28.226</td>
    </tr>
    <tr>
      <th>ANLQIDQINTDLNLER</th>
      <td>27.860</td>
      <td>28.733</td>
      <td>28.535</td>
      <td>27.932</td>
      <td>27.617</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.981</td>
      <td>30.037</td>
      <td>29.301</td>
      <td>29.290</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>28.346</td>
      <td>29.529</td>
      <td>29.261</td>
      <td>27.899</td>
      <td>28.561</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGKPVICATQMLESMIK</th>
      <td>30.878</td>
      <td>32.100</td>
      <td>31.672</td>
      <td>31.451</td>
      <td>30.789</td>
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
      <th>IAFAITAIK</th>
      <td>30.131</td>
      <td>29.930</td>
      <td>29.764</td>
      <td>30.297</td>
      <td>30.030</td>
    </tr>
    <tr>
      <th>IFTSIGEDYDER</th>
      <td>27.825</td>
      <td>28.053</td>
      <td>28.317</td>
      <td>28.100</td>
      <td>27.903</td>
    </tr>
    <tr>
      <th>LYGSAGPPPTGEEDTAEK</th>
      <td>27.519</td>
      <td>27.649</td>
      <td>27.945</td>
      <td>27.687</td>
      <td>27.610</td>
    </tr>
    <tr>
      <th>THIQDNHDGTYTVAYVPDVTGR</th>
      <td>28.003</td>
      <td>27.993</td>
      <td>27.812</td>
      <td>27.870</td>
      <td>27.775</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
      <td>31.820</td>
      <td>31.680</td>
      <td>32.365</td>
      <td>31.614</td>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.115
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.114
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.256
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.132
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.254
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_66_1.png)
    



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
      <td>0.112</td>
      <td>0.076</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.174</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.048</td>
      <td>-0.133</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.268</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.021</td>
      <td>0.184</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_68_1.png)
    



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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_69_1.png)
    


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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR</th>
      <th>VFDAIMNFK</th>
      <th>VHLVGIDIFTGK</th>
      <th>VHVIFNYK</th>
      <th>VIMVTGDHPITAK</th>
      <th>VMTIAPGLFGTPLLTSLPEK</th>
      <th>VNQIGSVTESLQACK</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK</th>
      <th>YGINTTDIFQTVDLWEGK</th>
      <th>YNEQHVPGSPFTAR</th>
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
      <td>28.868</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.860</td>
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
      <td>30.878</td>
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
      <td>28.429</td>
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
      <td>28.969</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.108</td>
      <td>27.375</td>
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
      <td>28.649</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.211</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.038</td>
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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VHLVGIDIFTGK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIMVTGDHPITAK_na</th>
      <th>VMTIAPGLFGTPLLTSLPEK_na</th>
      <th>VNQIGSVTESLQACK_na</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YNEQHVPGSPFTAR_na</th>
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
      <td>0.054</td>
      <td>-0.478</td>
      <td>-0.024</td>
      <td>-0.837</td>
      <td>0.161</td>
      <td>0.222</td>
      <td>-0.342</td>
      <td>-1.166</td>
      <td>-1.076</td>
      <td>-0.379</td>
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
      <td>0.112</td>
      <td>0.287</td>
      <td>-0.118</td>
      <td>-1.057</td>
      <td>-0.872</td>
      <td>0.151</td>
      <td>-0.518</td>
      <td>-1.004</td>
      <td>-1.041</td>
      <td>-0.641</td>
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
      <td>0.054</td>
      <td>0.080</td>
      <td>-0.921</td>
      <td>-1.076</td>
      <td>-0.293</td>
      <td>-0.919</td>
      <td>0.616</td>
      <td>-0.378</td>
      <td>-0.470</td>
      <td>-0.763</td>
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
      <td>0.054</td>
      <td>-0.468</td>
      <td>-0.721</td>
      <td>0.155</td>
      <td>0.161</td>
      <td>0.296</td>
      <td>-0.275</td>
      <td>-0.623</td>
      <td>-0.420</td>
      <td>-0.730</td>
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
      <td>-0.116</td>
      <td>0.420</td>
      <td>-0.217</td>
      <td>0.553</td>
      <td>0.161</td>
      <td>0.152</td>
      <td>0.534</td>
      <td>-0.589</td>
      <td>0.018</td>
      <td>0.307</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
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
      <td>0.283</td>
      <td>0.518</td>
      <td>0.276</td>
      <td>0.266</td>
      <td>0.267</td>
      <td>1.572</td>
      <td>-0.410</td>
      <td>0.540</td>
      <td>0.396</td>
      <td>0.734</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.309</td>
      <td>0.107</td>
      <td>-0.221</td>
      <td>-0.044</td>
      <td>-0.383</td>
      <td>-0.292</td>
      <td>0.263</td>
      <td>0.111</td>
      <td>-0.568</td>
      <td>-0.310</td>
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
      <td>0.717</td>
      <td>0.578</td>
      <td>-0.038</td>
      <td>-0.069</td>
      <td>0.049</td>
      <td>-0.301</td>
      <td>0.251</td>
      <td>0.218</td>
      <td>-0.162</td>
      <td>-0.006</td>
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
      <td>0.368</td>
      <td>0.573</td>
      <td>0.034</td>
      <td>0.155</td>
      <td>0.066</td>
      <td>-0.439</td>
      <td>0.210</td>
      <td>0.026</td>
      <td>-0.192</td>
      <td>0.061</td>
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
      <td>0.146</td>
      <td>-0.664</td>
      <td>1.170</td>
      <td>0.155</td>
      <td>1.013</td>
      <td>-0.372</td>
      <td>0.162</td>
      <td>-0.008</td>
      <td>-0.347</td>
      <td>0.061</td>
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
<p>991 rows × 100 columns</p>
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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VHLVGIDIFTGK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIMVTGDHPITAK_na</th>
      <th>VMTIAPGLFGTPLLTSLPEK_na</th>
      <th>VNQIGSVTESLQACK_na</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YNEQHVPGSPFTAR_na</th>
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
      <td>0.059</td>
      <td>-0.423</td>
      <td>-0.023</td>
      <td>-0.770</td>
      <td>0.172</td>
      <td>0.188</td>
      <td>-0.280</td>
      <td>-1.100</td>
      <td>-1.018</td>
      <td>-0.351</td>
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
      <td>0.112</td>
      <td>0.304</td>
      <td>-0.112</td>
      <td>-0.976</td>
      <td>-0.798</td>
      <td>0.121</td>
      <td>-0.439</td>
      <td>-0.946</td>
      <td>-0.985</td>
      <td>-0.598</td>
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
      <td>0.059</td>
      <td>0.108</td>
      <td>-0.872</td>
      <td>-0.995</td>
      <td>-0.254</td>
      <td>-0.890</td>
      <td>0.592</td>
      <td>-0.352</td>
      <td>-0.443</td>
      <td>-0.714</td>
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
      <td>0.059</td>
      <td>-0.413</td>
      <td>-0.683</td>
      <td>0.165</td>
      <td>0.172</td>
      <td>0.258</td>
      <td>-0.219</td>
      <td>-0.584</td>
      <td>-0.396</td>
      <td>-0.682</td>
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
      <td>-0.099</td>
      <td>0.431</td>
      <td>-0.206</td>
      <td>0.540</td>
      <td>0.172</td>
      <td>0.122</td>
      <td>0.517</td>
      <td>-0.552</td>
      <td>0.019</td>
      <td>0.297</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
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
      <td>0.270</td>
      <td>0.525</td>
      <td>0.261</td>
      <td>0.269</td>
      <td>0.271</td>
      <td>1.462</td>
      <td>-0.341</td>
      <td>0.518</td>
      <td>0.378</td>
      <td>0.700</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.294</td>
      <td>0.134</td>
      <td>-0.210</td>
      <td>-0.022</td>
      <td>-0.339</td>
      <td>-0.298</td>
      <td>0.271</td>
      <td>0.112</td>
      <td>-0.537</td>
      <td>-0.285</td>
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
      <td>0.672</td>
      <td>0.582</td>
      <td>-0.037</td>
      <td>-0.046</td>
      <td>0.067</td>
      <td>-0.307</td>
      <td>0.260</td>
      <td>0.214</td>
      <td>-0.152</td>
      <td>0.002</td>
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
      <td>0.349</td>
      <td>0.576</td>
      <td>0.031</td>
      <td>0.165</td>
      <td>0.083</td>
      <td>-0.436</td>
      <td>0.223</td>
      <td>0.031</td>
      <td>-0.180</td>
      <td>0.065</td>
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
      <td>0.144</td>
      <td>-0.600</td>
      <td>1.106</td>
      <td>0.165</td>
      <td>0.972</td>
      <td>-0.373</td>
      <td>0.179</td>
      <td>-0.001</td>
      <td>-0.327</td>
      <td>0.065</td>
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
<p>991 rows × 100 columns</p>
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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
      <td>991.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.008</td>
      <td>0.032</td>
      <td>-0.001</td>
      <td>0.019</td>
      <td>0.021</td>
      <td>-0.022</td>
      <td>0.032</td>
      <td>0.006</td>
      <td>0.002</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.926</td>
      <td>0.952</td>
      <td>0.947</td>
      <td>0.943</td>
      <td>0.939</td>
      <td>0.945</td>
      <td>0.910</td>
      <td>0.949</td>
      <td>0.948</td>
      <td>0.945</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.066</td>
      <td>-5.044</td>
      <td>-4.574</td>
      <td>-5.097</td>
      <td>-5.006</td>
      <td>-3.306</td>
      <td>-5.589</td>
      <td>-4.356</td>
      <td>-4.933</td>
      <td>-4.177</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.366</td>
      <td>-0.233</td>
      <td>-0.474</td>
      <td>-0.330</td>
      <td>-0.331</td>
      <td>-0.585</td>
      <td>-0.530</td>
      <td>-0.365</td>
      <td>-0.414</td>
      <td>-0.437</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.059</td>
      <td>0.304</td>
      <td>-0.008</td>
      <td>0.165</td>
      <td>0.172</td>
      <td>-0.197</td>
      <td>0.179</td>
      <td>0.064</td>
      <td>0.019</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.500</td>
      <td>0.595</td>
      <td>0.514</td>
      <td>0.565</td>
      <td>0.578</td>
      <td>0.592</td>
      <td>0.663</td>
      <td>0.547</td>
      <td>0.479</td>
      <td>0.611</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.223</td>
      <td>2.233</td>
      <td>2.466</td>
      <td>2.681</td>
      <td>2.150</td>
      <td>1.967</td>
      <td>1.960</td>
      <td>2.244</td>
      <td>2.472</td>
      <td>2.007</td>
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




    ((#50) ['AGAGSATLSMAYAGAR','AGKPVICATQMLESMIK','ALTGGIAHLFK','AMVSEFLK','ANLQIDQINTDLNLER','ARFEELNADLFR','ASGPGLNTTGVPASLPVEFTIDAK','ATAVMPDGQFK','ATESGAQSAPLPMEGVDISPK','DALSDLALHFLNK'...],
     (#50) ['AGAGSATLSMAYAGAR_na','AGKPVICATQMLESMIK_na','ALTGGIAHLFK_na','AMVSEFLK_na','ANLQIDQINTDLNLER_na','ARFEELNADLFR_na','ASGPGLNTTGVPASLPVEFTIDAK_na','ATAVMPDGQFK_na','ATESGAQSAPLPMEGVDISPK_na','DALSDLALHFLNK_na'...])




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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>91.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.251</td>
      <td>-0.048</td>
      <td>0.044</td>
      <td>-0.006</td>
      <td>-0.054</td>
      <td>-0.173</td>
      <td>-0.122</td>
      <td>0.017</td>
      <td>0.019</td>
      <td>0.058</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.106</td>
      <td>1.072</td>
      <td>0.959</td>
      <td>1.073</td>
      <td>1.070</td>
      <td>1.044</td>
      <td>1.228</td>
      <td>0.950</td>
      <td>0.945</td>
      <td>1.024</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.966</td>
      <td>-4.492</td>
      <td>-2.211</td>
      <td>-4.267</td>
      <td>-5.093</td>
      <td>-3.369</td>
      <td>-5.704</td>
      <td>-3.252</td>
      <td>-2.466</td>
      <td>-3.441</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.717</td>
      <td>-0.384</td>
      <td>-0.496</td>
      <td>-0.489</td>
      <td>-0.473</td>
      <td>-0.823</td>
      <td>-0.706</td>
      <td>-0.366</td>
      <td>-0.510</td>
      <td>-0.410</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.128</td>
      <td>0.222</td>
      <td>0.061</td>
      <td>0.188</td>
      <td>0.055</td>
      <td>-0.277</td>
      <td>0.172</td>
      <td>0.079</td>
      <td>0.075</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.480</td>
      <td>0.663</td>
      <td>0.552</td>
      <td>0.716</td>
      <td>0.638</td>
      <td>0.497</td>
      <td>0.799</td>
      <td>0.660</td>
      <td>0.515</td>
      <td>0.720</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.537</td>
      <td>1.477</td>
      <td>1.921</td>
      <td>1.703</td>
      <td>1.765</td>
      <td>1.874</td>
      <td>1.560</td>
      <td>2.150</td>
      <td>2.181</td>
      <td>1.961</td>
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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VHLVGIDIFTGK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIMVTGDHPITAK_val</th>
      <th>VMTIAPGLFGTPLLTSLPEK_val</th>
      <th>VNQIGSVTESLQACK_val</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YNEQHVPGSPFTAR_val</th>
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
      <td>0.059</td>
      <td>-0.423</td>
      <td>-0.023</td>
      <td>-0.770</td>
      <td>0.172</td>
      <td>0.188</td>
      <td>-0.280</td>
      <td>-1.100</td>
      <td>-1.018</td>
      <td>-0.351</td>
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
      <td>0.112</td>
      <td>0.304</td>
      <td>-0.112</td>
      <td>-0.976</td>
      <td>-0.798</td>
      <td>0.121</td>
      <td>-0.439</td>
      <td>-0.946</td>
      <td>-0.985</td>
      <td>-0.598</td>
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
      <td>0.059</td>
      <td>0.108</td>
      <td>-0.872</td>
      <td>-0.995</td>
      <td>-0.254</td>
      <td>-0.890</td>
      <td>0.592</td>
      <td>-0.352</td>
      <td>-0.443</td>
      <td>-0.714</td>
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
      <td>0.059</td>
      <td>-0.413</td>
      <td>-0.683</td>
      <td>0.165</td>
      <td>0.172</td>
      <td>0.258</td>
      <td>-0.219</td>
      <td>-0.584</td>
      <td>-0.396</td>
      <td>-0.682</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.069</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.099</td>
      <td>0.431</td>
      <td>-0.206</td>
      <td>0.540</td>
      <td>0.172</td>
      <td>0.122</td>
      <td>0.517</td>
      <td>-0.552</td>
      <td>0.019</td>
      <td>0.297</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.286</td>
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
      <td>0.270</td>
      <td>0.525</td>
      <td>0.261</td>
      <td>0.269</td>
      <td>0.271</td>
      <td>1.462</td>
      <td>-0.341</td>
      <td>0.518</td>
      <td>0.378</td>
      <td>0.700</td>
      <td>...</td>
      <td>0.860</td>
      <td>0.905</td>
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
      <td>0.294</td>
      <td>0.134</td>
      <td>-0.210</td>
      <td>-0.022</td>
      <td>-0.339</td>
      <td>-0.298</td>
      <td>0.271</td>
      <td>0.112</td>
      <td>-0.537</td>
      <td>-0.285</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.401</td>
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
      <td>0.672</td>
      <td>0.582</td>
      <td>-0.037</td>
      <td>-0.046</td>
      <td>0.067</td>
      <td>-0.307</td>
      <td>0.260</td>
      <td>0.214</td>
      <td>-0.152</td>
      <td>0.002</td>
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
      <td>0.349</td>
      <td>0.576</td>
      <td>0.031</td>
      <td>0.165</td>
      <td>0.083</td>
      <td>-0.436</td>
      <td>0.223</td>
      <td>0.031</td>
      <td>-0.180</td>
      <td>0.065</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.308</td>
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
      <td>0.144</td>
      <td>-0.600</td>
      <td>1.106</td>
      <td>0.165</td>
      <td>0.972</td>
      <td>-0.373</td>
      <td>0.179</td>
      <td>-0.001</td>
      <td>-0.327</td>
      <td>0.065</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.537</td>
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
<p>991 rows × 150 columns</p>
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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VHLVGIDIFTGK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIMVTGDHPITAK_val</th>
      <th>VMTIAPGLFGTPLLTSLPEK_val</th>
      <th>VNQIGSVTESLQACK_val</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YNEQHVPGSPFTAR_val</th>
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
      <td>0.059</td>
      <td>-0.423</td>
      <td>-0.023</td>
      <td>-0.770</td>
      <td>0.172</td>
      <td>0.188</td>
      <td>-0.280</td>
      <td>-1.100</td>
      <td>-1.018</td>
      <td>-0.351</td>
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
      <td>0.112</td>
      <td>0.304</td>
      <td>-0.112</td>
      <td>-0.976</td>
      <td>-0.798</td>
      <td>0.121</td>
      <td>-0.439</td>
      <td>-0.946</td>
      <td>-0.985</td>
      <td>-0.598</td>
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
      <td>0.059</td>
      <td>0.108</td>
      <td>-0.872</td>
      <td>-0.995</td>
      <td>-0.254</td>
      <td>-0.890</td>
      <td>0.592</td>
      <td>-0.352</td>
      <td>-0.443</td>
      <td>-0.714</td>
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
      <td>0.059</td>
      <td>-0.413</td>
      <td>-0.683</td>
      <td>0.165</td>
      <td>0.172</td>
      <td>0.258</td>
      <td>-0.219</td>
      <td>-0.584</td>
      <td>-0.396</td>
      <td>-0.682</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.069</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.099</td>
      <td>0.431</td>
      <td>-0.206</td>
      <td>0.540</td>
      <td>0.172</td>
      <td>0.122</td>
      <td>0.517</td>
      <td>-0.552</td>
      <td>0.019</td>
      <td>0.297</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.286</td>
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
      <td>0.270</td>
      <td>0.525</td>
      <td>0.261</td>
      <td>0.269</td>
      <td>0.271</td>
      <td>1.462</td>
      <td>-0.341</td>
      <td>0.518</td>
      <td>0.378</td>
      <td>0.700</td>
      <td>...</td>
      <td>0.860</td>
      <td>0.905</td>
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
      <td>0.294</td>
      <td>0.134</td>
      <td>-0.210</td>
      <td>-0.022</td>
      <td>-0.339</td>
      <td>-0.298</td>
      <td>0.271</td>
      <td>0.112</td>
      <td>-0.537</td>
      <td>-0.285</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.401</td>
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
      <td>0.672</td>
      <td>0.582</td>
      <td>-0.037</td>
      <td>-0.046</td>
      <td>0.067</td>
      <td>-0.307</td>
      <td>0.260</td>
      <td>0.214</td>
      <td>-0.152</td>
      <td>0.002</td>
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
      <td>0.349</td>
      <td>0.576</td>
      <td>0.031</td>
      <td>0.165</td>
      <td>0.083</td>
      <td>-0.436</td>
      <td>0.223</td>
      <td>0.031</td>
      <td>-0.180</td>
      <td>0.065</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.308</td>
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
      <td>0.144</td>
      <td>-0.600</td>
      <td>1.106</td>
      <td>0.165</td>
      <td>0.972</td>
      <td>-0.373</td>
      <td>0.179</td>
      <td>-0.001</td>
      <td>-0.327</td>
      <td>0.065</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.537</td>
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
<p>991 rows × 150 columns</p>
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
      <th>AGAGSATLSMAYAGAR_val</th>
      <th>AGKPVICATQMLESMIK_val</th>
      <th>ALTGGIAHLFK_val</th>
      <th>AMVSEFLK_val</th>
      <th>ANLQIDQINTDLNLER_val</th>
      <th>ARFEELNADLFR_val</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK_val</th>
      <th>ATAVMPDGQFK_val</th>
      <th>ATESGAQSAPLPMEGVDISPK_val</th>
      <th>DALSDLALHFLNK_val</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VHLVGIDIFTGK_val</th>
      <th>VHVIFNYK_val</th>
      <th>VIMVTGDHPITAK_val</th>
      <th>VMTIAPGLFGTPLLTSLPEK_val</th>
      <th>VNQIGSVTESLQACK_val</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_val</th>
      <th>YGINTTDIFQTVDLWEGK_val</th>
      <th>YNEQHVPGSPFTAR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>91.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>...</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>97.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.251</td>
      <td>-0.048</td>
      <td>0.044</td>
      <td>-0.006</td>
      <td>-0.054</td>
      <td>-0.173</td>
      <td>-0.122</td>
      <td>0.017</td>
      <td>0.019</td>
      <td>0.058</td>
      <td>...</td>
      <td>0.075</td>
      <td>-0.134</td>
      <td>0.152</td>
      <td>0.090</td>
      <td>-0.023</td>
      <td>-0.032</td>
      <td>-0.115</td>
      <td>0.185</td>
      <td>0.021</td>
      <td>0.083</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.106</td>
      <td>1.072</td>
      <td>0.959</td>
      <td>1.073</td>
      <td>1.070</td>
      <td>1.044</td>
      <td>1.228</td>
      <td>0.950</td>
      <td>0.945</td>
      <td>1.024</td>
      <td>...</td>
      <td>0.946</td>
      <td>0.909</td>
      <td>0.844</td>
      <td>0.935</td>
      <td>0.969</td>
      <td>1.061</td>
      <td>1.041</td>
      <td>0.789</td>
      <td>0.867</td>
      <td>1.088</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.966</td>
      <td>-4.492</td>
      <td>-2.211</td>
      <td>-4.267</td>
      <td>-5.093</td>
      <td>-3.369</td>
      <td>-5.704</td>
      <td>-3.252</td>
      <td>-2.466</td>
      <td>-3.441</td>
      <td>...</td>
      <td>-2.626</td>
      <td>-2.743</td>
      <td>-2.446</td>
      <td>-2.135</td>
      <td>-3.797</td>
      <td>-3.701</td>
      <td>-4.063</td>
      <td>-1.798</td>
      <td>-2.230</td>
      <td>-3.204</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.717</td>
      <td>-0.384</td>
      <td>-0.496</td>
      <td>-0.489</td>
      <td>-0.473</td>
      <td>-0.823</td>
      <td>-0.706</td>
      <td>-0.366</td>
      <td>-0.510</td>
      <td>-0.410</td>
      <td>...</td>
      <td>-0.641</td>
      <td>-0.636</td>
      <td>-0.257</td>
      <td>-0.431</td>
      <td>-0.404</td>
      <td>-0.502</td>
      <td>-0.602</td>
      <td>-0.386</td>
      <td>-0.479</td>
      <td>-0.061</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.128</td>
      <td>0.222</td>
      <td>0.061</td>
      <td>0.188</td>
      <td>0.055</td>
      <td>-0.277</td>
      <td>0.172</td>
      <td>0.079</td>
      <td>0.075</td>
      <td>0.070</td>
      <td>...</td>
      <td>0.034</td>
      <td>-0.308</td>
      <td>0.205</td>
      <td>-0.033</td>
      <td>-0.013</td>
      <td>0.028</td>
      <td>0.084</td>
      <td>0.166</td>
      <td>0.059</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.480</td>
      <td>0.663</td>
      <td>0.552</td>
      <td>0.716</td>
      <td>0.638</td>
      <td>0.497</td>
      <td>0.799</td>
      <td>0.660</td>
      <td>0.515</td>
      <td>0.720</td>
      <td>...</td>
      <td>0.822</td>
      <td>0.546</td>
      <td>0.706</td>
      <td>0.802</td>
      <td>0.517</td>
      <td>0.590</td>
      <td>0.572</td>
      <td>0.797</td>
      <td>0.653</td>
      <td>0.749</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.537</td>
      <td>1.477</td>
      <td>1.921</td>
      <td>1.703</td>
      <td>1.765</td>
      <td>1.874</td>
      <td>1.560</td>
      <td>2.150</td>
      <td>2.181</td>
      <td>1.961</td>
      <td>...</td>
      <td>1.821</td>
      <td>1.726</td>
      <td>1.657</td>
      <td>1.999</td>
      <td>2.039</td>
      <td>1.663</td>
      <td>1.577</td>
      <td>1.585</td>
      <td>1.573</td>
      <td>1.338</td>
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
      <th>AGAGSATLSMAYAGAR_na</th>
      <th>AGKPVICATQMLESMIK_na</th>
      <th>ALTGGIAHLFK_na</th>
      <th>AMVSEFLK_na</th>
      <th>ANLQIDQINTDLNLER_na</th>
      <th>ARFEELNADLFR_na</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK_na</th>
      <th>ATAVMPDGQFK_na</th>
      <th>ATESGAQSAPLPMEGVDISPK_na</th>
      <th>DALSDLALHFLNK_na</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VHLVGIDIFTGK_na</th>
      <th>VHVIFNYK_na</th>
      <th>VIMVTGDHPITAK_na</th>
      <th>VMTIAPGLFGTPLLTSLPEK_na</th>
      <th>VNQIGSVTESLQACK_na</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK_na</th>
      <th>YGINTTDIFQTVDLWEGK_na</th>
      <th>YNEQHVPGSPFTAR_na</th>
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
      <td>False</td>
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
      <td>False</td>
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
<p>991 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x0000024BFF047040>
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




    
![png](latent_2D_75_10_files/latent_2D_75_10_108_2.png)
    


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
      <td>0.949784</td>
      <td>0.747942</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.648808</td>
      <td>0.354572</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.471884</td>
      <td>0.325581</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.386685</td>
      <td>0.305682</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.342606</td>
      <td>0.300665</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.318043</td>
      <td>0.294439</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.301442</td>
      <td>0.288085</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.288688</td>
      <td>0.291285</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.282717</td>
      <td>0.287684</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.278774</td>
      <td>0.286280</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_112_1.png)
    



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




    TensorBase(0.2864)




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
      <th>AGAGSATLSMAYAGAR</th>
      <td>28.868</td>
      <td>28.930</td>
      <td>28.869</td>
      <td>28.985</td>
      <td>28.226</td>
      <td>27.951</td>
    </tr>
    <tr>
      <th>ANLQIDQINTDLNLER</th>
      <td>27.860</td>
      <td>28.733</td>
      <td>28.535</td>
      <td>27.932</td>
      <td>27.617</td>
      <td>27.489</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.981</td>
      <td>30.037</td>
      <td>29.301</td>
      <td>29.290</td>
      <td>28.946</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>28.346</td>
      <td>29.529</td>
      <td>29.261</td>
      <td>27.899</td>
      <td>28.561</td>
      <td>28.426</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGKPVICATQMLESMIK</th>
      <td>30.878</td>
      <td>32.100</td>
      <td>31.672</td>
      <td>31.451</td>
      <td>30.789</td>
      <td>31.315</td>
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
      <th>IAFAITAIK</th>
      <td>30.131</td>
      <td>29.930</td>
      <td>29.764</td>
      <td>30.297</td>
      <td>30.030</td>
      <td>30.094</td>
    </tr>
    <tr>
      <th>IFTSIGEDYDER</th>
      <td>27.825</td>
      <td>28.053</td>
      <td>28.317</td>
      <td>28.100</td>
      <td>27.903</td>
      <td>27.778</td>
    </tr>
    <tr>
      <th>LYGSAGPPPTGEEDTAEK</th>
      <td>27.519</td>
      <td>27.649</td>
      <td>27.945</td>
      <td>27.687</td>
      <td>27.610</td>
      <td>27.363</td>
    </tr>
    <tr>
      <th>THIQDNHDGTYTVAYVPDVTGR</th>
      <td>28.003</td>
      <td>27.993</td>
      <td>27.812</td>
      <td>27.870</td>
      <td>27.775</td>
      <td>27.751</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
      <td>31.820</td>
      <td>31.680</td>
      <td>32.365</td>
      <td>31.614</td>
      <td>31.775</td>
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
      <td>0.682</td>
      <td>-0.391</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.661</td>
      <td>-0.511</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.942</td>
      <td>-0.760</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.000</td>
      <td>-0.021</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.819</td>
      <td>-0.755</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_122_1.png)
    



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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_123_1.png)
    


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
      <th>AGAGSATLSMAYAGAR</th>
      <th>AGKPVICATQMLESMIK</th>
      <th>ALTGGIAHLFK</th>
      <th>AMVSEFLK</th>
      <th>ANLQIDQINTDLNLER</th>
      <th>ARFEELNADLFR</th>
      <th>ASGPGLNTTGVPASLPVEFTIDAK</th>
      <th>ATAVMPDGQFK</th>
      <th>ATESGAQSAPLPMEGVDISPK</th>
      <th>DALSDLALHFLNK</th>
      <th>...</th>
      <th>VALVYGQMNEPPGAR</th>
      <th>VFDAIMNFK</th>
      <th>VHLVGIDIFTGK</th>
      <th>VHVIFNYK</th>
      <th>VIMVTGDHPITAK</th>
      <th>VMTIAPGLFGTPLLTSLPEK</th>
      <th>VNQIGSVTESLQACK</th>
      <th>WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK</th>
      <th>YGINTTDIFQTVDLWEGK</th>
      <th>YNEQHVPGSPFTAR</th>
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
      <td>0.695</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.617</td>
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
      <td>0.615</td>
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
      <td>0.637</td>
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
      <td>0.708</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.563</td>
      <td>0.558</td>
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
      <td>0.548</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.683</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.522</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.643</td>
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
    
    Optimizer used: <function Adam at 0x0000024BFF047040>
    Loss function: <function loss_fct_vae at 0x0000024BFF065940>
    
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




    
![png](latent_2D_75_10_files/latent_2D_75_10_136_2.png)
    



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
      <td>1985.144531</td>
      <td>221.347229</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1947.587036</td>
      <td>213.203735</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1889.162231</td>
      <td>205.512177</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1840.202515</td>
      <td>198.488541</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1808.789551</td>
      <td>196.093262</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1787.738892</td>
      <td>197.587097</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1772.630981</td>
      <td>195.934479</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1762.803589</td>
      <td>195.448959</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1756.211548</td>
      <td>195.680557</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1750.741211</td>
      <td>195.781769</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_138_1.png)
    


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




    tensor(3070.2314)




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
      <th>AGAGSATLSMAYAGAR</th>
      <td>28.868</td>
      <td>28.930</td>
      <td>28.869</td>
      <td>28.985</td>
      <td>28.226</td>
      <td>27.951</td>
      <td>29.006</td>
    </tr>
    <tr>
      <th>ANLQIDQINTDLNLER</th>
      <td>27.860</td>
      <td>28.733</td>
      <td>28.535</td>
      <td>27.932</td>
      <td>27.617</td>
      <td>27.489</td>
      <td>28.644</td>
    </tr>
    <tr>
      <th>DHENIVIAK</th>
      <td>29.417</td>
      <td>29.981</td>
      <td>30.037</td>
      <td>29.301</td>
      <td>29.290</td>
      <td>28.946</td>
      <td>30.058</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>28.346</td>
      <td>29.529</td>
      <td>29.261</td>
      <td>27.899</td>
      <td>28.561</td>
      <td>28.426</td>
      <td>29.328</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGKPVICATQMLESMIK</th>
      <td>30.878</td>
      <td>32.100</td>
      <td>31.672</td>
      <td>31.451</td>
      <td>30.789</td>
      <td>31.315</td>
      <td>31.746</td>
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
      <th>IAFAITAIK</th>
      <td>30.131</td>
      <td>29.930</td>
      <td>29.764</td>
      <td>30.297</td>
      <td>30.030</td>
      <td>30.094</td>
      <td>29.870</td>
    </tr>
    <tr>
      <th>IFTSIGEDYDER</th>
      <td>27.825</td>
      <td>28.053</td>
      <td>28.317</td>
      <td>28.100</td>
      <td>27.903</td>
      <td>27.778</td>
      <td>28.488</td>
    </tr>
    <tr>
      <th>LYGSAGPPPTGEEDTAEK</th>
      <td>27.519</td>
      <td>27.649</td>
      <td>27.945</td>
      <td>27.687</td>
      <td>27.610</td>
      <td>27.363</td>
      <td>28.000</td>
    </tr>
    <tr>
      <th>THIQDNHDGTYTVAYVPDVTGR</th>
      <td>28.003</td>
      <td>27.993</td>
      <td>27.812</td>
      <td>27.870</td>
      <td>27.775</td>
      <td>27.751</td>
      <td>27.964</td>
    </tr>
    <tr>
      <th>VHLVGIDIFTGK</th>
      <td>32.256</td>
      <td>31.820</td>
      <td>31.680</td>
      <td>32.365</td>
      <td>31.614</td>
      <td>31.775</td>
      <td>31.746</td>
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
      <td>0.084</td>
      <td>-0.126</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.001</td>
      <td>0.046</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.092</td>
      <td>-0.112</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.061</td>
      <td>-0.084</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.233</td>
      <td>-0.210</td>
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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_146_1.png)
    



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
    


    
![png](latent_2D_75_10_files/latent_2D_75_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'ANLQIDQINTDLNLER'), ('20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007', 'DPFAHLPK'), ('20190111_QE8_nLC1_ASD_QC_HeLa_02', 'GPLMMYISK'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'MDATANDVPSPYEVR'), ('20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01', 'VALVYGQMNEPPGAR'), ('20190129_QE1_nLC2_GP_QC_MNT_HELA_02', 'SCMLTGTPESVQSAK'), ('20190204_QE2_NLC10_ANHO_QC_MNT_HELA_02', 'DIISDTSGDFRK'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm2', 'SAEFLLHMLK'), ('20190207_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01', 'AGAGSATLSMAYAGAR'), ('20190207_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01', 'DYGVYLEDSGHTLR'), ('20190220_QE2_NLC1_GP_QC_MNT_HELA_01', 'VMTIAPGLFGTPLLTSLPEK'), ('20190221_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'FIQENIFGICPHMTEDNK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_1', 'ANLQIDQINTDLNLER'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_25', 'TVTNAVVTVPAYFNDSQR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_27', 'TVTNAVVTVPAYFNDSQR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_15_24', 'ANLQIDQINTDLNLER'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_27', 'VFDAIMNFK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_18_30', 'VFDAIMNFK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_2', 'ALTGGIAHLFK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03', 'AGKPVICATQMLESMIK'), ('20190306_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'VNQIGSVTESLQACK'), ('20190310_QE2_NLC1_GP_MNT_HELA_01', 'DIISDTSGDFRK'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_01_20190317211403', 'ARFEELNADLFR'), ('20190320_QE9_nLC0_AnMu_MNT_Hela_50cm_01', 'THILLFLPK'), ('20190323_QE8_nLC14_RS_QC_MNT_Hela_50cm', 'IAFAITAIK'), ('20190401_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'IFTSIGEDYDER'), ('20190408_QE1_nLC2_GP_MNT_QC_hela_02_20190408131505', 'IFTSIGEDYDER'), ('20190415_QE8_nLC14_AL_QC_HeLa_02', 'TVTNAVVTVPAYFNDSQR'), ('20190425_QE9_nLC0_LiNi_QC_45cm_HeLa_01', 'AGAGSATLSMAYAGAR'), ('20190425_QE9_nLC0_LiNi_QC_45cm_HeLa_02', 'AGAGSATLSMAYAGAR'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'AGAGSATLSMAYAGAR'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'GAGTGGLGLAVEGPSEAK'), ('20190429_QX0_ChDe_MA_HeLa_500ng_LC07_1_BR14', 'YNEQHVPGSPFTAR'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15', 'VIMVTGDHPITAK'), ('20190506_QX8_MiWi_MA_HeLa_500ng_old', 'YNEQHVPGSPFTAR'), ('20190508_QE10_Evosep_LiNi_QC_EVOMNT_Hela_44min_1', 'YNEQHVPGSPFTAR'), ('20190508_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'YNEQHVPGSPFTAR'), ('20190508_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff_1', 'YNEQHVPGSPFTAR'), ('20190513_QE7_nLC7_MEM_QC_MNT_HeLa_01', 'DALSDLALHFLNK'), ('20190513_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'VIMVTGDHPITAK'), ('20190513_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'DALSDLALHFLNK'), ('20190514_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'ALTGGIAHLFK'), ('20190514_QX4_JiYu_MA_HeLa_500ng', 'IFTSIGEDYDER'), ('20190514_QX4_JiYu_MA_HeLa_500ng_BR14', 'SGPFGQIFRPDNFVFGQSGAGNNWAK'), ('20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02', 'TLSDYNIQK'), ('20190521_QX7_MaMu_MA_HeLaBr14_500ng', 'KLEEEQIILEDQNCK'), ('20190521_QX8_MiWi_MA_HeLa_BR14_500ng', 'KLEEEQIILEDQNCK'), ('20190527_QX4_IgPa_MA_HeLa_500ng', 'ASGPGLNTTGVPASLPVEFTIDAK'), ('20190604_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_01', 'ALTGGIAHLFK'), ('20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_01', 'VNQIGSVTESLQACK'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'ISMPDVDLHLK'), ('20190621_QE2_NLC1_GP_QC_MNT_HELA_01', 'ISMPDVDLHLK'), ('20190623_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_MUC_02', 'ALTGGIAHLFK'), ('20190702_QE3_nLC5_GF_QC_MNT_Hela_03', 'KLEEEQIILEDQNCK'), ('20190702_QE3_nLC5_TSB_QC_MNT_HELA_01', 'WSGPLSLQEVDEQPQHPLHVTYAGAAVDELGK'), ('20190702_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'AGKPVICATQMLESMIK'), ('20190708_QX7_MaMu_MA_HeLa_Br14_500ng', 'ISMPDLDLNLK'), ('20190708_QX7_MaMu_MA_HeLa_Br14_500ng', 'VALVYGQMNEPPGAR'), ('20190708_QX8_AnPi_MA_HeLa_BR14_500ng', 'VALVYGQMNEPPGAR'), ('20190709_QE3_nLC5_GF_QC_MNT_Hela_01', 'LYGSAGPPPTGEEDTAEK'), ('20190709_QE3_nLC5_GF_QC_MNT_Hela_02', 'LYGSAGPPPTGEEDTAEK'), ('20190719_QE1_nLC13_GP_QC_MNT_HELA_01', 'ANLQIDQINTDLNLER'), ('20190722_QX4_StEb_MA_HeLa_500ng', 'ISMPDLDLNLK'), ('20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar_2', 'ASGPGLNTTGVPASLPVEFTIDAK'), ('20190723_QX3_MiWi_MA_Hela_500ng_LC15', 'LQAALDDEEAGGRPAMEPGNGSLDLGGDSAGR'), ('20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_1', 'LMVALAK'), ('20190726_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'AMVSEFLK'), ('20190726_QX8_ChSc_MA_HeLa_500ng', 'GAGTGGLGLAVEGPSEAK'), ('20190731_QE8_nLC14_ASD_QC_MNT_HeLa_02', 'IAFAITAIK'), ('20190731_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_1', 'TVTNAVVTVPAYFNDSQR'), ('20190801_QX3_StEb_MA_Hela_500ng_LC15', 'SCMLTGTPESVQSAK'), ('20190802_QX7_AlRe_MA_HeLa_Br14_500ng', 'TFSHELSDFGLESTAGEIPVVAIR')]
    




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
      <td>0.489</td>
      <td>0.532</td>
      <td>1.571</td>
      <td>1.826</td>
      <td>1.938</td>
      <td>2.016</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.452</td>
      <td>0.477</td>
      <td>0.863</td>
      <td>0.993</td>
      <td>1.030</td>
      <td>1.007</td>
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
