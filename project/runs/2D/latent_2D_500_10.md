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
n_feat = 500
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
    VNVPVIGGHAGK                 990
    ELVYPPDYNPEGK                997
    HIMGQNVADYMR                 999
    SLEDQVEMLR                   982
    SLHDALCVIR                   995
    LMELHGEGSSSGK                992
    LMIEMDGTENK                  999
    ALPFWNEEIVPQIK               937
    TLHPDLGTDK                   997
    IVSRPEELREDDVGTGAGLLEIK      997
    KHPDASVNFSEFSK               998
    YADLTEDQLPSCESLK             998
    TEFLSFMNTELAAFTK             994
    NALESYAFNMK                  993
    IVVVTAGVR                    981
    VVVAENFDEIVNNENK             998
    NMDPLNDNIATLLHQSSDK          996
    LIAPVAEEEATVPNNK             997
    LVNHFVEEFK                   980
    GIHPTIISESFQK                999
    AMVSEFLK                     984
    AYHEQLSVAEITNACFEPANQMVK   1,000
    TAVVVGTITDDVR                999
    GLVEPVDVVDNADGTQTVNYVPSR     948
    DSYVGDEAQSK                  999
    TGTAEMSSILEER                999
    NAGVEGSLIVEK                 992
    YLTVAAVFR                    993
    LSVLGAITSVQQR                959
    NTDEMVELR                    996
    TLGILGLGR                    963
    GVNLPGAAVDLPAVSEK          1,000
    HQPTAIIAK                    980
    RLAPEYEAAATR                 953
    STGEAFVQFASQEIAEK            992
    RGEAHLAVNDFELAR              992
    IIQLLDDYPK                   963
    ILDQGEDFPASEMTR              993
    HELLQPFNVLYEK                996
    ELAPYDENWFYTR                956
    YDDMATCMK                    984
    IFVGGLSPDTPEEK               987
    LNNLVLFDK                    956
    SYCAEIAHNVSSK                990
    IVLLDSSLEYK                  987
    AHSSMVGVNLPQK              1,000
    VSFELFADK                    975
    RFPGYDSESK                   992
    NSVTPDMMEEMYK                993
    IYELAAGGTAVGTGLNTR           992
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
      <th>VNVPVIGGHAGK</th>
      <td>27.985</td>
    </tr>
    <tr>
      <th>ELVYPPDYNPEGK</th>
      <td>26.558</td>
    </tr>
    <tr>
      <th>HIMGQNVADYMR</th>
      <td>28.806</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>28.126</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>IVLLDSSLEYK</th>
      <td>28.844</td>
    </tr>
    <tr>
      <th>AHSSMVGVNLPQK</th>
      <td>28.602</td>
    </tr>
    <tr>
      <th>RFPGYDSESK</th>
      <td>28.980</td>
    </tr>
    <tr>
      <th>NSVTPDMMEEMYK</th>
      <td>28.778</td>
    </tr>
    <tr>
      <th>IYELAAGGTAVGTGLNTR</th>
      <td>26.758</td>
    </tr>
  </tbody>
</table>
<p>49332 rows × 1 columns</p>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_24_1.png)
    



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
      <td>0.987</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.020</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.880</td>
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
      <th>VNVPVIGGHAGK</th>
      <td>27.985</td>
    </tr>
    <tr>
      <th>ELVYPPDYNPEGK</th>
      <td>26.558</td>
    </tr>
    <tr>
      <th>HIMGQNVADYMR</th>
      <td>28.806</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
    </tr>
    <tr>
      <th>SLHDALCVIR</th>
      <td>28.126</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>IVLLDSSLEYK</th>
      <td>28.844</td>
    </tr>
    <tr>
      <th>AHSSMVGVNLPQK</th>
      <td>28.602</td>
    </tr>
    <tr>
      <th>RFPGYDSESK</th>
      <td>28.980</td>
    </tr>
    <tr>
      <th>NSVTPDMMEEMYK</th>
      <td>28.778</td>
    </tr>
    <tr>
      <th>IYELAAGGTAVGTGLNTR</th>
      <td>26.758</td>
    </tr>
  </tbody>
</table>
<p>49332 rows × 1 columns</p>
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
      <th>AHSSMVGVNLPQK</th>
      <td>29.949</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>AHSSMVGVNLPQK</th>
      <td>28.333</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_01</th>
      <th>AHSSMVGVNLPQK</th>
      <td>27.104</td>
    </tr>
    <tr>
      <th>20190528_QX1_PhGe_MA_HeLa_DMSO_500ng_LC14_190528164924</th>
      <th>AHSSMVGVNLPQK</th>
      <td>32.006</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_4</th>
      <th>AHSSMVGVNLPQK</th>
      <td>30.812</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190405_QE1_nLC2_GP_MNT_QC_hela_01</th>
      <th>YLTVAAVFR</th>
      <td>30.268</td>
    </tr>
    <tr>
      <th>20190706_QX4_MiWi_MA_HeLa_500ng</th>
      <th>YLTVAAVFR</th>
      <td>32.232</td>
    </tr>
    <tr>
      <th>20190527_QE4_LC12_AS_QC_MNT_HeLa_01</th>
      <th>YLTVAAVFR</th>
      <td>30.311</td>
    </tr>
    <tr>
      <th>20190514_QX6_ChDe_MA_HeLa_Br14_500ng_LC09</th>
      <th>YLTVAAVFR</th>
      <td>31.461</td>
    </tr>
    <tr>
      <th>20190619_QE7_nLC7_AP_QC_MNT_HeLa_02</th>
      <th>YLTVAAVFR</th>
      <td>30.368</td>
    </tr>
  </tbody>
</table>
<p>44399 rows × 1 columns</p>
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
    Shape in validation: (990, 50)
    




    ((990, 50), (990, 50))



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
      <th>AHSSMVGVNLPQK</th>
      <td>29.899</td>
      <td>30.250</td>
      <td>30.245</td>
      <td>30.008</td>
    </tr>
    <tr>
      <th>DSYVGDEAQSK</th>
      <td>32.783</td>
      <td>33.542</td>
      <td>33.310</td>
      <td>33.189</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>28.098</td>
    </tr>
    <tr>
      <th>TLGILGLGR</th>
      <td>28.345</td>
      <td>28.815</td>
      <td>28.585</td>
      <td>28.547</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>29.217</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>30.196</td>
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
      <th>VVVAENFDEIVNNENK</th>
      <td>29.613</td>
      <td>29.629</td>
      <td>29.761</td>
      <td>29.744</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <td>31.539</td>
      <td>31.468</td>
      <td>31.570</td>
      <td>31.623</td>
    </tr>
    <tr>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>30.671</td>
      <td>29.922</td>
      <td>30.016</td>
      <td>30.115</td>
    </tr>
    <tr>
      <th>LSVLGAITSVQQR</th>
      <td>27.476</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>27.425</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
      <td>30.274</td>
      <td>30.238</td>
      <td>30.494</td>
    </tr>
  </tbody>
</table>
<p>4933 rows × 4 columns</p>
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
      <th>20181230_QE6_nLC6_CSC_QC_HeLa_03</th>
      <th>KHPDASVNFSEFSK</th>
      <td>30.219</td>
      <td>30.205</td>
      <td>30.232</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190107_QE10_nLC0_KS_QC_MNT_HeLa_01</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <td>31.313</td>
      <td>31.238</td>
      <td>31.486</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190110_QE8_nLC14_JM_QC_MNT_HeLa_02</th>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.318</td>
      <td>30.274</td>
      <td>30.238</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_01</th>
      <th>LVNHFVEEFK</th>
      <td>30.359</td>
      <td>28.285</td>
      <td>29.359</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01</th>
      <th>SLEDQVEMLR</th>
      <td>28.443</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_02</th>
      <th>SLEDQVEMLR</th>
      <td>28.568</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_01</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <td>30.907</td>
      <td>31.238</td>
      <td>31.486</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190205_QE7_nLC7_MEM_QC_MNT_HeLa_03</th>
      <th>HQPTAIIAK</th>
      <td>33.074</td>
      <td>32.208</td>
      <td>32.028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190205_QE7_nLC7_MEM_QC_MNT_HeLa_04</th>
      <th>HQPTAIIAK</th>
      <td>33.122</td>
      <td>32.208</td>
      <td>32.028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_03</th>
      <th>HQPTAIIAK</th>
      <td>31.785</td>
      <td>32.208</td>
      <td>32.028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190219_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>VSFELFADK</th>
      <td>30.071</td>
      <td>31.015</td>
      <td>31.382</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_1_20190225173940</th>
      <th>LVNHFVEEFK</th>
      <td>27.129</td>
      <td>28.285</td>
      <td>29.359</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_2</th>
      <th>LVNHFVEEFK</th>
      <td>27.446</td>
      <td>28.285</td>
      <td>29.359</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE9_nLC0_RS_MNT_Hela_02</th>
      <th>VSFELFADK</th>
      <td>29.607</td>
      <td>31.015</td>
      <td>31.382</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190228_QE4_LC12_JE_QC_MNT_HeLa_01</th>
      <th>VNVPVIGGHAGK</th>
      <td>26.222</td>
      <td>27.439</td>
      <td>28.314</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190301_QE1_nLC2_ANHO_QC_MNT_HELA_01_20190303025443</th>
      <th>IIQLLDDYPK</th>
      <td>30.927</td>
      <td>30.289</td>
      <td>30.334</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190304_QE10_nLC0_KS_QC_MNT_HeLa_02</th>
      <th>LSVLGAITSVQQR</th>
      <td>27.695</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190318_QE2_NLC1_AB_MNT_HELA_04</th>
      <th>AHSSMVGVNLPQK</th>
      <td>27.985</td>
      <td>30.250</td>
      <td>30.245</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190402_QE6_LC6_AS_QC_MNT_HeLa_03</th>
      <th>YADLTEDQLPSCESLK</th>
      <td>30.281</td>
      <td>29.916</td>
      <td>30.035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190402_QE7_nLC3_AL_QC_MNT_HeLa_01</th>
      <th>YADLTEDQLPSCESLK</th>
      <td>28.349</td>
      <td>29.916</td>
      <td>30.035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190408_QE1_nLC2_GP_MNT_QC_hela_02_20190408131505</th>
      <th>NAGVEGSLIVEK</th>
      <td>30.255</td>
      <td>30.729</td>
      <td>30.952</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190409_QE1_nLC2_ANHO_MNT_QC_hela_02</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>30.556</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190423_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff</th>
      <th>YADLTEDQLPSCESLK</th>
      <td>31.712</td>
      <td>29.916</td>
      <td>30.035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190423_QX7_JuSc_MA_HeLaBr14_500ng_LC02</th>
      <th>LNNLVLFDK</th>
      <td>32.189</td>
      <td>30.242</td>
      <td>30.313</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190425_QX4_JoSw_MA_HeLa_500ng_BR13_standard</th>
      <th>YLTVAAVFR</th>
      <td>30.961</td>
      <td>30.440</td>
      <td>30.569</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02</th>
      <th>RLAPEYEAAATR</th>
      <td>29.287</td>
      <td>28.819</td>
      <td>28.546</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336</th>
      <th>LSVLGAITSVQQR</th>
      <td>29.485</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190508_QE8_nLC14_ASD_QC_MNT_HeLa_01</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>25.961</td>
      <td>27.146</td>
      <td>27.151</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190510_QE2_NLC1_GP_MNT_HELA_02</th>
      <th>KHPDASVNFSEFSK</th>
      <td>29.617</td>
      <td>30.205</td>
      <td>30.232</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>SYCAEIAHNVSSK</th>
      <td>30.987</td>
      <td>30.573</td>
      <td>30.489</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QX7_ChDe_MA_HeLaBr14_500ng</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <td>29.255</td>
      <td>27.146</td>
      <td>27.151</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QX8_MiWi_MA_HeLa_BR14_500ng</th>
      <th>VSFELFADK</th>
      <td>33.204</td>
      <td>31.015</td>
      <td>31.382</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190514_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14</th>
      <th>SYCAEIAHNVSSK</th>
      <td>31.794</td>
      <td>30.573</td>
      <td>30.489</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QX3_AsJa_MA_Hela_500ng_LC15</th>
      <th>IIQLLDDYPK</th>
      <td>31.893</td>
      <td>30.289</td>
      <td>30.334</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QX4_JiYu_MA_HeLa_500ng_BR14</th>
      <th>IIQLLDDYPK</th>
      <td>31.713</td>
      <td>30.289</td>
      <td>30.334</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190522_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14_190524170803</th>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>32.297</td>
      <td>29.922</td>
      <td>30.016</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX7_IgPa_MA_HeLa_Br14_500ng</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>32.525</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190618_QX4_JiYu_MA_HeLa_500ng_centroid</th>
      <th>TGTAEMSSILEER</th>
      <td>27.211</td>
      <td>30.089</td>
      <td>30.100</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01</th>
      <th>NAGVEGSLIVEK</th>
      <td>30.486</td>
      <td>30.729</td>
      <td>30.952</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE6_LC4_AS_QC_MNT_HeLa_02</th>
      <th>RFPGYDSESK</th>
      <td>30.027</td>
      <td>29.007</td>
      <td>29.082</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE6_LC4_AS_QC_MNT_HeLa_03</th>
      <th>RFPGYDSESK</th>
      <td>28.646</td>
      <td>29.007</td>
      <td>29.082</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_02</th>
      <th>DSYVGDEAQSK</th>
      <td>34.087</td>
      <td>33.542</td>
      <td>33.310</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190627_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>29.664</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190628_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</th>
      <th>YDDMATCMK</th>
      <td>28.010</td>
      <td>28.358</td>
      <td>28.418</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>SYCAEIAHNVSSK</th>
      <td>31.655</td>
      <td>30.573</td>
      <td>30.489</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190705_QX0_AnBr_MA_HeLa_500ng_LC07_01_190707104639</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>32.939</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190708_QE9_nLC2_AGF_QC_MNT_HeLa_01_20190709092142</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>31.120</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01</th>
      <th>ELAPYDENWFYTR</th>
      <td>30.610</td>
      <td>29.427</td>
      <td>29.245</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190716_QE1_nLC13_ANHO_QC_MNT_HELA_01</th>
      <th>LVNHFVEEFK</th>
      <td>27.221</td>
      <td>28.285</td>
      <td>29.359</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190726_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_4thcolumn_1</th>
      <th>VNVPVIGGHAGK</th>
      <td>30.991</td>
      <td>27.439</td>
      <td>28.314</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190801_QX3_StEb_MA_Hela_500ng_LC15</th>
      <th>NMDPLNDNIATLLHQSSDK</th>
      <td>28.940</td>
      <td>29.917</td>
      <td>29.589</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>SLEDQVEMLR</th>
      <td>30.574</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
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
      <td>ALPFWNEEIVPQIK</td>
      <td>29.497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AMVSEFLK</td>
      <td>28.042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AYHEQLSVAEITNACFEPANQMVK</td>
      <td>30.245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ELAPYDENWFYTR</td>
      <td>27.436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ELVYPPDYNPEGK</td>
      <td>26.558</td>
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
      <td>AHSSMVGVNLPQK</td>
      <td>29.899</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DSYVGDEAQSK</td>
      <td>32.783</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>SLEDQVEMLR</td>
      <td>28.105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TLGILGLGR</td>
      <td>28.345</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>ALPFWNEEIVPQIK</td>
      <td>29.217</td>
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
      <td>20190412_QE6_LC6_AS_QC_MNT_HeLa_01</td>
      <td>LNNLVLFDK</td>
      <td>30.486</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190718_QE9_nLC9_NHS_MNT_HELA_50cm_01</td>
      <td>YDDMATCMK</td>
      <td>26.540</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190621_QX2_SeVW_MA_HeLa_500ng_LC05</td>
      <td>HELLQPFNVLYEK</td>
      <td>30.399</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190422_QX8_JuSc_MA_HeLa_500ng_1</td>
      <td>KHPDASVNFSEFSK</td>
      <td>30.495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190205_QE7_nLC7_MEM_QC_MNT_HeLa_03</td>
      <td>SLHDALCVIR</td>
      <td>30.043</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190702_QE3_nLC5_GF_QC_MNT_Hela_01</td>
      <td>NTDEMVELR</td>
      <td>27.828</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190417_QE4_LC12_JE-IAH_QC_MNT_HeLa_03</td>
      <td>VSFELFADK</td>
      <td>30.373</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190211_QE4_nLC12_SIS_QC_MNT_Hela_2</td>
      <td>RFPGYDSESK</td>
      <td>28.667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190618_QX3_LiSc_MA_Hela_500ng_LC15</td>
      <td>NTDEMVELR</td>
      <td>30.943</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190528_QX6_AsJa_MA_HeLa_Br14_500ng_LC09</td>
      <td>HIMGQNVADYMR</td>
      <td>30.559</td>
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
      <td>20190606_QE4_LC12_JE_QC_MNT_HeLa_03</td>
      <td>RGEAHLAVNDFELAR</td>
      <td>28.706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190510_QE2_NLC1_GP_MNT_HELA_02</td>
      <td>TLGILGLGR</td>
      <td>28.640</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190726_QE7_nLC7_MEM_QC_MNT_HeLa_03</td>
      <td>AYHEQLSVAEITNACFEPANQMVK</td>
      <td>31.619</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190621_QX4_JoMu_MA_HeLa_500ng_190621161214</td>
      <td>TGTAEMSSILEER</td>
      <td>31.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190225_QE10_PhGe_Evosep_88min_HeLa_6</td>
      <td>AHSSMVGVNLPQK</td>
      <td>29.437</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190626_QE7_nLC7_DS_QC_MNT_HeLa_02</td>
      <td>VNVPVIGGHAGK</td>
      <td>26.298</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190626_QX6_ChDe_MA_HeLa_500ng_LC09</td>
      <td>TEFLSFMNTELAAFTK</td>
      <td>32.885</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190803_QE9_nLC13_RG_SA_HeLa_50cm_350ng</td>
      <td>YDDMATCMK</td>
      <td>28.988</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190715_QE2_NLC1_ANHO_MNT_HELA_02</td>
      <td>LMIEMDGTENK</td>
      <td>32.776</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01</td>
      <td>AHSSMVGVNLPQK</td>
      <td>31.777</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (991, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1372, 155)



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
     'n_samples': 991,
     'y_range': (20, 36)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1982       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 991        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,126
    Total trainable params: 3,126
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001CBC0506040>
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
      <td>1.746584</td>
      <td>1.643620</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.664948</td>
      <td>0.672359</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.571682</td>
      <td>0.615115</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.514574</td>
      <td>0.563398</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.547034</td>
      <td>0.532022</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.461233</td>
      <td>0.518302</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.432974</td>
      <td>0.509214</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.416380</td>
      <td>0.506805</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.377849</td>
      <td>0.502489</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.439591</td>
      <td>0.502159</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_58_1.png)
    


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
      <th>3,161</th>
      <td>634</td>
      <td>34</td>
      <td>28.706</td>
    </tr>
    <tr>
      <th>2,512</th>
      <td>502</td>
      <td>43</td>
      <td>28.640</td>
    </tr>
    <tr>
      <th>4,616</th>
      <td>928</td>
      <td>4</td>
      <td>31.619</td>
    </tr>
    <tr>
      <th>3,556</th>
      <td>717</td>
      <td>42</td>
      <td>31.227</td>
    </tr>
    <tr>
      <th>1,136</th>
      <td>217</td>
      <td>1</td>
      <td>29.437</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>10</td>
      <td>31.245</td>
    </tr>
    <tr>
      <th>484</th>
      <td>97</td>
      <td>3</td>
      <td>27.915</td>
    </tr>
    <tr>
      <th>2,518</th>
      <td>503</td>
      <td>37</td>
      <td>27.225</td>
    </tr>
    <tr>
      <th>1,407</th>
      <td>276</td>
      <td>44</td>
      <td>32.527</td>
    </tr>
    <tr>
      <th>1,068</th>
      <td>204</td>
      <td>45</td>
      <td>28.599</td>
    </tr>
  </tbody>
</table>
<p>4933 rows × 3 columns</p>
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
      <th>AHSSMVGVNLPQK</th>
      <td>29.899</td>
      <td>30.250</td>
      <td>30.245</td>
      <td>30.008</td>
      <td>29.380</td>
    </tr>
    <tr>
      <th>DSYVGDEAQSK</th>
      <td>32.783</td>
      <td>33.542</td>
      <td>33.310</td>
      <td>33.189</td>
      <td>32.801</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>28.098</td>
      <td>27.852</td>
    </tr>
    <tr>
      <th>TLGILGLGR</th>
      <td>28.345</td>
      <td>28.815</td>
      <td>28.585</td>
      <td>28.547</td>
      <td>27.590</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>29.217</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>30.196</td>
      <td>30.150</td>
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
      <th>VVVAENFDEIVNNENK</th>
      <td>29.613</td>
      <td>29.629</td>
      <td>29.761</td>
      <td>29.744</td>
      <td>29.211</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <td>31.539</td>
      <td>31.468</td>
      <td>31.570</td>
      <td>31.623</td>
      <td>31.089</td>
    </tr>
    <tr>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>30.671</td>
      <td>29.922</td>
      <td>30.016</td>
      <td>30.115</td>
      <td>29.536</td>
    </tr>
    <tr>
      <th>LSVLGAITSVQQR</th>
      <td>27.476</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>27.425</td>
      <td>27.084</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
      <td>30.274</td>
      <td>30.238</td>
      <td>30.494</td>
      <td>30.053</td>
    </tr>
  </tbody>
</table>
<p>4933 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.036
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.062
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.184
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.110
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.162
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_66_1.png)
    



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
      <td>-0.231</td>
      <td>-0.116</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.230</td>
      <td>-0.149</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.077</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.041</td>
      <td>-0.205</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.059</td>
      <td>-0.077</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_68_1.png)
    



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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_69_1.png)
    


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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK</th>
      <th>TGTAEMSSILEER</th>
      <th>TLGILGLGR</th>
      <th>TLHPDLGTDK</th>
      <th>VNVPVIGGHAGK</th>
      <th>VSFELFADK</th>
      <th>VVVAENFDEIVNNENK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YDDMATCMK</th>
      <th>YLTVAAVFR</th>
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
      <td>29.899</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.783</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.345</td>
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
      <td>29.217</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26.450</td>
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
      <td>31.245</td>
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
      <td>30.178</td>
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
      <td>30.452</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.954</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.494</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>29.912</td>
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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLGILGLGR_na</th>
      <th>TLHPDLGTDK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VSFELFADK_na</th>
      <th>VVVAENFDEIVNNENK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YDDMATCMK_na</th>
      <th>YLTVAAVFR_na</th>
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
      <td>0.004</td>
      <td>-1.021</td>
      <td>-0.843</td>
      <td>-1.000</td>
      <td>0.197</td>
      <td>-1.528</td>
      <td>-1.082</td>
      <td>-1.328</td>
      <td>0.895</td>
      <td>-0.768</td>
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
      <td>-0.186</td>
      <td>0.160</td>
      <td>-1.063</td>
      <td>-0.965</td>
      <td>-0.430</td>
      <td>-1.560</td>
      <td>-0.079</td>
      <td>-1.195</td>
      <td>0.715</td>
      <td>-0.760</td>
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
      <td>-0.248</td>
      <td>0.021</td>
      <td>-1.083</td>
      <td>-0.733</td>
      <td>0.156</td>
      <td>-0.526</td>
      <td>0.880</td>
      <td>-0.647</td>
      <td>-0.080</td>
      <td>-0.169</td>
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
      <td>0.105</td>
      <td>-1.198</td>
      <td>-0.791</td>
      <td>-0.070</td>
      <td>-0.031</td>
      <td>-0.972</td>
      <td>0.430</td>
      <td>-1.200</td>
      <td>1.129</td>
      <td>0.024</td>
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
      <td>-0.142</td>
      <td>0.033</td>
      <td>0.552</td>
      <td>-0.528</td>
      <td>0.197</td>
      <td>0.527</td>
      <td>-0.571</td>
      <td>-1.153</td>
      <td>-0.003</td>
      <td>-0.212</td>
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
      <td>-0.222</td>
      <td>0.720</td>
      <td>0.263</td>
      <td>1.474</td>
      <td>0.216</td>
      <td>0.683</td>
      <td>0.695</td>
      <td>0.519</td>
      <td>-0.003</td>
      <td>0.902</td>
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
      <td>0.583</td>
      <td>-0.071</td>
      <td>-0.047</td>
      <td>0.097</td>
      <td>0.259</td>
      <td>-0.140</td>
      <td>-0.744</td>
      <td>-0.155</td>
      <td>-0.364</td>
      <td>-0.626</td>
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
      <td>0.391</td>
      <td>0.160</td>
      <td>-0.073</td>
      <td>0.049</td>
      <td>0.463</td>
      <td>0.212</td>
      <td>-0.262</td>
      <td>-0.048</td>
      <td>-0.414</td>
      <td>-0.672</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>-1.705</td>
      <td>0.067</td>
      <td>-2.988</td>
      <td>-0.070</td>
      <td>0.389</td>
      <td>0.043</td>
      <td>-0.079</td>
      <td>-0.007</td>
      <td>-0.756</td>
      <td>-0.700</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-1.502</td>
      <td>-3.233</td>
      <td>0.041</td>
      <td>-0.070</td>
      <td>0.170</td>
      <td>-0.058</td>
      <td>-0.348</td>
      <td>-0.051</td>
      <td>-1.005</td>
      <td>-0.530</td>
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
  </tbody>
</table>
<p>990 rows × 100 columns</p>
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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLGILGLGR_na</th>
      <th>TLHPDLGTDK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VSFELFADK_na</th>
      <th>VVVAENFDEIVNNENK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YDDMATCMK_na</th>
      <th>YLTVAAVFR_na</th>
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
      <td>0.004</td>
      <td>-0.912</td>
      <td>-0.775</td>
      <td>-0.956</td>
      <td>0.208</td>
      <td>-1.398</td>
      <td>-1.033</td>
      <td>-1.268</td>
      <td>0.826</td>
      <td>-0.747</td>
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
      <td>-0.176</td>
      <td>0.175</td>
      <td>-0.983</td>
      <td>-0.923</td>
      <td>-0.387</td>
      <td>-1.427</td>
      <td>-0.083</td>
      <td>-1.141</td>
      <td>0.660</td>
      <td>-0.740</td>
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
      <td>-0.235</td>
      <td>0.046</td>
      <td>-1.001</td>
      <td>-0.703</td>
      <td>0.170</td>
      <td>-0.468</td>
      <td>0.824</td>
      <td>-0.623</td>
      <td>-0.074</td>
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
      <td>False</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.100</td>
      <td>-1.074</td>
      <td>-0.726</td>
      <td>-0.074</td>
      <td>-0.008</td>
      <td>-0.882</td>
      <td>0.398</td>
      <td>-1.147</td>
      <td>1.043</td>
      <td>0.005</td>
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
      <td>-0.134</td>
      <td>0.058</td>
      <td>0.539</td>
      <td>-0.508</td>
      <td>0.208</td>
      <td>0.508</td>
      <td>-0.549</td>
      <td>-1.102</td>
      <td>-0.004</td>
      <td>-0.220</td>
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
      <td>-0.210</td>
      <td>0.690</td>
      <td>0.267</td>
      <td>1.391</td>
      <td>0.226</td>
      <td>0.653</td>
      <td>0.650</td>
      <td>0.483</td>
      <td>-0.004</td>
      <td>0.839</td>
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
      <td>0.553</td>
      <td>-0.038</td>
      <td>-0.025</td>
      <td>0.085</td>
      <td>0.268</td>
      <td>-0.110</td>
      <td>-0.713</td>
      <td>-0.156</td>
      <td>-0.337</td>
      <td>-0.613</td>
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
      <td>0.371</td>
      <td>0.175</td>
      <td>-0.049</td>
      <td>0.039</td>
      <td>0.461</td>
      <td>0.217</td>
      <td>-0.257</td>
      <td>-0.054</td>
      <td>-0.383</td>
      <td>-0.656</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>-1.617</td>
      <td>0.089</td>
      <td>-2.796</td>
      <td>-0.074</td>
      <td>0.391</td>
      <td>0.060</td>
      <td>-0.083</td>
      <td>-0.016</td>
      <td>-0.699</td>
      <td>-0.682</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-1.424</td>
      <td>-2.947</td>
      <td>0.058</td>
      <td>-0.074</td>
      <td>0.183</td>
      <td>-0.034</td>
      <td>-0.338</td>
      <td>-0.057</td>
      <td>-0.929</td>
      <td>-0.521</td>
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
  </tbody>
</table>
<p>990 rows × 100 columns</p>
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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
      <td>990.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000</td>
      <td>0.028</td>
      <td>0.019</td>
      <td>-0.007</td>
      <td>0.021</td>
      <td>0.020</td>
      <td>-0.009</td>
      <td>-0.009</td>
      <td>-0.001</td>
      <td>-0.018</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.949</td>
      <td>0.921</td>
      <td>0.943</td>
      <td>0.949</td>
      <td>0.950</td>
      <td>0.928</td>
      <td>0.947</td>
      <td>0.948</td>
      <td>0.924</td>
      <td>0.950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.657</td>
      <td>-6.112</td>
      <td>-5.117</td>
      <td>-3.437</td>
      <td>-6.591</td>
      <td>-5.496</td>
      <td>-4.020</td>
      <td>-4.384</td>
      <td>-4.563</td>
      <td>-4.854</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.425</td>
      <td>-0.308</td>
      <td>-0.335</td>
      <td>-0.474</td>
      <td>-0.181</td>
      <td>-0.328</td>
      <td>-0.558</td>
      <td>-0.481</td>
      <td>-0.515</td>
      <td>-0.584</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.004</td>
      <td>0.175</td>
      <td>0.167</td>
      <td>-0.074</td>
      <td>0.208</td>
      <td>0.141</td>
      <td>-0.083</td>
      <td>-0.088</td>
      <td>-0.004</td>
      <td>-0.179</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.568</td>
      <td>0.564</td>
      <td>0.555</td>
      <td>0.437</td>
      <td>0.533</td>
      <td>0.536</td>
      <td>0.699</td>
      <td>0.524</td>
      <td>0.548</td>
      <td>0.609</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.222</td>
      <td>2.078</td>
      <td>2.686</td>
      <td>2.761</td>
      <td>1.699</td>
      <td>1.936</td>
      <td>2.185</td>
      <td>2.044</td>
      <td>2.134</td>
      <td>2.462</td>
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




    ((#50) ['AHSSMVGVNLPQK','ALPFWNEEIVPQIK','AMVSEFLK','AYHEQLSVAEITNACFEPANQMVK','DSYVGDEAQSK','ELAPYDENWFYTR','ELVYPPDYNPEGK','GIHPTIISESFQK','GLVEPVDVVDNADGTQTVNYVPSR','GVNLPGAAVDLPAVSEK'...],
     (#50) ['AHSSMVGVNLPQK_na','ALPFWNEEIVPQIK_na','AMVSEFLK_na','AYHEQLSVAEITNACFEPANQMVK_na','DSYVGDEAQSK_na','ELAPYDENWFYTR_na','ELVYPPDYNPEGK_na','GIHPTIISESFQK_na','GLVEPVDVVDNADGTQTVNYVPSR_na','GVNLPGAAVDLPAVSEK_na'...])




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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>94.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>95.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.037</td>
      <td>0.017</td>
      <td>-0.015</td>
      <td>-0.083</td>
      <td>0.022</td>
      <td>0.030</td>
      <td>0.093</td>
      <td>0.065</td>
      <td>0.229</td>
      <td>0.012</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.019</td>
      <td>0.933</td>
      <td>1.094</td>
      <td>1.026</td>
      <td>0.844</td>
      <td>0.898</td>
      <td>1.051</td>
      <td>1.072</td>
      <td>0.864</td>
      <td>1.082</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.711</td>
      <td>-3.268</td>
      <td>-4.378</td>
      <td>-3.530</td>
      <td>-3.215</td>
      <td>-2.460</td>
      <td>-3.230</td>
      <td>-4.345</td>
      <td>-2.342</td>
      <td>-2.847</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.629</td>
      <td>-0.431</td>
      <td>-0.392</td>
      <td>-0.583</td>
      <td>-0.262</td>
      <td>-0.442</td>
      <td>-0.486</td>
      <td>-0.501</td>
      <td>-0.415</td>
      <td>-0.670</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.040</td>
      <td>0.204</td>
      <td>0.143</td>
      <td>-0.011</td>
      <td>0.155</td>
      <td>0.053</td>
      <td>0.060</td>
      <td>0.004</td>
      <td>0.178</td>
      <td>-0.130</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.759</td>
      <td>0.655</td>
      <td>0.693</td>
      <td>0.568</td>
      <td>0.560</td>
      <td>0.648</td>
      <td>0.904</td>
      <td>0.934</td>
      <td>0.979</td>
      <td>1.065</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.850</td>
      <td>1.659</td>
      <td>2.128</td>
      <td>2.095</td>
      <td>1.584</td>
      <td>1.847</td>
      <td>1.939</td>
      <td>1.912</td>
      <td>2.038</td>
      <td>2.148</td>
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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLGILGLGR_val</th>
      <th>TLHPDLGTDK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VSFELFADK_val</th>
      <th>VVVAENFDEIVNNENK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YDDMATCMK_val</th>
      <th>YLTVAAVFR_val</th>
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
      <td>0.004</td>
      <td>-0.912</td>
      <td>-0.775</td>
      <td>-0.956</td>
      <td>0.208</td>
      <td>-1.398</td>
      <td>-1.033</td>
      <td>-1.268</td>
      <td>0.826</td>
      <td>-0.747</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.243</td>
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
      <td>-0.176</td>
      <td>0.175</td>
      <td>-0.983</td>
      <td>-0.923</td>
      <td>-0.387</td>
      <td>-1.427</td>
      <td>-0.083</td>
      <td>-1.141</td>
      <td>0.660</td>
      <td>-0.740</td>
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
      <td>-0.235</td>
      <td>0.046</td>
      <td>-1.001</td>
      <td>-0.703</td>
      <td>0.170</td>
      <td>-0.468</td>
      <td>0.824</td>
      <td>-0.623</td>
      <td>-0.074</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.100</td>
      <td>-1.074</td>
      <td>-0.726</td>
      <td>-0.074</td>
      <td>-0.008</td>
      <td>-0.882</td>
      <td>0.398</td>
      <td>-1.147</td>
      <td>1.043</td>
      <td>0.005</td>
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
      <td>-0.088</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.134</td>
      <td>0.058</td>
      <td>0.539</td>
      <td>-0.508</td>
      <td>0.208</td>
      <td>0.508</td>
      <td>-0.549</td>
      <td>-1.102</td>
      <td>-0.004</td>
      <td>-0.220</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.169</td>
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
      <td>-0.210</td>
      <td>0.690</td>
      <td>0.267</td>
      <td>1.391</td>
      <td>0.226</td>
      <td>0.653</td>
      <td>0.650</td>
      <td>0.483</td>
      <td>-0.004</td>
      <td>0.839</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.213</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.553</td>
      <td>-0.038</td>
      <td>-0.025</td>
      <td>0.085</td>
      <td>0.268</td>
      <td>-0.110</td>
      <td>-0.713</td>
      <td>-0.156</td>
      <td>-0.337</td>
      <td>-0.613</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.319</td>
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
      <td>0.371</td>
      <td>0.175</td>
      <td>-0.049</td>
      <td>0.039</td>
      <td>0.461</td>
      <td>0.217</td>
      <td>-0.257</td>
      <td>-0.054</td>
      <td>-0.383</td>
      <td>-0.656</td>
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
      <td>-1.617</td>
      <td>0.089</td>
      <td>-2.796</td>
      <td>-0.074</td>
      <td>0.391</td>
      <td>0.060</td>
      <td>-0.083</td>
      <td>-0.016</td>
      <td>-0.699</td>
      <td>-0.682</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.041</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-1.424</td>
      <td>-2.947</td>
      <td>0.058</td>
      <td>-0.074</td>
      <td>0.183</td>
      <td>-0.034</td>
      <td>-0.338</td>
      <td>-0.057</td>
      <td>-0.929</td>
      <td>-0.521</td>
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
<p>990 rows × 150 columns</p>
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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLGILGLGR_val</th>
      <th>TLHPDLGTDK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VSFELFADK_val</th>
      <th>VVVAENFDEIVNNENK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YDDMATCMK_val</th>
      <th>YLTVAAVFR_val</th>
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
      <td>0.004</td>
      <td>-0.912</td>
      <td>-0.775</td>
      <td>-0.956</td>
      <td>0.208</td>
      <td>-1.398</td>
      <td>-1.033</td>
      <td>-1.268</td>
      <td>0.826</td>
      <td>-0.747</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.243</td>
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
      <td>-0.176</td>
      <td>0.175</td>
      <td>-0.983</td>
      <td>-0.923</td>
      <td>-0.387</td>
      <td>-1.427</td>
      <td>-0.083</td>
      <td>-1.141</td>
      <td>0.660</td>
      <td>-0.740</td>
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
      <td>-0.235</td>
      <td>0.046</td>
      <td>-1.001</td>
      <td>-0.703</td>
      <td>0.170</td>
      <td>-0.468</td>
      <td>0.824</td>
      <td>-0.623</td>
      <td>-0.074</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.100</td>
      <td>-1.074</td>
      <td>-0.726</td>
      <td>-0.074</td>
      <td>-0.008</td>
      <td>-0.882</td>
      <td>0.398</td>
      <td>-1.147</td>
      <td>1.043</td>
      <td>0.005</td>
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
      <td>-0.088</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.134</td>
      <td>0.058</td>
      <td>0.539</td>
      <td>-0.508</td>
      <td>0.208</td>
      <td>0.508</td>
      <td>-0.549</td>
      <td>-1.102</td>
      <td>-0.004</td>
      <td>-0.220</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.169</td>
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
      <td>-0.210</td>
      <td>0.690</td>
      <td>0.267</td>
      <td>1.391</td>
      <td>0.226</td>
      <td>0.653</td>
      <td>0.650</td>
      <td>0.483</td>
      <td>-0.004</td>
      <td>0.839</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.213</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.553</td>
      <td>-0.038</td>
      <td>-0.025</td>
      <td>0.085</td>
      <td>0.268</td>
      <td>-0.110</td>
      <td>-0.713</td>
      <td>-0.156</td>
      <td>-0.337</td>
      <td>-0.613</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.319</td>
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
      <td>0.371</td>
      <td>0.175</td>
      <td>-0.049</td>
      <td>0.039</td>
      <td>0.461</td>
      <td>0.217</td>
      <td>-0.257</td>
      <td>-0.054</td>
      <td>-0.383</td>
      <td>-0.656</td>
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
      <td>-1.617</td>
      <td>0.089</td>
      <td>-2.796</td>
      <td>-0.074</td>
      <td>0.391</td>
      <td>0.060</td>
      <td>-0.083</td>
      <td>-0.016</td>
      <td>-0.699</td>
      <td>-0.682</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.041</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>-1.424</td>
      <td>-2.947</td>
      <td>0.058</td>
      <td>-0.074</td>
      <td>0.183</td>
      <td>-0.034</td>
      <td>-0.338</td>
      <td>-0.057</td>
      <td>-0.929</td>
      <td>-0.521</td>
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
<p>990 rows × 150 columns</p>
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
      <th>AHSSMVGVNLPQK_val</th>
      <th>ALPFWNEEIVPQIK_val</th>
      <th>AMVSEFLK_val</th>
      <th>AYHEQLSVAEITNACFEPANQMVK_val</th>
      <th>DSYVGDEAQSK_val</th>
      <th>ELAPYDENWFYTR_val</th>
      <th>ELVYPPDYNPEGK_val</th>
      <th>GIHPTIISESFQK_val</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR_val</th>
      <th>GVNLPGAAVDLPAVSEK_val</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLGILGLGR_val</th>
      <th>TLHPDLGTDK_val</th>
      <th>VNVPVIGGHAGK_val</th>
      <th>VSFELFADK_val</th>
      <th>VVVAENFDEIVNNENK_val</th>
      <th>YADLTEDQLPSCESLK_val</th>
      <th>YDDMATCMK_val</th>
      <th>YLTVAAVFR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>94.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>96.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>97.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.037</td>
      <td>0.017</td>
      <td>-0.015</td>
      <td>-0.083</td>
      <td>0.022</td>
      <td>0.030</td>
      <td>0.093</td>
      <td>0.065</td>
      <td>0.229</td>
      <td>0.012</td>
      <td>...</td>
      <td>0.065</td>
      <td>-0.132</td>
      <td>-0.085</td>
      <td>-0.046</td>
      <td>-0.077</td>
      <td>0.081</td>
      <td>0.021</td>
      <td>-0.021</td>
      <td>0.122</td>
      <td>0.126</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.019</td>
      <td>0.933</td>
      <td>1.094</td>
      <td>1.026</td>
      <td>0.844</td>
      <td>0.898</td>
      <td>1.051</td>
      <td>1.072</td>
      <td>0.864</td>
      <td>1.082</td>
      <td>...</td>
      <td>1.044</td>
      <td>1.154</td>
      <td>1.020</td>
      <td>0.995</td>
      <td>1.060</td>
      <td>1.031</td>
      <td>1.016</td>
      <td>1.049</td>
      <td>0.909</td>
      <td>0.911</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.711</td>
      <td>-3.268</td>
      <td>-4.378</td>
      <td>-3.530</td>
      <td>-3.215</td>
      <td>-2.460</td>
      <td>-3.230</td>
      <td>-4.345</td>
      <td>-2.342</td>
      <td>-2.847</td>
      <td>...</td>
      <td>-5.080</td>
      <td>-4.509</td>
      <td>-3.358</td>
      <td>-2.913</td>
      <td>-2.016</td>
      <td>-4.380</td>
      <td>-2.806</td>
      <td>-3.860</td>
      <td>-3.574</td>
      <td>-1.749</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.629</td>
      <td>-0.431</td>
      <td>-0.392</td>
      <td>-0.583</td>
      <td>-0.262</td>
      <td>-0.442</td>
      <td>-0.486</td>
      <td>-0.501</td>
      <td>-0.415</td>
      <td>-0.670</td>
      <td>...</td>
      <td>-0.338</td>
      <td>-0.614</td>
      <td>-0.496</td>
      <td>-0.550</td>
      <td>-0.885</td>
      <td>-0.521</td>
      <td>-0.611</td>
      <td>-0.630</td>
      <td>-0.373</td>
      <td>-0.424</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.040</td>
      <td>0.204</td>
      <td>0.143</td>
      <td>-0.011</td>
      <td>0.155</td>
      <td>0.053</td>
      <td>0.060</td>
      <td>0.004</td>
      <td>0.178</td>
      <td>-0.130</td>
      <td>...</td>
      <td>0.193</td>
      <td>0.012</td>
      <td>0.252</td>
      <td>-0.006</td>
      <td>-0.388</td>
      <td>-0.243</td>
      <td>-0.097</td>
      <td>-0.083</td>
      <td>0.061</td>
      <td>-0.032</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.759</td>
      <td>0.655</td>
      <td>0.693</td>
      <td>0.568</td>
      <td>0.560</td>
      <td>0.648</td>
      <td>0.904</td>
      <td>0.934</td>
      <td>0.979</td>
      <td>1.065</td>
      <td>...</td>
      <td>0.758</td>
      <td>0.560</td>
      <td>0.566</td>
      <td>0.727</td>
      <td>1.078</td>
      <td>1.118</td>
      <td>0.614</td>
      <td>0.712</td>
      <td>0.823</td>
      <td>0.983</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.850</td>
      <td>1.659</td>
      <td>2.128</td>
      <td>2.095</td>
      <td>1.584</td>
      <td>1.847</td>
      <td>1.939</td>
      <td>1.912</td>
      <td>2.038</td>
      <td>2.148</td>
      <td>...</td>
      <td>1.640</td>
      <td>1.842</td>
      <td>1.695</td>
      <td>1.563</td>
      <td>1.796</td>
      <td>1.806</td>
      <td>2.140</td>
      <td>2.136</td>
      <td>1.936</td>
      <td>1.989</td>
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
      <th>AHSSMVGVNLPQK_na</th>
      <th>ALPFWNEEIVPQIK_na</th>
      <th>AMVSEFLK_na</th>
      <th>AYHEQLSVAEITNACFEPANQMVK_na</th>
      <th>DSYVGDEAQSK_na</th>
      <th>ELAPYDENWFYTR_na</th>
      <th>ELVYPPDYNPEGK_na</th>
      <th>GIHPTIISESFQK_na</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR_na</th>
      <th>GVNLPGAAVDLPAVSEK_na</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLGILGLGR_na</th>
      <th>TLHPDLGTDK_na</th>
      <th>VNVPVIGGHAGK_na</th>
      <th>VSFELFADK_na</th>
      <th>VVVAENFDEIVNNENK_na</th>
      <th>YADLTEDQLPSCESLK_na</th>
      <th>YDDMATCMK_na</th>
      <th>YLTVAAVFR_na</th>
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
      <td>False</td>
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
      <td>False</td>
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
      <td>False</td>
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
<p>990 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x000001CBC0506040>
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




    
![png](latent_2D_500_10_files/latent_2D_500_10_108_2.png)
    


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
      <td>0.999554</td>
      <td>0.817122</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.692016</td>
      <td>0.399036</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.492404</td>
      <td>0.321330</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.391310</td>
      <td>0.311825</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.338603</td>
      <td>0.302370</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.308028</td>
      <td>0.294634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.289118</td>
      <td>0.291346</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.277582</td>
      <td>0.286686</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.269348</td>
      <td>0.287499</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.264055</td>
      <td>0.288058</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4933, 4933)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.2884)




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
      <th>AHSSMVGVNLPQK</th>
      <td>29.899</td>
      <td>30.250</td>
      <td>30.245</td>
      <td>30.008</td>
      <td>29.380</td>
      <td>29.453</td>
    </tr>
    <tr>
      <th>DSYVGDEAQSK</th>
      <td>32.783</td>
      <td>33.542</td>
      <td>33.310</td>
      <td>33.189</td>
      <td>32.801</td>
      <td>32.745</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>28.098</td>
      <td>27.852</td>
      <td>27.936</td>
    </tr>
    <tr>
      <th>TLGILGLGR</th>
      <td>28.345</td>
      <td>28.815</td>
      <td>28.585</td>
      <td>28.547</td>
      <td>27.590</td>
      <td>28.133</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>29.217</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>30.196</td>
      <td>30.150</td>
      <td>30.167</td>
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
      <th>VVVAENFDEIVNNENK</th>
      <td>29.613</td>
      <td>29.629</td>
      <td>29.761</td>
      <td>29.744</td>
      <td>29.211</td>
      <td>29.232</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <td>31.539</td>
      <td>31.468</td>
      <td>31.570</td>
      <td>31.623</td>
      <td>31.089</td>
      <td>31.205</td>
    </tr>
    <tr>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>30.671</td>
      <td>29.922</td>
      <td>30.016</td>
      <td>30.115</td>
      <td>29.536</td>
      <td>29.588</td>
    </tr>
    <tr>
      <th>LSVLGAITSVQQR</th>
      <td>27.476</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>27.425</td>
      <td>27.084</td>
      <td>27.249</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
      <td>30.274</td>
      <td>30.238</td>
      <td>30.494</td>
      <td>30.053</td>
      <td>30.145</td>
    </tr>
  </tbody>
</table>
<p>4933 rows × 6 columns</p>
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
      <td>-0.526</td>
      <td>0.257</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.318</td>
      <td>0.146</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.744</td>
      <td>0.578</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.362</td>
      <td>0.302</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.675</td>
      <td>0.523</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_122_1.png)
    



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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_123_1.png)
    


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
      <th>AHSSMVGVNLPQK</th>
      <th>ALPFWNEEIVPQIK</th>
      <th>AMVSEFLK</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <th>DSYVGDEAQSK</th>
      <th>ELAPYDENWFYTR</th>
      <th>ELVYPPDYNPEGK</th>
      <th>GIHPTIISESFQK</th>
      <th>GLVEPVDVVDNADGTQTVNYVPSR</th>
      <th>GVNLPGAAVDLPAVSEK</th>
      <th>...</th>
      <th>TEFLSFMNTELAAFTK</th>
      <th>TGTAEMSSILEER</th>
      <th>TLGILGLGR</th>
      <th>TLHPDLGTDK</th>
      <th>VNVPVIGGHAGK</th>
      <th>VSFELFADK</th>
      <th>VVVAENFDEIVNNENK</th>
      <th>YADLTEDQLPSCESLK</th>
      <th>YDDMATCMK</th>
      <th>YLTVAAVFR</th>
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
      <td>0.633</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.733</td>
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
      <td>0.612</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.466</td>
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
      <td>0.640</td>
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
      <td>0.393</td>
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
      <td>0.634</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.756</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.824</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.669</td>
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
    
    Optimizer used: <function Adam at 0x000001CBC0506040>
    Loss function: <function loss_fct_vae at 0x000001CBC0524940>
    
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




    
![png](latent_2D_500_10_files/latent_2D_500_10_136_2.png)
    



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
      <td>2030.367920</td>
      <td>221.667953</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2000.650146</td>
      <td>214.327179</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1950.947998</td>
      <td>207.411331</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1903.320557</td>
      <td>203.803909</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1865.540894</td>
      <td>202.047745</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1838.540039</td>
      <td>200.467682</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1819.487061</td>
      <td>197.413116</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1805.375244</td>
      <td>198.887466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1795.638062</td>
      <td>198.276871</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1787.440063</td>
      <td>198.039719</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4933)




```python
len(pred[0])
```




    4933




```python
learn.loss_func(pred, target)
```




    tensor(3108.5349)




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
      <th>AHSSMVGVNLPQK</th>
      <td>29.899</td>
      <td>30.250</td>
      <td>30.245</td>
      <td>30.008</td>
      <td>29.380</td>
      <td>29.453</td>
      <td>30.340</td>
    </tr>
    <tr>
      <th>DSYVGDEAQSK</th>
      <td>32.783</td>
      <td>33.542</td>
      <td>33.310</td>
      <td>33.189</td>
      <td>32.801</td>
      <td>32.745</td>
      <td>32.688</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>28.105</td>
      <td>28.950</td>
      <td>28.810</td>
      <td>28.098</td>
      <td>27.852</td>
      <td>27.936</td>
      <td>28.863</td>
    </tr>
    <tr>
      <th>TLGILGLGR</th>
      <td>28.345</td>
      <td>28.815</td>
      <td>28.585</td>
      <td>28.547</td>
      <td>27.590</td>
      <td>28.133</td>
      <td>28.478</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALPFWNEEIVPQIK</th>
      <td>29.217</td>
      <td>31.081</td>
      <td>30.827</td>
      <td>30.196</td>
      <td>30.150</td>
      <td>30.167</td>
      <td>30.949</td>
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
      <th>VVVAENFDEIVNNENK</th>
      <td>29.613</td>
      <td>29.629</td>
      <td>29.761</td>
      <td>29.744</td>
      <td>29.211</td>
      <td>29.232</td>
      <td>29.719</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AYHEQLSVAEITNACFEPANQMVK</th>
      <td>31.539</td>
      <td>31.468</td>
      <td>31.570</td>
      <td>31.623</td>
      <td>31.089</td>
      <td>31.205</td>
      <td>31.596</td>
    </tr>
    <tr>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>30.671</td>
      <td>29.922</td>
      <td>30.016</td>
      <td>30.115</td>
      <td>29.536</td>
      <td>29.588</td>
      <td>30.063</td>
    </tr>
    <tr>
      <th>LSVLGAITSVQQR</th>
      <td>27.476</td>
      <td>27.545</td>
      <td>27.684</td>
      <td>27.425</td>
      <td>27.084</td>
      <td>27.249</td>
      <td>27.673</td>
    </tr>
    <tr>
      <th>STGEAFVQFASQEIAEK</th>
      <td>30.364</td>
      <td>30.274</td>
      <td>30.238</td>
      <td>30.494</td>
      <td>30.053</td>
      <td>30.145</td>
      <td>30.304</td>
    </tr>
  </tbody>
</table>
<p>4933 rows × 7 columns</p>
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
      <td>-0.117</td>
      <td>-0.239</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.150</td>
      <td>-0.235</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.042</td>
      <td>-0.039</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.034</td>
      <td>-0.083</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.045</td>
      <td>-0.163</td>
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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_146_1.png)
    



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
    


    
![png](latent_2D_500_10_files/latent_2D_500_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181230_QE6_nLC6_CSC_QC_HeLa_03', 'KHPDASVNFSEFSK'), ('20190107_QE10_nLC0_KS_QC_MNT_HeLa_01', 'GVNLPGAAVDLPAVSEK'), ('20190110_QE8_nLC14_JM_QC_MNT_HeLa_02', 'STGEAFVQFASQEIAEK'), ('20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_01', 'LVNHFVEEFK'), ('20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01', 'SLEDQVEMLR'), ('20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_02', 'SLEDQVEMLR'), ('20190204_QE6_nLC6_MPL_QC_MNT_HeLa_01', 'GVNLPGAAVDLPAVSEK'), ('20190205_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'HQPTAIIAK'), ('20190205_QE7_nLC7_MEM_QC_MNT_HeLa_04', 'HQPTAIIAK'), ('20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_03', 'HQPTAIIAK'), ('20190219_QE2_NLC1_GP_QC_MNT_HELA_01', 'VSFELFADK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_1_20190225173940', 'LVNHFVEEFK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_2', 'LVNHFVEEFK'), ('20190225_QE9_nLC0_RS_MNT_Hela_02', 'VSFELFADK'), ('20190228_QE4_LC12_JE_QC_MNT_HeLa_01', 'VNVPVIGGHAGK'), ('20190301_QE1_nLC2_ANHO_QC_MNT_HELA_01_20190303025443', 'IIQLLDDYPK'), ('20190304_QE10_nLC0_KS_QC_MNT_HeLa_02', 'LSVLGAITSVQQR'), ('20190318_QE2_NLC1_AB_MNT_HELA_04', 'AHSSMVGVNLPQK'), ('20190402_QE6_LC6_AS_QC_MNT_HeLa_03', 'YADLTEDQLPSCESLK'), ('20190402_QE7_nLC3_AL_QC_MNT_HeLa_01', 'YADLTEDQLPSCESLK'), ('20190408_QE1_nLC2_GP_MNT_QC_hela_02_20190408131505', 'NAGVEGSLIVEK'), ('20190409_QE1_nLC2_ANHO_MNT_QC_hela_02', 'ALPFWNEEIVPQIK'), ('20190423_QX2_FlMe_MA_HeLa_500ng_LC05_CTCDoff', 'YADLTEDQLPSCESLK'), ('20190423_QX7_JuSc_MA_HeLaBr14_500ng_LC02', 'LNNLVLFDK'), ('20190425_QX4_JoSw_MA_HeLa_500ng_BR13_standard', 'YLTVAAVFR'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'RLAPEYEAAATR'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336', 'LSVLGAITSVQQR'), ('20190508_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'GLVEPVDVVDNADGTQTVNYVPSR'), ('20190510_QE2_NLC1_GP_MNT_HELA_02', 'KHPDASVNFSEFSK'), ('20190513_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'SYCAEIAHNVSSK'), ('20190513_QX7_ChDe_MA_HeLaBr14_500ng', 'GLVEPVDVVDNADGTQTVNYVPSR'), ('20190513_QX8_MiWi_MA_HeLa_BR14_500ng', 'VSFELFADK'), ('20190514_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'SYCAEIAHNVSSK'), ('20190515_QX3_AsJa_MA_Hela_500ng_LC15', 'IIQLLDDYPK'), ('20190515_QX4_JiYu_MA_HeLa_500ng_BR14', 'IIQLLDDYPK'), ('20190522_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14_190524170803', 'IVSRPEELREDDVGTGAGLLEIK'), ('20190611_QX7_IgPa_MA_HeLa_Br14_500ng', 'ALPFWNEEIVPQIK'), ('20190618_QX4_JiYu_MA_HeLa_500ng_centroid', 'TGTAEMSSILEER'), ('20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01', 'NAGVEGSLIVEK'), ('20190625_QE6_LC4_AS_QC_MNT_HeLa_02', 'RFPGYDSESK'), ('20190625_QE6_LC4_AS_QC_MNT_HeLa_03', 'RFPGYDSESK'), ('20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_02', 'DSYVGDEAQSK'), ('20190627_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'ALPFWNEEIVPQIK'), ('20190628_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'YDDMATCMK'), ('20190701_QX8_AnPi_MA_HeLa_BR14_500ng', 'SYCAEIAHNVSSK'), ('20190705_QX0_AnBr_MA_HeLa_500ng_LC07_01_190707104639', 'ALPFWNEEIVPQIK'), ('20190708_QE9_nLC2_AGF_QC_MNT_HeLa_01_20190709092142', 'ALPFWNEEIVPQIK'), ('20190712_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_01', 'ELAPYDENWFYTR'), ('20190716_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'LVNHFVEEFK'), ('20190726_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_4thcolumn_1', 'VNVPVIGGHAGK'), ('20190801_QX3_StEb_MA_Hela_500ng_LC15', 'NMDPLNDNIATLLHQSSDK'), ('20190803_QX8_AnPi_MA_HeLa_BR14_500ng', 'SLEDQVEMLR')]
    




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
      <td>0.502</td>
      <td>0.542</td>
      <td>1.505</td>
      <td>1.980</td>
      <td>2.023</td>
      <td>2.099</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.434</td>
      <td>0.463</td>
      <td>0.854</td>
      <td>1.065</td>
      <td>1.067</td>
      <td>1.052</td>
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
