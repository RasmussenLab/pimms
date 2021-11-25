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
n_feat = 25
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
    ELGITALHIK                          994
    LVINGNPITIFQERDPSK                  975
    VICILSHPIK                          995
    SQLDIIIHSLK                         997
    KFDQLLAEEK                          995
    GEMMDLQHGSLFLR                      989
    AGFAGDDAPR                          992
    LHIIEVGTPPTGNQPFPK                  998
    DGQAMLWDLNEGK                       998
    VFITDDFHDMMPK                       994
    SDALETLGFLNHYQMK                    999
    AGLQFPVGR                           992
    LFPLIQAMHPTLAGK                     995
    RGFAFVTFDDHDSVDK                    984
    LSPEELLLR                           978
    LYSPSQIGAFVLMK                      978
    GAEAANVTGPGGVPVQGSK                 988
    SLAGSSGPGASSGTSGDHGELVVR            997
    ALDVMVSTFHK                         999
    TPELNLDQFHDK                      1,000
    TGVELGKPTHFTVNAK                    936
    VHSFPTLK                            993
    ALLFIPR                             991
    LVAIVDPHIK                          997
    GLTSVINQK                           994
    HVVQSISTQQEK                        993
    IGIEIIK                             991
    TAVVVGTITDDVR                       999
    LDIDSPPITAR                         988
    AHQVVEDGYEFFAK                      998
    FDASFFGVHPK                         977
    LLDFGSLSNLQVTQPTVGMNFK              960
    EMDRETLIDVAR                        983
    EHDPVGQMVNNPK                       989
    YTPSGQAGAAASESLFVSNHAY              999
    PLRLPLQDVYK                         947
    VSSDNVADLHEK                        994
    ISMPDLDLNLK                         980
    FHQLDIDDLQSIR                       988
    LTNSMMMHGR                          953
    LHQLAMQQSHFPMTHGNTGFSGIESSSPEVK     995
    LISWYDNEFGYSNR                    1,000
    LHFFMPGFAPLTSR                    1,000
    FQSSHHPTDITSLDQYVER                 998
    AITIAGVPQSVTECVK                    996
    VDATEESDLAQQYGVR                    995
    ILPTLEAVAALGNK                      999
    SCMLTGTPESVQSAK                     976
    ESEPQAAAEPAEAK                      970
    SEMEVQDAELK                         991
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
      <th>ELGITALHIK</th>
      <td>29.679</td>
    </tr>
    <tr>
      <th>LVINGNPITIFQERDPSK</th>
      <td>31.978</td>
    </tr>
    <tr>
      <th>VICILSHPIK</th>
      <td>27.624</td>
    </tr>
    <tr>
      <th>SQLDIIIHSLK</th>
      <td>29.018</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>VDATEESDLAQQYGVR</th>
      <td>27.885</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
    </tr>
    <tr>
      <th>SCMLTGTPESVQSAK</th>
      <td>27.876</td>
    </tr>
    <tr>
      <th>ESEPQAAAEPAEAK</th>
      <td>30.445</td>
    </tr>
    <tr>
      <th>SEMEVQDAELK</th>
      <td>29.430</td>
    </tr>
  </tbody>
</table>
<p>49407 rows × 1 columns</p>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_24_1.png)
    



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
      <th>ELGITALHIK</th>
      <td>29.679</td>
    </tr>
    <tr>
      <th>LVINGNPITIFQERDPSK</th>
      <td>31.978</td>
    </tr>
    <tr>
      <th>VICILSHPIK</th>
      <td>27.624</td>
    </tr>
    <tr>
      <th>SQLDIIIHSLK</th>
      <td>29.018</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>VDATEESDLAQQYGVR</th>
      <td>27.885</td>
    </tr>
    <tr>
      <th>ILPTLEAVAALGNK</th>
      <td>29.576</td>
    </tr>
    <tr>
      <th>SCMLTGTPESVQSAK</th>
      <td>27.876</td>
    </tr>
    <tr>
      <th>ESEPQAAAEPAEAK</th>
      <td>30.445</td>
    </tr>
    <tr>
      <th>SEMEVQDAELK</th>
      <td>29.430</td>
    </tr>
  </tbody>
</table>
<p>49407 rows × 1 columns</p>
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
      <th>20190411_QE6_LC6_AS_QC_MNT_HeLa_03</th>
      <th>AGFAGDDAPR</th>
      <td>33.696</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>AGFAGDDAPR</th>
      <td>34.210</td>
    </tr>
    <tr>
      <th>20190624_QE4_nLC12_MM_QC_MNT_HELA_02_20190626221327</th>
      <th>AGFAGDDAPR</th>
      <td>34.050</td>
    </tr>
    <tr>
      <th>20190528_QX1_PhGe_MA_HeLa_DMSO_500ng_LC14</th>
      <th>AGFAGDDAPR</th>
      <td>34.970</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>AGFAGDDAPR</th>
      <td>34.011</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01_20190430180750</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
      <td>28.798</td>
    </tr>
    <tr>
      <th>20190405_QE1_nLC2_GP_MNT_QC_hela_01</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
      <td>28.602</td>
    </tr>
    <tr>
      <th>20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
      <td>32.772</td>
    </tr>
    <tr>
      <th>20190215_QE7_nLC3_CK_QC_MNT_HeLa_01</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
      <td>26.749</td>
    </tr>
    <tr>
      <th>20190618_QX3_LiSc_MA_Hela_500ng_LC15</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
      <td>30.471</td>
    </tr>
  </tbody>
</table>
<p>44468 rows × 1 columns</p>
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
      <th>AGFAGDDAPR</th>
      <td>33.072</td>
      <td>33.805</td>
      <td>33.738</td>
      <td>33.444</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.018</td>
      <td>31.845</td>
      <td>31.491</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
      <td>29.711</td>
      <td>29.488</td>
      <td>29.216</td>
    </tr>
    <tr>
      <th>PLRLPLQDVYK</th>
      <td>31.968</td>
      <td>29.610</td>
      <td>29.858</td>
      <td>30.742</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGLQFPVGR</th>
      <td>32.984</td>
      <td>33.692</td>
      <td>33.616</td>
      <td>33.508</td>
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
      <th>TAVVVGTITDDVR</th>
      <td>27.987</td>
      <td>29.196</td>
      <td>29.654</td>
      <td>28.679</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AITIAGVPQSVTECVK</th>
      <td>29.010</td>
      <td>29.707</td>
      <td>29.737</td>
      <td>29.483</td>
    </tr>
    <tr>
      <th>GEMMDLQHGSLFLR</th>
      <td>28.109</td>
      <td>27.962</td>
      <td>29.430</td>
      <td>27.849</td>
    </tr>
    <tr>
      <th>SDALETLGFLNHYQMK</th>
      <td>30.265</td>
      <td>30.019</td>
      <td>29.868</td>
      <td>30.286</td>
    </tr>
    <tr>
      <th>TGVELGKPTHFTVNAK</th>
      <td>31.341</td>
      <td>30.564</td>
      <td>29.986</td>
      <td>31.537</td>
    </tr>
  </tbody>
</table>
<p>4939 rows × 4 columns</p>
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
      <th>20190104_QE6_nLC6_MM_QC_MNT_HELA_02_190108210418</th>
      <th>PLRLPLQDVYK</th>
      <td>29.264</td>
      <td>29.610</td>
      <td>29.858</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190107_QE5_nLC5_DS_QC_MNT_HeLa_FlashPack_03</th>
      <th>HVVQSISTQQEK</th>
      <td>28.840</td>
      <td>28.746</td>
      <td>28.603</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190115_QE5_nLC5_RJC_MNT_HeLa_02</th>
      <th>LISWYDNEFGYSNR</th>
      <td>31.520</td>
      <td>31.364</td>
      <td>31.377</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01</th>
      <th>SQLDIIIHSLK</th>
      <td>29.257</td>
      <td>29.238</td>
      <td>29.034</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190129_QE1_nLC2_GP_QC_MNT_HELA_01</th>
      <th>VSSDNVADLHEK</th>
      <td>28.045</td>
      <td>28.196</td>
      <td>28.359</td>
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
      <th>20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05</th>
      <th>ELGITALHIK</th>
      <td>31.602</td>
      <td>30.507</td>
      <td>30.503</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX6_MaTa_MA_HeLa_500ng_LC09</th>
      <th>FHQLDIDDLQSIR</th>
      <td>28.990</td>
      <td>28.524</td>
      <td>28.309</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>RGFAFVTFDDHDSVDK</th>
      <td>30.346</td>
      <td>30.090</td>
      <td>29.791</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190804_QX0_AsJa_MA_HeLa_500ng_LC07_01</th>
      <th>RGFAFVTFDDHDSVDK</th>
      <td>30.247</td>
      <td>30.090</td>
      <td>29.791</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <th>TAVVVGTITDDVR</th>
      <td>28.212</td>
      <td>29.196</td>
      <td>29.654</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 4 columns</p>
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
      <td>AGLQFPVGR</td>
      <td>33.093</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AHQVVEDGYEFFAK</td>
      <td>27.137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AITIAGVPQSVTECVK</td>
      <td>27.892</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALLFIPR</td>
      <td>30.571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DGQAMLWDLNEGK</td>
      <td>27.510</td>
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
      <td>AGFAGDDAPR</td>
      <td>33.072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALDVMVSTFHK</td>
      <td>30.949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>KFDQLLAEEK</td>
      <td>28.574</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>PLRLPLQDVYK</td>
      <td>31.968</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>AGLQFPVGR</td>
      <td>32.984</td>
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
      <td>20190514_QX4_JiYu_MA_HeLa_500ng</td>
      <td>LSPEELLLR</td>
      <td>30.627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190630_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</td>
      <td>AGFAGDDAPR</td>
      <td>33.409</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04</td>
      <td>LHFFMPGFAPLTSR</td>
      <td>33.306</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190521_QX6_AsJa_MA_HeLa_Br14_500ng_LC09_20190522134621</td>
      <td>LISWYDNEFGYSNR</td>
      <td>32.247</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190611_QE4_LC12_JE_QC_MNT_HeLa_01</td>
      <td>EHDPVGQMVNNPK</td>
      <td>28.539</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190611_QE4_LC12_JE_QC_MNT_HeLa_03</td>
      <td>VICILSHPIK</td>
      <td>28.873</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190315_QE2_NLC1_GP_MNT_HELA_01</td>
      <td>LYSPSQIGAFVLMK</td>
      <td>29.590</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190514_QE8_nLC13_AGF_QC_MNT_HeLa_01</td>
      <td>LSPEELLLR</td>
      <td>27.724</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190219_QE4_nLC12_SIS_QC_MNT_Hela_2</td>
      <td>ILPTLEAVAALGNK</td>
      <td>29.634</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190722_QX2_IgPa_MA_HeLa_500ng_CTCDoff_LC05</td>
      <td>PLRLPLQDVYK</td>
      <td>32.153</td>
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
      <td>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_01</td>
      <td>LFPLIQAMHPTLAGK</td>
      <td>29.475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190625_QX7_IgPa_MA_HeLa_Br14_500ng</td>
      <td>ILPTLEAVAALGNK</td>
      <td>30.023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190717_QX8_ChSc_MA_HeLa_500ng</td>
      <td>GAEAANVTGPGGVPVQGSK</td>
      <td>31.583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190429_QX4_ChDe_MA_HeLa_500ng_BR13_standard</td>
      <td>GEMMDLQHGSLFLR</td>
      <td>33.718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</td>
      <td>ALDVMVSTFHK</td>
      <td>32.522</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190515_QE4_LC12_AS_QC_MNT_HeLa_01</td>
      <td>SEMEVQDAELK</td>
      <td>29.580</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190420_QE8_nLC14_RG_QC_HeLa_01</td>
      <td>HVVQSISTQQEK</td>
      <td>28.902</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190313_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>EMDRETLIDVAR</td>
      <td>27.634</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190416_QE7_nLC3_OOE_QC_MNT_HeLa_250ng_RO-002</td>
      <td>FQSSHHPTDITSLDQYVER</td>
      <td>29.640</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190530_QX4_IgPa_MA_HeLa_500ng</td>
      <td>YTPSGQAGAAASESLFVSNHAY</td>
      <td>32.547</td>
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




    (1377, 155)



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
     'y_range': (19, 36)}
    








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
    
    Optimizer used: <function Adam at 0x00000225FB995040>
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
      <td>2.254934</td>
      <td>2.049533</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.937010</td>
      <td>0.876902</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.698513</td>
      <td>0.675700</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.636022</td>
      <td>0.649592</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.588784</td>
      <td>0.602336</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.597380</td>
      <td>0.583953</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.508708</td>
      <td>0.574144</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.517343</td>
      <td>0.567747</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.468729</td>
      <td>0.566257</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.472479</td>
      <td>0.566331</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_58_1.png)
    


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
      <th>3,514</th>
      <td>709</td>
      <td>24</td>
      <td>29.475</td>
    </tr>
    <tr>
      <th>3,692</th>
      <td>743</td>
      <td>20</td>
      <td>30.023</td>
    </tr>
    <tr>
      <th>4,391</th>
      <td>885</td>
      <td>15</td>
      <td>31.583</td>
    </tr>
    <tr>
      <th>2,265</th>
      <td>455</td>
      <td>16</td>
      <td>33.718</td>
    </tr>
    <tr>
      <th>756</th>
      <td>156</td>
      <td>5</td>
      <td>32.522</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3,348</th>
      <td>678</td>
      <td>7</td>
      <td>30.434</td>
    </tr>
    <tr>
      <th>4,911</th>
      <td>987</td>
      <td>36</td>
      <td>29.696</td>
    </tr>
    <tr>
      <th>4,571</th>
      <td>924</td>
      <td>44</td>
      <td>29.470</td>
    </tr>
    <tr>
      <th>1,262</th>
      <td>255</td>
      <td>26</td>
      <td>29.728</td>
    </tr>
    <tr>
      <th>301</th>
      <td>63</td>
      <td>23</td>
      <td>30.918</td>
    </tr>
  </tbody>
</table>
<p>4939 rows × 3 columns</p>
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
      <th>AGFAGDDAPR</th>
      <td>33.072</td>
      <td>33.805</td>
      <td>33.738</td>
      <td>33.444</td>
      <td>33.212</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.018</td>
      <td>31.845</td>
      <td>31.491</td>
      <td>31.276</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
      <td>29.711</td>
      <td>29.488</td>
      <td>29.216</td>
      <td>28.813</td>
    </tr>
    <tr>
      <th>PLRLPLQDVYK</th>
      <td>31.968</td>
      <td>29.610</td>
      <td>29.858</td>
      <td>30.742</td>
      <td>28.871</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGLQFPVGR</th>
      <td>32.984</td>
      <td>33.692</td>
      <td>33.616</td>
      <td>33.508</td>
      <td>33.256</td>
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
      <th>TAVVVGTITDDVR</th>
      <td>27.987</td>
      <td>29.196</td>
      <td>29.654</td>
      <td>28.679</td>
      <td>29.122</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AITIAGVPQSVTECVK</th>
      <td>29.010</td>
      <td>29.707</td>
      <td>29.737</td>
      <td>29.483</td>
      <td>29.578</td>
    </tr>
    <tr>
      <th>GEMMDLQHGSLFLR</th>
      <td>28.109</td>
      <td>27.962</td>
      <td>29.430</td>
      <td>27.849</td>
      <td>27.371</td>
    </tr>
    <tr>
      <th>SDALETLGFLNHYQMK</th>
      <td>30.265</td>
      <td>30.019</td>
      <td>29.868</td>
      <td>30.286</td>
      <td>29.853</td>
    </tr>
    <tr>
      <th>TGVELGKPTHFTVNAK</th>
      <td>31.341</td>
      <td>30.564</td>
      <td>29.986</td>
      <td>31.537</td>
      <td>31.203</td>
    </tr>
  </tbody>
</table>
<p>4939 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.117
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.104
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.204
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.126
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.185
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_66_1.png)
    



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
      <td>-0.165</td>
      <td>-0.010</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.209</td>
      <td>-0.053</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.108</td>
      <td>-0.086</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.172</td>
      <td>-0.146</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.232</td>
      <td>0.079</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_68_1.png)
    



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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_69_1.png)
    


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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK</th>
      <th>TAVVVGTITDDVR</th>
      <th>TGVELGKPTHFTVNAK</th>
      <th>TPELNLDQFHDK</th>
      <th>VDATEESDLAQQYGVR</th>
      <th>VFITDDFHDMMPK</th>
      <th>VHSFPTLK</th>
      <th>VICILSHPIK</th>
      <th>VSSDNVADLHEK</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
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
      <td>33.072</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.949</td>
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
      <td>32.984</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.572</td>
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
      <td>28.724</td>
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
      <td>28.346</td>
      <td>NaN</td>
      <td>28.466</td>
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
      <td>29.491</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.910</td>
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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK_na</th>
      <th>TAVVVGTITDDVR_na</th>
      <th>TGVELGKPTHFTVNAK_na</th>
      <th>TPELNLDQFHDK_na</th>
      <th>VDATEESDLAQQYGVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VHSFPTLK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VSSDNVADLHEK_na</th>
      <th>YTPSGQAGAAASESLFVSNHAY_na</th>
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
      <td>0.068</td>
      <td>-0.646</td>
      <td>-1.433</td>
      <td>-1.503</td>
      <td>0.183</td>
      <td>-0.862</td>
      <td>-1.186</td>
      <td>-1.364</td>
      <td>-0.774</td>
      <td>-0.158</td>
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
      <td>-0.464</td>
      <td>0.083</td>
      <td>-1.272</td>
      <td>-1.423</td>
      <td>-0.540</td>
      <td>-1.035</td>
      <td>0.018</td>
      <td>-1.779</td>
      <td>-0.521</td>
      <td>-0.335</td>
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
      <td>-0.223</td>
      <td>0.364</td>
      <td>-0.654</td>
      <td>-0.203</td>
      <td>-0.331</td>
      <td>-0.395</td>
      <td>-0.665</td>
      <td>0.111</td>
      <td>-0.146</td>
      <td>-0.857</td>
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
      <td>0.097</td>
      <td>-0.146</td>
      <td>-0.577</td>
      <td>-0.021</td>
      <td>-5.646</td>
      <td>-0.869</td>
      <td>-1.150</td>
      <td>-1.344</td>
      <td>-0.960</td>
      <td>-0.329</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.715</td>
      <td>-0.560</td>
      <td>-0.644</td>
      <td>-0.914</td>
      <td>-1.334</td>
      <td>-0.642</td>
      <td>-0.641</td>
      <td>-0.690</td>
      <td>0.003</td>
      <td>-0.643</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</th>
      <td>0.481</td>
      <td>0.218</td>
      <td>0.391</td>
      <td>0.281</td>
      <td>-1.249</td>
      <td>0.241</td>
      <td>0.244</td>
      <td>0.706</td>
      <td>0.263</td>
      <td>0.859</td>
      <td>...</td>
      <td>True</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>0.931</td>
      <td>0.648</td>
      <td>0.436</td>
      <td>0.670</td>
      <td>-1.484</td>
      <td>-0.051</td>
      <td>0.505</td>
      <td>0.995</td>
      <td>0.003</td>
      <td>1.057</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.057</td>
      <td>-0.446</td>
      <td>-0.126</td>
      <td>-0.204</td>
      <td>0.187</td>
      <td>-0.439</td>
      <td>-0.054</td>
      <td>0.392</td>
      <td>-0.180</td>
      <td>-0.043</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>-0.223</td>
      <td>-0.297</td>
      <td>0.481</td>
      <td>-0.021</td>
      <td>0.508</td>
      <td>-0.294</td>
      <td>0.018</td>
      <td>0.143</td>
      <td>-0.046</td>
      <td>-0.135</td>
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
      <td>-0.284</td>
      <td>-0.243</td>
      <td>0.265</td>
      <td>-0.021</td>
      <td>0.127</td>
      <td>-0.240</td>
      <td>-0.500</td>
      <td>0.289</td>
      <td>-0.028</td>
      <td>-0.040</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK_na</th>
      <th>TAVVVGTITDDVR_na</th>
      <th>TGVELGKPTHFTVNAK_na</th>
      <th>TPELNLDQFHDK_na</th>
      <th>VDATEESDLAQQYGVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VHSFPTLK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VSSDNVADLHEK_na</th>
      <th>YTPSGQAGAAASESLFVSNHAY_na</th>
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
      <td>0.071</td>
      <td>-0.601</td>
      <td>-1.351</td>
      <td>-1.424</td>
      <td>0.193</td>
      <td>-0.820</td>
      <td>-1.121</td>
      <td>-1.288</td>
      <td>-0.732</td>
      <td>-0.166</td>
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
      <td>-0.430</td>
      <td>0.088</td>
      <td>-1.199</td>
      <td>-1.349</td>
      <td>-0.493</td>
      <td>-0.983</td>
      <td>0.019</td>
      <td>-1.679</td>
      <td>-0.492</td>
      <td>-0.333</td>
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
      <td>-0.203</td>
      <td>0.354</td>
      <td>-0.613</td>
      <td>-0.195</td>
      <td>-0.294</td>
      <td>-0.379</td>
      <td>-0.628</td>
      <td>0.103</td>
      <td>-0.137</td>
      <td>-0.824</td>
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
      <td>-0.128</td>
      <td>-0.540</td>
      <td>-0.023</td>
      <td>-5.341</td>
      <td>-0.826</td>
      <td>-1.088</td>
      <td>-1.269</td>
      <td>-0.908</td>
      <td>-0.327</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.667</td>
      <td>-0.520</td>
      <td>-0.604</td>
      <td>-0.868</td>
      <td>-1.247</td>
      <td>-0.612</td>
      <td>-0.605</td>
      <td>-0.652</td>
      <td>0.003</td>
      <td>-0.623</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</th>
      <td>0.463</td>
      <td>0.215</td>
      <td>0.377</td>
      <td>0.264</td>
      <td>-1.166</td>
      <td>0.222</td>
      <td>0.233</td>
      <td>0.664</td>
      <td>0.249</td>
      <td>0.791</td>
      <td>...</td>
      <td>True</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>0.887</td>
      <td>0.622</td>
      <td>0.419</td>
      <td>0.632</td>
      <td>-1.389</td>
      <td>-0.055</td>
      <td>0.480</td>
      <td>0.936</td>
      <td>0.003</td>
      <td>0.978</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.046</td>
      <td>-0.412</td>
      <td>-0.113</td>
      <td>-0.196</td>
      <td>0.198</td>
      <td>-0.421</td>
      <td>-0.049</td>
      <td>0.367</td>
      <td>-0.169</td>
      <td>-0.058</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>-0.203</td>
      <td>-0.271</td>
      <td>0.462</td>
      <td>-0.023</td>
      <td>0.502</td>
      <td>-0.283</td>
      <td>0.019</td>
      <td>0.133</td>
      <td>-0.043</td>
      <td>-0.144</td>
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
      <td>-0.260</td>
      <td>-0.220</td>
      <td>0.257</td>
      <td>-0.023</td>
      <td>0.140</td>
      <td>-0.233</td>
      <td>-0.472</td>
      <td>0.270</td>
      <td>-0.026</td>
      <td>-0.055</td>
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
<p>992 rows × 100 columns</p>
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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
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
      <td>0.008</td>
      <td>0.010</td>
      <td>0.006</td>
      <td>-0.002</td>
      <td>0.020</td>
      <td>-0.006</td>
      <td>0.002</td>
      <td>-0.002</td>
      <td>0.000</td>
      <td>-0.017</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.945</td>
      <td>0.945</td>
      <td>0.948</td>
      <td>0.947</td>
      <td>0.950</td>
      <td>0.945</td>
      <td>0.948</td>
      <td>0.943</td>
      <td>0.946</td>
      <td>0.942</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.894</td>
      <td>-9.684</td>
      <td>-5.132</td>
      <td>-4.344</td>
      <td>-5.341</td>
      <td>-5.232</td>
      <td>-4.834</td>
      <td>-4.180</td>
      <td>-6.134</td>
      <td>-3.507</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.322</td>
      <td>-0.392</td>
      <td>-0.430</td>
      <td>-0.491</td>
      <td>-0.407</td>
      <td>-0.420</td>
      <td>-0.444</td>
      <td>-0.390</td>
      <td>-0.444</td>
      <td>-0.506</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.071</td>
      <td>0.088</td>
      <td>0.061</td>
      <td>-0.023</td>
      <td>0.193</td>
      <td>-0.055</td>
      <td>0.019</td>
      <td>-0.017</td>
      <td>0.003</td>
      <td>-0.148</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.546</td>
      <td>0.536</td>
      <td>0.528</td>
      <td>0.470</td>
      <td>0.615</td>
      <td>0.522</td>
      <td>0.441</td>
      <td>0.418</td>
      <td>0.505</td>
      <td>0.419</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.034</td>
      <td>1.877</td>
      <td>2.633</td>
      <td>2.219</td>
      <td>1.932</td>
      <td>2.163</td>
      <td>2.109</td>
      <td>2.211</td>
      <td>2.309</td>
      <td>2.255</td>
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




    ((#50) ['AGFAGDDAPR','AGLQFPVGR','AHQVVEDGYEFFAK','AITIAGVPQSVTECVK','ALDVMVSTFHK','ALLFIPR','DGQAMLWDLNEGK','EHDPVGQMVNNPK','ELGITALHIK','EMDRETLIDVAR'...],
     (#50) ['AGFAGDDAPR_na','AGLQFPVGR_na','AHQVVEDGYEFFAK_na','AITIAGVPQSVTECVK_na','ALDVMVSTFHK_na','ALLFIPR_na','DGQAMLWDLNEGK_na','EHDPVGQMVNNPK_na','ELGITALHIK_na','EMDRETLIDVAR_na'...])




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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.113</td>
      <td>-0.063</td>
      <td>-0.215</td>
      <td>0.133</td>
      <td>-0.123</td>
      <td>0.185</td>
      <td>-0.040</td>
      <td>0.257</td>
      <td>0.023</td>
      <td>-0.138</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.938</td>
      <td>1.424</td>
      <td>1.129</td>
      <td>0.910</td>
      <td>1.192</td>
      <td>0.875</td>
      <td>1.054</td>
      <td>1.012</td>
      <td>1.073</td>
      <td>1.032</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.663</td>
      <td>-10.948</td>
      <td>-3.992</td>
      <td>-3.441</td>
      <td>-4.096</td>
      <td>-1.960</td>
      <td>-5.027</td>
      <td>-2.557</td>
      <td>-3.544</td>
      <td>-3.120</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.500</td>
      <td>-0.350</td>
      <td>-0.680</td>
      <td>-0.424</td>
      <td>-0.648</td>
      <td>-0.417</td>
      <td>-0.450</td>
      <td>-0.233</td>
      <td>-0.557</td>
      <td>-0.694</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.133</td>
      <td>0.177</td>
      <td>-0.089</td>
      <td>0.081</td>
      <td>0.143</td>
      <td>0.073</td>
      <td>0.045</td>
      <td>0.115</td>
      <td>0.096</td>
      <td>-0.148</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.824</td>
      <td>0.628</td>
      <td>0.345</td>
      <td>0.792</td>
      <td>0.597</td>
      <td>0.787</td>
      <td>0.436</td>
      <td>1.046</td>
      <td>0.711</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.939</td>
      <td>1.732</td>
      <td>2.318</td>
      <td>2.118</td>
      <td>1.727</td>
      <td>1.915</td>
      <td>2.029</td>
      <td>2.136</td>
      <td>2.056</td>
      <td>1.904</td>
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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK_val</th>
      <th>TAVVVGTITDDVR_val</th>
      <th>TGVELGKPTHFTVNAK_val</th>
      <th>TPELNLDQFHDK_val</th>
      <th>VDATEESDLAQQYGVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VHSFPTLK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VSSDNVADLHEK_val</th>
      <th>YTPSGQAGAAASESLFVSNHAY_val</th>
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
      <td>0.071</td>
      <td>-0.601</td>
      <td>-1.351</td>
      <td>-1.424</td>
      <td>0.193</td>
      <td>-0.820</td>
      <td>-1.121</td>
      <td>-1.288</td>
      <td>-0.732</td>
      <td>-0.166</td>
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
      <td>-0.430</td>
      <td>0.088</td>
      <td>-1.199</td>
      <td>-1.349</td>
      <td>-0.493</td>
      <td>-0.983</td>
      <td>0.019</td>
      <td>-1.679</td>
      <td>-0.492</td>
      <td>-0.333</td>
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
      <td>-0.203</td>
      <td>0.354</td>
      <td>-0.613</td>
      <td>-0.195</td>
      <td>-0.294</td>
      <td>-0.379</td>
      <td>-0.628</td>
      <td>0.103</td>
      <td>-0.137</td>
      <td>-0.824</td>
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
      <td>-0.128</td>
      <td>-0.540</td>
      <td>-0.023</td>
      <td>-5.341</td>
      <td>-0.826</td>
      <td>-1.088</td>
      <td>-1.269</td>
      <td>-0.908</td>
      <td>-0.327</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.461</td>
      <td>NaN</td>
      <td>-0.498</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.667</td>
      <td>-0.520</td>
      <td>-0.604</td>
      <td>-0.868</td>
      <td>-1.247</td>
      <td>-0.612</td>
      <td>-0.605</td>
      <td>-0.652</td>
      <td>0.003</td>
      <td>-0.623</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.183</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</th>
      <td>0.463</td>
      <td>0.215</td>
      <td>0.377</td>
      <td>0.264</td>
      <td>-1.166</td>
      <td>0.222</td>
      <td>0.233</td>
      <td>0.664</td>
      <td>0.249</td>
      <td>0.791</td>
      <td>...</td>
      <td>-0.429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>0.887</td>
      <td>0.622</td>
      <td>0.419</td>
      <td>0.632</td>
      <td>-1.389</td>
      <td>-0.055</td>
      <td>0.480</td>
      <td>0.936</td>
      <td>0.003</td>
      <td>0.978</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.308</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.046</td>
      <td>-0.412</td>
      <td>-0.113</td>
      <td>-0.196</td>
      <td>0.198</td>
      <td>-0.421</td>
      <td>-0.049</td>
      <td>0.367</td>
      <td>-0.169</td>
      <td>-0.058</td>
      <td>...</td>
      <td>0.894</td>
      <td>-0.864</td>
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
      <td>-0.203</td>
      <td>-0.271</td>
      <td>0.462</td>
      <td>-0.023</td>
      <td>0.502</td>
      <td>-0.283</td>
      <td>0.019</td>
      <td>0.133</td>
      <td>-0.043</td>
      <td>-0.144</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.999</td>
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
      <td>-0.260</td>
      <td>-0.220</td>
      <td>0.257</td>
      <td>-0.023</td>
      <td>0.140</td>
      <td>-0.233</td>
      <td>-0.472</td>
      <td>0.270</td>
      <td>-0.026</td>
      <td>-0.055</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.808</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK_val</th>
      <th>TAVVVGTITDDVR_val</th>
      <th>TGVELGKPTHFTVNAK_val</th>
      <th>TPELNLDQFHDK_val</th>
      <th>VDATEESDLAQQYGVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VHSFPTLK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VSSDNVADLHEK_val</th>
      <th>YTPSGQAGAAASESLFVSNHAY_val</th>
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
      <td>0.071</td>
      <td>-0.601</td>
      <td>-1.351</td>
      <td>-1.424</td>
      <td>0.193</td>
      <td>-0.820</td>
      <td>-1.121</td>
      <td>-1.288</td>
      <td>-0.732</td>
      <td>-0.166</td>
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
      <td>-0.430</td>
      <td>0.088</td>
      <td>-1.199</td>
      <td>-1.349</td>
      <td>-0.493</td>
      <td>-0.983</td>
      <td>0.019</td>
      <td>-1.679</td>
      <td>-0.492</td>
      <td>-0.333</td>
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
      <td>-0.203</td>
      <td>0.354</td>
      <td>-0.613</td>
      <td>-0.195</td>
      <td>-0.294</td>
      <td>-0.379</td>
      <td>-0.628</td>
      <td>0.103</td>
      <td>-0.137</td>
      <td>-0.824</td>
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
      <td>-0.128</td>
      <td>-0.540</td>
      <td>-0.023</td>
      <td>-5.341</td>
      <td>-0.826</td>
      <td>-1.088</td>
      <td>-1.269</td>
      <td>-0.908</td>
      <td>-0.327</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.461</td>
      <td>NaN</td>
      <td>-0.498</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.667</td>
      <td>-0.520</td>
      <td>-0.604</td>
      <td>-0.868</td>
      <td>-1.247</td>
      <td>-0.612</td>
      <td>-0.605</td>
      <td>-0.652</td>
      <td>0.003</td>
      <td>-0.623</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.183</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</th>
      <td>0.463</td>
      <td>0.215</td>
      <td>0.377</td>
      <td>0.264</td>
      <td>-1.166</td>
      <td>0.222</td>
      <td>0.233</td>
      <td>0.664</td>
      <td>0.249</td>
      <td>0.791</td>
      <td>...</td>
      <td>-0.429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_02</th>
      <td>0.887</td>
      <td>0.622</td>
      <td>0.419</td>
      <td>0.632</td>
      <td>-1.389</td>
      <td>-0.055</td>
      <td>0.480</td>
      <td>0.936</td>
      <td>0.003</td>
      <td>0.978</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.308</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.207</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.046</td>
      <td>-0.412</td>
      <td>-0.113</td>
      <td>-0.196</td>
      <td>0.198</td>
      <td>-0.421</td>
      <td>-0.049</td>
      <td>0.367</td>
      <td>-0.169</td>
      <td>-0.058</td>
      <td>...</td>
      <td>0.894</td>
      <td>-0.864</td>
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
      <td>-0.203</td>
      <td>-0.271</td>
      <td>0.462</td>
      <td>-0.023</td>
      <td>0.502</td>
      <td>-0.283</td>
      <td>0.019</td>
      <td>0.133</td>
      <td>-0.043</td>
      <td>-0.144</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.999</td>
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
      <td>-0.260</td>
      <td>-0.220</td>
      <td>0.257</td>
      <td>-0.023</td>
      <td>0.140</td>
      <td>-0.233</td>
      <td>-0.472</td>
      <td>0.270</td>
      <td>-0.026</td>
      <td>-0.055</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.808</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>AGFAGDDAPR_val</th>
      <th>AGLQFPVGR_val</th>
      <th>AHQVVEDGYEFFAK_val</th>
      <th>AITIAGVPQSVTECVK_val</th>
      <th>ALDVMVSTFHK_val</th>
      <th>ALLFIPR_val</th>
      <th>DGQAMLWDLNEGK_val</th>
      <th>EHDPVGQMVNNPK_val</th>
      <th>ELGITALHIK_val</th>
      <th>EMDRETLIDVAR_val</th>
      <th>...</th>
      <th>SQLDIIIHSLK_val</th>
      <th>TAVVVGTITDDVR_val</th>
      <th>TGVELGKPTHFTVNAK_val</th>
      <th>TPELNLDQFHDK_val</th>
      <th>VDATEESDLAQQYGVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VHSFPTLK_val</th>
      <th>VICILSHPIK_val</th>
      <th>VSSDNVADLHEK_val</th>
      <th>YTPSGQAGAAASESLFVSNHAY_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>...</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>94.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.113</td>
      <td>-0.063</td>
      <td>-0.215</td>
      <td>0.133</td>
      <td>-0.123</td>
      <td>0.185</td>
      <td>-0.040</td>
      <td>0.257</td>
      <td>0.023</td>
      <td>-0.138</td>
      <td>...</td>
      <td>-0.084</td>
      <td>-0.179</td>
      <td>-0.079</td>
      <td>0.112</td>
      <td>0.016</td>
      <td>0.042</td>
      <td>0.077</td>
      <td>0.047</td>
      <td>-0.224</td>
      <td>-0.120</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.938</td>
      <td>1.424</td>
      <td>1.129</td>
      <td>0.910</td>
      <td>1.192</td>
      <td>0.875</td>
      <td>1.054</td>
      <td>1.012</td>
      <td>1.073</td>
      <td>1.032</td>
      <td>...</td>
      <td>1.094</td>
      <td>0.889</td>
      <td>0.920</td>
      <td>0.916</td>
      <td>0.916</td>
      <td>1.049</td>
      <td>1.019</td>
      <td>0.900</td>
      <td>1.105</td>
      <td>0.851</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.663</td>
      <td>-10.948</td>
      <td>-3.992</td>
      <td>-3.441</td>
      <td>-4.096</td>
      <td>-1.960</td>
      <td>-5.027</td>
      <td>-2.557</td>
      <td>-3.544</td>
      <td>-3.120</td>
      <td>...</td>
      <td>-5.236</td>
      <td>-2.063</td>
      <td>-2.150</td>
      <td>-2.643</td>
      <td>-2.274</td>
      <td>-2.326</td>
      <td>-2.654</td>
      <td>-2.771</td>
      <td>-3.734</td>
      <td>-2.248</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.500</td>
      <td>-0.350</td>
      <td>-0.680</td>
      <td>-0.424</td>
      <td>-0.648</td>
      <td>-0.417</td>
      <td>-0.450</td>
      <td>-0.233</td>
      <td>-0.557</td>
      <td>-0.694</td>
      <td>...</td>
      <td>-0.580</td>
      <td>-0.780</td>
      <td>-0.925</td>
      <td>-0.404</td>
      <td>-0.551</td>
      <td>-0.519</td>
      <td>-0.526</td>
      <td>-0.289</td>
      <td>-0.752</td>
      <td>-0.628</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.133</td>
      <td>0.177</td>
      <td>-0.089</td>
      <td>0.081</td>
      <td>0.143</td>
      <td>0.073</td>
      <td>0.045</td>
      <td>0.115</td>
      <td>0.096</td>
      <td>-0.148</td>
      <td>...</td>
      <td>0.124</td>
      <td>-0.314</td>
      <td>0.240</td>
      <td>-0.122</td>
      <td>-0.172</td>
      <td>-0.154</td>
      <td>-0.022</td>
      <td>0.044</td>
      <td>-0.203</td>
      <td>-0.236</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.824</td>
      <td>0.628</td>
      <td>0.345</td>
      <td>0.792</td>
      <td>0.597</td>
      <td>0.787</td>
      <td>0.436</td>
      <td>1.046</td>
      <td>0.711</td>
      <td>0.267</td>
      <td>...</td>
      <td>0.659</td>
      <td>0.180</td>
      <td>0.674</td>
      <td>0.825</td>
      <td>0.574</td>
      <td>0.944</td>
      <td>0.808</td>
      <td>0.436</td>
      <td>0.378</td>
      <td>0.256</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.939</td>
      <td>1.732</td>
      <td>2.318</td>
      <td>2.118</td>
      <td>1.727</td>
      <td>1.915</td>
      <td>2.029</td>
      <td>2.136</td>
      <td>2.056</td>
      <td>1.904</td>
      <td>...</td>
      <td>1.456</td>
      <td>1.963</td>
      <td>1.254</td>
      <td>1.895</td>
      <td>2.070</td>
      <td>1.915</td>
      <td>2.105</td>
      <td>2.140</td>
      <td>1.797</td>
      <td>2.103</td>
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
      <th>AGFAGDDAPR_na</th>
      <th>AGLQFPVGR_na</th>
      <th>AHQVVEDGYEFFAK_na</th>
      <th>AITIAGVPQSVTECVK_na</th>
      <th>ALDVMVSTFHK_na</th>
      <th>ALLFIPR_na</th>
      <th>DGQAMLWDLNEGK_na</th>
      <th>EHDPVGQMVNNPK_na</th>
      <th>ELGITALHIK_na</th>
      <th>EMDRETLIDVAR_na</th>
      <th>...</th>
      <th>SQLDIIIHSLK_na</th>
      <th>TAVVVGTITDDVR_na</th>
      <th>TGVELGKPTHFTVNAK_na</th>
      <th>TPELNLDQFHDK_na</th>
      <th>VDATEESDLAQQYGVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VHSFPTLK_na</th>
      <th>VICILSHPIK_na</th>
      <th>VSSDNVADLHEK_na</th>
      <th>YTPSGQAGAAASESLFVSNHAY_na</th>
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
      <td>False</td>
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
      <th>20190805_QE10_nLC0_LiNi_MNT_45cm_HeLa_MUC_01</th>
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
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
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
    
    Optimizer used: <function Adam at 0x00000225FB995040>
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




    
![png](latent_2D_25_10_files/latent_2D_25_10_108_2.png)
    


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
      <td>0.991807</td>
      <td>0.867890</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.727856</td>
      <td>0.456160</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.528797</td>
      <td>0.363936</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.423685</td>
      <td>0.335669</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.366255</td>
      <td>0.331358</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.334844</td>
      <td>0.326942</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.314026</td>
      <td>0.324456</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.301615</td>
      <td>0.321752</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.294528</td>
      <td>0.321690</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.289721</td>
      <td>0.319077</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4939, 4939)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3211)




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
      <th>AGFAGDDAPR</th>
      <td>33.072</td>
      <td>33.805</td>
      <td>33.738</td>
      <td>33.444</td>
      <td>33.212</td>
      <td>33.117</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.018</td>
      <td>31.845</td>
      <td>31.491</td>
      <td>31.276</td>
      <td>31.330</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
      <td>29.711</td>
      <td>29.488</td>
      <td>29.216</td>
      <td>28.813</td>
      <td>29.155</td>
    </tr>
    <tr>
      <th>PLRLPLQDVYK</th>
      <td>31.968</td>
      <td>29.610</td>
      <td>29.858</td>
      <td>30.742</td>
      <td>28.871</td>
      <td>28.770</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGLQFPVGR</th>
      <td>32.984</td>
      <td>33.692</td>
      <td>33.616</td>
      <td>33.508</td>
      <td>33.256</td>
      <td>32.906</td>
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
      <th>TAVVVGTITDDVR</th>
      <td>27.987</td>
      <td>29.196</td>
      <td>29.654</td>
      <td>28.679</td>
      <td>29.122</td>
      <td>28.985</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AITIAGVPQSVTECVK</th>
      <td>29.010</td>
      <td>29.707</td>
      <td>29.737</td>
      <td>29.483</td>
      <td>29.578</td>
      <td>29.496</td>
    </tr>
    <tr>
      <th>GEMMDLQHGSLFLR</th>
      <td>28.109</td>
      <td>27.962</td>
      <td>29.430</td>
      <td>27.849</td>
      <td>27.371</td>
      <td>27.291</td>
    </tr>
    <tr>
      <th>SDALETLGFLNHYQMK</th>
      <td>30.265</td>
      <td>30.019</td>
      <td>29.868</td>
      <td>30.286</td>
      <td>29.853</td>
      <td>29.888</td>
    </tr>
    <tr>
      <th>TGVELGKPTHFTVNAK</th>
      <td>31.341</td>
      <td>30.564</td>
      <td>29.986</td>
      <td>31.537</td>
      <td>31.203</td>
      <td>31.318</td>
    </tr>
  </tbody>
</table>
<p>4939 rows × 6 columns</p>
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
      <td>0.767</td>
      <td>-0.413</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.607</td>
      <td>-0.268</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.709</td>
      <td>-0.553</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.162</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.586</td>
      <td>-0.503</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_122_1.png)
    



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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_123_1.png)
    


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
      <th>AGFAGDDAPR</th>
      <th>AGLQFPVGR</th>
      <th>AHQVVEDGYEFFAK</th>
      <th>AITIAGVPQSVTECVK</th>
      <th>ALDVMVSTFHK</th>
      <th>ALLFIPR</th>
      <th>DGQAMLWDLNEGK</th>
      <th>EHDPVGQMVNNPK</th>
      <th>ELGITALHIK</th>
      <th>EMDRETLIDVAR</th>
      <th>...</th>
      <th>SQLDIIIHSLK</th>
      <th>TAVVVGTITDDVR</th>
      <th>TGVELGKPTHFTVNAK</th>
      <th>TPELNLDQFHDK</th>
      <th>VDATEESDLAQQYGVR</th>
      <th>VFITDDFHDMMPK</th>
      <th>VHSFPTLK</th>
      <th>VICILSHPIK</th>
      <th>VSSDNVADLHEK</th>
      <th>YTPSGQAGAAASESLFVSNHAY</th>
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
      <td>0.748</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.597</td>
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
      <td>0.775</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.541</td>
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
      <td>0.543</td>
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
      <td>0.571</td>
      <td>NaN</td>
      <td>0.466</td>
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
      <td>0.620</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.649</td>
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
    
    Optimizer used: <function Adam at 0x00000225FB995040>
    Loss function: <function loss_fct_vae at 0x00000225FB9BD940>
    
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








    SuggestedLRs(valley=0.002511886414140463)




    
![png](latent_2D_25_10_files/latent_2D_25_10_136_2.png)
    



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
      <td>2019.737305</td>
      <td>228.973099</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1983.392090</td>
      <td>230.442017</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1923.045654</td>
      <td>228.875580</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1873.471313</td>
      <td>220.013748</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1839.788208</td>
      <td>218.747375</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1815.103149</td>
      <td>216.189972</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1797.097290</td>
      <td>213.053421</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1784.711060</td>
      <td>210.055328</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1774.542114</td>
      <td>210.626602</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1768.424805</td>
      <td>210.654099</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4939)




```python
len(pred[0])
```




    4939




```python
learn.loss_func(pred, target)
```




    tensor(3319.4282)




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
      <th>AGFAGDDAPR</th>
      <td>33.072</td>
      <td>33.805</td>
      <td>33.738</td>
      <td>33.444</td>
      <td>33.212</td>
      <td>33.117</td>
      <td>32.966</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.018</td>
      <td>31.845</td>
      <td>31.491</td>
      <td>31.276</td>
      <td>31.330</td>
      <td>31.843</td>
    </tr>
    <tr>
      <th>KFDQLLAEEK</th>
      <td>28.574</td>
      <td>29.711</td>
      <td>29.488</td>
      <td>29.216</td>
      <td>28.813</td>
      <td>29.155</td>
      <td>29.501</td>
    </tr>
    <tr>
      <th>PLRLPLQDVYK</th>
      <td>31.968</td>
      <td>29.610</td>
      <td>29.858</td>
      <td>30.742</td>
      <td>28.871</td>
      <td>28.770</td>
      <td>29.905</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AGLQFPVGR</th>
      <td>32.984</td>
      <td>33.692</td>
      <td>33.616</td>
      <td>33.508</td>
      <td>33.256</td>
      <td>32.906</td>
      <td>33.521</td>
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
      <th>TAVVVGTITDDVR</th>
      <td>27.987</td>
      <td>29.196</td>
      <td>29.654</td>
      <td>28.679</td>
      <td>29.122</td>
      <td>28.985</td>
      <td>29.556</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AITIAGVPQSVTECVK</th>
      <td>29.010</td>
      <td>29.707</td>
      <td>29.737</td>
      <td>29.483</td>
      <td>29.578</td>
      <td>29.496</td>
      <td>29.832</td>
    </tr>
    <tr>
      <th>GEMMDLQHGSLFLR</th>
      <td>28.109</td>
      <td>27.962</td>
      <td>29.430</td>
      <td>27.849</td>
      <td>27.371</td>
      <td>27.291</td>
      <td>29.560</td>
    </tr>
    <tr>
      <th>SDALETLGFLNHYQMK</th>
      <td>30.265</td>
      <td>30.019</td>
      <td>29.868</td>
      <td>30.286</td>
      <td>29.853</td>
      <td>29.888</td>
      <td>29.842</td>
    </tr>
    <tr>
      <th>TGVELGKPTHFTVNAK</th>
      <td>31.341</td>
      <td>30.564</td>
      <td>29.986</td>
      <td>31.537</td>
      <td>31.203</td>
      <td>31.318</td>
      <td>30.011</td>
    </tr>
  </tbody>
</table>
<p>4939 rows × 7 columns</p>
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
      <td>0.561</td>
      <td>-0.016</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.518</td>
      <td>-0.076</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.271</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.202</td>
      <td>0.122</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.210</td>
      <td>-0.024</td>
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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_146_1.png)
    



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
    


    
![png](latent_2D_25_10_files/latent_2D_25_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20190104_QE6_nLC6_MM_QC_MNT_HELA_02_190108210418', 'PLRLPLQDVYK'), ('20190107_QE5_nLC5_DS_QC_MNT_HeLa_FlashPack_03', 'HVVQSISTQQEK'), ('20190115_QE5_nLC5_RJC_MNT_HeLa_02', 'LISWYDNEFGYSNR'), ('20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01', 'SQLDIIIHSLK'), ('20190129_QE1_nLC2_GP_QC_MNT_HELA_01', 'VSSDNVADLHEK'), ('20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01', 'ESEPQAAAEPAEAK'), ('20190204_QE4_LC12_SCL_QC_MNT_HeLa_02', 'ALDVMVSTFHK'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm2', 'FHQLDIDDLQSIR'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm3', 'FHQLDIDDLQSIR'), ('20190211_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'FDASFFGVHPK'), ('20190220_QE3_nLC7_TSB_QC_MNT_HELA_01', 'VSSDNVADLHEK'), ('20190221_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'LHFFMPGFAPLTSR'), ('20190221_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'LHFFMPGFAPLTSR'), ('20190228_QE4_LC12_JE_QC_MNT_HeLa_01', 'LDIDSPPITAR'), ('20190228_QE4_LC12_JE_QC_MNT_HeLa_02', 'LDIDSPPITAR'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_01', 'TPELNLDQFHDK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_02', 'TPELNLDQFHDK'), ('20190305_QE8_nLC14_RG_QC_MNT_50cm_HELA_01', 'AGLQFPVGR'), ('20190306_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'AGLQFPVGR'), ('20190306_QE1_nLC2_ANHO_QC_MNT_HELA_01_20190308134131', 'HVVQSISTQQEK'), ('20190402_QE1_nLC2_GP_MNT_QC_hela_01', 'FQSSHHPTDITSLDQYVER'), ('20190402_QE6_LC6_AS_QC_MNT_HeLa_02', 'TPELNLDQFHDK'), ('20190409_QE1_nLC2_ANHO_MNT_QC_hela_01', 'ISMPDLDLNLK'), ('20190415_QE3_nLC5_DS_QC_MNT_HeLa_02', 'LTNSMMMHGR'), ('20190417_QX4_JoSw_MA_HeLa_500ng_BR14_new', 'LTNSMMMHGR'), ('20190423_QX6_MaTa_MA_HeLa_Br14_500ng_DIA_LC09', 'LFPLIQAMHPTLAGK'), ('20190502_QX7_ChDe_MA_HeLa_500ng', 'LVAIVDPHIK'), ('20190507_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_01', 'PLRLPLQDVYK'), ('20190507_QX6_ChDe_MA_HeLa_Br13_500ng_LC09', 'IGIEIIK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_06', 'SCMLTGTPESVQSAK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_07', 'VSSDNVADLHEK'), ('20190513_QE6_LC4_IAH_QC_MNT_HeLa_03', 'LDIDSPPITAR'), ('20190514_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'LISWYDNEFGYSNR'), ('20190517_QX0_AlRe_MA_HeLa_500ng_LC07_1_BR14', 'VSSDNVADLHEK'), ('20190520_QX1_JoMu_MA_HeLa_500ng_LC10_DMSO', 'IGIEIIK'), ('20190522_QE3_nLC3_AP_QC_MNT_HeLa_03', 'ESEPQAAAEPAEAK'), ('20190606_QE4_LC12_JE_QC_MNT_HeLa_01b', 'LVAIVDPHIK'), ('20190614_QX3_JoSw_MA_Hela_500ng_LC15_190709210149', 'EHDPVGQMVNNPK'), ('20190618_QX4_JiYu_MA_HeLa_500ng', 'EMDRETLIDVAR'), ('20190619_QX7_IgPa_MA_HeLa_Br14_500ng_190619192949', 'ILPTLEAVAALGNK'), ('20190621_QX4_JoMu_MA_HeLa_500ng_190621161214', 'TAVVVGTITDDVR'), ('20190623_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_MUC_02', 'TGVELGKPTHFTVNAK'), ('20190624_QE6_LC4_AS_QC_MNT_HeLa_03', 'LDIDSPPITAR'), ('20190624_QX4_JiYu_MA_HeLa_500ng', 'ALLFIPR'), ('20190625_QX7_IgPa_MA_HeLa_Br14_500ng', 'ILPTLEAVAALGNK'), ('20190626_QX6_ChDe_MA_HeLa_500ng_LC09', 'RGFAFVTFDDHDSVDK'), ('20190629_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'SDALETLGFLNHYQMK'), ('20190701_QX6_MaTa_MA_HeLa_500ng_LC09', 'HVVQSISTQQEK'), ('20190707_QX3_MaTa_MA_Hela_500ng_LC15', 'VICILSHPIK'), ('20190708_QE10_nLC0_LiNi_QC_MNT_15cm_HeLa_MUC_01', 'VICILSHPIK'), ('20190708_QE4_LC12_IAH_QC_MNT_HeLa_01', 'ESEPQAAAEPAEAK'), ('20190708_QX7_MaMu_MA_HeLa_Br14_500ng', 'LLDFGSLSNLQVTQPTVGMNFK'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'LHIIEVGTPPTGNQPFPK'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05', 'LHIIEVGTPPTGNQPFPK'), ('20190712_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'VHSFPTLK'), ('20190712_QE1_nLC13_ANHO_QC_MNT_HELA_03', 'VHSFPTLK'), ('20190717_QE6_LC4_SCL_QC_MNT_Hela_03', 'ISMPDLDLNLK'), ('20190719_QX1_JoMu_MA_HeLa_500ng_LC10', 'LYSPSQIGAFVLMK'), ('20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_1', 'LVINGNPITIFQERDPSK'), ('20190726_QE7_nLC7_MEM_QC_MNT_HeLa_01', 'LISWYDNEFGYSNR'), ('20190730_QX6_AsJa_MA_HeLa_500ng_LC09', 'HVVQSISTQQEK'), ('20190731_QE8_nLC14_ASD_QC_MNT_HeLa_03', 'GEMMDLQHGSLFLR'), ('20190802_QE9_nLC13_RG_QC_MNT_HeLa_MUC_50cm_2', 'ELGITALHIK'), ('20190802_QX2_OzKa_MA_HeLa_500ng_CTCDoff_LC05', 'ELGITALHIK'), ('20190802_QX6_MaTa_MA_HeLa_500ng_LC09', 'FHQLDIDDLQSIR'), ('20190803_QX8_AnPi_MA_HeLa_BR14_500ng', 'RGFAFVTFDDHDSVDK'), ('20190804_QX0_AsJa_MA_HeLa_500ng_LC07_01', 'RGFAFVTFDDHDSVDK'), ('20190805_QE1_nLC2_AB_MNT_HELA_01', 'TAVVVGTITDDVR')]
    




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
      <td>0.566</td>
      <td>0.609</td>
      <td>1.546</td>
      <td>1.607</td>
      <td>2.001</td>
      <td>2.101</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.483</td>
      <td>0.501</td>
      <td>0.855</td>
      <td>0.941</td>
      <td>1.050</td>
      <td>1.030</td>
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
