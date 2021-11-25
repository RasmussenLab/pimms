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
    AIDDNMSLDEIEK                   968
    FAQPGSFEYEYAMR                  961
    TLTAVHDAILEDLVFPSEIVGK          960
    ITLQDVVSHSK                     992
    VEFMDDTSR                       981
    RAGELTEDEVER                    993
    FNEEHIPDSPFVVPVASPSGDAR         981
    ALTSEIALLQSR                    995
    AHGPGLEGGLVGKPAEFTIDTK          964
    IVEVLLMK                        993
    EQIVPKPEEEVAQK                  993
    VVSQYSSLLSPMSVNAVMK             997
    AQIHDLVLVGGSTR                  981
    SGDSEVYQLGDVSQK                 970
    VLAMSGDPNYLHR                   974
    VFITDDFHDMMPK                   994
    GYISPYFINTSK                    971
    ILQDGGLQVVEK                    992
    EGHLSPDIVAEQK                   996
    LDPHLVLDQLR                     990
    NMMAACDPR                       997
    IGLFGGAGVGK                     960
    GLVLGPIHK                       986
    HEQILVLDPPTDLK                  981
    NIDNPALADIYTEHAHQVVVAK          991
    ELAEDGYSGVEVR                   999
    ARFEELCSDLFR                    997
    TAFQEALDAAGDK                 1,000
    ADEGISFR                        984
    EGNDLYHEMIESGVINLK              982
    EIIDLVLDR                       990
    TFVNITPAEVGVLVGK                957
    YMACCLLYR                       958
    GIGMGNIGPAGMGMEGIGFGINK         987
    SEMEVQDAELK                     991
    AFGYYGPLR                       954
    NTGIICTIGPASR                   954
    FLSQPFQVAEVFTGHMGK              983
    TVAGGAWTYNTTSAVTVK              992
    HIYYITGETK                      997
    IHVSDQELQSANASVDDSRLEELK        987
    MELQEIQLK                       991
    FVINYDYPNSSEDYIHR               997
    DSTLIMQLLR                      996
    IPSAVGYQPTLATDMGTMQER           924
    IAGYVTHLMK                      994
    KIEPELDGSAQVTSHDASTNGLINFIK     997
    EEASDYLELDTIK                   980
    VEPGLGADNSVVR                   974
    THEAQIQEMR                      986
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
      <th>AIDDNMSLDEIEK</th>
      <td>26.945</td>
    </tr>
    <tr>
      <th>FAQPGSFEYEYAMR</th>
      <td>27.780</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
    </tr>
    <tr>
      <th>ITLQDVVSHSK</th>
      <td>27.466</td>
    </tr>
    <tr>
      <th>VEFMDDTSR</th>
      <td>28.817</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
    </tr>
    <tr>
      <th>EEASDYLELDTIK</th>
      <td>29.041</td>
    </tr>
    <tr>
      <th>VEPGLGADNSVVR</th>
      <td>29.902</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>29.699</td>
    </tr>
  </tbody>
</table>
<p>49112 rows × 1 columns</p>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_24_1.png)
    



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
      <td>0.982</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.024</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.980</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.990</td>
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
      <th>AIDDNMSLDEIEK</th>
      <td>26.945</td>
    </tr>
    <tr>
      <th>FAQPGSFEYEYAMR</th>
      <td>27.780</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
    </tr>
    <tr>
      <th>ITLQDVVSHSK</th>
      <td>27.466</td>
    </tr>
    <tr>
      <th>VEFMDDTSR</th>
      <td>28.817</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>IAGYVTHLMK</th>
      <td>30.712</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
    </tr>
    <tr>
      <th>EEASDYLELDTIK</th>
      <td>29.041</td>
    </tr>
    <tr>
      <th>VEPGLGADNSVVR</th>
      <td>29.902</td>
    </tr>
    <tr>
      <th>THEAQIQEMR</th>
      <td>29.699</td>
    </tr>
  </tbody>
</table>
<p>49112 rows × 1 columns</p>
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
      <th>ADEGISFR</th>
      <td>31.443</td>
    </tr>
    <tr>
      <th>20190730_QE6_nLC4_MPL_QC_MNT_HeLa_01</th>
      <th>ADEGISFR</th>
      <td>32.155</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_02</th>
      <th>ADEGISFR</th>
      <td>26.847</td>
    </tr>
    <tr>
      <th>20190528_QX1_PhGe_MA_HeLa_DMSO_500ng_LC14_190528164924</th>
      <th>ADEGISFR</th>
      <td>33.451</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>ADEGISFR</th>
      <td>32.090</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190803_QE8_nLC14_ASD_QC_MNT_HeLa</th>
      <th>YMACCLLYR</th>
      <td>28.626</td>
    </tr>
    <tr>
      <th>20190724_QX0_MePh_MA_HeLa_500ng_LC07_01</th>
      <th>YMACCLLYR</th>
      <td>31.556</td>
    </tr>
    <tr>
      <th>20190204_QE8_nLC14_RG_QC_HeLa_15cm_02</th>
      <th>YMACCLLYR</th>
      <td>29.894</td>
    </tr>
    <tr>
      <th>20190612_QX3_JoMu_MA_HeLa_500ng_LC15_uPAC200cm</th>
      <th>YMACCLLYR</th>
      <td>31.391</td>
    </tr>
    <tr>
      <th>20190609_QX8_MiWi_MA_HeLa_BR14_500ng</th>
      <th>YMACCLLYR</th>
      <td>26.511</td>
    </tr>
  </tbody>
</table>
<p>44203 rows × 1 columns</p>
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
    Shape in validation: (994, 50)
    




    ((994, 50), (994, 50))



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
      <th>ADEGISFR</th>
      <td>30.461</td>
      <td>31.743</td>
      <td>31.775</td>
      <td>31.118</td>
    </tr>
    <tr>
      <th>ALTSEIALLQSR</th>
      <td>27.244</td>
      <td>28.492</td>
      <td>28.482</td>
      <td>27.467</td>
    </tr>
    <tr>
      <th>EGNDLYHEMIESGVINLK</th>
      <td>27.597</td>
      <td>28.637</td>
      <td>28.841</td>
      <td>27.366</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
      <td>30.036</td>
      <td>29.682</td>
      <td>29.769</td>
    </tr>
    <tr>
      <th>VLAMSGDPNYLHR</th>
      <td>27.667</td>
      <td>28.729</td>
      <td>28.549</td>
      <td>27.706</td>
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
      <th>FVINYDYPNSSEDYIHR</th>
      <td>28.390</td>
      <td>28.837</td>
      <td>28.853</td>
      <td>28.490</td>
    </tr>
    <tr>
      <th>HIYYITGETK</th>
      <td>30.932</td>
      <td>30.755</td>
      <td>30.692</td>
      <td>30.868</td>
    </tr>
    <tr>
      <th>IHVSDQELQSANASVDDSRLEELK</th>
      <td>28.929</td>
      <td>28.922</td>
      <td>29.252</td>
      <td>29.313</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
      <td>30.012</td>
      <td>29.913</td>
      <td>30.597</td>
    </tr>
    <tr>
      <th>NMMAACDPR</th>
      <td>32.531</td>
      <td>32.091</td>
      <td>32.015</td>
      <td>32.285</td>
    </tr>
  </tbody>
</table>
<p>4909 rows × 4 columns</p>
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
      <th>ALTSEIALLQSR</th>
      <td>27.441</td>
      <td>28.492</td>
      <td>28.482</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181227_QE6_nLC6_CSC_QC_HeLa_02</th>
      <th>SGDSEVYQLGDVSQK</th>
      <td>28.562</td>
      <td>27.819</td>
      <td>28.079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</th>
      <th>ELAEDGYSGVEVR</th>
      <td>29.437</td>
      <td>29.305</td>
      <td>29.527</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190115_QE2_NLC10_TW_QC_MNT_HeLa_01</th>
      <th>ITLQDVVSHSK</th>
      <td>27.326</td>
      <td>28.156</td>
      <td>28.136</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190115_QE5_nLC5_RJC_MNT_HeLa_01</th>
      <th>YMACCLLYR</th>
      <td>29.433</td>
      <td>29.193</td>
      <td>29.171</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190129_QE1_nLC2_GP_QC_MNT_HELA_01</th>
      <th>EQIVPKPEEEVAQK</th>
      <td>29.390</td>
      <td>30.085</td>
      <td>30.309</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190131_QE10_nLC0_NHS_MNT_HELA_50cm_02</th>
      <th>FAQPGSFEYEYAMR</th>
      <td>25.513</td>
      <td>26.295</td>
      <td>26.916</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190201_QE10_nLC0_NHS_MNT_HELA_45cm_01</th>
      <th>FLSQPFQVAEVFTGHMGK</th>
      <td>30.397</td>
      <td>31.597</td>
      <td>31.237</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01</th>
      <th>GYISPYFINTSK</th>
      <td>28.945</td>
      <td>29.545</td>
      <td>29.954</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04</th>
      <th>TAFQEALDAAGDK</th>
      <td>32.373</td>
      <td>32.281</td>
      <td>32.362</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190207_QE2_NLC10_GP_MNT_HeLa_01</th>
      <th>AFGYYGPLR</th>
      <td>29.946</td>
      <td>29.547</td>
      <td>29.626</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190214_QE4_LC12_SCL_QC_MNT_HeLa_03</th>
      <th>DSTLIMQLLR</th>
      <td>28.647</td>
      <td>32.249</td>
      <td>31.987</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190219_QE3_Evo2_UHG_QC_MNT_HELA_01_190219173213</th>
      <th>MELQEIQLK</th>
      <td>29.337</td>
      <td>30.147</td>
      <td>30.028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_5</th>
      <th>AIDDNMSLDEIEK</th>
      <td>28.684</td>
      <td>28.974</td>
      <td>29.005</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_27</th>
      <th>VVSQYSSLLSPMSVNAVMK</th>
      <td>27.289</td>
      <td>29.086</td>
      <td>28.795</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_23</th>
      <th>VVSQYSSLLSPMSVNAVMK</th>
      <td>25.883</td>
      <td>29.086</td>
      <td>28.795</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190301_QE1_nLC2_ANHO_QC_MNT_HELA_01_20190303025443</th>
      <th>TFVNITPAEVGVLVGK</th>
      <td>33.723</td>
      <td>33.049</td>
      <td>32.773</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190305_QE8_nLC14_ASD_QC_MNT_50cm_HELA_02</th>
      <th>AIDDNMSLDEIEK</th>
      <td>25.668</td>
      <td>28.974</td>
      <td>29.005</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ILQDGGLQVVEK</th>
      <td>26.163</td>
      <td>29.414</td>
      <td>29.260</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190305_QE8_nLC14_RG_QC_MNT_50cm_HELA_01</th>
      <th>AIDDNMSLDEIEK</th>
      <td>29.755</td>
      <td>28.974</td>
      <td>29.005</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190408_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>VVSQYSSLLSPMSVNAVMK</th>
      <td>29.266</td>
      <td>29.086</td>
      <td>28.795</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190417_QX4_JoSw_MA_HeLa_500ng_BR14_new</th>
      <th>YMACCLLYR</th>
      <td>31.200</td>
      <td>29.193</td>
      <td>29.171</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190418_QX8_JuSc_MA_HeLa_500ng_1</th>
      <th>AFGYYGPLR</th>
      <td>30.938</td>
      <td>29.547</td>
      <td>29.626</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190422_QE4_LC12_JE-IAH_QC_MNT_HeLa_02</th>
      <th>IAGYVTHLMK</th>
      <td>29.506</td>
      <td>30.492</td>
      <td>30.388</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01_20190425184929</th>
      <th>IVEVLLMK</th>
      <td>28.597</td>
      <td>28.559</td>
      <td>28.433</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190507_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_01</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <td>30.051</td>
      <td>28.898</td>
      <td>28.599</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TFVNITPAEVGVLVGK</th>
      <td>33.702</td>
      <td>33.049</td>
      <td>32.773</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190509_QE8_nLC14_AGF_QC_MNT_HeLa_01</th>
      <th>EEASDYLELDTIK</th>
      <td>27.548</td>
      <td>29.273</td>
      <td>29.159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QE6_LC4_IAH_QC_MNT_HeLa_03</th>
      <th>EGHLSPDIVAEQK</th>
      <td>30.252</td>
      <td>30.071</td>
      <td>30.036</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QE2_NLC1_GP_MNT_HELA_01</th>
      <th>MELQEIQLK</th>
      <td>28.719</td>
      <td>30.147</td>
      <td>30.028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QE4_LC12_AS_QC_MNT_HeLa_01</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <td>30.392</td>
      <td>28.898</td>
      <td>28.599</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02</th>
      <th>EEASDYLELDTIK</th>
      <td>29.531</td>
      <td>29.273</td>
      <td>29.159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190531_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>DSTLIMQLLR</th>
      <td>26.974</td>
      <td>32.249</td>
      <td>31.987</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190604_QX8_MiWi_MA_HeLa_BR14_500ng</th>
      <th>LDPHLVLDQLR</th>
      <td>29.866</td>
      <td>27.455</td>
      <td>27.819</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20190605_QX0_MePh_MA_HeLa_500ng_LC07_1_BR14</th>
      <th>AQIHDLVLVGGSTR</th>
      <td>32.926</td>
      <td>29.151</td>
      <td>29.433</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>LDPHLVLDQLR</th>
      <td>31.135</td>
      <td>27.455</td>
      <td>27.819</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190605_QX3_ChDe_MA_Hela_500ng_LC15</th>
      <th>EEASDYLELDTIK</th>
      <td>30.672</td>
      <td>29.273</td>
      <td>29.159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625163359</th>
      <th>GLVLGPIHK</th>
      <td>30.800</td>
      <td>29.967</td>
      <td>29.888</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX0_MePh_MA_HeLa_500ng_LC07_5</th>
      <th>FLSQPFQVAEVFTGHMGK</th>
      <td>33.167</td>
      <td>31.597</td>
      <td>31.237</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>EIIDLVLDR</th>
      <td>32.440</td>
      <td>32.246</td>
      <td>32.189</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX3_LiSc_MA_Hela_500ng_LC15</th>
      <th>EIIDLVLDR</th>
      <td>33.181</td>
      <td>32.246</td>
      <td>32.189</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX4_JiYu_MA_HeLa_500ng</th>
      <th>EIIDLVLDR</th>
      <td>32.210</td>
      <td>32.246</td>
      <td>32.189</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02</th>
      <th>NMMAACDPR</th>
      <td>33.643</td>
      <td>32.091</td>
      <td>32.015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190615_QX4_JiYu_MA_HeLa_500ng</th>
      <th>VLAMSGDPNYLHR</th>
      <td>30.540</td>
      <td>28.729</td>
      <td>28.549</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190617_QE_LC_UHG_QC_MNT_HELA_04</th>
      <th>IAGYVTHLMK</th>
      <td>30.240</td>
      <td>30.492</td>
      <td>30.388</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03</th>
      <th>LDPHLVLDQLR</th>
      <td>26.880</td>
      <td>27.455</td>
      <td>27.819</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE9_nLC0_RG_MNT_Hela_MUC_50cm_2</th>
      <th>EGHLSPDIVAEQK</th>
      <td>28.237</td>
      <td>30.071</td>
      <td>30.036</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190626_QX7_IgPa_MA_HeLa_Br14_500ng</th>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>27.113</td>
      <td>30.036</td>
      <td>29.682</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190628_QE2_NLC1_TL_QC_MNT_HELA_05</th>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.605</td>
      <td>30.012</td>
      <td>29.913</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03</th>
      <th>GLVLGPIHK</th>
      <td>29.355</td>
      <td>29.967</td>
      <td>29.888</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190708_QX7_MaMu_MA_HeLa_Br14_500ng</th>
      <th>EGHLSPDIVAEQK</th>
      <td>31.589</td>
      <td>30.071</td>
      <td>30.036</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190708_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>ILQDGGLQVVEK</th>
      <td>29.453</td>
      <td>29.414</td>
      <td>29.260</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190709_QX2_JoMu_MA_HeLa_500ng_LC05</th>
      <th>SGDSEVYQLGDVSQK</th>
      <td>29.196</td>
      <td>27.819</td>
      <td>28.079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190709_QX2_JoMu_MA_HeLa_500ng_LC05_190709143552</th>
      <th>SGDSEVYQLGDVSQK</th>
      <td>28.517</td>
      <td>27.819</td>
      <td>28.079</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190709_QX6_MaTa_MA_HeLa_500ng_LC09_20190709155356</th>
      <th>ARFEELCSDLFR</th>
      <td>30.894</td>
      <td>30.166</td>
      <td>30.123</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190722_QE8_nLC0_BDA_QC_HeLa_50cm_02</th>
      <th>EGNDLYHEMIESGVINLK</th>
      <td>30.126</td>
      <td>28.637</td>
      <td>28.841</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2</th>
      <th>HIYYITGETK</th>
      <td>29.943</td>
      <td>30.755</td>
      <td>30.692</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190725_QX0_MePh_MA_HeLa_500ng_LC07_01</th>
      <th>TFVNITPAEVGVLVGK</th>
      <td>34.177</td>
      <td>33.049</td>
      <td>32.773</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QX8_ChSc_MA_HeLa_500ng</th>
      <th>EIIDLVLDR</th>
      <td>33.449</td>
      <td>32.246</td>
      <td>32.189</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>FVINYDYPNSSEDYIHR</th>
      <td>28.973</td>
      <td>28.837</td>
      <td>28.853</td>
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
      <td>AFGYYGPLR</td>
      <td>28.703</td>
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
      <td>AIDDNMSLDEIEK</td>
      <td>26.945</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AQIHDLVLVGGSTR</td>
      <td>29.171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ARFEELCSDLFR</td>
      <td>28.689</td>
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
      <td>ADEGISFR</td>
      <td>30.461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALTSEIALLQSR</td>
      <td>27.244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EGNDLYHEMIESGVINLK</td>
      <td>27.597</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TLTAVHDAILEDLVFPSEIVGK</td>
      <td>30.168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>VLAMSGDPNYLHR</td>
      <td>27.667</td>
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
      <td>20190208_QE3_nLC3_KBE_QC_MNT_HeLa_01</td>
      <td>EEASDYLELDTIK</td>
      <td>28.406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190719_QX1_JoMu_MA_HeLa_500ng_LC10</td>
      <td>IVEVLLMK</td>
      <td>29.830</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190515_QE4_LC12_AS_QC_MNT_HeLa_01</td>
      <td>FLSQPFQVAEVFTGHMGK</td>
      <td>32.412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190709_QE3_nLC5_GF_QC_MNT_Hela_02</td>
      <td>KIEPELDGSAQVTSHDASTNGLINFIK</td>
      <td>29.837</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190731_QE8_nLC14_ASD_QC_MNT_HeLa_02</td>
      <td>NTGIICTIGPASR</td>
      <td>31.336</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190425_QX8_JuSc_MA_HeLa_500ng_1</td>
      <td>ARFEELCSDLFR</td>
      <td>29.537</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190723_QE4_LC12_IAH_QC_MNT_HeLa_03</td>
      <td>THEAQIQEMR</td>
      <td>29.049</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190802_QE3_nLC3_DBJ_AMV_QC_MNT_HELA_02</td>
      <td>GLVLGPIHK</td>
      <td>29.302</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190425_QX4_JoSw_MA_HeLa_500ng_BR13_standard_190425181909</td>
      <td>FVINYDYPNSSEDYIHR</td>
      <td>31.340</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625163359</td>
      <td>FLSQPFQVAEVFTGHMGK</td>
      <td>31.984</td>
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
      <td>20190717_QX3_OzKa_MA_Hela_500ng_LC15_190721144939</td>
      <td>ALTSEIALLQSR</td>
      <td>30.092</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190121_QE5_nLC5_AH_QC_MNT_HeLa_250ng_01</td>
      <td>NTGIICTIGPASR</td>
      <td>31.225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190625_QE1_nLC2_GP_QC_MNT_HELA_03</td>
      <td>THEAQIQEMR</td>
      <td>30.063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190702_QE10_nLC0_FaCo_QC_MNT_HeLa_MUC</td>
      <td>TFVNITPAEVGVLVGK</td>
      <td>33.509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190719_QE1_nLC13_GP_QC_MNT_HELA_01</td>
      <td>FAQPGSFEYEYAMR</td>
      <td>23.897</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190213_QE3_nLC3_UH_QC_MNT_HeLa_02</td>
      <td>ALTSEIALLQSR</td>
      <td>27.276</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190624_QX4_JiYu_MA_HeLa_500ng</td>
      <td>FVINYDYPNSSEDYIHR</td>
      <td>29.791</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190129_QE8_nLC14_FaCo_QC_MNT_50cm_Hela_20190129205246</td>
      <td>TFVNITPAEVGVLVGK</td>
      <td>33.786</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190624_QE4_nLC12_MM_QC_MNT_HELA_01_20190625144904</td>
      <td>DSTLIMQLLR</td>
      <td>32.149</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190611_QE4_LC12_JE_QC_MNT_HeLa_01</td>
      <td>EGHLSPDIVAEQK</td>
      <td>29.666</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (995, 51)




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
     'n_samples': 995,
     'y_range': (20, 35)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1990       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 995        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,138
    Total trainable params: 3,138
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x00000297F9587040>
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
      <td>2.378061</td>
      <td>2.036471</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.853891</td>
      <td>0.877336</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.714594</td>
      <td>0.739279</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.667353</td>
      <td>0.702194</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.612311</td>
      <td>0.687951</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.590427</td>
      <td>0.661613</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.570451</td>
      <td>0.656601</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.594977</td>
      <td>0.650413</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.558643</td>
      <td>0.649032</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.510968</td>
      <td>0.648707</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_58_1.png)
    


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
      <th>4,366</th>
      <td>885</td>
      <td>5</td>
      <td>30.092</td>
    </tr>
    <tr>
      <th>350</th>
      <td>74</td>
      <td>36</td>
      <td>31.225</td>
    </tr>
    <tr>
      <th>3,594</th>
      <td>734</td>
      <td>42</td>
      <td>30.063</td>
    </tr>
    <tr>
      <th>3,925</th>
      <td>798</td>
      <td>41</td>
      <td>33.509</td>
    </tr>
    <tr>
      <th>4,393</th>
      <td>891</td>
      <td>15</td>
      <td>23.897</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2,606</th>
      <td>531</td>
      <td>27</td>
      <td>29.648</td>
    </tr>
    <tr>
      <th>3,901</th>
      <td>794</td>
      <td>35</td>
      <td>33.233</td>
    </tr>
    <tr>
      <th>925</th>
      <td>192</td>
      <td>5</td>
      <td>27.273</td>
    </tr>
    <tr>
      <th>126</th>
      <td>27</td>
      <td>10</td>
      <td>27.762</td>
    </tr>
    <tr>
      <th>4,731</th>
      <td>959</td>
      <td>9</td>
      <td>29.992</td>
    </tr>
  </tbody>
</table>
<p>4909 rows × 3 columns</p>
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
      <th>ADEGISFR</th>
      <td>30.461</td>
      <td>31.743</td>
      <td>31.775</td>
      <td>31.118</td>
      <td>31.234</td>
    </tr>
    <tr>
      <th>ALTSEIALLQSR</th>
      <td>27.244</td>
      <td>28.492</td>
      <td>28.482</td>
      <td>27.467</td>
      <td>27.818</td>
    </tr>
    <tr>
      <th>EGNDLYHEMIESGVINLK</th>
      <td>27.597</td>
      <td>28.637</td>
      <td>28.841</td>
      <td>27.366</td>
      <td>27.543</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
      <td>30.036</td>
      <td>29.682</td>
      <td>29.769</td>
      <td>28.650</td>
    </tr>
    <tr>
      <th>VLAMSGDPNYLHR</th>
      <td>27.667</td>
      <td>28.729</td>
      <td>28.549</td>
      <td>27.706</td>
      <td>27.530</td>
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
      <th>FVINYDYPNSSEDYIHR</th>
      <td>28.390</td>
      <td>28.837</td>
      <td>28.853</td>
      <td>28.490</td>
      <td>28.621</td>
    </tr>
    <tr>
      <th>HIYYITGETK</th>
      <td>30.932</td>
      <td>30.755</td>
      <td>30.692</td>
      <td>30.868</td>
      <td>30.391</td>
    </tr>
    <tr>
      <th>IHVSDQELQSANASVDDSRLEELK</th>
      <td>28.929</td>
      <td>28.922</td>
      <td>29.252</td>
      <td>29.313</td>
      <td>28.729</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
      <td>30.012</td>
      <td>29.913</td>
      <td>30.597</td>
      <td>29.905</td>
    </tr>
    <tr>
      <th>NMMAACDPR</th>
      <td>32.531</td>
      <td>32.091</td>
      <td>32.015</td>
      <td>32.285</td>
      <td>31.843</td>
    </tr>
  </tbody>
</table>
<p>4909 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.132
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.133
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.270
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.173
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.158
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_66_1.png)
    



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
      <td>-0.157</td>
      <td>-0.079</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.198</td>
      <td>-0.069</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.004</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.185</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.193</td>
      <td>-0.341</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_68_1.png)
    



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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_69_1.png)
    


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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK</th>
      <th>THEAQIQEMR</th>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <th>TVAGGAWTYNTTSAVTVK</th>
      <th>VEFMDDTSR</th>
      <th>VEPGLGADNSVVR</th>
      <th>VFITDDFHDMMPK</th>
      <th>VLAMSGDPNYLHR</th>
      <th>VVSQYSSLLSPMSVNAVMK</th>
      <th>YMACCLLYR</th>
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
      <td>30.461</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.244</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.168</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>28.705</td>
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
      <td>29.391</td>
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
      <td>29.147</td>
      <td>27.471</td>
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
      <td>27.441</td>
      <td>28.800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.640</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.330</td>
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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>THEAQIQEMR_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>VEFMDDTSR_na</th>
      <th>VEPGLGADNSVVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAMSGDPNYLHR_na</th>
      <th>VVSQYSSLLSPMSVNAVMK_na</th>
      <th>YMACCLLYR_na</th>
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
      <td>-0.027</td>
      <td>-0.935</td>
      <td>-0.835</td>
      <td>-1.365</td>
      <td>0.007</td>
      <td>-0.091</td>
      <td>-1.265</td>
      <td>-0.197</td>
      <td>-0.576</td>
      <td>-1.207</td>
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
      <td>-0.969</td>
      <td>-0.069</td>
      <td>-0.769</td>
      <td>-0.338</td>
      <td>-1.233</td>
      <td>-0.041</td>
      <td>-0.821</td>
      <td>-0.087</td>
      <td>-0.524</td>
      <td>-1.085</td>
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
      <td>-0.251</td>
      <td>-0.558</td>
      <td>-0.168</td>
      <td>-0.498</td>
      <td>-0.488</td>
      <td>-0.283</td>
      <td>-0.918</td>
      <td>-0.367</td>
      <td>-0.004</td>
      <td>-0.039</td>
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
      <td>-0.650</td>
      <td>-0.776</td>
      <td>-1.038</td>
      <td>-0.017</td>
      <td>0.007</td>
      <td>0.328</td>
      <td>-0.580</td>
      <td>-0.088</td>
      <td>-0.569</td>
      <td>-0.519</td>
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
      <td>-0.867</td>
      <td>-1.120</td>
      <td>-0.776</td>
      <td>-0.285</td>
      <td>0.007</td>
      <td>-0.099</td>
      <td>-0.961</td>
      <td>0.796</td>
      <td>0.087</td>
      <td>-1.180</td>
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
      <td>0.332</td>
      <td>0.487</td>
      <td>-0.581</td>
      <td>0.916</td>
      <td>0.634</td>
      <td>1.130</td>
      <td>0.649</td>
      <td>0.990</td>
      <td>0.087</td>
      <td>0.198</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.299</td>
      <td>-0.004</td>
      <td>0.380</td>
      <td>-0.264</td>
      <td>-0.192</td>
      <td>0.036</td>
      <td>-0.244</td>
      <td>-1.920</td>
      <td>0.069</td>
      <td>0.276</td>
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
      <td>0.094</td>
      <td>-0.069</td>
      <td>0.441</td>
      <td>-0.062</td>
      <td>-0.175</td>
      <td>0.219</td>
      <td>0.120</td>
      <td>0.271</td>
      <td>0.129</td>
      <td>0.364</td>
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
      <td>0.253</td>
      <td>-0.396</td>
      <td>0.526</td>
      <td>-0.151</td>
      <td>-0.030</td>
      <td>0.207</td>
      <td>0.044</td>
      <td>0.212</td>
      <td>-0.163</td>
      <td>-2.097</td>
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
      <td>0.245</td>
      <td>-0.277</td>
      <td>0.844</td>
      <td>-0.017</td>
      <td>-0.060</td>
      <td>-0.331</td>
      <td>0.034</td>
      <td>0.279</td>
      <td>-0.113</td>
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
  </tbody>
</table>
<p>994 rows × 100 columns</p>
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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>THEAQIQEMR_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>VEFMDDTSR_na</th>
      <th>VEPGLGADNSVVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAMSGDPNYLHR_na</th>
      <th>VVSQYSSLLSPMSVNAVMK_na</th>
      <th>YMACCLLYR_na</th>
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
      <td>-0.028</td>
      <td>-0.877</td>
      <td>-0.749</td>
      <td>-1.276</td>
      <td>0.008</td>
      <td>-0.098</td>
      <td>-1.194</td>
      <td>-0.169</td>
      <td>-0.530</td>
      <td>-1.139</td>
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
      <td>-0.915</td>
      <td>-0.074</td>
      <td>-0.688</td>
      <td>-0.317</td>
      <td>-1.166</td>
      <td>-0.051</td>
      <td>-0.773</td>
      <td>-0.065</td>
      <td>-0.481</td>
      <td>-1.024</td>
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
      <td>-0.239</td>
      <td>-0.528</td>
      <td>-0.126</td>
      <td>-0.467</td>
      <td>-0.461</td>
      <td>-0.278</td>
      <td>-0.865</td>
      <td>-0.331</td>
      <td>0.007</td>
      <td>-0.034</td>
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
      <td>-0.615</td>
      <td>-0.730</td>
      <td>-0.939</td>
      <td>-0.019</td>
      <td>0.008</td>
      <td>0.296</td>
      <td>-0.546</td>
      <td>-0.066</td>
      <td>-0.524</td>
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
      <td>False</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.819</td>
      <td>-1.049</td>
      <td>-0.694</td>
      <td>-0.269</td>
      <td>0.008</td>
      <td>-0.105</td>
      <td>-0.906</td>
      <td>0.772</td>
      <td>0.092</td>
      <td>-1.114</td>
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
      <td>0.309</td>
      <td>0.441</td>
      <td>-0.511</td>
      <td>0.852</td>
      <td>0.601</td>
      <td>1.050</td>
      <td>0.619</td>
      <td>0.956</td>
      <td>0.092</td>
      <td>0.191</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.278</td>
      <td>-0.014</td>
      <td>0.386</td>
      <td>-0.248</td>
      <td>-0.180</td>
      <td>0.022</td>
      <td>-0.227</td>
      <td>-1.802</td>
      <td>0.076</td>
      <td>0.264</td>
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
      <td>0.085</td>
      <td>-0.074</td>
      <td>0.444</td>
      <td>-0.060</td>
      <td>-0.165</td>
      <td>0.194</td>
      <td>0.118</td>
      <td>0.274</td>
      <td>0.132</td>
      <td>0.348</td>
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
      <td>0.234</td>
      <td>-0.377</td>
      <td>0.523</td>
      <td>-0.143</td>
      <td>-0.028</td>
      <td>0.183</td>
      <td>0.046</td>
      <td>0.219</td>
      <td>-0.142</td>
      <td>-1.982</td>
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
      <td>0.228</td>
      <td>-0.267</td>
      <td>0.820</td>
      <td>-0.019</td>
      <td>-0.056</td>
      <td>-0.324</td>
      <td>0.036</td>
      <td>0.282</td>
      <td>-0.095</td>
      <td>0.030</td>
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
<p>994 rows × 100 columns</p>
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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
      <td>994.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.003</td>
      <td>-0.011</td>
      <td>0.031</td>
      <td>-0.002</td>
      <td>0.001</td>
      <td>-0.012</td>
      <td>0.004</td>
      <td>0.017</td>
      <td>0.011</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.941</td>
      <td>0.927</td>
      <td>0.935</td>
      <td>0.933</td>
      <td>0.947</td>
      <td>0.940</td>
      <td>0.947</td>
      <td>0.948</td>
      <td>0.940</td>
      <td>0.947</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.425</td>
      <td>-5.230</td>
      <td>-4.809</td>
      <td>-3.521</td>
      <td>-4.573</td>
      <td>-3.373</td>
      <td>-6.177</td>
      <td>-5.305</td>
      <td>-5.693</td>
      <td>-6.199</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.508</td>
      <td>-0.482</td>
      <td>-0.245</td>
      <td>-0.405</td>
      <td>-0.406</td>
      <td>-0.473</td>
      <td>-0.423</td>
      <td>-0.243</td>
      <td>-0.323</td>
      <td>-0.357</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.028</td>
      <td>-0.074</td>
      <td>0.236</td>
      <td>-0.019</td>
      <td>0.008</td>
      <td>-0.105</td>
      <td>0.036</td>
      <td>0.166</td>
      <td>0.092</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.506</td>
      <td>0.443</td>
      <td>0.605</td>
      <td>0.420</td>
      <td>0.495</td>
      <td>0.913</td>
      <td>0.492</td>
      <td>0.547</td>
      <td>0.481</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.126</td>
      <td>2.400</td>
      <td>2.215</td>
      <td>2.006</td>
      <td>2.611</td>
      <td>1.685</td>
      <td>2.016</td>
      <td>1.752</td>
      <td>2.071</td>
      <td>2.258</td>
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




    ((#50) ['ADEGISFR','AFGYYGPLR','AHGPGLEGGLVGKPAEFTIDTK','AIDDNMSLDEIEK','ALTSEIALLQSR','AQIHDLVLVGGSTR','ARFEELCSDLFR','DSTLIMQLLR','EEASDYLELDTIK','EGHLSPDIVAEQK'...],
     (#50) ['ADEGISFR_na','AFGYYGPLR_na','AHGPGLEGGLVGKPAEFTIDTK_na','AIDDNMSLDEIEK_na','ALTSEIALLQSR_na','AQIHDLVLVGGSTR_na','ARFEELCSDLFR_na','DSTLIMQLLR_na','EEASDYLELDTIK_na','EGHLSPDIVAEQK_na'...])




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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.000</td>
      <td>95.000</td>
      <td>96.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.045</td>
      <td>-0.092</td>
      <td>0.070</td>
      <td>0.010</td>
      <td>-0.026</td>
      <td>0.080</td>
      <td>0.115</td>
      <td>0.088</td>
      <td>-0.005</td>
      <td>-0.042</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.213</td>
      <td>1.050</td>
      <td>1.017</td>
      <td>1.048</td>
      <td>1.025</td>
      <td>0.968</td>
      <td>1.063</td>
      <td>1.016</td>
      <td>0.994</td>
      <td>0.899</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.939</td>
      <td>-3.905</td>
      <td>-3.662</td>
      <td>-3.999</td>
      <td>-3.519</td>
      <td>-2.117</td>
      <td>-5.782</td>
      <td>-3.189</td>
      <td>-3.917</td>
      <td>-2.192</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.670</td>
      <td>-0.563</td>
      <td>-0.552</td>
      <td>-0.439</td>
      <td>-0.637</td>
      <td>-0.509</td>
      <td>-0.411</td>
      <td>-0.261</td>
      <td>-0.414</td>
      <td>-0.585</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.151</td>
      <td>0.001</td>
      <td>0.334</td>
      <td>0.006</td>
      <td>-0.039</td>
      <td>-0.060</td>
      <td>0.155</td>
      <td>0.276</td>
      <td>0.225</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.946</td>
      <td>0.503</td>
      <td>0.744</td>
      <td>0.704</td>
      <td>0.684</td>
      <td>1.059</td>
      <td>0.766</td>
      <td>0.728</td>
      <td>0.549</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.078</td>
      <td>2.062</td>
      <td>1.677</td>
      <td>1.929</td>
      <td>1.982</td>
      <td>1.641</td>
      <td>1.842</td>
      <td>1.652</td>
      <td>2.308</td>
      <td>1.870</td>
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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>THEAQIQEMR_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>VEFMDDTSR_val</th>
      <th>VEPGLGADNSVVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAMSGDPNYLHR_val</th>
      <th>VVSQYSSLLSPMSVNAVMK_val</th>
      <th>YMACCLLYR_val</th>
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
      <td>-0.028</td>
      <td>-0.877</td>
      <td>-0.749</td>
      <td>-1.276</td>
      <td>0.008</td>
      <td>-0.098</td>
      <td>-1.194</td>
      <td>-0.169</td>
      <td>-0.530</td>
      <td>-1.139</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.245</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.625</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.915</td>
      <td>-0.074</td>
      <td>-0.688</td>
      <td>-0.317</td>
      <td>-1.166</td>
      <td>-0.051</td>
      <td>-0.773</td>
      <td>-0.065</td>
      <td>-0.481</td>
      <td>-1.024</td>
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
      <td>0.157</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.239</td>
      <td>-0.528</td>
      <td>-0.126</td>
      <td>-0.467</td>
      <td>-0.461</td>
      <td>-0.278</td>
      <td>-0.865</td>
      <td>-0.331</td>
      <td>0.007</td>
      <td>-0.034</td>
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
      <td>-0.615</td>
      <td>-0.730</td>
      <td>-0.939</td>
      <td>-0.019</td>
      <td>0.008</td>
      <td>0.296</td>
      <td>-0.546</td>
      <td>-0.066</td>
      <td>-0.524</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.819</td>
      <td>-1.049</td>
      <td>-0.694</td>
      <td>-0.269</td>
      <td>0.008</td>
      <td>-0.105</td>
      <td>-0.906</td>
      <td>0.772</td>
      <td>0.092</td>
      <td>-1.114</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.150</td>
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
      <td>0.309</td>
      <td>0.441</td>
      <td>-0.511</td>
      <td>0.852</td>
      <td>0.601</td>
      <td>1.050</td>
      <td>0.619</td>
      <td>0.956</td>
      <td>0.092</td>
      <td>0.191</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.278</td>
      <td>-0.014</td>
      <td>0.386</td>
      <td>-0.248</td>
      <td>-0.180</td>
      <td>0.022</td>
      <td>-0.227</td>
      <td>-1.802</td>
      <td>0.076</td>
      <td>0.264</td>
      <td>...</td>
      <td>-0.098</td>
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
      <td>0.085</td>
      <td>-0.074</td>
      <td>0.444</td>
      <td>-0.060</td>
      <td>-0.165</td>
      <td>0.194</td>
      <td>0.118</td>
      <td>0.274</td>
      <td>0.132</td>
      <td>0.348</td>
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
      <td>0.234</td>
      <td>-0.377</td>
      <td>0.523</td>
      <td>-0.143</td>
      <td>-0.028</td>
      <td>0.183</td>
      <td>0.046</td>
      <td>0.219</td>
      <td>-0.142</td>
      <td>-1.982</td>
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
      <td>0.228</td>
      <td>-0.267</td>
      <td>0.820</td>
      <td>-0.019</td>
      <td>-0.056</td>
      <td>-0.324</td>
      <td>0.036</td>
      <td>0.282</td>
      <td>-0.095</td>
      <td>0.030</td>
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
<p>994 rows × 150 columns</p>
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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>THEAQIQEMR_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>VEFMDDTSR_val</th>
      <th>VEPGLGADNSVVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAMSGDPNYLHR_val</th>
      <th>VVSQYSSLLSPMSVNAVMK_val</th>
      <th>YMACCLLYR_val</th>
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
      <td>-0.028</td>
      <td>-0.877</td>
      <td>-0.749</td>
      <td>-1.276</td>
      <td>0.008</td>
      <td>-0.098</td>
      <td>-1.194</td>
      <td>-0.169</td>
      <td>-0.530</td>
      <td>-1.139</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.245</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.625</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.915</td>
      <td>-0.074</td>
      <td>-0.688</td>
      <td>-0.317</td>
      <td>-1.166</td>
      <td>-0.051</td>
      <td>-0.773</td>
      <td>-0.065</td>
      <td>-0.481</td>
      <td>-1.024</td>
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
      <td>0.157</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.239</td>
      <td>-0.528</td>
      <td>-0.126</td>
      <td>-0.467</td>
      <td>-0.461</td>
      <td>-0.278</td>
      <td>-0.865</td>
      <td>-0.331</td>
      <td>0.007</td>
      <td>-0.034</td>
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
      <td>-0.615</td>
      <td>-0.730</td>
      <td>-0.939</td>
      <td>-0.019</td>
      <td>0.008</td>
      <td>0.296</td>
      <td>-0.546</td>
      <td>-0.066</td>
      <td>-0.524</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.819</td>
      <td>-1.049</td>
      <td>-0.694</td>
      <td>-0.269</td>
      <td>0.008</td>
      <td>-0.105</td>
      <td>-0.906</td>
      <td>0.772</td>
      <td>0.092</td>
      <td>-1.114</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.150</td>
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
      <td>0.309</td>
      <td>0.441</td>
      <td>-0.511</td>
      <td>0.852</td>
      <td>0.601</td>
      <td>1.050</td>
      <td>0.619</td>
      <td>0.956</td>
      <td>0.092</td>
      <td>0.191</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.278</td>
      <td>-0.014</td>
      <td>0.386</td>
      <td>-0.248</td>
      <td>-0.180</td>
      <td>0.022</td>
      <td>-0.227</td>
      <td>-1.802</td>
      <td>0.076</td>
      <td>0.264</td>
      <td>...</td>
      <td>-0.098</td>
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
      <td>0.085</td>
      <td>-0.074</td>
      <td>0.444</td>
      <td>-0.060</td>
      <td>-0.165</td>
      <td>0.194</td>
      <td>0.118</td>
      <td>0.274</td>
      <td>0.132</td>
      <td>0.348</td>
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
      <td>0.234</td>
      <td>-0.377</td>
      <td>0.523</td>
      <td>-0.143</td>
      <td>-0.028</td>
      <td>0.183</td>
      <td>0.046</td>
      <td>0.219</td>
      <td>-0.142</td>
      <td>-1.982</td>
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
      <td>0.228</td>
      <td>-0.267</td>
      <td>0.820</td>
      <td>-0.019</td>
      <td>-0.056</td>
      <td>-0.324</td>
      <td>0.036</td>
      <td>0.282</td>
      <td>-0.095</td>
      <td>0.030</td>
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
<p>994 rows × 150 columns</p>
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
      <th>ADEGISFR_val</th>
      <th>AFGYYGPLR_val</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK_val</th>
      <th>AIDDNMSLDEIEK_val</th>
      <th>ALTSEIALLQSR_val</th>
      <th>AQIHDLVLVGGSTR_val</th>
      <th>ARFEELCSDLFR_val</th>
      <th>DSTLIMQLLR_val</th>
      <th>EEASDYLELDTIK_val</th>
      <th>EGHLSPDIVAEQK_val</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>THEAQIQEMR_val</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_val</th>
      <th>TVAGGAWTYNTTSAVTVK_val</th>
      <th>VEFMDDTSR_val</th>
      <th>VEPGLGADNSVVR_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VLAMSGDPNYLHR_val</th>
      <th>VVSQYSSLLSPMSVNAVMK_val</th>
      <th>YMACCLLYR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98.000</td>
      <td>95.000</td>
      <td>96.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>98.000</td>
      <td>100.000</td>
      <td>...</td>
      <td>96.000</td>
      <td>99.000</td>
      <td>96.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>97.000</td>
      <td>99.000</td>
      <td>97.000</td>
      <td>100.000</td>
      <td>96.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.045</td>
      <td>-0.092</td>
      <td>0.070</td>
      <td>0.010</td>
      <td>-0.026</td>
      <td>0.080</td>
      <td>0.115</td>
      <td>0.088</td>
      <td>-0.005</td>
      <td>-0.042</td>
      <td>...</td>
      <td>0.037</td>
      <td>-0.045</td>
      <td>0.026</td>
      <td>-0.110</td>
      <td>0.159</td>
      <td>-0.183</td>
      <td>0.007</td>
      <td>-0.102</td>
      <td>-0.122</td>
      <td>0.040</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.213</td>
      <td>1.050</td>
      <td>1.017</td>
      <td>1.048</td>
      <td>1.025</td>
      <td>0.968</td>
      <td>1.063</td>
      <td>1.016</td>
      <td>0.994</td>
      <td>0.899</td>
      <td>...</td>
      <td>0.892</td>
      <td>1.002</td>
      <td>0.920</td>
      <td>0.822</td>
      <td>0.900</td>
      <td>1.072</td>
      <td>1.074</td>
      <td>1.065</td>
      <td>1.066</td>
      <td>0.832</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.939</td>
      <td>-3.905</td>
      <td>-3.662</td>
      <td>-3.999</td>
      <td>-3.519</td>
      <td>-2.117</td>
      <td>-5.782</td>
      <td>-3.189</td>
      <td>-3.917</td>
      <td>-2.192</td>
      <td>...</td>
      <td>-2.741</td>
      <td>-3.685</td>
      <td>-2.439</td>
      <td>-2.349</td>
      <td>-1.650</td>
      <td>-4.326</td>
      <td>-2.764</td>
      <td>-3.292</td>
      <td>-3.746</td>
      <td>-2.468</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.670</td>
      <td>-0.563</td>
      <td>-0.552</td>
      <td>-0.439</td>
      <td>-0.637</td>
      <td>-0.509</td>
      <td>-0.411</td>
      <td>-0.261</td>
      <td>-0.414</td>
      <td>-0.585</td>
      <td>...</td>
      <td>-0.379</td>
      <td>-0.425</td>
      <td>-0.473</td>
      <td>-0.577</td>
      <td>-0.515</td>
      <td>-0.608</td>
      <td>-0.693</td>
      <td>-0.570</td>
      <td>-0.717</td>
      <td>-0.314</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.151</td>
      <td>0.001</td>
      <td>0.334</td>
      <td>0.006</td>
      <td>-0.039</td>
      <td>-0.060</td>
      <td>0.155</td>
      <td>0.276</td>
      <td>0.225</td>
      <td>0.039</td>
      <td>...</td>
      <td>0.301</td>
      <td>0.139</td>
      <td>0.219</td>
      <td>-0.189</td>
      <td>-0.136</td>
      <td>0.009</td>
      <td>-0.121</td>
      <td>0.033</td>
      <td>0.189</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.946</td>
      <td>0.503</td>
      <td>0.744</td>
      <td>0.704</td>
      <td>0.684</td>
      <td>1.059</td>
      <td>0.766</td>
      <td>0.728</td>
      <td>0.549</td>
      <td>0.508</td>
      <td>...</td>
      <td>0.665</td>
      <td>0.670</td>
      <td>0.578</td>
      <td>0.368</td>
      <td>1.163</td>
      <td>0.483</td>
      <td>0.821</td>
      <td>0.598</td>
      <td>0.627</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.078</td>
      <td>2.062</td>
      <td>1.677</td>
      <td>1.929</td>
      <td>1.982</td>
      <td>1.641</td>
      <td>1.842</td>
      <td>1.652</td>
      <td>2.308</td>
      <td>1.870</td>
      <td>...</td>
      <td>1.216</td>
      <td>1.539</td>
      <td>1.767</td>
      <td>2.014</td>
      <td>1.780</td>
      <td>1.726</td>
      <td>2.121</td>
      <td>1.673</td>
      <td>1.435</td>
      <td>1.984</td>
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
      <th>ADEGISFR_na</th>
      <th>AFGYYGPLR_na</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK_na</th>
      <th>AIDDNMSLDEIEK_na</th>
      <th>ALTSEIALLQSR_na</th>
      <th>AQIHDLVLVGGSTR_na</th>
      <th>ARFEELCSDLFR_na</th>
      <th>DSTLIMQLLR_na</th>
      <th>EEASDYLELDTIK_na</th>
      <th>EGHLSPDIVAEQK_na</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>THEAQIQEMR_na</th>
      <th>TLTAVHDAILEDLVFPSEIVGK_na</th>
      <th>TVAGGAWTYNTTSAVTVK_na</th>
      <th>VEFMDDTSR_na</th>
      <th>VEPGLGADNSVVR_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VLAMSGDPNYLHR_na</th>
      <th>VVSQYSSLLSPMSVNAVMK_na</th>
      <th>YMACCLLYR_na</th>
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
      <td>False</td>
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
<p>994 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x00000297F9587040>
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




    
![png](latent_2D_100_10_files/latent_2D_100_10_108_2.png)
    


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
      <td>0.945455</td>
      <td>0.766780</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.644612</td>
      <td>0.422244</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.496172</td>
      <td>0.411279</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.421464</td>
      <td>0.383860</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.381563</td>
      <td>0.378475</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.358796</td>
      <td>0.375551</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.340705</td>
      <td>0.362119</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.329361</td>
      <td>0.361454</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.321927</td>
      <td>0.363134</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.316460</td>
      <td>0.361538</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4909, 4909)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3605)




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
      <th>ADEGISFR</th>
      <td>30.461</td>
      <td>31.743</td>
      <td>31.775</td>
      <td>31.118</td>
      <td>31.234</td>
      <td>31.018</td>
    </tr>
    <tr>
      <th>ALTSEIALLQSR</th>
      <td>27.244</td>
      <td>28.492</td>
      <td>28.482</td>
      <td>27.467</td>
      <td>27.818</td>
      <td>27.547</td>
    </tr>
    <tr>
      <th>EGNDLYHEMIESGVINLK</th>
      <td>27.597</td>
      <td>28.637</td>
      <td>28.841</td>
      <td>27.366</td>
      <td>27.543</td>
      <td>27.818</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
      <td>30.036</td>
      <td>29.682</td>
      <td>29.769</td>
      <td>28.650</td>
      <td>28.664</td>
    </tr>
    <tr>
      <th>VLAMSGDPNYLHR</th>
      <td>27.667</td>
      <td>28.729</td>
      <td>28.549</td>
      <td>27.706</td>
      <td>27.530</td>
      <td>27.275</td>
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
      <th>FVINYDYPNSSEDYIHR</th>
      <td>28.390</td>
      <td>28.837</td>
      <td>28.853</td>
      <td>28.490</td>
      <td>28.621</td>
      <td>28.605</td>
    </tr>
    <tr>
      <th>HIYYITGETK</th>
      <td>30.932</td>
      <td>30.755</td>
      <td>30.692</td>
      <td>30.868</td>
      <td>30.391</td>
      <td>30.358</td>
    </tr>
    <tr>
      <th>IHVSDQELQSANASVDDSRLEELK</th>
      <td>28.929</td>
      <td>28.922</td>
      <td>29.252</td>
      <td>29.313</td>
      <td>28.729</td>
      <td>28.719</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
      <td>30.012</td>
      <td>29.913</td>
      <td>30.597</td>
      <td>29.905</td>
      <td>30.004</td>
    </tr>
    <tr>
      <th>NMMAACDPR</th>
      <td>32.531</td>
      <td>32.091</td>
      <td>32.015</td>
      <td>32.285</td>
      <td>31.843</td>
      <td>31.923</td>
    </tr>
  </tbody>
</table>
<p>4909 rows × 6 columns</p>
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
      <td>0.033</td>
      <td>-0.076</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.029</td>
      <td>-0.049</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.733</td>
      <td>-0.766</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.594</td>
      <td>-0.636</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.241</td>
      <td>-0.118</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_122_1.png)
    



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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_123_1.png)
    


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
      <th>ADEGISFR</th>
      <th>AFGYYGPLR</th>
      <th>AHGPGLEGGLVGKPAEFTIDTK</th>
      <th>AIDDNMSLDEIEK</th>
      <th>ALTSEIALLQSR</th>
      <th>AQIHDLVLVGGSTR</th>
      <th>ARFEELCSDLFR</th>
      <th>DSTLIMQLLR</th>
      <th>EEASDYLELDTIK</th>
      <th>EGHLSPDIVAEQK</th>
      <th>...</th>
      <th>TFVNITPAEVGVLVGK</th>
      <th>THEAQIQEMR</th>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <th>TVAGGAWTYNTTSAVTVK</th>
      <th>VEFMDDTSR</th>
      <th>VEPGLGADNSVVR</th>
      <th>VFITDDFHDMMPK</th>
      <th>VLAMSGDPNYLHR</th>
      <th>VVSQYSSLLSPMSVNAVMK</th>
      <th>YMACCLLYR</th>
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
      <td>0.617</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.498</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.715</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.630</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>0.571</td>
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
      <td>0.671</td>
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
      <td>0.653</td>
      <td>0.524</td>
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
      <td>0.520</td>
      <td>0.620</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.679</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.732</td>
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
    
    Optimizer used: <function Adam at 0x00000297F9587040>
    Loss function: <function loss_fct_vae at 0x00000297F95A5940>
    
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




    
![png](latent_2D_100_10_files/latent_2D_100_10_136_2.png)
    



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
      <td>1994.865479</td>
      <td>218.006882</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1959.040894</td>
      <td>213.487366</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1894.172363</td>
      <td>215.847855</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1840.316040</td>
      <td>209.491745</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1804.649292</td>
      <td>200.816116</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1779.109497</td>
      <td>202.638031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1762.020020</td>
      <td>199.662079</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1749.379639</td>
      <td>197.001617</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1739.365479</td>
      <td>197.190643</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1732.340454</td>
      <td>197.389496</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4909)




```python
len(pred[0])
```




    4909




```python
learn.loss_func(pred, target)
```




    tensor(3110.6082)




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
      <th>ADEGISFR</th>
      <td>30.461</td>
      <td>31.743</td>
      <td>31.775</td>
      <td>31.118</td>
      <td>31.234</td>
      <td>31.018</td>
      <td>31.958</td>
    </tr>
    <tr>
      <th>ALTSEIALLQSR</th>
      <td>27.244</td>
      <td>28.492</td>
      <td>28.482</td>
      <td>27.467</td>
      <td>27.818</td>
      <td>27.547</td>
      <td>28.586</td>
    </tr>
    <tr>
      <th>EGNDLYHEMIESGVINLK</th>
      <td>27.597</td>
      <td>28.637</td>
      <td>28.841</td>
      <td>27.366</td>
      <td>27.543</td>
      <td>27.818</td>
      <td>28.857</td>
    </tr>
    <tr>
      <th>TLTAVHDAILEDLVFPSEIVGK</th>
      <td>30.168</td>
      <td>30.036</td>
      <td>29.682</td>
      <td>29.769</td>
      <td>28.650</td>
      <td>28.664</td>
      <td>29.708</td>
    </tr>
    <tr>
      <th>VLAMSGDPNYLHR</th>
      <td>27.667</td>
      <td>28.729</td>
      <td>28.549</td>
      <td>27.706</td>
      <td>27.530</td>
      <td>27.275</td>
      <td>28.674</td>
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
      <th>FVINYDYPNSSEDYIHR</th>
      <td>28.390</td>
      <td>28.837</td>
      <td>28.853</td>
      <td>28.490</td>
      <td>28.621</td>
      <td>28.605</td>
      <td>28.710</td>
    </tr>
    <tr>
      <th>HIYYITGETK</th>
      <td>30.932</td>
      <td>30.755</td>
      <td>30.692</td>
      <td>30.868</td>
      <td>30.391</td>
      <td>30.358</td>
      <td>30.423</td>
    </tr>
    <tr>
      <th>IHVSDQELQSANASVDDSRLEELK</th>
      <td>28.929</td>
      <td>28.922</td>
      <td>29.252</td>
      <td>29.313</td>
      <td>28.729</td>
      <td>28.719</td>
      <td>29.014</td>
    </tr>
    <tr>
      <th>KIEPELDGSAQVTSHDASTNGLINFIK</th>
      <td>30.358</td>
      <td>30.012</td>
      <td>29.913</td>
      <td>30.597</td>
      <td>29.905</td>
      <td>30.004</td>
      <td>29.703</td>
    </tr>
    <tr>
      <th>NMMAACDPR</th>
      <td>32.531</td>
      <td>32.091</td>
      <td>32.015</td>
      <td>32.285</td>
      <td>31.843</td>
      <td>31.923</td>
      <td>31.940</td>
    </tr>
  </tbody>
</table>
<p>4909 rows × 7 columns</p>
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
      <td>-0.071</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.040</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.024</td>
      <td>0.088</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.070</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.047</td>
      <td>0.167</td>
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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_146_1.png)
    



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
    


    
![png](latent_2D_100_10_files/latent_2D_100_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'ALTSEIALLQSR'), ('20181227_QE6_nLC6_CSC_QC_HeLa_02', 'SGDSEVYQLGDVSQK'), ('20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007', 'ELAEDGYSGVEVR'), ('20190115_QE2_NLC10_TW_QC_MNT_HeLa_01', 'ITLQDVVSHSK'), ('20190115_QE5_nLC5_RJC_MNT_HeLa_01', 'YMACCLLYR'), ('20190129_QE1_nLC2_GP_QC_MNT_HELA_01', 'EQIVPKPEEEVAQK'), ('20190131_QE10_nLC0_NHS_MNT_HELA_50cm_02', 'FAQPGSFEYEYAMR'), ('20190201_QE10_nLC0_NHS_MNT_HELA_45cm_01', 'FLSQPFQVAEVFTGHMGK'), ('20190203_QE3_nLC3_KBE_QC_MNT_HeLa_01', 'GYISPYFINTSK'), ('20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04', 'TAFQEALDAAGDK'), ('20190207_QE2_NLC10_GP_MNT_HeLa_01', 'AFGYYGPLR'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_03', 'DSTLIMQLLR'), ('20190219_QE3_Evo2_UHG_QC_MNT_HELA_01_190219173213', 'MELQEIQLK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_5', 'AIDDNMSLDEIEK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_27', 'VVSQYSSLLSPMSVNAVMK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_23', 'VVSQYSSLLSPMSVNAVMK'), ('20190301_QE1_nLC2_ANHO_QC_MNT_HELA_01_20190303025443', 'TFVNITPAEVGVLVGK'), ('20190305_QE8_nLC14_ASD_QC_MNT_50cm_HELA_02', 'AIDDNMSLDEIEK'), ('20190305_QE8_nLC14_ASD_QC_MNT_50cm_HELA_02', 'ILQDGGLQVVEK'), ('20190305_QE8_nLC14_RG_QC_MNT_50cm_HELA_01', 'AIDDNMSLDEIEK'), ('20190408_QE4_LC12_IAH_QC_MNT_HeLa_02', 'VVSQYSSLLSPMSVNAVMK'), ('20190417_QX4_JoSw_MA_HeLa_500ng_BR14_new', 'YMACCLLYR'), ('20190418_QX8_JuSc_MA_HeLa_500ng_1', 'AFGYYGPLR'), ('20190422_QE4_LC12_JE-IAH_QC_MNT_HeLa_02', 'IAGYVTHLMK'), ('20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01_20190425184929', 'IVEVLLMK'), ('20190507_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_01', 'AHGPGLEGGLVGKPAEFTIDTK'), ('20190507_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_01', 'TFVNITPAEVGVLVGK'), ('20190509_QE8_nLC14_AGF_QC_MNT_HeLa_01', 'EEASDYLELDTIK'), ('20190513_QE6_LC4_IAH_QC_MNT_HeLa_03', 'EGHLSPDIVAEQK'), ('20190515_QE2_NLC1_GP_MNT_HELA_01', 'MELQEIQLK'), ('20190515_QE4_LC12_AS_QC_MNT_HeLa_01', 'AHGPGLEGGLVGKPAEFTIDTK'), ('20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_02', 'EEASDYLELDTIK'), ('20190531_QE2_NLC1_GP_QC_MNT_HELA_01', 'DSTLIMQLLR'), ('20190604_QX8_MiWi_MA_HeLa_BR14_500ng', 'LDPHLVLDQLR'), ('20190605_QX0_MePh_MA_HeLa_500ng_LC07_1_BR14', 'AQIHDLVLVGGSTR'), ('20190605_QX0_MePh_MA_HeLa_500ng_LC07_1_BR14', 'LDPHLVLDQLR'), ('20190605_QX3_ChDe_MA_Hela_500ng_LC15', 'EEASDYLELDTIK'), ('20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625163359', 'GLVLGPIHK'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_5', 'FLSQPFQVAEVFTGHMGK'), ('20190611_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'EIIDLVLDR'), ('20190611_QX3_LiSc_MA_Hela_500ng_LC15', 'EIIDLVLDR'), ('20190611_QX4_JiYu_MA_HeLa_500ng', 'EIIDLVLDR'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02', 'NMMAACDPR'), ('20190615_QX4_JiYu_MA_HeLa_500ng', 'VLAMSGDPNYLHR'), ('20190617_QE_LC_UHG_QC_MNT_HELA_04', 'IAGYVTHLMK'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'LDPHLVLDQLR'), ('20190625_QE9_nLC0_RG_MNT_Hela_MUC_50cm_2', 'EGHLSPDIVAEQK'), ('20190626_QX7_IgPa_MA_HeLa_Br14_500ng', 'TLTAVHDAILEDLVFPSEIVGK'), ('20190628_QE2_NLC1_TL_QC_MNT_HELA_05', 'KIEPELDGSAQVTSHDASTNGLINFIK'), ('20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03', 'GLVLGPIHK'), ('20190708_QX7_MaMu_MA_HeLa_Br14_500ng', 'EGHLSPDIVAEQK'), ('20190708_QX8_AnPi_MA_HeLa_BR14_500ng', 'ILQDGGLQVVEK'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05', 'SGDSEVYQLGDVSQK'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05_190709143552', 'SGDSEVYQLGDVSQK'), ('20190709_QX6_MaTa_MA_HeLa_500ng_LC09_20190709155356', 'ARFEELCSDLFR'), ('20190722_QE8_nLC0_BDA_QC_HeLa_50cm_02', 'EGNDLYHEMIESGVINLK'), ('20190725_QE9_nLC9_RG_QC_MNT_HeLa_MUC_50cm_2', 'HIYYITGETK'), ('20190725_QX0_MePh_MA_HeLa_500ng_LC07_01', 'TFVNITPAEVGVLVGK'), ('20190731_QX8_ChSc_MA_HeLa_500ng', 'EIIDLVLDR'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'FVINYDYPNSSEDYIHR')]
    




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
      <th>intensity_pred_vae</th>
      <th>replicates</th>
      <th>train_average</th>
      <th>train_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSE</th>
      <td>0.649</td>
      <td>0.683</td>
      <td>1.691</td>
      <td>1.740</td>
      <td>2.031</td>
      <td>2.095</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.483</td>
      <td>0.502</td>
      <td>0.949</td>
      <td>0.897</td>
      <td>1.056</td>
      <td>1.040</td>
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
