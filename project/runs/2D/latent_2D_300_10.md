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
    LYSVSYLLK                   961
    SDIGEVILVGGMTR              994
    GASQAGMTGYGMPR              964
    ALDVMVSTFHK                 999
    QITVNDLPVGR                 999
    KYEQGFITDPVVLSPK            998
    IPEISIQDMTAQVTSPSGK         994
    ETTDTDTADQVIASFK            980
    IIAPPERK                    940
    TIGGGDDSFTTFFCETGAGK        995
    HLNEIDLFHCIDPNDSK           997
    EILVGDVGQTVDDPYATFVK        990
    VVVQVLAEEPEAVLK             996
    EALLSSAVDHGSDEVK            968
    YHTSQSGDEMTSLSEYVSR       1,000
    TPAQYDASELK                 982
    YAPSEAGLHEMDIR              987
    DSNNLCLHFNPR                996
    HQPTAIIAK                   980
    YLMEEDEDAYKK                985
    IRYESLTDPSK                 998
    ISMPDFDLHLK                 999
    TIGTGLVTNTLAMTEEEK          989
    HQEGEIFDTEK                 998
    VFDAIMNFK                   990
    PMFIVNTNVPR               1,000
    VFITDDFHDMMPK               994
    IIYGGSVTGATCK               977
    ALIAAQYSGAQVR               986
    DDEFTHLYTLIVRPDNTYEVK       954
    GAGTGGLGLAVEGPSEAK          944
    VIVVGNPANTNCLTASK           966
    GQAAVQQLQAEGLSPR            995
    GCITIIGGGDTATCCAK           998
    IISNASCTTNCLAPLAK         1,000
    EALTYDGALLGDR               980
    FDDAVVQSDMK                 998
    LLCGLLAER                   973
    ILDSVGIEADDDRLNK          1,000
    IVSRPEELREDDVGTGAGLLEIK     997
    LVIITAGAR                   998
    ETNLDSLPLVDTHSK             975
    SYELPDGQVITIGNER          1,000
    KYEDICPSTHNMDVPNIK        1,000
    DFTVSAMHGDMDQK              997
    FNADEFEDMVAEK               991
    FDTGNLCMVTGGANLGR           999
    LSPPYSSPQEFAQDVGR           995
    HEQILVLDPPTDLK              981
    ELISNSSDALDK                983
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
      <th>LYSVSYLLK</th>
      <td>28.338</td>
    </tr>
    <tr>
      <th>SDIGEVILVGGMTR</th>
      <td>28.157</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>28.539</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>29.828</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>DFTVSAMHGDMDQK</th>
      <td>30.706</td>
    </tr>
    <tr>
      <th>FNADEFEDMVAEK</th>
      <td>30.555</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>30.138</td>
    </tr>
    <tr>
      <th>LSPPYSSPQEFAQDVGR</th>
      <td>26.405</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
    </tr>
  </tbody>
</table>
<p>49360 rows × 1 columns</p>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_24_1.png)
    



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
      <td>0.987</td>
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
      <th>LYSVSYLLK</th>
      <td>28.338</td>
    </tr>
    <tr>
      <th>SDIGEVILVGGMTR</th>
      <td>28.157</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>28.539</td>
    </tr>
    <tr>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>29.828</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>DFTVSAMHGDMDQK</th>
      <td>30.706</td>
    </tr>
    <tr>
      <th>FNADEFEDMVAEK</th>
      <td>30.555</td>
    </tr>
    <tr>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>30.138</td>
    </tr>
    <tr>
      <th>LSPPYSSPQEFAQDVGR</th>
      <td>26.405</td>
    </tr>
    <tr>
      <th>HEQILVLDPPTDLK</th>
      <td>28.024</td>
    </tr>
  </tbody>
</table>
<p>49360 rows × 1 columns</p>
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
      <th>20190411_QE6_LC6_AS_QC_MNT_HeLa_02</th>
      <th>ALDVMVSTFHK</th>
      <td>32.214</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>ALDVMVSTFHK</th>
      <td>32.347</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_02</th>
      <th>ALDVMVSTFHK</th>
      <td>29.249</td>
    </tr>
    <tr>
      <th>20190528_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>ALDVMVSTFHK</th>
      <td>29.901</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>ALDVMVSTFHK</th>
      <td>32.522</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190318_QE2_NLC1_AB_MNT_HELA_06</th>
      <th>YLMEEDEDAYKK</th>
      <td>30.146</td>
    </tr>
    <tr>
      <th>20190802_QX7_AlRe_MA_HeLa_Br14_500ng</th>
      <th>YLMEEDEDAYKK</th>
      <td>29.587</td>
    </tr>
    <tr>
      <th>20190524_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>YLMEEDEDAYKK</th>
      <td>31.313</td>
    </tr>
    <tr>
      <th>20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03</th>
      <th>YLMEEDEDAYKK</th>
      <td>29.685</td>
    </tr>
    <tr>
      <th>20190510_QE2_NLC1_GP_MNT_HELA_01_20190510174108</th>
      <th>YLMEEDEDAYKK</th>
      <td>28.633</td>
    </tr>
  </tbody>
</table>
<p>44424 rows × 1 columns</p>
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
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.015</td>
      <td>31.827</td>
      <td>31.491</td>
    </tr>
    <tr>
      <th>DSNNLCLHFNPR</th>
      <td>30.046</td>
      <td>30.842</td>
      <td>30.635</td>
      <td>30.399</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALIAAQYSGAQVR</th>
      <td>29.496</td>
      <td>30.535</td>
      <td>30.473</td>
      <td>29.931</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>29.006</td>
      <td>28.341</td>
      <td>28.219</td>
      <td>28.219</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.734</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>30.011</td>
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
      <th>GQAAVQQLQAEGLSPR</th>
      <td>26.757</td>
      <td>27.069</td>
      <td>27.457</td>
      <td>26.385</td>
    </tr>
    <tr>
      <th>ILDSVGIEADDDRLNK</th>
      <td>31.576</td>
      <td>31.573</td>
      <td>31.581</td>
      <td>31.516</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>32.130</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>32.349</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>32.338</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>31.604</td>
    </tr>
    <tr>
      <th>YAPSEAGLHEMDIR</th>
      <td>29.643</td>
      <td>29.426</td>
      <td>29.014</td>
      <td>29.642</td>
    </tr>
  </tbody>
</table>
<p>4936 rows × 4 columns</p>
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
      <th>DSNNLCLHFNPR</th>
      <td>30.136</td>
      <td>30.842</td>
      <td>30.635</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181230_QE6_nLC6_CSC_QC_HeLa_03</th>
      <th>HQEGEIFDTEK</th>
      <td>29.947</td>
      <td>29.886</td>
      <td>30.106</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190114_QE7_nLC7_AL_QC_MNT_HeLa_01</th>
      <th>IPEISIQDMTAQVTSPSGK</th>
      <td>30.848</td>
      <td>29.329</td>
      <td>29.045</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <td>33.075</td>
      <td>32.504</td>
      <td>32.252</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190128_QE3_nLC3_MJ_MNT_HeLa_02</th>
      <th>IRYESLTDPSK</th>
      <td>31.549</td>
      <td>31.829</td>
      <td>31.892</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_01</th>
      <th>FDTGNLCMVTGGANLGR</th>
      <td>29.916</td>
      <td>30.130</td>
      <td>30.074</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>ISMPDFDLHLK</th>
      <td>30.005</td>
      <td>29.897</td>
      <td>29.719</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190214_QE4_LC12_SCL_QC_MNT_HeLa_01</th>
      <th>KYEDICPSTHNMDVPNIK</th>
      <td>32.461</td>
      <td>32.340</td>
      <td>32.274</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190214_QE4_LC12_SCL_QC_MNT_HeLa_02</th>
      <th>KYEDICPSTHNMDVPNIK</th>
      <td>32.541</td>
      <td>32.340</td>
      <td>32.274</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_23</th>
      <th>HLNEIDLFHCIDPNDSK</th>
      <td>30.262</td>
      <td>30.423</td>
      <td>30.430</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190301_QE7_nLC7_DS_QC_MNT_HeLa_01</th>
      <th>GAGTGGLGLAVEGPSEAK</th>
      <td>30.458</td>
      <td>29.238</td>
      <td>29.081</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190301_QE7_nLC7_DS_QC_MNT_HeLa_01_20190301161023</th>
      <th>GAGTGGLGLAVEGPSEAK</th>
      <td>30.458</td>
      <td>29.238</td>
      <td>29.081</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03</th>
      <th>IVSRPEELREDDVGTGAGLLEIK</th>
      <td>28.002</td>
      <td>29.926</td>
      <td>30.017</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190318_QE2_NLC1_AB_MNT_HELA_02</th>
      <th>ILDSVGIEADDDRLNK</th>
      <td>32.127</td>
      <td>31.573</td>
      <td>31.581</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190318_QE2_NLC1_AB_MNT_HELA_03</th>
      <th>ALDVMVSTFHK</th>
      <td>32.028</td>
      <td>32.015</td>
      <td>31.827</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190404_QE7_nLC3_AL_QC_MNT_HeLa_02</th>
      <th>TIGGGDDSFTTFFCETGAGK</th>
      <td>27.522</td>
      <td>28.590</td>
      <td>28.419</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190405_QE1_nLC2_GP_MNT_QC_hela_01</th>
      <th>TIGGGDDSFTTFFCETGAGK</th>
      <td>27.839</td>
      <td>28.590</td>
      <td>28.419</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190502_QE8_nLC14_AGF_QC_MNT_HeLa_01</th>
      <th>LVIITAGAR</th>
      <td>31.652</td>
      <td>31.780</td>
      <td>31.729</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190506_QX8_MiWi_MA_HeLa_500ng_new</th>
      <th>GASQAGMTGYGMPR</th>
      <td>24.619</td>
      <td>28.341</td>
      <td>28.219</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190510_QE2_NLC1_GP_MNT_HELA_02</th>
      <th>HQEGEIFDTEK</th>
      <td>29.596</td>
      <td>29.886</td>
      <td>30.106</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QE6_LC4_IAH_QC_MNT_HeLa_03</th>
      <th>VFDAIMNFK</th>
      <td>28.999</td>
      <td>29.261</td>
      <td>29.861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190513_QE7_nLC7_MEM_QC_MNT_HeLa_03</th>
      <th>VFITDDFHDMMPK</th>
      <td>27.601</td>
      <td>28.707</td>
      <td>28.898</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QX4_JiYu_MA_HeLa_500ng_BR14</th>
      <th>SDIGEVILVGGMTR</th>
      <td>29.887</td>
      <td>28.538</td>
      <td>28.795</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190522_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14</th>
      <th>IISNASCTTNCLAPLAK</th>
      <td>34.071</td>
      <td>33.165</td>
      <td>33.003</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190524_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>FNADEFEDMVAEK</th>
      <td>29.574</td>
      <td>29.823</td>
      <td>29.692</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190530_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>LVIITAGAR</th>
      <td>33.284</td>
      <td>31.780</td>
      <td>31.729</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_01</th>
      <th>GAGTGGLGLAVEGPSEAK</th>
      <td>28.111</td>
      <td>29.238</td>
      <td>29.081</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625212127</th>
      <th>LYSVSYLLK</th>
      <td>26.889</td>
      <td>29.205</td>
      <td>28.697</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02</th>
      <th>ELISNSSDALDK</th>
      <td>33.041</td>
      <td>30.339</td>
      <td>30.678</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190618_QX3_LiSc_MA_Hela_500ng_LC15</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>32.012</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190619_QE1_nLC2_GP_QC_MNT_HELA_01</th>
      <th>HQPTAIIAK</th>
      <td>32.182</td>
      <td>32.232</td>
      <td>32.042</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190620_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>TPAQYDASELK</th>
      <td>32.309</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190620_QX1_JoMu_MA_HeLa__500ng_LC10</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>32.205</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_01</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>27.885</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QX4_JoMu_MA_HeLa_500ng</th>
      <th>LLCGLLAER</th>
      <td>24.183</td>
      <td>31.854</td>
      <td>31.710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190624_QE4_nLC12_MM_QC_MNT_HELA_01_20190626192509</th>
      <th>KYEQGFITDPVVLSPK</th>
      <td>28.608</td>
      <td>28.768</td>
      <td>28.783</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_02</th>
      <th>HEQILVLDPPTDLK</th>
      <td>28.437</td>
      <td>27.776</td>
      <td>27.687</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190626_QX6_ChDe_MA_HeLa_500ng_LC09</th>
      <th>VIVVGNPANTNCLTASK</th>
      <td>30.464</td>
      <td>29.301</td>
      <td>29.142</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190627_QX3_MaMu_MA_Hela_500ng_LC15</th>
      <th>LLCGLLAER</th>
      <td>31.269</td>
      <td>31.854</td>
      <td>31.710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190628_QE1_nLC13_ANHO_QC_MNT_HELA_03</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.155</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190630_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1</th>
      <th>IIYGGSVTGATCK</th>
      <td>30.723</td>
      <td>31.425</td>
      <td>31.319</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <th>DFTVSAMHGDMDQK</th>
      <td>31.081</td>
      <td>30.338</td>
      <td>30.296</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QX2_LiSc_MA_HeLa_500ng_LC05</th>
      <th>IIAPPERK</th>
      <td>32.429</td>
      <td>33.599</td>
      <td>33.304</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190709_QE1_nLC13_ANHO_QC_MNT_HELA_01</th>
      <th>ALIAAQYSGAQVR</th>
      <td>29.847</td>
      <td>30.535</td>
      <td>30.473</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190712_QE8_nLC14_AnMu_QC_MNT_50cm_Hela_01</th>
      <th>VFITDDFHDMMPK</th>
      <td>28.884</td>
      <td>28.707</td>
      <td>28.898</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190712_QE9_nLC9_NHS_MNT_HELA_50cm_MUC_01</th>
      <th>VFITDDFHDMMPK</th>
      <td>28.587</td>
      <td>28.707</td>
      <td>28.898</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190715_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <th>FDDAVVQSDMK</th>
      <td>29.279</td>
      <td>29.752</td>
      <td>29.810</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190715_QE8_nLC14_RG_QC_MNT_50cm_Hela_02</th>
      <th>EALLSSAVDHGSDEVK</th>
      <td>30.462</td>
      <td>29.758</td>
      <td>29.845</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190716_QX6_MaTa_MA_HeLa_500ng_LC09</th>
      <th>HQPTAIIAK</th>
      <td>32.626</td>
      <td>32.232</td>
      <td>32.042</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190717_QX3_OzKa_MA_Hela_500ng_LC15_190720214645</th>
      <th>EALTYDGALLGDR</th>
      <td>30.686</td>
      <td>29.077</td>
      <td>29.071</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190719_QE2_NLC1_ANHO_MNT_HELA_02</th>
      <th>LSPPYSSPQEFAQDVGR</th>
      <td>27.287</td>
      <td>27.569</td>
      <td>27.627</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190722_QE8_nLC0_BDA_QC_HeLa_50cm_02</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <td>30.177</td>
      <td>31.086</td>
      <td>30.820</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar</th>
      <th>QITVNDLPVGR</th>
      <td>33.070</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190726_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_4thcolumn_1</th>
      <th>IISNASCTTNCLAPLAK</th>
      <td>33.371</td>
      <td>33.165</td>
      <td>33.003</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190731_QX8_ChSc_MA_HeLa_500ng</th>
      <th>FDDAVVQSDMK</th>
      <td>31.499</td>
      <td>29.752</td>
      <td>29.810</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX6_MaTa_MA_HeLa_500ng_LC09</th>
      <th>QITVNDLPVGR</th>
      <td>33.009</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_300ng</th>
      <th>TPAQYDASELK</th>
      <td>31.184</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>LYSVSYLLK</th>
      <td>26.381</td>
      <td>29.205</td>
      <td>28.697</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>LYSVSYLLK</th>
      <td>27.455</td>
      <td>29.205</td>
      <td>28.697</td>
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
      <td>ALIAAQYSGAQVR</td>
      <td>29.414</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DDEFTHLYTLIVRPDNTYEVK</td>
      <td>31.332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DFTVSAMHGDMDQK</td>
      <td>28.586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EALLSSAVDHGSDEVK</td>
      <td>28.830</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EALTYDGALLGDR</td>
      <td>27.815</td>
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
      <td>ALDVMVSTFHK</td>
      <td>30.949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DSNNLCLHFNPR</td>
      <td>30.046</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>ALIAAQYSGAQVR</td>
      <td>29.496</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>GASQAGMTGYGMPR</td>
      <td>29.006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181221_QE8_nLC0_NHS_MNT_HeLa_01</td>
      <td>ETNLDSLPLVDTHSK</td>
      <td>28.734</td>
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
      <td>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04</td>
      <td>VFITDDFHDMMPK</td>
      <td>28.321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190702_QX0_AnBr_MA_HeLa_500ng_LC07_01_190702180001</td>
      <td>EILVGDVGQTVDDPYATFVK</td>
      <td>33.391</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190509_QE2_NLC1_GP_MNT_HELA_01</td>
      <td>KYEQGFITDPVVLSPK</td>
      <td>28.641</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190726_QX8_ChSc_MA_HeLa_500ng</td>
      <td>IVSRPEELREDDVGTGAGLLEIK</td>
      <td>32.583</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181230_QE6_nLC6_CSC_QC_HeLa_03</td>
      <td>LSPPYSSPQEFAQDVGR</td>
      <td>27.747</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03</td>
      <td>IISNASCTTNCLAPLAK</td>
      <td>33.532</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01_20190430180750</td>
      <td>ALDVMVSTFHK</td>
      <td>31.297</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_2</td>
      <td>HLNEIDLFHCIDPNDSK</td>
      <td>24.956</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190426_QX1_JoMu_MA_HeLa_500ng_LC11</td>
      <td>DSNNLCLHFNPR</td>
      <td>31.449</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190723_QE4_LC12_IAH_QC_MNT_HeLa_01</td>
      <td>EALLSSAVDHGSDEVK</td>
      <td>28.772</td>
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
      <td>20190208_QE2_NLC1_AB_QC_MNT_HELA_2</td>
      <td>ELISNSSDALDK</td>
      <td>30.242</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190426_QX1_JoMu_MA_HeLa_500ng_LC11</td>
      <td>LYSVSYLLK</td>
      <td>28.036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190502_QX7_ChDe_MA_HeLaBr14_500ng</td>
      <td>IVSRPEELREDDVGTGAGLLEIK</td>
      <td>33.486</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190503_QX1_LiSc_MA_HeLa_500ng_LC10</td>
      <td>VFDAIMNFK</td>
      <td>31.359</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190201_QE10_nLC0_NHS_MNT_HELA_45cm_01</td>
      <td>GASQAGMTGYGMPR</td>
      <td>27.141</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190201_QE1_nLC2_GP_QC_MNT_HELA_01</td>
      <td>EALTYDGALLGDR</td>
      <td>28.108</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190207_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_01</td>
      <td>TIGTGLVTNTLAMTEEEK</td>
      <td>27.743</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190315_QE2_NLC1_GP_MNT_HELA_01</td>
      <td>KYEDICPSTHNMDVPNIK</td>
      <td>31.987</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190710_QE1_nLC13_ANHO_QC_MNT_HELA_01</td>
      <td>DFTVSAMHGDMDQK</td>
      <td>30.191</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190629_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</td>
      <td>QITVNDLPVGR</td>
      <td>31.899</td>
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




    (1382, 155)



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
     'y_range': (20, 38)}
    








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
    
    Optimizer used: <function Adam at 0x000001D66C536040>
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
      <td>1.426021</td>
      <td>1.240089</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.507862</td>
      <td>0.477302</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.471286</td>
      <td>0.460707</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.544459</td>
      <td>0.441807</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.397767</td>
      <td>0.439586</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.441695</td>
      <td>0.420181</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.435240</td>
      <td>0.414824</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.421531</td>
      <td>0.405282</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.452503</td>
      <td>0.402932</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.419101</td>
      <td>0.402654</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_58_1.png)
    


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
      <th>769</th>
      <td>155</td>
      <td>9</td>
      <td>30.242</td>
    </tr>
    <tr>
      <th>2,188</th>
      <td>446</td>
      <td>36</td>
      <td>28.036</td>
    </tr>
    <tr>
      <th>2,290</th>
      <td>466</td>
      <td>30</td>
      <td>33.486</td>
    </tr>
    <tr>
      <th>2,317</th>
      <td>470</td>
      <td>44</td>
      <td>31.359</td>
    </tr>
    <tr>
      <th>498</th>
      <td>108</td>
      <td>16</td>
      <td>27.141</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4,421</th>
      <td>900</td>
      <td>37</td>
      <td>30.280</td>
    </tr>
    <tr>
      <th>1,114</th>
      <td>222</td>
      <td>5</td>
      <td>30.761</td>
    </tr>
    <tr>
      <th>1,835</th>
      <td>374</td>
      <td>38</td>
      <td>32.288</td>
    </tr>
    <tr>
      <th>352</th>
      <td>73</td>
      <td>27</td>
      <td>30.354</td>
    </tr>
    <tr>
      <th>1,018</th>
      <td>203</td>
      <td>25</td>
      <td>31.256</td>
    </tr>
  </tbody>
</table>
<p>4936 rows × 3 columns</p>
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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.015</td>
      <td>31.827</td>
      <td>31.491</td>
      <td>31.059</td>
    </tr>
    <tr>
      <th>DSNNLCLHFNPR</th>
      <td>30.046</td>
      <td>30.842</td>
      <td>30.635</td>
      <td>30.399</td>
      <td>29.991</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALIAAQYSGAQVR</th>
      <td>29.496</td>
      <td>30.535</td>
      <td>30.473</td>
      <td>29.931</td>
      <td>29.835</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>29.006</td>
      <td>28.341</td>
      <td>28.219</td>
      <td>28.219</td>
      <td>27.404</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.734</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>30.011</td>
      <td>28.484</td>
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
      <th>GQAAVQQLQAEGLSPR</th>
      <td>26.757</td>
      <td>27.069</td>
      <td>27.457</td>
      <td>26.385</td>
      <td>26.802</td>
    </tr>
    <tr>
      <th>ILDSVGIEADDDRLNK</th>
      <td>31.576</td>
      <td>31.573</td>
      <td>31.581</td>
      <td>31.516</td>
      <td>32.065</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>32.130</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>32.349</td>
      <td>32.164</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>32.338</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>31.604</td>
      <td>31.815</td>
    </tr>
    <tr>
      <th>YAPSEAGLHEMDIR</th>
      <td>29.643</td>
      <td>29.426</td>
      <td>29.014</td>
      <td>29.642</td>
      <td>29.554</td>
    </tr>
  </tbody>
</table>
<p>4936 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.060
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.072
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.227
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.135
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.176
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_66_1.png)
    



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
      <td>-0.002</td>
      <td>-0.040</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.076</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.101</td>
      <td>-0.132</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.032</td>
      <td>-0.059</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.067</td>
      <td>-0.095</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_68_1.png)
    



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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_69_1.png)
    


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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK</th>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <th>TPAQYDASELK</th>
      <th>VFDAIMNFK</th>
      <th>VFITDDFHDMMPK</th>
      <th>VIVVGNPANTNCLTASK</th>
      <th>VVVQVLAEEPEAVLK</th>
      <th>YAPSEAGLHEMDIR</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <th>YLMEEDEDAYKK</th>
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
      <td>30.949</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.046</td>
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
      <td>29.496</td>
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
      <td>28.734</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.331</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.755</td>
      <td>30.354</td>
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
      <td>30.136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.733</td>
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
      <td>23.993</td>
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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VVVQVLAEEPEAVLK_na</th>
      <th>YAPSEAGLHEMDIR_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
      <th>YLMEEDEDAYKK_na</th>
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
      <td>0.191</td>
      <td>-0.997</td>
      <td>0.379</td>
      <td>-1.286</td>
      <td>0.198</td>
      <td>-0.778</td>
      <td>-1.068</td>
      <td>-1.222</td>
      <td>-0.656</td>
      <td>-0.040</td>
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
      <td>-0.501</td>
      <td>0.051</td>
      <td>0.325</td>
      <td>-1.306</td>
      <td>-0.648</td>
      <td>-0.998</td>
      <td>-1.189</td>
      <td>-1.512</td>
      <td>-0.632</td>
      <td>-0.098</td>
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
      <td>-0.300</td>
      <td>-0.030</td>
      <td>-0.469</td>
      <td>-0.773</td>
      <td>0.098</td>
      <td>-0.800</td>
      <td>-0.317</td>
      <td>0.331</td>
      <td>-0.400</td>
      <td>-0.266</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-5.424</td>
      <td>-0.678</td>
      <td>0.214</td>
      <td>0.029</td>
      <td>0.198</td>
      <td>-0.558</td>
      <td>-0.914</td>
      <td>-1.002</td>
      <td>-0.291</td>
      <td>0.457</td>
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
      <td>-1.267</td>
      <td>-0.949</td>
      <td>1.478</td>
      <td>-1.220</td>
      <td>0.198</td>
      <td>-0.824</td>
      <td>-0.838</td>
      <td>-0.282</td>
      <td>-0.216</td>
      <td>-0.165</td>
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
      <td>-1.412</td>
      <td>0.686</td>
      <td>0.426</td>
      <td>0.551</td>
      <td>-0.407</td>
      <td>0.968</td>
      <td>0.620</td>
      <td>-0.368</td>
      <td>0.942</td>
      <td>1.147</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.200</td>
      <td>-0.075</td>
      <td>0.552</td>
      <td>0.540</td>
      <td>0.724</td>
      <td>0.027</td>
      <td>-0.839</td>
      <td>0.079</td>
      <td>-0.534</td>
      <td>-0.374</td>
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
      <td>0.318</td>
      <td>0.108</td>
      <td>0.876</td>
      <td>0.426</td>
      <td>0.960</td>
      <td>-0.036</td>
      <td>-0.622</td>
      <td>0.370</td>
      <td>-0.369</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.508</td>
      <td>0.170</td>
      <td>0.971</td>
      <td>0.029</td>
      <td>0.720</td>
      <td>-0.063</td>
      <td>0.004</td>
      <td>0.175</td>
      <td>-0.350</td>
      <td>-0.448</td>
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
      <td>0.141</td>
      <td>0.105</td>
      <td>0.871</td>
      <td>0.029</td>
      <td>0.686</td>
      <td>0.086</td>
      <td>0.004</td>
      <td>0.174</td>
      <td>-0.216</td>
      <td>-0.589</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VVVQVLAEEPEAVLK_na</th>
      <th>YAPSEAGLHEMDIR_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
      <th>YLMEEDEDAYKK_na</th>
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
      <td>0.202</td>
      <td>-0.933</td>
      <td>0.380</td>
      <td>-1.215</td>
      <td>0.209</td>
      <td>-0.734</td>
      <td>-1.002</td>
      <td>-1.132</td>
      <td>-0.646</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.456</td>
      <td>0.055</td>
      <td>0.329</td>
      <td>-1.234</td>
      <td>-0.593</td>
      <td>-0.939</td>
      <td>-1.116</td>
      <td>-1.407</td>
      <td>-0.623</td>
      <td>-0.127</td>
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
      <td>-0.264</td>
      <td>-0.022</td>
      <td>-0.408</td>
      <td>-0.729</td>
      <td>0.115</td>
      <td>-0.754</td>
      <td>-0.297</td>
      <td>0.338</td>
      <td>-0.404</td>
      <td>-0.285</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
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
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-5.132</td>
      <td>-0.632</td>
      <td>0.227</td>
      <td>0.030</td>
      <td>0.209</td>
      <td>-0.529</td>
      <td>-0.857</td>
      <td>-0.924</td>
      <td>-0.301</td>
      <td>0.396</td>
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
      <td>-1.183</td>
      <td>-0.888</td>
      <td>1.401</td>
      <td>-1.152</td>
      <td>0.209</td>
      <td>-0.777</td>
      <td>-0.786</td>
      <td>-0.242</td>
      <td>-0.230</td>
      <td>-0.190</td>
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
      <td>-1.321</td>
      <td>0.652</td>
      <td>0.424</td>
      <td>0.525</td>
      <td>-0.364</td>
      <td>0.895</td>
      <td>0.582</td>
      <td>-0.324</td>
      <td>0.862</td>
      <td>1.045</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>0.210</td>
      <td>-0.064</td>
      <td>0.541</td>
      <td>0.514</td>
      <td>0.708</td>
      <td>0.017</td>
      <td>-0.788</td>
      <td>0.098</td>
      <td>-0.531</td>
      <td>-0.387</td>
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
      <td>0.323</td>
      <td>0.108</td>
      <td>0.842</td>
      <td>0.406</td>
      <td>0.933</td>
      <td>-0.042</td>
      <td>-0.583</td>
      <td>0.374</td>
      <td>-0.374</td>
      <td>-0.439</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_03</th>
      <td>0.503</td>
      <td>0.166</td>
      <td>0.930</td>
      <td>0.030</td>
      <td>0.705</td>
      <td>-0.067</td>
      <td>0.005</td>
      <td>0.189</td>
      <td>-0.357</td>
      <td>-0.457</td>
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
      <td>0.154</td>
      <td>0.105</td>
      <td>0.837</td>
      <td>0.030</td>
      <td>0.673</td>
      <td>0.072</td>
      <td>0.005</td>
      <td>0.188</td>
      <td>-0.230</td>
      <td>-0.589</td>
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
  </tbody>
</table>
<p>996 rows × 100 columns</p>
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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
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
      <td>0.020</td>
      <td>0.006</td>
      <td>0.028</td>
      <td>0.003</td>
      <td>0.022</td>
      <td>-0.008</td>
      <td>0.001</td>
      <td>0.024</td>
      <td>-0.027</td>
      <td>-0.035</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.950</td>
      <td>0.942</td>
      <td>0.930</td>
      <td>0.947</td>
      <td>0.949</td>
      <td>0.934</td>
      <td>0.939</td>
      <td>0.947</td>
      <td>0.944</td>
      <td>0.942</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.132</td>
      <td>-5.838</td>
      <td>-6.249</td>
      <td>-5.623</td>
      <td>-4.905</td>
      <td>-4.384</td>
      <td>-4.546</td>
      <td>-5.618</td>
      <td>-4.765</td>
      <td>-2.690</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.397</td>
      <td>-0.375</td>
      <td>-0.308</td>
      <td>-0.380</td>
      <td>-0.289</td>
      <td>-0.454</td>
      <td>-0.466</td>
      <td>-0.327</td>
      <td>-0.514</td>
      <td>-0.560</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.202</td>
      <td>0.055</td>
      <td>0.197</td>
      <td>0.030</td>
      <td>0.209</td>
      <td>-0.063</td>
      <td>0.005</td>
      <td>0.219</td>
      <td>-0.230</td>
      <td>-0.285</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.605</td>
      <td>0.535</td>
      <td>0.561</td>
      <td>0.476</td>
      <td>0.587</td>
      <td>0.407</td>
      <td>0.489</td>
      <td>0.650</td>
      <td>0.716</td>
      <td>0.792</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.883</td>
      <td>2.091</td>
      <td>1.868</td>
      <td>1.995</td>
      <td>1.920</td>
      <td>2.086</td>
      <td>2.180</td>
      <td>1.719</td>
      <td>2.068</td>
      <td>2.309</td>
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




    ((#50) ['ALDVMVSTFHK','ALIAAQYSGAQVR','DDEFTHLYTLIVRPDNTYEVK','DFTVSAMHGDMDQK','DSNNLCLHFNPR','EALLSSAVDHGSDEVK','EALTYDGALLGDR','EILVGDVGQTVDDPYATFVK','ELISNSSDALDK','ETNLDSLPLVDTHSK'...],
     (#50) ['ALDVMVSTFHK_na','ALIAAQYSGAQVR_na','DDEFTHLYTLIVRPDNTYEVK_na','DFTVSAMHGDMDQK_na','DSNNLCLHFNPR_na','EALLSSAVDHGSDEVK_na','EALTYDGALLGDR_na','EILVGDVGQTVDDPYATFVK_na','ELISNSSDALDK_na','ETNLDSLPLVDTHSK_na'...])




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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>97.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.077</td>
      <td>0.092</td>
      <td>-0.128</td>
      <td>-0.031</td>
      <td>0.015</td>
      <td>-0.019</td>
      <td>0.211</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828</td>
      <td>0.986</td>
      <td>1.045</td>
      <td>0.805</td>
      <td>0.944</td>
      <td>0.924</td>
      <td>0.807</td>
      <td>0.916</td>
      <td>1.007</td>
      <td>0.961</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.886</td>
      <td>-4.603</td>
      <td>-3.591</td>
      <td>-2.292</td>
      <td>-2.837</td>
      <td>-3.210</td>
      <td>-2.917</td>
      <td>-2.879</td>
      <td>-2.735</td>
      <td>-1.956</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.340</td>
      <td>-0.342</td>
      <td>-0.486</td>
      <td>-0.520</td>
      <td>-0.484</td>
      <td>-0.590</td>
      <td>-0.233</td>
      <td>-0.480</td>
      <td>-0.658</td>
      <td>-0.630</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.201</td>
      <td>0.151</td>
      <td>0.059</td>
      <td>0.030</td>
      <td>0.249</td>
      <td>-0.068</td>
      <td>0.069</td>
      <td>0.281</td>
      <td>-0.220</td>
      <td>-0.285</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.669</td>
      <td>0.618</td>
      <td>0.545</td>
      <td>0.483</td>
      <td>0.666</td>
      <td>0.477</td>
      <td>0.784</td>
      <td>0.701</td>
      <td>1.159</td>
      <td>1.157</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.495</td>
      <td>1.917</td>
      <td>1.540</td>
      <td>1.623</td>
      <td>1.539</td>
      <td>1.817</td>
      <td>2.149</td>
      <td>1.529</td>
      <td>1.908</td>
      <td>1.698</td>
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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VVVQVLAEEPEAVLK_val</th>
      <th>YAPSEAGLHEMDIR_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
      <th>YLMEEDEDAYKK_val</th>
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
      <td>0.202</td>
      <td>-0.933</td>
      <td>0.380</td>
      <td>-1.215</td>
      <td>0.209</td>
      <td>-0.734</td>
      <td>-1.002</td>
      <td>-1.132</td>
      <td>-0.646</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.456</td>
      <td>0.055</td>
      <td>0.329</td>
      <td>-1.234</td>
      <td>-0.593</td>
      <td>-0.939</td>
      <td>-1.116</td>
      <td>-1.407</td>
      <td>-0.623</td>
      <td>-0.127</td>
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
      <td>-0.264</td>
      <td>-0.022</td>
      <td>-0.408</td>
      <td>-0.729</td>
      <td>0.115</td>
      <td>-0.754</td>
      <td>-0.297</td>
      <td>0.338</td>
      <td>-0.404</td>
      <td>-0.285</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.227</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-5.132</td>
      <td>-0.632</td>
      <td>0.227</td>
      <td>0.030</td>
      <td>0.209</td>
      <td>-0.529</td>
      <td>-0.857</td>
      <td>-0.924</td>
      <td>-0.301</td>
      <td>0.396</td>
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
      <td>-1.183</td>
      <td>-0.888</td>
      <td>1.401</td>
      <td>-1.152</td>
      <td>0.209</td>
      <td>-0.777</td>
      <td>-0.786</td>
      <td>-0.242</td>
      <td>-0.230</td>
      <td>-0.190</td>
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
      <td>-4.180</td>
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
      <td>-1.321</td>
      <td>0.652</td>
      <td>0.424</td>
      <td>0.525</td>
      <td>-0.364</td>
      <td>0.895</td>
      <td>0.582</td>
      <td>-0.324</td>
      <td>0.862</td>
      <td>1.045</td>
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
      <td>0.210</td>
      <td>-0.064</td>
      <td>0.541</td>
      <td>0.514</td>
      <td>0.708</td>
      <td>0.017</td>
      <td>-0.788</td>
      <td>0.098</td>
      <td>-0.531</td>
      <td>-0.387</td>
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
      <td>0.323</td>
      <td>0.108</td>
      <td>0.842</td>
      <td>0.406</td>
      <td>0.933</td>
      <td>-0.042</td>
      <td>-0.583</td>
      <td>0.374</td>
      <td>-0.374</td>
      <td>-0.439</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.061</td>
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
      <td>0.503</td>
      <td>0.166</td>
      <td>0.930</td>
      <td>0.030</td>
      <td>0.705</td>
      <td>-0.067</td>
      <td>0.005</td>
      <td>0.189</td>
      <td>-0.357</td>
      <td>-0.457</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.468</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.154</td>
      <td>0.105</td>
      <td>0.837</td>
      <td>0.030</td>
      <td>0.673</td>
      <td>0.072</td>
      <td>0.005</td>
      <td>0.188</td>
      <td>-0.230</td>
      <td>-0.589</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.451</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VVVQVLAEEPEAVLK_val</th>
      <th>YAPSEAGLHEMDIR_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
      <th>YLMEEDEDAYKK_val</th>
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
      <td>0.202</td>
      <td>-0.933</td>
      <td>0.380</td>
      <td>-1.215</td>
      <td>0.209</td>
      <td>-0.734</td>
      <td>-1.002</td>
      <td>-1.132</td>
      <td>-0.646</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.456</td>
      <td>0.055</td>
      <td>0.329</td>
      <td>-1.234</td>
      <td>-0.593</td>
      <td>-0.939</td>
      <td>-1.116</td>
      <td>-1.407</td>
      <td>-0.623</td>
      <td>-0.127</td>
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
      <td>-0.264</td>
      <td>-0.022</td>
      <td>-0.408</td>
      <td>-0.729</td>
      <td>0.115</td>
      <td>-0.754</td>
      <td>-0.297</td>
      <td>0.338</td>
      <td>-0.404</td>
      <td>-0.285</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.227</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-5.132</td>
      <td>-0.632</td>
      <td>0.227</td>
      <td>0.030</td>
      <td>0.209</td>
      <td>-0.529</td>
      <td>-0.857</td>
      <td>-0.924</td>
      <td>-0.301</td>
      <td>0.396</td>
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
      <td>-1.183</td>
      <td>-0.888</td>
      <td>1.401</td>
      <td>-1.152</td>
      <td>0.209</td>
      <td>-0.777</td>
      <td>-0.786</td>
      <td>-0.242</td>
      <td>-0.230</td>
      <td>-0.190</td>
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
      <td>-4.180</td>
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
      <td>-1.321</td>
      <td>0.652</td>
      <td>0.424</td>
      <td>0.525</td>
      <td>-0.364</td>
      <td>0.895</td>
      <td>0.582</td>
      <td>-0.324</td>
      <td>0.862</td>
      <td>1.045</td>
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
      <td>0.210</td>
      <td>-0.064</td>
      <td>0.541</td>
      <td>0.514</td>
      <td>0.708</td>
      <td>0.017</td>
      <td>-0.788</td>
      <td>0.098</td>
      <td>-0.531</td>
      <td>-0.387</td>
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
      <td>0.323</td>
      <td>0.108</td>
      <td>0.842</td>
      <td>0.406</td>
      <td>0.933</td>
      <td>-0.042</td>
      <td>-0.583</td>
      <td>0.374</td>
      <td>-0.374</td>
      <td>-0.439</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.061</td>
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
      <td>0.503</td>
      <td>0.166</td>
      <td>0.930</td>
      <td>0.030</td>
      <td>0.705</td>
      <td>-0.067</td>
      <td>0.005</td>
      <td>0.189</td>
      <td>-0.357</td>
      <td>-0.457</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.468</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <td>0.154</td>
      <td>0.105</td>
      <td>0.837</td>
      <td>0.030</td>
      <td>0.673</td>
      <td>0.072</td>
      <td>0.005</td>
      <td>0.188</td>
      <td>-0.230</td>
      <td>-0.589</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.451</td>
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>ALDVMVSTFHK_val</th>
      <th>ALIAAQYSGAQVR_val</th>
      <th>DDEFTHLYTLIVRPDNTYEVK_val</th>
      <th>DFTVSAMHGDMDQK_val</th>
      <th>DSNNLCLHFNPR_val</th>
      <th>EALLSSAVDHGSDEVK_val</th>
      <th>EALTYDGALLGDR_val</th>
      <th>EILVGDVGQTVDDPYATFVK_val</th>
      <th>ELISNSSDALDK_val</th>
      <th>ETNLDSLPLVDTHSK_val</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_val</th>
      <th>TIGTGLVTNTLAMTEEEK_val</th>
      <th>TPAQYDASELK_val</th>
      <th>VFDAIMNFK_val</th>
      <th>VFITDDFHDMMPK_val</th>
      <th>VIVVGNPANTNCLTASK_val</th>
      <th>VVVQVLAEEPEAVLK_val</th>
      <th>YAPSEAGLHEMDIR_val</th>
      <th>YHTSQSGDEMTSLSEYVSR_val</th>
      <th>YLMEEDEDAYKK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>97.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>97.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.077</td>
      <td>0.092</td>
      <td>-0.128</td>
      <td>-0.031</td>
      <td>0.015</td>
      <td>-0.019</td>
      <td>0.211</td>
      <td>0.020</td>
      <td>0.114</td>
      <td>0.036</td>
      <td>...</td>
      <td>0.067</td>
      <td>-0.107</td>
      <td>-0.051</td>
      <td>0.154</td>
      <td>-0.131</td>
      <td>0.160</td>
      <td>-0.024</td>
      <td>-0.003</td>
      <td>-0.029</td>
      <td>0.037</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828</td>
      <td>0.986</td>
      <td>1.045</td>
      <td>0.805</td>
      <td>0.944</td>
      <td>0.924</td>
      <td>0.807</td>
      <td>0.916</td>
      <td>1.007</td>
      <td>0.961</td>
      <td>...</td>
      <td>0.902</td>
      <td>0.955</td>
      <td>1.095</td>
      <td>1.061</td>
      <td>1.019</td>
      <td>0.993</td>
      <td>1.077</td>
      <td>0.944</td>
      <td>1.011</td>
      <td>0.941</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.886</td>
      <td>-4.603</td>
      <td>-3.591</td>
      <td>-2.292</td>
      <td>-2.837</td>
      <td>-3.210</td>
      <td>-2.917</td>
      <td>-2.879</td>
      <td>-2.735</td>
      <td>-1.956</td>
      <td>...</td>
      <td>-2.566</td>
      <td>-3.869</td>
      <td>-4.779</td>
      <td>-1.996</td>
      <td>-3.310</td>
      <td>-3.120</td>
      <td>-5.229</td>
      <td>-3.948</td>
      <td>-2.535</td>
      <td>-4.180</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.340</td>
      <td>-0.342</td>
      <td>-0.486</td>
      <td>-0.520</td>
      <td>-0.484</td>
      <td>-0.590</td>
      <td>-0.233</td>
      <td>-0.480</td>
      <td>-0.658</td>
      <td>-0.630</td>
      <td>...</td>
      <td>-0.437</td>
      <td>-0.602</td>
      <td>-0.595</td>
      <td>-0.609</td>
      <td>-0.788</td>
      <td>-0.329</td>
      <td>-0.578</td>
      <td>-0.181</td>
      <td>-0.703</td>
      <td>-0.333</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.201</td>
      <td>0.151</td>
      <td>0.059</td>
      <td>0.030</td>
      <td>0.249</td>
      <td>-0.068</td>
      <td>0.069</td>
      <td>0.281</td>
      <td>-0.220</td>
      <td>-0.285</td>
      <td>...</td>
      <td>0.008</td>
      <td>-0.124</td>
      <td>0.148</td>
      <td>-0.327</td>
      <td>-0.262</td>
      <td>0.285</td>
      <td>0.130</td>
      <td>0.294</td>
      <td>-0.150</td>
      <td>0.196</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.669</td>
      <td>0.618</td>
      <td>0.545</td>
      <td>0.483</td>
      <td>0.666</td>
      <td>0.477</td>
      <td>0.784</td>
      <td>0.701</td>
      <td>1.159</td>
      <td>1.157</td>
      <td>...</td>
      <td>0.677</td>
      <td>0.304</td>
      <td>0.722</td>
      <td>1.363</td>
      <td>0.589</td>
      <td>0.908</td>
      <td>0.688</td>
      <td>0.536</td>
      <td>0.749</td>
      <td>0.680</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.495</td>
      <td>1.917</td>
      <td>1.540</td>
      <td>1.623</td>
      <td>1.539</td>
      <td>1.817</td>
      <td>2.149</td>
      <td>1.529</td>
      <td>1.908</td>
      <td>1.698</td>
      <td>...</td>
      <td>1.937</td>
      <td>1.906</td>
      <td>1.598</td>
      <td>1.780</td>
      <td>1.942</td>
      <td>1.775</td>
      <td>1.690</td>
      <td>1.355</td>
      <td>2.039</td>
      <td>1.398</td>
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
      <th>ALDVMVSTFHK_na</th>
      <th>ALIAAQYSGAQVR_na</th>
      <th>DDEFTHLYTLIVRPDNTYEVK_na</th>
      <th>DFTVSAMHGDMDQK_na</th>
      <th>DSNNLCLHFNPR_na</th>
      <th>EALLSSAVDHGSDEVK_na</th>
      <th>EALTYDGALLGDR_na</th>
      <th>EILVGDVGQTVDDPYATFVK_na</th>
      <th>ELISNSSDALDK_na</th>
      <th>ETNLDSLPLVDTHSK_na</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK_na</th>
      <th>TIGTGLVTNTLAMTEEEK_na</th>
      <th>TPAQYDASELK_na</th>
      <th>VFDAIMNFK_na</th>
      <th>VFITDDFHDMMPK_na</th>
      <th>VIVVGNPANTNCLTASK_na</th>
      <th>VVVQVLAEEPEAVLK_na</th>
      <th>YAPSEAGLHEMDIR_na</th>
      <th>YHTSQSGDEMTSLSEYVSR_na</th>
      <th>YLMEEDEDAYKK_na</th>
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
      <td>False</td>
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
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
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
      <td>False</td>
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
    
    Optimizer used: <function Adam at 0x000001D66C536040>
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




    
![png](latent_2D_300_10_files/latent_2D_300_10_108_2.png)
    


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
      <td>0.944794</td>
      <td>0.689589</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.647858</td>
      <td>0.344155</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.466461</td>
      <td>0.308917</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.384956</td>
      <td>0.301918</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.340464</td>
      <td>0.285868</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.314145</td>
      <td>0.285941</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.295345</td>
      <td>0.273876</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.284893</td>
      <td>0.268750</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.276837</td>
      <td>0.267274</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.272195</td>
      <td>0.269358</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4936, 4936)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.2721)




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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.015</td>
      <td>31.827</td>
      <td>31.491</td>
      <td>31.059</td>
      <td>31.041</td>
    </tr>
    <tr>
      <th>DSNNLCLHFNPR</th>
      <td>30.046</td>
      <td>30.842</td>
      <td>30.635</td>
      <td>30.399</td>
      <td>29.991</td>
      <td>30.196</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALIAAQYSGAQVR</th>
      <td>29.496</td>
      <td>30.535</td>
      <td>30.473</td>
      <td>29.931</td>
      <td>29.835</td>
      <td>29.502</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>29.006</td>
      <td>28.341</td>
      <td>28.219</td>
      <td>28.219</td>
      <td>27.404</td>
      <td>27.200</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.734</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>30.011</td>
      <td>28.484</td>
      <td>28.416</td>
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
      <th>GQAAVQQLQAEGLSPR</th>
      <td>26.757</td>
      <td>27.069</td>
      <td>27.457</td>
      <td>26.385</td>
      <td>26.802</td>
      <td>26.944</td>
    </tr>
    <tr>
      <th>ILDSVGIEADDDRLNK</th>
      <td>31.576</td>
      <td>31.573</td>
      <td>31.581</td>
      <td>31.516</td>
      <td>32.065</td>
      <td>31.682</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>32.130</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>32.349</td>
      <td>32.164</td>
      <td>32.270</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>32.338</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>31.604</td>
      <td>31.815</td>
      <td>32.051</td>
    </tr>
    <tr>
      <th>YAPSEAGLHEMDIR</th>
      <td>29.643</td>
      <td>29.426</td>
      <td>29.014</td>
      <td>29.642</td>
      <td>29.554</td>
      <td>29.733</td>
    </tr>
  </tbody>
</table>
<p>4936 rows × 6 columns</p>
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
      <td>0.500</td>
      <td>-0.934</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.490</td>
      <td>-0.935</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.834</td>
      <td>-0.884</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.628</td>
      <td>-0.905</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.696</td>
      <td>-0.923</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_122_1.png)
    



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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_123_1.png)
    


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
      <th>ALDVMVSTFHK</th>
      <th>ALIAAQYSGAQVR</th>
      <th>DDEFTHLYTLIVRPDNTYEVK</th>
      <th>DFTVSAMHGDMDQK</th>
      <th>DSNNLCLHFNPR</th>
      <th>EALLSSAVDHGSDEVK</th>
      <th>EALTYDGALLGDR</th>
      <th>EILVGDVGQTVDDPYATFVK</th>
      <th>ELISNSSDALDK</th>
      <th>ETNLDSLPLVDTHSK</th>
      <th>...</th>
      <th>TIGGGDDSFTTFFCETGAGK</th>
      <th>TIGTGLVTNTLAMTEEEK</th>
      <th>TPAQYDASELK</th>
      <th>VFDAIMNFK</th>
      <th>VFITDDFHDMMPK</th>
      <th>VIVVGNPANTNCLTASK</th>
      <th>VVVQVLAEEPEAVLK</th>
      <th>YAPSEAGLHEMDIR</th>
      <th>YHTSQSGDEMTSLSEYVSR</th>
      <th>YLMEEDEDAYKK</th>
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
      <td>0.597</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.631</td>
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
      <td>0.628</td>
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
      <td>0.444</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.748</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.594</td>
      <td>0.677</td>
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
      <td>0.645</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.604</td>
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
      <td>0.020</td>
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
    
    Optimizer used: <function Adam at 0x000001D66C536040>
    Loss function: <function loss_fct_vae at 0x000001D66C553940>
    
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








    SuggestedLRs(valley=0.001737800776027143)




    
![png](latent_2D_300_10_files/latent_2D_300_10_136_2.png)
    



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
      <td>2007.889893</td>
      <td>218.604111</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1980.962524</td>
      <td>213.734177</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1939.623169</td>
      <td>204.017441</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1893.503052</td>
      <td>198.501846</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1853.009399</td>
      <td>194.931259</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1821.802002</td>
      <td>192.881714</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1797.123413</td>
      <td>191.036530</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1778.493530</td>
      <td>190.886398</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1765.164673</td>
      <td>190.703674</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1755.589233</td>
      <td>190.658005</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4936)




```python
len(pred[0])
```




    4936




```python
learn.loss_func(pred, target)
```




    tensor(3016.6853)




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
      <th rowspan="2" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ALDVMVSTFHK</th>
      <td>30.949</td>
      <td>32.015</td>
      <td>31.827</td>
      <td>31.491</td>
      <td>31.059</td>
      <td>31.041</td>
      <td>31.869</td>
    </tr>
    <tr>
      <th>DSNNLCLHFNPR</th>
      <td>30.046</td>
      <td>30.842</td>
      <td>30.635</td>
      <td>30.399</td>
      <td>29.991</td>
      <td>30.196</td>
      <td>30.797</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>ALIAAQYSGAQVR</th>
      <td>29.496</td>
      <td>30.535</td>
      <td>30.473</td>
      <td>29.931</td>
      <td>29.835</td>
      <td>29.502</td>
      <td>30.627</td>
    </tr>
    <tr>
      <th>GASQAGMTGYGMPR</th>
      <td>29.006</td>
      <td>28.341</td>
      <td>28.219</td>
      <td>28.219</td>
      <td>27.404</td>
      <td>27.200</td>
      <td>28.347</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <th>ETNLDSLPLVDTHSK</th>
      <td>28.734</td>
      <td>29.123</td>
      <td>29.727</td>
      <td>30.011</td>
      <td>28.484</td>
      <td>28.416</td>
      <td>29.828</td>
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
      <th>GQAAVQQLQAEGLSPR</th>
      <td>26.757</td>
      <td>27.069</td>
      <td>27.457</td>
      <td>26.385</td>
      <td>26.802</td>
      <td>26.944</td>
      <td>27.594</td>
    </tr>
    <tr>
      <th>ILDSVGIEADDDRLNK</th>
      <td>31.576</td>
      <td>31.573</td>
      <td>31.581</td>
      <td>31.516</td>
      <td>32.065</td>
      <td>31.682</td>
      <td>30.776</td>
    </tr>
    <tr>
      <th>QITVNDLPVGR</th>
      <td>32.130</td>
      <td>32.300</td>
      <td>32.204</td>
      <td>32.349</td>
      <td>32.164</td>
      <td>32.270</td>
      <td>31.360</td>
    </tr>
    <tr>
      <th>TPAQYDASELK</th>
      <td>32.338</td>
      <td>31.807</td>
      <td>31.638</td>
      <td>31.604</td>
      <td>31.815</td>
      <td>32.051</td>
      <td>31.242</td>
    </tr>
    <tr>
      <th>YAPSEAGLHEMDIR</th>
      <td>29.643</td>
      <td>29.426</td>
      <td>29.014</td>
      <td>29.642</td>
      <td>29.554</td>
      <td>29.733</td>
      <td>29.035</td>
    </tr>
  </tbody>
</table>
<p>4936 rows × 7 columns</p>
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
      <td>-0.120</td>
      <td>0.021</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.083</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.088</td>
      <td>-0.093</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.143</td>
      <td>-0.003</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.183</td>
      <td>0.010</td>
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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_146_1.png)
    



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
    


    
![png](latent_2D_300_10_files/latent_2D_300_10_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01', 'DSNNLCLHFNPR'), ('20181230_QE6_nLC6_CSC_QC_HeLa_03', 'HQEGEIFDTEK'), ('20190114_QE7_nLC7_AL_QC_MNT_HeLa_01', 'IPEISIQDMTAQVTSPSGK'), ('20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03', 'EILVGDVGQTVDDPYATFVK'), ('20190128_QE3_nLC3_MJ_MNT_HeLa_02', 'IRYESLTDPSK'), ('20190204_QE6_nLC6_MPL_QC_MNT_HeLa_01', 'FDTGNLCMVTGGANLGR'), ('20190208_QE2_NLC1_AB_QC_MNT_HELA_3', 'ISMPDFDLHLK'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_01', 'KYEDICPSTHNMDVPNIK'), ('20190214_QE4_LC12_SCL_QC_MNT_HeLa_02', 'KYEDICPSTHNMDVPNIK'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_16_23', 'HLNEIDLFHCIDPNDSK'), ('20190301_QE7_nLC7_DS_QC_MNT_HeLa_01', 'GAGTGGLGLAVEGPSEAK'), ('20190301_QE7_nLC7_DS_QC_MNT_HeLa_01_20190301161023', 'GAGTGGLGLAVEGPSEAK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03', 'IVSRPEELREDDVGTGAGLLEIK'), ('20190318_QE2_NLC1_AB_MNT_HELA_02', 'ILDSVGIEADDDRLNK'), ('20190318_QE2_NLC1_AB_MNT_HELA_03', 'ALDVMVSTFHK'), ('20190404_QE7_nLC3_AL_QC_MNT_HeLa_02', 'TIGGGDDSFTTFFCETGAGK'), ('20190405_QE1_nLC2_GP_MNT_QC_hela_01', 'TIGGGDDSFTTFFCETGAGK'), ('20190502_QE8_nLC14_AGF_QC_MNT_HeLa_01', 'LVIITAGAR'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'GASQAGMTGYGMPR'), ('20190510_QE2_NLC1_GP_MNT_HELA_02', 'HQEGEIFDTEK'), ('20190513_QE6_LC4_IAH_QC_MNT_HeLa_03', 'VFDAIMNFK'), ('20190513_QE7_nLC7_MEM_QC_MNT_HeLa_03', 'VFITDDFHDMMPK'), ('20190515_QX4_JiYu_MA_HeLa_500ng_BR14', 'SDIGEVILVGGMTR'), ('20190522_QX0_MaPe_MA_HeLa_500ng_LC07_1_BR14', 'IISNASCTTNCLAPLAK'), ('20190524_QE4_LC12_IAH_QC_MNT_HeLa_02', 'FNADEFEDMVAEK'), ('20190530_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'LVIITAGAR'), ('20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_01', 'GAGTGGLGLAVEGPSEAK'), ('20190609_QX8_MiWi_MA_HeLa_BR14_500ng_190625212127', 'LYSVSYLLK'), ('20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02', 'ELISNSSDALDK'), ('20190618_QX3_LiSc_MA_Hela_500ng_LC15', 'ETNLDSLPLVDTHSK'), ('20190619_QE1_nLC2_GP_QC_MNT_HELA_01', 'HQPTAIIAK'), ('20190620_QE2_NLC1_GP_QC_MNT_HELA_01', 'TPAQYDASELK'), ('20190620_QX1_JoMu_MA_HeLa__500ng_LC10', 'ETNLDSLPLVDTHSK'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'ETNLDSLPLVDTHSK'), ('20190621_QX4_JoMu_MA_HeLa_500ng', 'LLCGLLAER'), ('20190624_QE4_nLC12_MM_QC_MNT_HELA_01_20190626192509', 'KYEQGFITDPVVLSPK'), ('20190625_QE8_nLC14_GP_QC_MNT_15cm_Hela_02', 'HEQILVLDPPTDLK'), ('20190626_QX6_ChDe_MA_HeLa_500ng_LC09', 'VIVVGNPANTNCLTASK'), ('20190627_QX3_MaMu_MA_Hela_500ng_LC15', 'LLCGLLAER'), ('20190628_QE1_nLC13_ANHO_QC_MNT_HELA_03', 'ETNLDSLPLVDTHSK'), ('20190630_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'IIYGGSVTGATCK'), ('20190701_QE4_LC12_IAH_QC_MNT_HeLa_01', 'DFTVSAMHGDMDQK'), ('20190701_QX2_LiSc_MA_HeLa_500ng_LC05', 'IIAPPERK'), ('20190709_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'ALIAAQYSGAQVR'), ('20190712_QE8_nLC14_AnMu_QC_MNT_50cm_Hela_01', 'VFITDDFHDMMPK'), ('20190712_QE9_nLC9_NHS_MNT_HELA_50cm_MUC_01', 'VFITDDFHDMMPK'), ('20190715_QE4_LC12_IAH_QC_MNT_HeLa_01', 'FDDAVVQSDMK'), ('20190715_QE8_nLC14_RG_QC_MNT_50cm_Hela_02', 'EALLSSAVDHGSDEVK'), ('20190716_QX6_MaTa_MA_HeLa_500ng_LC09', 'HQPTAIIAK'), ('20190717_QX3_OzKa_MA_Hela_500ng_LC15_190720214645', 'EALTYDGALLGDR'), ('20190719_QE2_NLC1_ANHO_MNT_HELA_02', 'LSPPYSSPQEFAQDVGR'), ('20190722_QE8_nLC0_BDA_QC_HeLa_50cm_02', 'DDEFTHLYTLIVRPDNTYEVK'), ('20190723_QX1_JoMu_MA_HeLa_500ng_LC10_pack-2000bar', 'QITVNDLPVGR'), ('20190726_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_4thcolumn_1', 'IISNASCTTNCLAPLAK'), ('20190731_QX8_ChSc_MA_HeLa_500ng', 'FDDAVVQSDMK'), ('20190802_QX6_MaTa_MA_HeLa_500ng_LC09', 'QITVNDLPVGR'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_300ng', 'TPAQYDASELK'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'LYSVSYLLK'), ('20190803_QX8_AnPi_MA_HeLa_BR14_500ng', 'LYSVSYLLK')]
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intensity_pred_collab</th>
      <th>intensity_pred_dae</th>
      <th>replicates</th>
      <th>train_average</th>
      <th>train_median</th>
      <th>intensity_pred_vae</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MSE</th>
      <td>0.403</td>
      <td>0.435</td>
      <td>1.394</td>
      <td>1.668</td>
      <td>1.731</td>
      <td>1.772</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.395</td>
      <td>0.419</td>
      <td>0.814</td>
      <td>0.958</td>
      <td>0.942</td>
      <td>1.013</td>
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
