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
    SLAGSSGPGASSGTSGDHGELVVR     997
    LVSNHSLHETSSVFVDSLTK         984
    LFIGGLNTETNEK                993
    SNFAEALAAHK                  997
    DSIVHQAGMLK                  991
    AAYLQETGKPLDETLK             999
    SQIHDIVLVGGSTR             1,000
    VACIGAWHPAR                  961
    TFCQLILDPIFK                 959
    YDDMATCMK                    984
    MAPYQGPDAVPGALDYK          1,000
    TAVETAVLLLR                  987
    IITLTGPTNAIFK                963
    DNSTMGYMMAK                  937
    SLHDAIMIVR                   985
    ILATPPQEDAPSVDIANIR        1,000
    TNQELQEINR                   986
    EDQTEYLEER                   986
    TLGILGLGR                    963
    IFGVTTLDIVR                  951
    AVLVDLEPGTMDSVR              999
    ALDTMNFDVIK                  979
    SPYTVTVGQACNPSACR            953
    RAGELTEDEVER                 993
    TAFQEALDAAGDK              1,000
    GSGNLEAIHIIK                 977
    IVSRPEELREDDVGTGAGLLEIK      997
    GIPHLVTHDAR                1,000
    MLDAEDIVNTARPDEK             992
    ESSETPDQFMTADETR             969
    SDALETLGFLNHYQMK             999
    SDVLELTDDNFESR               983
    DYGNSPLHR                    983
    LSFQHDPETSVLVLR              993
    ILLAELEQLK                   992
    AVFVDLEPTVIDEVR              953
    KYEDICPSTHNMDVPNIK         1,000
    TISHVIIGLK                   993
    AVAQALEVIPR                  990
    HLAGLGLTEAIDK                998
    TFVNITPAEVGVLVGK             957
    SEIDLFNIRK                   951
    TELEDTLDSTAAQQELR            990
    GHFGPINSVAFHPDGK             992
    LSLEGDHSTPPSAYGSVK         1,000
    AGVNTVTTLVENKK               966
    VLGTSVESIMATEDR              990
    LSDGVAVLK                    989
    DILLRPELEELR                 992
    LSPPYSSPQEFAQDVGR            995
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
      <th>SLAGSSGPGASSGTSGDHGELVVR</th>
      <td>29.101</td>
    </tr>
    <tr>
      <th>LVSNHSLHETSSVFVDSLTK</th>
      <td>29.257</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>28.728</td>
    </tr>
    <tr>
      <th>SNFAEALAAHK</th>
      <td>29.618</td>
    </tr>
    <tr>
      <th>DSIVHQAGMLK</th>
      <td>26.859</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AGVNTVTTLVENKK</th>
      <td>30.173</td>
    </tr>
    <tr>
      <th>VLGTSVESIMATEDR</th>
      <td>27.540</td>
    </tr>
    <tr>
      <th>LSDGVAVLK</th>
      <td>31.702</td>
    </tr>
    <tr>
      <th>DILLRPELEELR</th>
      <td>27.801</td>
    </tr>
    <tr>
      <th>LSPPYSSPQEFAQDVGR</th>
      <td>26.405</td>
    </tr>
  </tbody>
</table>
<p>49188 rows × 1 columns</p>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_24_1.png)
    



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
      <td>0.026</td>
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
      <th>SLAGSSGPGASSGTSGDHGELVVR</th>
      <td>29.101</td>
    </tr>
    <tr>
      <th>LVSNHSLHETSSVFVDSLTK</th>
      <td>29.257</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>28.728</td>
    </tr>
    <tr>
      <th>SNFAEALAAHK</th>
      <td>29.618</td>
    </tr>
    <tr>
      <th>DSIVHQAGMLK</th>
      <td>26.859</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>AGVNTVTTLVENKK</th>
      <td>30.173</td>
    </tr>
    <tr>
      <th>VLGTSVESIMATEDR</th>
      <td>27.540</td>
    </tr>
    <tr>
      <th>LSDGVAVLK</th>
      <td>31.702</td>
    </tr>
    <tr>
      <th>DILLRPELEELR</th>
      <td>27.801</td>
    </tr>
    <tr>
      <th>LSPPYSSPQEFAQDVGR</th>
      <td>26.405</td>
    </tr>
  </tbody>
</table>
<p>49188 rows × 1 columns</p>
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
      <th>AAYLQETGKPLDETLK</th>
      <td>28.117</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>AAYLQETGKPLDETLK</th>
      <td>29.394</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_02</th>
      <th>AAYLQETGKPLDETLK</th>
      <td>25.263</td>
    </tr>
    <tr>
      <th>20190528_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>AAYLQETGKPLDETLK</th>
      <td>29.657</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>AAYLQETGKPLDETLK</th>
      <td>28.857</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190717_QX3_OzKa_MA_Hela_500ng_LC15_190720214645</th>
      <th>YDDMATCMK</th>
      <td>30.811</td>
    </tr>
    <tr>
      <th>20190219_QE10_nLC14_FaCo_QC_HeLa_50cm_20190219185517</th>
      <th>YDDMATCMK</th>
      <td>28.469</td>
    </tr>
    <tr>
      <th>20190204_QE9_nLC9_NHS_MNT_HELA_45cm_Newcolm_02</th>
      <th>YDDMATCMK</th>
      <td>27.720</td>
    </tr>
    <tr>
      <th>20190204_QE4_LC12_SCL_QC_MNT_HeLa_01</th>
      <th>YDDMATCMK</th>
      <td>28.590</td>
    </tr>
    <tr>
      <th>20190131_QE6_LC6_AS_MNT_HeLa_01</th>
      <th>YDDMATCMK</th>
      <td>27.818</td>
    </tr>
  </tbody>
</table>
<p>44270 rows × 1 columns</p>
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
    Shape in validation: (997, 50)
    




    ((997, 50), (997, 50))



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
      <th>AAYLQETGKPLDETLK</th>
      <td>28.703</td>
      <td>28.909</td>
      <td>29.133</td>
      <td>28.444</td>
    </tr>
    <tr>
      <th>AVFVDLEPTVIDEVR</th>
      <td>32.479</td>
      <td>32.781</td>
      <td>32.496</td>
      <td>32.449</td>
    </tr>
    <tr>
      <th>EDQTEYLEER</th>
      <td>30.817</td>
      <td>31.696</td>
      <td>31.780</td>
      <td>31.238</td>
    </tr>
    <tr>
      <th>SLHDAIMIVR</th>
      <td>27.467</td>
      <td>28.452</td>
      <td>28.578</td>
      <td>27.372</td>
    </tr>
    <tr>
      <th>VACIGAWHPAR</th>
      <td>28.940</td>
      <td>30.359</td>
      <td>30.262</td>
      <td>29.478</td>
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
      <th>DILLRPELEELR</th>
      <td>27.801</td>
      <td>27.788</td>
      <td>27.747</td>
      <td>28.212</td>
    </tr>
    <tr>
      <th>DYGNSPLHR</th>
      <td>28.299</td>
      <td>28.399</td>
      <td>28.869</td>
      <td>28.484</td>
    </tr>
    <tr>
      <th>IITLTGPTNAIFK</th>
      <td>29.647</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>29.786</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>29.555</td>
      <td>29.769</td>
      <td>29.673</td>
      <td>29.663</td>
    </tr>
    <tr>
      <th>SPYTVTVGQACNPSACR</th>
      <td>28.526</td>
      <td>28.393</td>
      <td>28.237</td>
      <td>28.659</td>
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
      <th rowspan="2" valign="top">20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-052</th>
      <th>ILLAELEQLK</th>
      <td>32.512</td>
      <td>32.279</td>
      <td>32.106</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SLAGSSGPGASSGTSGDHGELVVR</th>
      <td>30.218</td>
      <td>30.587</td>
      <td>30.745</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-053</th>
      <th>ILLAELEQLK</th>
      <td>32.572</td>
      <td>32.279</td>
      <td>32.106</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190107_QE5_nLC5_DS_QC_MNT_HeLa_FlashPack_03</th>
      <th>IITLTGPTNAIFK</th>
      <td>29.401</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190108_QE1_nLC2_MB_QC_MNT_HELA_old_01</th>
      <th>DYGNSPLHR</th>
      <td>28.052</td>
      <td>28.399</td>
      <td>28.869</td>
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
      <th>20190722_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_3rdcolumn_4</th>
      <th>SDVLELTDDNFESR</th>
      <td>29.654</td>
      <td>29.275</td>
      <td>29.288</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190722_QX3_MiWi_MA_Hela_500ng_LC15</th>
      <th>TFVNITPAEVGVLVGK</th>
      <td>33.549</td>
      <td>33.075</td>
      <td>32.797</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190729_QX3_MiWi_MA_Hela_500ng_LC15</th>
      <th>TLGILGLGR</th>
      <td>29.599</td>
      <td>28.820</td>
      <td>28.572</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>IITLTGPTNAIFK</th>
      <td>31.492</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QX8_AnPi_MA_HeLa_BR14_500ng</th>
      <th>SLHDAIMIVR</th>
      <td>30.376</td>
      <td>28.452</td>
      <td>28.578</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 4 columns</p>
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
      <td>AGVNTVTTLVENKK</td>
      <td>28.355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>ALDTMNFDVIK</td>
      <td>28.452</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVAQALEVIPR</td>
      <td>28.257</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVLVDLEPGTMDSVR</td>
      <td>30.155</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DILLRPELEELR</td>
      <td>27.630</td>
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
      <td>AAYLQETGKPLDETLK</td>
      <td>28.703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVFVDLEPTVIDEVR</td>
      <td>32.479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>EDQTEYLEER</td>
      <td>30.817</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>SLHDAIMIVR</td>
      <td>27.467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>VACIGAWHPAR</td>
      <td>28.940</td>
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
      <td>20190313_QE1_nLC2_GP_QC_MNT_HELA_02_20190315065501</td>
      <td>IVSRPEELREDDVGTGAGLLEIK</td>
      <td>29.975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190611_QE4_LC12_JE_QC_MNT_HeLa_02</td>
      <td>MLDAEDIVNTARPDEK</td>
      <td>30.878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190515_QE2_NLC1_GP_MNT_HELA_01</td>
      <td>SLHDAIMIVR</td>
      <td>27.207</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_02</td>
      <td>TLGILGLGR</td>
      <td>28.871</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>SNFAEALAAHK</td>
      <td>29.471</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190228_QE4_LC12_JE_QC_MNT_HeLa_02</td>
      <td>SDVLELTDDNFESR</td>
      <td>28.544</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190530_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</td>
      <td>LFIGGLNTETNEK</td>
      <td>31.401</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190312_QE8_nLC14_LiNi_QC_MNT_50cm_HELA_02</td>
      <td>TFCQLILDPIFK</td>
      <td>30.610</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190606_QE4_LC12_JE_QC_MNT_HeLa_01</td>
      <td>ILATPPQEDAPSVDIANIR</td>
      <td>30.635</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20181227_QE6_nLC6_CSC_QC_MNT_HeLa_01</td>
      <td>AVAQALEVIPR</td>
      <td>27.937</td>
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
      <td>20190131_QE6_LC6_AS_MNT_HeLa_02</td>
      <td>TLGILGLGR</td>
      <td>28.858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190626_QX6_ChDe_MA_HeLa_500ng_LC09</td>
      <td>SLHDAIMIVR</td>
      <td>31.050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190111_QE2_NLC10_ANHO_QC_MNT_HELA_01</td>
      <td>VLGTSVESIMATEDR</td>
      <td>27.758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190731_QX8_ChSc_MA_HeLa_500ng</td>
      <td>LFIGGLNTETNEK</td>
      <td>30.852</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190402_QE1_nLC2_GP_MNT_QC_hela_01</td>
      <td>TISHVIIGLK</td>
      <td>27.552</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190702_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1</td>
      <td>LSDGVAVLK</td>
      <td>31.737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190114_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02</td>
      <td>TNQELQEINR</td>
      <td>31.190</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190327_QE6_LC6_SCL_QC_MNT_Hela_01</td>
      <td>RAGELTEDEVER</td>
      <td>30.221</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190331_QE10_nLC13_LiNi_QC_45cm_HeLa_01_20190401142408</td>
      <td>TISHVIIGLK</td>
      <td>28.727</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190303_QE9_nLC0_FaCo_QC_MNT_Hela_50cm_newcol_20190304104835</td>
      <td>ILATPPQEDAPSVDIANIR</td>
      <td>29.567</td>
    </tr>
  </tbody>
</table>



```python
len(collab.dls.classes['Sample ID']), len(collab.dls.classes['peptide'])
```




    (998, 51)




```python
len(collab.dls.train), len(collab.dls.valid)  # mini-batches
```




    (1378, 154)



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
     'n_samples': 998,
     'y_range': (21, 36)}
    








    EmbeddingDotBias (Input shape: 32 x 2)
    ============================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ============================================================================
                         32 x 2              
    Embedding                                 1996       True      
    Embedding                                 102        True      
    ____________________________________________________________________________
                         32 x 1              
    Embedding                                 998        True      
    Embedding                                 51         True      
    ____________________________________________________________________________
    
    Total params: 3,147
    Total trainable params: 3,147
    Total non-trainable params: 0
    
    Optimizer used: <function Adam at 0x000001C51BF85040>
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
      <td>1.782035</td>
      <td>1.569148</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.638906</td>
      <td>0.634213</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.539817</td>
      <td>0.520697</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.488163</td>
      <td>0.498128</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.460502</td>
      <td>0.493117</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.534325</td>
      <td>0.484686</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.464376</td>
      <td>0.473312</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.457070</td>
      <td>0.466557</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.465899</td>
      <td>0.467532</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.465885</td>
      <td>0.467376</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_58_1.png)
    


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
      <th>488</th>
      <td>103</td>
      <td>46</td>
      <td>28.858</td>
    </tr>
    <tr>
      <th>3,678</th>
      <td>754</td>
      <td>36</td>
      <td>31.050</td>
    </tr>
    <tr>
      <th>206</th>
      <td>42</td>
      <td>49</td>
      <td>27.758</td>
    </tr>
    <tr>
      <th>4,753</th>
      <td>965</td>
      <td>23</td>
      <td>30.852</td>
    </tr>
    <tr>
      <th>1,669</th>
      <td>337</td>
      <td>45</td>
      <td>27.552</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>845</th>
      <td>171</td>
      <td>44</td>
      <td>31.769</td>
    </tr>
    <tr>
      <th>4,004</th>
      <td>815</td>
      <td>40</td>
      <td>33.270</td>
    </tr>
    <tr>
      <th>3,534</th>
      <td>725</td>
      <td>50</td>
      <td>28.106</td>
    </tr>
    <tr>
      <th>4,376</th>
      <td>890</td>
      <td>44</td>
      <td>31.764</td>
    </tr>
    <tr>
      <th>4,169</th>
      <td>849</td>
      <td>32</td>
      <td>31.903</td>
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
      <th>AAYLQETGKPLDETLK</th>
      <td>28.703</td>
      <td>28.909</td>
      <td>29.133</td>
      <td>28.444</td>
      <td>28.277</td>
    </tr>
    <tr>
      <th>AVFVDLEPTVIDEVR</th>
      <td>32.479</td>
      <td>32.781</td>
      <td>32.496</td>
      <td>32.449</td>
      <td>31.831</td>
    </tr>
    <tr>
      <th>EDQTEYLEER</th>
      <td>30.817</td>
      <td>31.696</td>
      <td>31.780</td>
      <td>31.238</td>
      <td>30.927</td>
    </tr>
    <tr>
      <th>SLHDAIMIVR</th>
      <td>27.467</td>
      <td>28.452</td>
      <td>28.578</td>
      <td>27.372</td>
      <td>27.545</td>
    </tr>
    <tr>
      <th>VACIGAWHPAR</th>
      <td>28.940</td>
      <td>30.359</td>
      <td>30.262</td>
      <td>29.478</td>
      <td>29.468</td>
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
      <th>DILLRPELEELR</th>
      <td>27.801</td>
      <td>27.788</td>
      <td>27.747</td>
      <td>28.212</td>
      <td>27.766</td>
    </tr>
    <tr>
      <th>DYGNSPLHR</th>
      <td>28.299</td>
      <td>28.399</td>
      <td>28.869</td>
      <td>28.484</td>
      <td>27.842</td>
    </tr>
    <tr>
      <th>IITLTGPTNAIFK</th>
      <td>29.647</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>29.786</td>
      <td>29.786</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>29.555</td>
      <td>29.769</td>
      <td>29.673</td>
      <td>29.663</td>
      <td>29.857</td>
    </tr>
    <tr>
      <th>SPYTVTVGQACNPSACR</th>
      <td>28.526</td>
      <td>28.393</td>
      <td>28.237</td>
      <td>28.659</td>
      <td>29.340</td>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.067
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.064
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.198
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.120
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.153
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_66_1.png)
    



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
      <td>-0.049</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.075</td>
      <td>0.047</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.081</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.093</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.048</td>
      <td>0.266</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_68_1.png)
    



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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_69_1.png)
    


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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR</th>
      <th>TELEDTLDSTAAQQELR</th>
      <th>TFCQLILDPIFK</th>
      <th>TFVNITPAEVGVLVGK</th>
      <th>TISHVIIGLK</th>
      <th>TLGILGLGR</th>
      <th>TNQELQEINR</th>
      <th>VACIGAWHPAR</th>
      <th>VLGTSVESIMATEDR</th>
      <th>YDDMATCMK</th>
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
      <td>28.703</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.479</td>
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
      <td>28.940</td>
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
      <td>30.140</td>
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
      <td>28.333</td>
      <td>31.753</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.566</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.815</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.876</td>
      <td>30.679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26.810</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>28.438</td>
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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR_na</th>
      <th>TELEDTLDSTAAQQELR_na</th>
      <th>TFCQLILDPIFK_na</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>TLGILGLGR_na</th>
      <th>TNQELQEINR_na</th>
      <th>VACIGAWHPAR_na</th>
      <th>VLGTSVESIMATEDR_na</th>
      <th>YDDMATCMK_na</th>
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
      <td>-0.133</td>
      <td>-0.983</td>
      <td>-0.851</td>
      <td>-0.243</td>
      <td>0.177</td>
      <td>-0.989</td>
      <td>-0.109</td>
      <td>-0.188</td>
      <td>-1.262</td>
      <td>-0.678</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.371</td>
      <td>0.273</td>
      <td>-0.817</td>
      <td>-0.233</td>
      <td>-0.046</td>
      <td>-1.269</td>
      <td>-0.149</td>
      <td>0.119</td>
      <td>-3.370</td>
      <td>-0.120</td>
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
      <td>-0.513</td>
      <td>0.188</td>
      <td>-0.829</td>
      <td>-0.576</td>
      <td>-0.080</td>
      <td>-0.397</td>
      <td>-0.955</td>
      <td>-0.868</td>
      <td>-0.326</td>
      <td>-0.843</td>
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
      <td>-0.509</td>
      <td>-0.972</td>
      <td>-0.680</td>
      <td>-0.300</td>
      <td>0.177</td>
      <td>-0.551</td>
      <td>0.042</td>
      <td>0.076</td>
      <td>-1.069</td>
      <td>-0.239</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.006</td>
      <td>-0.972</td>
      <td>-0.393</td>
      <td>0.126</td>
      <td>0.177</td>
      <td>-0.036</td>
      <td>0.361</td>
      <td>0.090</td>
      <td>0.122</td>
      <td>-0.364</td>
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
      <td>1.214</td>
      <td>-3.046</td>
      <td>0.617</td>
      <td>1.161</td>
      <td>1.238</td>
      <td>0.762</td>
      <td>0.436</td>
      <td>1.212</td>
      <td>0.122</td>
      <td>1.190</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.243</td>
      <td>0.315</td>
      <td>-0.443</td>
      <td>-0.828</td>
      <td>-0.316</td>
      <td>-0.377</td>
      <td>0.183</td>
      <td>-0.373</td>
      <td>0.663</td>
      <td>-0.215</td>
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
      <td>0.157</td>
      <td>-0.070</td>
      <td>-0.340</td>
      <td>-0.728</td>
      <td>0.095</td>
      <td>-0.295</td>
      <td>0.612</td>
      <td>-0.478</td>
      <td>0.431</td>
      <td>-0.167</td>
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
      <td>0.044</td>
      <td>0.712</td>
      <td>0.053</td>
      <td>-0.300</td>
      <td>-0.013</td>
      <td>-0.373</td>
      <td>0.209</td>
      <td>-0.515</td>
      <td>0.568</td>
      <td>-0.215</td>
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
      <td>0.188</td>
      <td>0.442</td>
      <td>0.087</td>
      <td>-0.300</td>
      <td>-0.019</td>
      <td>-0.463</td>
      <td>0.032</td>
      <td>-0.874</td>
      <td>0.364</td>
      <td>-0.239</td>
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
<p>997 rows × 100 columns</p>
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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR_na</th>
      <th>TELEDTLDSTAAQQELR_na</th>
      <th>TFCQLILDPIFK_na</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>TLGILGLGR_na</th>
      <th>TNQELQEINR_na</th>
      <th>VACIGAWHPAR_na</th>
      <th>VLGTSVESIMATEDR_na</th>
      <th>YDDMATCMK_na</th>
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
      <td>-0.141</td>
      <td>-0.883</td>
      <td>-0.799</td>
      <td>-0.265</td>
      <td>0.192</td>
      <td>-0.941</td>
      <td>-0.099</td>
      <td>-0.239</td>
      <td>-1.179</td>
      <td>-0.669</td>
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
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.367</td>
      <td>0.295</td>
      <td>-0.766</td>
      <td>-0.256</td>
      <td>-0.015</td>
      <td>-1.207</td>
      <td>-0.138</td>
      <td>0.046</td>
      <td>-3.171</td>
      <td>-0.143</td>
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
      <td>-0.501</td>
      <td>0.215</td>
      <td>-0.778</td>
      <td>-0.582</td>
      <td>-0.047</td>
      <td>-0.381</td>
      <td>-0.899</td>
      <td>-0.872</td>
      <td>-0.294</td>
      <td>-0.825</td>
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
      <td>-0.498</td>
      <td>-0.872</td>
      <td>-0.638</td>
      <td>-0.320</td>
      <td>0.192</td>
      <td>-0.526</td>
      <td>0.044</td>
      <td>0.007</td>
      <td>-0.996</td>
      <td>-0.255</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.009</td>
      <td>-0.872</td>
      <td>-0.368</td>
      <td>0.085</td>
      <td>0.192</td>
      <td>-0.037</td>
      <td>0.345</td>
      <td>0.020</td>
      <td>0.129</td>
      <td>-0.373</td>
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
      <td>1.138</td>
      <td>-2.816</td>
      <td>0.579</td>
      <td>1.067</td>
      <td>1.176</td>
      <td>0.719</td>
      <td>0.416</td>
      <td>1.063</td>
      <td>0.129</td>
      <td>1.094</td>
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
      <td>True</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.245</td>
      <td>0.334</td>
      <td>-0.415</td>
      <td>-0.820</td>
      <td>-0.266</td>
      <td>-0.361</td>
      <td>0.176</td>
      <td>-0.412</td>
      <td>0.641</td>
      <td>-0.232</td>
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
      <td>0.135</td>
      <td>-0.027</td>
      <td>-0.318</td>
      <td>-0.726</td>
      <td>0.115</td>
      <td>-0.284</td>
      <td>0.582</td>
      <td>-0.509</td>
      <td>0.421</td>
      <td>-0.187</td>
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
      <td>0.028</td>
      <td>0.706</td>
      <td>0.051</td>
      <td>-0.320</td>
      <td>0.015</td>
      <td>-0.357</td>
      <td>0.202</td>
      <td>-0.544</td>
      <td>0.551</td>
      <td>-0.232</td>
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
      <td>0.164</td>
      <td>0.453</td>
      <td>0.082</td>
      <td>-0.320</td>
      <td>0.010</td>
      <td>-0.443</td>
      <td>0.034</td>
      <td>-0.878</td>
      <td>0.358</td>
      <td>-0.255</td>
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
<p>997 rows × 100 columns</p>
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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
      <td>997.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.014</td>
      <td>0.039</td>
      <td>0.000</td>
      <td>-0.035</td>
      <td>0.027</td>
      <td>-0.004</td>
      <td>0.004</td>
      <td>-0.064</td>
      <td>0.014</td>
      <td>-0.029</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.949</td>
      <td>0.938</td>
      <td>0.939</td>
      <td>0.949</td>
      <td>0.929</td>
      <td>0.949</td>
      <td>0.945</td>
      <td>0.931</td>
      <td>0.946</td>
      <td>0.945</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.738</td>
      <td>-4.901</td>
      <td>-4.602</td>
      <td>-3.254</td>
      <td>-5.119</td>
      <td>-5.169</td>
      <td>-5.141</td>
      <td>-2.509</td>
      <td>-4.583</td>
      <td>-3.229</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.465</td>
      <td>-0.100</td>
      <td>-0.423</td>
      <td>-0.632</td>
      <td>-0.298</td>
      <td>-0.442</td>
      <td>-0.390</td>
      <td>-0.658</td>
      <td>-0.362</td>
      <td>-0.622</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.141</td>
      <td>0.295</td>
      <td>0.003</td>
      <td>-0.320</td>
      <td>0.192</td>
      <td>-0.037</td>
      <td>0.034</td>
      <td>-0.412</td>
      <td>0.129</td>
      <td>-0.255</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.454</td>
      <td>0.576</td>
      <td>0.427</td>
      <td>0.883</td>
      <td>0.539</td>
      <td>0.492</td>
      <td>0.499</td>
      <td>0.677</td>
      <td>0.604</td>
      <td>0.784</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.085</td>
      <td>1.624</td>
      <td>1.941</td>
      <td>2.086</td>
      <td>1.758</td>
      <td>1.967</td>
      <td>2.723</td>
      <td>1.903</td>
      <td>1.959</td>
      <td>2.106</td>
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




    ((#50) ['AAYLQETGKPLDETLK','AGVNTVTTLVENKK','ALDTMNFDVIK','AVAQALEVIPR','AVFVDLEPTVIDEVR','AVLVDLEPGTMDSVR','DILLRPELEELR','DNSTMGYMMAK','DSIVHQAGMLK','DYGNSPLHR'...],
     (#50) ['AAYLQETGKPLDETLK_na','AGVNTVTTLVENKK_na','ALDTMNFDVIK_na','AVAQALEVIPR_na','AVFVDLEPTVIDEVR_na','AVLVDLEPGTMDSVR_na','DILLRPELEELR_na','DNSTMGYMMAK_na','DSIVHQAGMLK_na','DYGNSPLHR_na'...])




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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>94.000</td>
      <td>99.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.122</td>
      <td>0.055</td>
      <td>0.016</td>
      <td>-0.054</td>
      <td>0.041</td>
      <td>0.122</td>
      <td>0.129</td>
      <td>0.103</td>
      <td>-0.071</td>
      <td>-0.082</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.053</td>
      <td>0.859</td>
      <td>0.937</td>
      <td>0.927</td>
      <td>0.750</td>
      <td>0.725</td>
      <td>0.901</td>
      <td>0.959</td>
      <td>1.007</td>
      <td>0.865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.930</td>
      <td>-3.385</td>
      <td>-2.696</td>
      <td>-1.418</td>
      <td>-2.291</td>
      <td>-1.538</td>
      <td>-3.025</td>
      <td>-1.802</td>
      <td>-2.849</td>
      <td>-1.759</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.654</td>
      <td>-0.185</td>
      <td>-0.516</td>
      <td>-0.761</td>
      <td>-0.446</td>
      <td>-0.374</td>
      <td>-0.354</td>
      <td>-0.624</td>
      <td>-0.770</td>
      <td>-0.575</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.193</td>
      <td>0.195</td>
      <td>-0.010</td>
      <td>-0.387</td>
      <td>0.168</td>
      <td>0.007</td>
      <td>0.052</td>
      <td>-0.279</td>
      <td>0.118</td>
      <td>-0.322</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.225</td>
      <td>0.574</td>
      <td>0.601</td>
      <td>0.934</td>
      <td>0.537</td>
      <td>0.506</td>
      <td>0.735</td>
      <td>1.091</td>
      <td>0.666</td>
      <td>0.244</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.913</td>
      <td>1.260</td>
      <td>1.706</td>
      <td>1.682</td>
      <td>1.425</td>
      <td>1.807</td>
      <td>1.848</td>
      <td>1.777</td>
      <td>1.782</td>
      <td>1.710</td>
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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR_val</th>
      <th>TELEDTLDSTAAQQELR_val</th>
      <th>TFCQLILDPIFK_val</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>TLGILGLGR_val</th>
      <th>TNQELQEINR_val</th>
      <th>VACIGAWHPAR_val</th>
      <th>VLGTSVESIMATEDR_val</th>
      <th>YDDMATCMK_val</th>
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
      <td>-0.141</td>
      <td>-0.883</td>
      <td>-0.799</td>
      <td>-0.265</td>
      <td>0.192</td>
      <td>-0.941</td>
      <td>-0.099</td>
      <td>-0.239</td>
      <td>-1.179</td>
      <td>-0.669</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.036</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.367</td>
      <td>0.295</td>
      <td>-0.766</td>
      <td>-0.256</td>
      <td>-0.015</td>
      <td>-1.207</td>
      <td>-0.138</td>
      <td>0.046</td>
      <td>-3.171</td>
      <td>-0.143</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.501</td>
      <td>0.215</td>
      <td>-0.778</td>
      <td>-0.582</td>
      <td>-0.047</td>
      <td>-0.381</td>
      <td>-0.899</td>
      <td>-0.872</td>
      <td>-0.294</td>
      <td>-0.825</td>
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
      <td>-0.498</td>
      <td>-0.872</td>
      <td>-0.638</td>
      <td>-0.320</td>
      <td>0.192</td>
      <td>-0.526</td>
      <td>0.044</td>
      <td>0.007</td>
      <td>-0.996</td>
      <td>-0.255</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.453</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.009</td>
      <td>-0.872</td>
      <td>-0.368</td>
      <td>0.085</td>
      <td>0.192</td>
      <td>-0.037</td>
      <td>0.345</td>
      <td>0.020</td>
      <td>0.129</td>
      <td>-0.373</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.347</td>
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
      <td>1.138</td>
      <td>-2.816</td>
      <td>0.579</td>
      <td>1.067</td>
      <td>1.176</td>
      <td>0.719</td>
      <td>0.416</td>
      <td>1.063</td>
      <td>0.129</td>
      <td>1.094</td>
      <td>...</td>
      <td>0.955</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.964</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.245</td>
      <td>0.334</td>
      <td>-0.415</td>
      <td>-0.820</td>
      <td>-0.266</td>
      <td>-0.361</td>
      <td>0.176</td>
      <td>-0.412</td>
      <td>0.641</td>
      <td>-0.232</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.144</td>
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
      <td>0.135</td>
      <td>-0.027</td>
      <td>-0.318</td>
      <td>-0.726</td>
      <td>0.115</td>
      <td>-0.284</td>
      <td>0.582</td>
      <td>-0.509</td>
      <td>0.421</td>
      <td>-0.187</td>
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
      <td>0.028</td>
      <td>0.706</td>
      <td>0.051</td>
      <td>-0.320</td>
      <td>0.015</td>
      <td>-0.357</td>
      <td>0.202</td>
      <td>-0.544</td>
      <td>0.551</td>
      <td>-0.232</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.299</td>
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
      <td>0.164</td>
      <td>0.453</td>
      <td>0.082</td>
      <td>-0.320</td>
      <td>0.010</td>
      <td>-0.443</td>
      <td>0.034</td>
      <td>-0.878</td>
      <td>0.358</td>
      <td>-0.255</td>
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
<p>997 rows × 150 columns</p>
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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR_val</th>
      <th>TELEDTLDSTAAQQELR_val</th>
      <th>TFCQLILDPIFK_val</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>TLGILGLGR_val</th>
      <th>TNQELQEINR_val</th>
      <th>VACIGAWHPAR_val</th>
      <th>VLGTSVESIMATEDR_val</th>
      <th>YDDMATCMK_val</th>
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
      <td>-0.141</td>
      <td>-0.883</td>
      <td>-0.799</td>
      <td>-0.265</td>
      <td>0.192</td>
      <td>-0.941</td>
      <td>-0.099</td>
      <td>-0.239</td>
      <td>-1.179</td>
      <td>-0.669</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.036</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.367</td>
      <td>0.295</td>
      <td>-0.766</td>
      <td>-0.256</td>
      <td>-0.015</td>
      <td>-1.207</td>
      <td>-0.138</td>
      <td>0.046</td>
      <td>-3.171</td>
      <td>-0.143</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.501</td>
      <td>0.215</td>
      <td>-0.778</td>
      <td>-0.582</td>
      <td>-0.047</td>
      <td>-0.381</td>
      <td>-0.899</td>
      <td>-0.872</td>
      <td>-0.294</td>
      <td>-0.825</td>
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
      <td>-0.498</td>
      <td>-0.872</td>
      <td>-0.638</td>
      <td>-0.320</td>
      <td>0.192</td>
      <td>-0.526</td>
      <td>0.044</td>
      <td>0.007</td>
      <td>-0.996</td>
      <td>-0.255</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.453</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.009</td>
      <td>-0.872</td>
      <td>-0.368</td>
      <td>0.085</td>
      <td>0.192</td>
      <td>-0.037</td>
      <td>0.345</td>
      <td>0.020</td>
      <td>0.129</td>
      <td>-0.373</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.347</td>
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
      <td>1.138</td>
      <td>-2.816</td>
      <td>0.579</td>
      <td>1.067</td>
      <td>1.176</td>
      <td>0.719</td>
      <td>0.416</td>
      <td>1.063</td>
      <td>0.129</td>
      <td>1.094</td>
      <td>...</td>
      <td>0.955</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.964</td>
    </tr>
    <tr>
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.245</td>
      <td>0.334</td>
      <td>-0.415</td>
      <td>-0.820</td>
      <td>-0.266</td>
      <td>-0.361</td>
      <td>0.176</td>
      <td>-0.412</td>
      <td>0.641</td>
      <td>-0.232</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.144</td>
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
      <td>0.135</td>
      <td>-0.027</td>
      <td>-0.318</td>
      <td>-0.726</td>
      <td>0.115</td>
      <td>-0.284</td>
      <td>0.582</td>
      <td>-0.509</td>
      <td>0.421</td>
      <td>-0.187</td>
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
      <td>0.028</td>
      <td>0.706</td>
      <td>0.051</td>
      <td>-0.320</td>
      <td>0.015</td>
      <td>-0.357</td>
      <td>0.202</td>
      <td>-0.544</td>
      <td>0.551</td>
      <td>-0.232</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.299</td>
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
      <td>0.164</td>
      <td>0.453</td>
      <td>0.082</td>
      <td>-0.320</td>
      <td>0.010</td>
      <td>-0.443</td>
      <td>0.034</td>
      <td>-0.878</td>
      <td>0.358</td>
      <td>-0.255</td>
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
<p>997 rows × 150 columns</p>
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
      <th>AAYLQETGKPLDETLK_val</th>
      <th>AGVNTVTTLVENKK_val</th>
      <th>ALDTMNFDVIK_val</th>
      <th>AVAQALEVIPR_val</th>
      <th>AVFVDLEPTVIDEVR_val</th>
      <th>AVLVDLEPGTMDSVR_val</th>
      <th>DILLRPELEELR_val</th>
      <th>DNSTMGYMMAK_val</th>
      <th>DSIVHQAGMLK_val</th>
      <th>DYGNSPLHR_val</th>
      <th>...</th>
      <th>TAVETAVLLLR_val</th>
      <th>TELEDTLDSTAAQQELR_val</th>
      <th>TFCQLILDPIFK_val</th>
      <th>TFVNITPAEVGVLVGK_val</th>
      <th>TISHVIIGLK_val</th>
      <th>TLGILGLGR_val</th>
      <th>TNQELQEINR_val</th>
      <th>VACIGAWHPAR_val</th>
      <th>VLGTSVESIMATEDR_val</th>
      <th>YDDMATCMK_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>97.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>95.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>94.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>...</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>96.000</td>
      <td>96.000</td>
      <td>99.000</td>
      <td>96.000</td>
      <td>99.000</td>
      <td>96.000</td>
      <td>99.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.122</td>
      <td>0.055</td>
      <td>0.016</td>
      <td>-0.054</td>
      <td>0.041</td>
      <td>0.122</td>
      <td>0.129</td>
      <td>0.103</td>
      <td>-0.071</td>
      <td>-0.082</td>
      <td>...</td>
      <td>0.047</td>
      <td>0.077</td>
      <td>0.020</td>
      <td>-0.157</td>
      <td>-0.124</td>
      <td>0.060</td>
      <td>0.006</td>
      <td>0.124</td>
      <td>-0.013</td>
      <td>-0.208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.053</td>
      <td>0.859</td>
      <td>0.937</td>
      <td>0.927</td>
      <td>0.750</td>
      <td>0.725</td>
      <td>0.901</td>
      <td>0.959</td>
      <td>1.007</td>
      <td>0.865</td>
      <td>...</td>
      <td>0.855</td>
      <td>0.880</td>
      <td>0.984</td>
      <td>1.197</td>
      <td>1.173</td>
      <td>0.900</td>
      <td>0.995</td>
      <td>0.878</td>
      <td>0.968</td>
      <td>1.043</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.930</td>
      <td>-3.385</td>
      <td>-2.696</td>
      <td>-1.418</td>
      <td>-2.291</td>
      <td>-1.538</td>
      <td>-3.025</td>
      <td>-1.802</td>
      <td>-2.849</td>
      <td>-1.759</td>
      <td>...</td>
      <td>-2.504</td>
      <td>-3.447</td>
      <td>-3.175</td>
      <td>-5.266</td>
      <td>-4.158</td>
      <td>-3.545</td>
      <td>-2.378</td>
      <td>-4.130</td>
      <td>-2.534</td>
      <td>-3.840</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.654</td>
      <td>-0.185</td>
      <td>-0.516</td>
      <td>-0.761</td>
      <td>-0.446</td>
      <td>-0.374</td>
      <td>-0.354</td>
      <td>-0.624</td>
      <td>-0.770</td>
      <td>-0.575</td>
      <td>...</td>
      <td>-0.501</td>
      <td>-0.313</td>
      <td>-0.320</td>
      <td>-0.539</td>
      <td>-0.662</td>
      <td>-0.123</td>
      <td>-0.699</td>
      <td>-0.343</td>
      <td>-0.622</td>
      <td>-0.809</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.193</td>
      <td>0.195</td>
      <td>-0.010</td>
      <td>-0.387</td>
      <td>0.168</td>
      <td>0.007</td>
      <td>0.052</td>
      <td>-0.279</td>
      <td>0.118</td>
      <td>-0.322</td>
      <td>...</td>
      <td>-0.132</td>
      <td>0.322</td>
      <td>0.226</td>
      <td>0.142</td>
      <td>-0.118</td>
      <td>0.251</td>
      <td>-0.180</td>
      <td>0.044</td>
      <td>-0.243</td>
      <td>-0.237</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.225</td>
      <td>0.574</td>
      <td>0.601</td>
      <td>0.934</td>
      <td>0.537</td>
      <td>0.506</td>
      <td>0.735</td>
      <td>1.091</td>
      <td>0.666</td>
      <td>0.244</td>
      <td>...</td>
      <td>0.454</td>
      <td>0.692</td>
      <td>0.588</td>
      <td>0.606</td>
      <td>0.533</td>
      <td>0.601</td>
      <td>0.852</td>
      <td>0.662</td>
      <td>0.718</td>
      <td>0.358</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.913</td>
      <td>1.260</td>
      <td>1.706</td>
      <td>1.682</td>
      <td>1.425</td>
      <td>1.807</td>
      <td>1.848</td>
      <td>1.777</td>
      <td>1.782</td>
      <td>1.710</td>
      <td>...</td>
      <td>1.900</td>
      <td>1.629</td>
      <td>1.517</td>
      <td>1.143</td>
      <td>2.830</td>
      <td>1.495</td>
      <td>1.941</td>
      <td>1.980</td>
      <td>1.885</td>
      <td>1.800</td>
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
      <th>AAYLQETGKPLDETLK_na</th>
      <th>AGVNTVTTLVENKK_na</th>
      <th>ALDTMNFDVIK_na</th>
      <th>AVAQALEVIPR_na</th>
      <th>AVFVDLEPTVIDEVR_na</th>
      <th>AVLVDLEPGTMDSVR_na</th>
      <th>DILLRPELEELR_na</th>
      <th>DNSTMGYMMAK_na</th>
      <th>DSIVHQAGMLK_na</th>
      <th>DYGNSPLHR_na</th>
      <th>...</th>
      <th>TAVETAVLLLR_na</th>
      <th>TELEDTLDSTAAQQELR_na</th>
      <th>TFCQLILDPIFK_na</th>
      <th>TFVNITPAEVGVLVGK_na</th>
      <th>TISHVIIGLK_na</th>
      <th>TLGILGLGR_na</th>
      <th>TNQELQEINR_na</th>
      <th>VACIGAWHPAR_na</th>
      <th>VLGTSVESIMATEDR_na</th>
      <th>YDDMATCMK_na</th>
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
      <td>False</td>
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
      <td>False</td>
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
<p>997 rows × 50 columns</p>
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
    
    Optimizer used: <function Adam at 0x000001C51BF85040>
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




    
![png](latent_2D_500_30_files/latent_2D_500_30_108_2.png)
    


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
      <td>0.952616</td>
      <td>0.716246</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.623235</td>
      <td>0.341137</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.460872</td>
      <td>0.317283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.383326</td>
      <td>0.315230</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.347163</td>
      <td>0.298944</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.320433</td>
      <td>0.286585</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.303074</td>
      <td>0.279390</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.287768</td>
      <td>0.278947</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.278835</td>
      <td>0.278753</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.272542</td>
      <td>0.274232</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_112_1.png)
    



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




    TensorBase(0.2765)




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
      <th>AAYLQETGKPLDETLK</th>
      <td>28.703</td>
      <td>28.909</td>
      <td>29.133</td>
      <td>28.444</td>
      <td>28.277</td>
      <td>27.823</td>
    </tr>
    <tr>
      <th>AVFVDLEPTVIDEVR</th>
      <td>32.479</td>
      <td>32.781</td>
      <td>32.496</td>
      <td>32.449</td>
      <td>31.831</td>
      <td>32.018</td>
    </tr>
    <tr>
      <th>EDQTEYLEER</th>
      <td>30.817</td>
      <td>31.696</td>
      <td>31.780</td>
      <td>31.238</td>
      <td>30.927</td>
      <td>30.616</td>
    </tr>
    <tr>
      <th>SLHDAIMIVR</th>
      <td>27.467</td>
      <td>28.452</td>
      <td>28.578</td>
      <td>27.372</td>
      <td>27.545</td>
      <td>27.135</td>
    </tr>
    <tr>
      <th>VACIGAWHPAR</th>
      <td>28.940</td>
      <td>30.359</td>
      <td>30.262</td>
      <td>29.478</td>
      <td>29.468</td>
      <td>29.242</td>
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
      <th>DILLRPELEELR</th>
      <td>27.801</td>
      <td>27.788</td>
      <td>27.747</td>
      <td>28.212</td>
      <td>27.766</td>
      <td>27.798</td>
    </tr>
    <tr>
      <th>DYGNSPLHR</th>
      <td>28.299</td>
      <td>28.399</td>
      <td>28.869</td>
      <td>28.484</td>
      <td>27.842</td>
      <td>27.984</td>
    </tr>
    <tr>
      <th>IITLTGPTNAIFK</th>
      <td>29.647</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>29.786</td>
      <td>29.786</td>
      <td>29.783</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>29.555</td>
      <td>29.769</td>
      <td>29.673</td>
      <td>29.663</td>
      <td>29.857</td>
      <td>29.759</td>
    </tr>
    <tr>
      <th>SPYTVTVGQACNPSACR</th>
      <td>28.526</td>
      <td>28.393</td>
      <td>28.237</td>
      <td>28.659</td>
      <td>29.340</td>
      <td>28.973</td>
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
      <td>-0.879</td>
      <td>0.626</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.850</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.847</td>
      <td>0.843</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.789</td>
      <td>0.682</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.824</td>
      <td>0.812</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_122_1.png)
    



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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_123_1.png)
    


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
      <th>AAYLQETGKPLDETLK</th>
      <th>AGVNTVTTLVENKK</th>
      <th>ALDTMNFDVIK</th>
      <th>AVAQALEVIPR</th>
      <th>AVFVDLEPTVIDEVR</th>
      <th>AVLVDLEPGTMDSVR</th>
      <th>DILLRPELEELR</th>
      <th>DNSTMGYMMAK</th>
      <th>DSIVHQAGMLK</th>
      <th>DYGNSPLHR</th>
      <th>...</th>
      <th>TAVETAVLLLR</th>
      <th>TELEDTLDSTAAQQELR</th>
      <th>TFCQLILDPIFK</th>
      <th>TFVNITPAEVGVLVGK</th>
      <th>TISHVIIGLK</th>
      <th>TLGILGLGR</th>
      <th>TNQELQEINR</th>
      <th>VACIGAWHPAR</th>
      <th>VLGTSVESIMATEDR</th>
      <th>YDDMATCMK</th>
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
      <td>0.596</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.743</td>
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
      <td>0.537</td>
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
      <td>0.660</td>
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
      <td>0.567</td>
      <td>0.672</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.702</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.580</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.879</td>
      <td>0.650</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.514</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.710</td>
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
    
    Optimizer used: <function Adam at 0x000001C51BF85040>
    Loss function: <function loss_fct_vae at 0x000001C51BFAD940>
    
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




    
![png](latent_2D_500_30_files/latent_2D_500_30_136_2.png)
    



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
      <td>1945.383545</td>
      <td>213.703842</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1911.575806</td>
      <td>203.301544</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1857.726562</td>
      <td>196.853546</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1819.366455</td>
      <td>194.975327</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1795.157104</td>
      <td>194.758865</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1778.124878</td>
      <td>196.234421</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1767.396240</td>
      <td>196.446121</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1759.447632</td>
      <td>196.550201</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1754.679932</td>
      <td>196.787064</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1750.562622</td>
      <td>196.790955</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_138_1.png)
    


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




    tensor(3106.3735)




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
      <th>AAYLQETGKPLDETLK</th>
      <td>28.703</td>
      <td>28.909</td>
      <td>29.133</td>
      <td>28.444</td>
      <td>28.277</td>
      <td>27.823</td>
      <td>29.216</td>
    </tr>
    <tr>
      <th>AVFVDLEPTVIDEVR</th>
      <td>32.479</td>
      <td>32.781</td>
      <td>32.496</td>
      <td>32.449</td>
      <td>31.831</td>
      <td>32.018</td>
      <td>32.611</td>
    </tr>
    <tr>
      <th>EDQTEYLEER</th>
      <td>30.817</td>
      <td>31.696</td>
      <td>31.780</td>
      <td>31.238</td>
      <td>30.927</td>
      <td>30.616</td>
      <td>31.929</td>
    </tr>
    <tr>
      <th>SLHDAIMIVR</th>
      <td>27.467</td>
      <td>28.452</td>
      <td>28.578</td>
      <td>27.372</td>
      <td>27.545</td>
      <td>27.135</td>
      <td>28.663</td>
    </tr>
    <tr>
      <th>VACIGAWHPAR</th>
      <td>28.940</td>
      <td>30.359</td>
      <td>30.262</td>
      <td>29.478</td>
      <td>29.468</td>
      <td>29.242</td>
      <td>30.433</td>
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
      <th>DILLRPELEELR</th>
      <td>27.801</td>
      <td>27.788</td>
      <td>27.747</td>
      <td>28.212</td>
      <td>27.766</td>
      <td>27.798</td>
      <td>27.711</td>
    </tr>
    <tr>
      <th>DYGNSPLHR</th>
      <td>28.299</td>
      <td>28.399</td>
      <td>28.869</td>
      <td>28.484</td>
      <td>27.842</td>
      <td>27.984</td>
      <td>28.776</td>
    </tr>
    <tr>
      <th>IITLTGPTNAIFK</th>
      <td>29.647</td>
      <td>29.899</td>
      <td>30.090</td>
      <td>29.786</td>
      <td>29.786</td>
      <td>29.783</td>
      <td>30.016</td>
    </tr>
    <tr>
      <th>LFIGGLNTETNEK</th>
      <td>29.555</td>
      <td>29.769</td>
      <td>29.673</td>
      <td>29.663</td>
      <td>29.857</td>
      <td>29.759</td>
      <td>29.683</td>
    </tr>
    <tr>
      <th>SPYTVTVGQACNPSACR</th>
      <td>28.526</td>
      <td>28.393</td>
      <td>28.237</td>
      <td>28.659</td>
      <td>29.340</td>
      <td>28.973</td>
      <td>28.317</td>
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
      <td>-0.025</td>
      <td>-0.094</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.102</td>
      <td>-0.154</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.197</td>
      <td>-0.257</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>-0.047</td>
      <td>-0.093</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.002</td>
      <td>0.014</td>
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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_146_1.png)
    



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
    


    
![png](latent_2D_500_30_files/latent_2D_500_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-052', 'ILLAELEQLK'), ('20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-052', 'SLAGSSGPGASSGTSGDHGELVVR'), ('20181229_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-053', 'ILLAELEQLK'), ('20190107_QE5_nLC5_DS_QC_MNT_HeLa_FlashPack_03', 'IITLTGPTNAIFK'), ('20190108_QE1_nLC2_MB_QC_MNT_HELA_old_01', 'DYGNSPLHR'), ('20190118_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'KYEDICPSTHNMDVPNIK'), ('20190118_QE9_nLC9_NHS_MNT_HELA_50cm_03', 'KYEDICPSTHNMDVPNIK'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_02', 'SLHDAIMIVR'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'SLHDAIMIVR'), ('20190129_QE10_nLC0_FM_QC_MNT_HeLa_50cm_01', 'GHFGPINSVAFHPDGK'), ('20190201_QE1_nLC2_GP_QC_MNT_HELA_01', 'HLAGLGLTEAIDK'), ('20190204_QE6_nLC6_MPL_QC_MNT_HeLa_04', 'GIPHLVTHDAR'), ('20190206_QE8_nLC0_ASD_QC_HeLa_50cm_20190206192638', 'LSFQHDPETSVLVLR'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm3', 'GHFGPINSVAFHPDGK'), ('20190207_QE8_nLC0_ASD_QC_HeLa_43cm4', 'GHFGPINSVAFHPDGK'), ('20190213_QE10_nLC0_KS_QC_MNT_HeLa_02', 'LFIGGLNTETNEK'), ('20190213_QE3_nLC3_UH_QC_MNT_HeLa_01', 'LFIGGLNTETNEK'), ('20190218_QE6_LC6_SCL_MVM_QC_MNT_Hela_03', 'IITLTGPTNAIFK'), ('20190222_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'TFCQLILDPIFK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_5', 'TELEDTLDSTAAQQELR'), ('20190226_QE10_PhGe_Evosep_88min-30cmCol-HeLa_14_27', 'LFIGGLNTETNEK'), ('20190303_QE9_nLC0_FaCo_QC_MNT_Hela_50cm_newcol_20190304104835', 'IFGVTTLDIVR'), ('20190308_QE3_nLC5_MR_QC_MNT_HeLa_CPRHEasyctcdon_04', 'SEIDLFNIRK'), ('20190308_QE9_nLC0_FaCo_QC_MNT_Hela_50cm', 'GSGNLEAIHIIK'), ('20190312_QE8_nLC14_LiNi_QC_MNT_50cm_HELA_02', 'ESSETPDQFMTADETR'), ('20190313_QE1_nLC2_GP_QC_MNT_HELA_02_20190318000412', 'VLGTSVESIMATEDR'), ('20190318_QE2_NLC1_AB_MNT_HELA_01', 'LSLEGDHSTPPSAYGSVK'), ('20190318_QE2_NLC1_AB_MNT_HELA_04', 'AAYLQETGKPLDETLK'), ('20190318_QE2_NLC1_AB_MNT_HELA_04', 'LSDGVAVLK'), ('20190326_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'TELEDTLDSTAAQQELR'), ('20190330_QE1_nLC2_GP_MNT_QC_hela_01', 'TFVNITPAEVGVLVGK'), ('20190402_QE6_LC6_AS_QC_MNT_HeLa_01', 'YDDMATCMK'), ('20190415_QE8_nLC14_AL_QC_MNT_HeLa_01', 'AVLVDLEPGTMDSVR'), ('20190415_QE8_nLC14_AL_QC_MNT_HeLa_01', 'TLGILGLGR'), ('20190422_QX3_MaTa_MA_Br14_Hela_500ng_LC15', 'VACIGAWHPAR'), ('20190423_QE8_nLC14_AGF_QC_MNT_HeLa_01_20190423191324', 'IFGVTTLDIVR'), ('20190424_QE2_NLC1_ANHO_MNT_HELA_01', 'SEIDLFNIRK'), ('20190425_QX7_ChDe_MA_HeLaBr14_500ng_LC02', 'TNQELQEINR'), ('20190429_QX3_ChDe_MA_Hela_500ng_LC15_190429151336', 'SPYTVTVGQACNPSACR'), ('20190506_QX4_JiYu_MA_HeLa_500ng_BR13_standard', 'GSGNLEAIHIIK'), ('20190506_QX6_ChDe_MA_HeLa_Br13_500ng_LC09', 'HLAGLGLTEAIDK'), ('20190506_QX8_MiWi_MA_HeLa_500ng_new', 'SEIDLFNIRK'), ('20190507_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_01', 'SEIDLFNIRK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'TFVNITPAEVGVLVGK'), ('20190510_QE1_nLC2_ANHO_QC_MNT_HELA_04', 'SNFAEALAAHK'), ('20190513_QE6_LC4_IAH_QC_MNT_HeLa_01', 'DYGNSPLHR'), ('20190515_QE2_NLC1_GP_MNT_HELA_01', 'LSPPYSSPQEFAQDVGR'), ('20190519_QE10_nLC13_LiNi_QC_MNT_15cm_HeLa_01', 'SNFAEALAAHK'), ('20190605_QX0_MePh_MA_HeLa_500ng_LC07_1_BR14', 'AVLVDLEPGTMDSVR'), ('20190606_QE4_LC12_JE_QC_MNT_HeLa_02b', 'AGVNTVTTLVENKK'), ('20190610_QX1_JoMu_MA_HeLa_DMSO_500ng_LC14', 'TFCQLILDPIFK'), ('20190611_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1', 'DSIVHQAGMLK'), ('20190611_QX3_LiSc_MA_Hela_500ng_LC15', 'DSIVHQAGMLK'), ('20190611_QX3_LiSc_MA_Hela_500ng_LC15', 'IVSRPEELREDDVGTGAGLLEIK'), ('20190614_QX3_JoSw_MA_Hela_500ng_LC15', 'AVFVDLEPTVIDEVR'), ('20190617_QE_LC_UHG_QC_MNT_HELA_03', 'RAGELTEDEVER'), ('20190621_QX3_MePh_MA_Hela_500ng_LC15_190621150413', 'ESSETPDQFMTADETR'), ('20190624_QE10_nLC14_LiNi_QC_MNT_15cm_HeLa_CPH_01', 'GHFGPINSVAFHPDGK'), ('20190624_QE4_nLC12_MM_QC_MNT_HELA_01', 'GHFGPINSVAFHPDGK'), ('20190626_QX6_ChDe_MA_HeLa_500ng_LC09', 'HLAGLGLTEAIDK'), ('20190629_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'DSIVHQAGMLK'), ('20190701_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'ILLAELEQLK'), ('20190701_QX8_AnPi_MA_HeLa_BR14_500ng', 'ILATPPQEDAPSVDIANIR'), ('20190702_QE3_nLC5_GF_QC_MNT_Hela_03', 'VACIGAWHPAR'), ('20190702_QE3_nLC5_TSB_QC_MNT_HELA_02', 'SDVLELTDDNFESR'), ('20190702_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'IFGVTTLDIVR'), ('20190709_QX1_JoMu_MA_HeLa_500ng_LC10', 'SLAGSSGPGASSGTSGDHGELVVR'), ('20190709_QX2_JoMu_MA_HeLa_500ng_LC05', 'TLGILGLGR'), ('20190710_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'AVLVDLEPGTMDSVR'), ('20190722_QE10_nLC0_LiNi_QC_45cm_HeLa_MUC_3rdcolumn_4', 'SDVLELTDDNFESR'), ('20190722_QX3_MiWi_MA_Hela_500ng_LC15', 'TFVNITPAEVGVLVGK'), ('20190729_QX3_MiWi_MA_Hela_500ng_LC15', 'TLGILGLGR'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'IITLTGPTNAIFK'), ('20190803_QX8_AnPi_MA_HeLa_BR14_500ng', 'SLHDAIMIVR')]
    




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
      <td>0.467</td>
      <td>0.489</td>
      <td>1.405</td>
      <td>1.551</td>
      <td>1.798</td>
      <td>1.877</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.421</td>
      <td>0.440</td>
      <td>0.820</td>
      <td>0.918</td>
      <td>0.999</td>
      <td>0.978</td>
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
