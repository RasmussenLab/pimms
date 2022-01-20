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
    DPFAHLPK                        990
    LTPEEEEILNK                     988
    LAPITSDPTEATAVGAVEASFK        1,000
    MPSLPSYK                        976
    ILNIFGVIK                       965
    MAPYQGPDAVPGALDYK             1,000
    LSFQHDPETSVLVLR                 993
    AVCMLSNTTAVAEAWAR               994
    KFDQLLAEEK                      995
    DNSTMGYMMAK                     937
    FVMQEEFSR                       968
    GLGTDEDSLIEIICSR                905
    LGGSAVISLEGKPL                  993
    YLTVAAVFR                       993
    LMIEMDGTENK                     999
    SLESLHSFVAAATK                  988
    RAPFDLFENR                      999
    KYEDICPSTHNMDVPNIK            1,000
    NIEDVIAQGIGK                    987
    TSIAIDTIINQK                    993
    ESYSVYVYK                       992
    DMFQETMEAMR                     988
    SEHPGLSIGDTAK                   975
    HIDSAHLYNNEEQVGLAIR             974
    EAPPMEKPEVVK                    981
    ADLLLSTQPGREEGSPLELER           998
    TLFGLHLSQK                      994
    EAYPGDVFYLHSR                   996
    HSQFIGYPITLFVEK                 957
    FWEVISDEHGIDPTGTYHGDSDLQLER     976
    LAALNPESNTAGLDIFAK              991
    KLEEEQIILEDQNCK                 996
    VTVLFAGQHIAK                    992
    FDASFFGVHPK                     977
    YMACCLLYR                       958
    VAPEEHPVLLTEAPLNPK            1,000
    AFDSGIIPMEFVNK                  964
    AMVSEFLK                        984
    IVLLDSSLEYK                     987
    EGIPPDQQR                       998
    IGEEQSAEDAEDGPPELLFIHGGHTAK     979
    AQIFANTVDNAR                    990
    DPVQEAWAEDVDLR                  986
    TGTAEMSSILEER                   999
    EMEAELEDERK                     981
    TLNDELEIIEGMK                   998
    NALESYAFNMK                     993
    NILFVITKPDVYK                   998
    IGAEVYHNLK                    1,000
    SLEDQVEMLR                      982
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
      <th>DPFAHLPK</th>
      <td>29.130</td>
    </tr>
    <tr>
      <th>LTPEEEEILNK</th>
      <td>29.541</td>
    </tr>
    <tr>
      <th>LAPITSDPTEATAVGAVEASFK</th>
      <td>31.321</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>29.889</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>27.185</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>TLNDELEIIEGMK</th>
      <td>32.288</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>30.210</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
    </tr>
    <tr>
      <th>IGAEVYHNLK</th>
      <td>33.064</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>29.126</td>
    </tr>
  </tbody>
</table>
<p>49247 rows × 1 columns</p>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_24_1.png)
    



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
      <td>0.985</td>
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
      <th>DPFAHLPK</th>
      <td>29.130</td>
    </tr>
    <tr>
      <th>LTPEEEEILNK</th>
      <td>29.541</td>
    </tr>
    <tr>
      <th>LAPITSDPTEATAVGAVEASFK</th>
      <td>31.321</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>29.889</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>27.185</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">20190805_QE1_nLC2_AB_MNT_HELA_04</th>
      <th>TLNDELEIIEGMK</th>
      <td>32.288</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>30.210</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
    </tr>
    <tr>
      <th>IGAEVYHNLK</th>
      <td>33.064</td>
    </tr>
    <tr>
      <th>SLEDQVEMLR</th>
      <td>29.126</td>
    </tr>
  </tbody>
</table>
<p>49247 rows × 1 columns</p>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>30.237</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_02</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>30.729</td>
    </tr>
    <tr>
      <th>20190624_QE6_LC4_AS_QC_MNT_HeLa_02</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>27.050</td>
    </tr>
    <tr>
      <th>20190528_QX2_SeVW_MA_HeLa_500ng_LC05_CTCDoff_1</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>29.972</td>
    </tr>
    <tr>
      <th>20190208_QE2_NLC1_AB_QC_MNT_HELA_3</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>30.956</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>20190411_QE3_nLC5_DS_QC_MNT_HeLa_02</th>
      <th>YMACCLLYR</th>
      <td>29.090</td>
    </tr>
    <tr>
      <th>20190603_QX4_JiYu_MA_HeLa_500ng</th>
      <th>YMACCLLYR</th>
      <td>30.585</td>
    </tr>
    <tr>
      <th>20190624_QX3_MaMu_MA_Hela_500ng_LC15</th>
      <th>YMACCLLYR</th>
      <td>30.704</td>
    </tr>
    <tr>
      <th>20190613_QX0_MePh_MA_HeLa_500ng_LC07_Stab_02</th>
      <th>YMACCLLYR</th>
      <td>31.069</td>
    </tr>
    <tr>
      <th>20190226_QE1_nLC2_AB_QC_MNT_HELA_04</th>
      <th>YMACCLLYR</th>
      <td>29.480</td>
    </tr>
  </tbody>
</table>
<p>44321 rows × 1 columns</p>
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
      <th rowspan="4" valign="top">20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>29.346</td>
      <td>30.578</td>
      <td>30.484</td>
      <td>29.659</td>
    </tr>
    <tr>
      <th>AVCMLSNTTAVAEAWAR</th>
      <td>27.524</td>
      <td>28.473</td>
      <td>28.625</td>
      <td>27.414</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>28.866</td>
      <td>30.433</td>
      <td>30.561</td>
      <td>29.379</td>
    </tr>
    <tr>
      <th>TGTAEMSSILEER</th>
      <td>29.238</td>
      <td>30.100</td>
      <td>30.101</td>
      <td>29.453</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>28.794</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>29.345</td>
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
      <th>DNSTMGYMMAK</th>
      <td>26.222</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>27.218</td>
    </tr>
    <tr>
      <th>GLGTDEDSLIEIICSR</th>
      <td>32.466</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>32.584</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>29.717</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>29.779</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>30.843</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>30.904</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
      <td>29.968</td>
      <td>29.761</td>
      <td>30.305</td>
    </tr>
  </tbody>
</table>
<p>4926 rows × 4 columns</p>
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
      <th>20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007</th>
      <th>HSQFIGYPITLFVEK</th>
      <td>30.231</td>
      <td>30.257</td>
      <td>30.142</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190107_QE2_NLC10_ANHO_QC_MNT_HELA_02</th>
      <th>GLGTDEDSLIEIICSR</th>
      <td>32.233</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE2_NLC10_ANHO_QC_MNT_HELA_03</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>33.864</td>
      <td>34.261</td>
      <td>33.494</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190111_QE8_nLC1_ASD_QC_HeLa_01</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLER</th>
      <td>29.093</td>
      <td>31.189</td>
      <td>30.922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190118_QE9_nLC9_NHS_MNT_HELA_50cm_04</th>
      <th>LTPEEEEILNK</th>
      <td>30.372</td>
      <td>30.491</td>
      <td>30.655</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03</th>
      <th>NALESYAFNMK</th>
      <td>30.809</td>
      <td>30.433</td>
      <td>30.561</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190129_QE8_nLC14_FaCo_QC_MNT_50cm_Hela_20190129205246</th>
      <th>MPSLPSYK</th>
      <td>31.929</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190201_QE10_nLC0_NHS_MNT_HELA_45cm_01</th>
      <th>DMFQETMEAMR</th>
      <td>27.614</td>
      <td>28.665</td>
      <td>28.306</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190219_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>AQIFANTVDNAR</th>
      <td>30.150</td>
      <td>30.806</td>
      <td>30.784</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190221_QE4_LC12_IAH_QC_MNT_HeLa_02</th>
      <th>FVMQEEFSR</th>
      <td>28.085</td>
      <td>28.516</td>
      <td>28.955</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190221_QE8_nLC9_JM_QC_MNT_HeLa_01</th>
      <th>LGGSAVISLEGKPL</th>
      <td>31.696</td>
      <td>30.984</td>
      <td>30.939</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_5</th>
      <th>GLGTDEDSLIEIICSR</th>
      <td>30.857</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_6</th>
      <th>LAPITSDPTEATAVGAVEASFK</th>
      <td>31.144</td>
      <td>32.122</td>
      <td>32.012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190225_QE10_PhGe_Evosep_88min_HeLa_8</th>
      <th>TSIAIDTIINQK</th>
      <td>28.959</td>
      <td>29.436</td>
      <td>29.458</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190226_QE1_nLC2_AB_QC_MNT_HELA_01</th>
      <th>EAYPGDVFYLHSR</th>
      <td>27.594</td>
      <td>27.835</td>
      <td>28.315</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190226_QE1_nLC2_AB_QC_MNT_HELA_02</th>
      <th>KFDQLLAEEK</th>
      <td>30.720</td>
      <td>29.711</td>
      <td>29.491</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03</th>
      <th>TLFGLHLSQK</th>
      <td>26.997</td>
      <td>28.376</td>
      <td>28.285</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190308_QE3_nLC5_MR_QC_MNT_HeLa_Easy_01</th>
      <th>FDASFFGVHPK</th>
      <td>24.118</td>
      <td>28.749</td>
      <td>28.809</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190308_QE3_nLC5_MR_QC_MNT_HeLa_Easyctcdon_02</th>
      <th>DPFAHLPK</th>
      <td>25.722</td>
      <td>27.830</td>
      <td>28.959</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190308_QE3_nLC5_MR_QC_MNT_Hela_CPRHEasy_03</th>
      <th>KFDQLLAEEK</th>
      <td>27.744</td>
      <td>29.711</td>
      <td>29.491</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190308_QE9_nLC0_FaCo_QC_MNT_Hela_50cm_newcol</th>
      <th>DNSTMGYMMAK</th>
      <td>25.830</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190315_QE2_NLC1_GP_MNT_HELA_01</th>
      <th>SLESLHSFVAAATK</th>
      <td>29.276</td>
      <td>28.385</td>
      <td>28.175</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190328_QE1_nLC2_GP_MNT_QC_hela_01</th>
      <th>HIDSAHLYNNEEQVGLAIR</th>
      <td>28.720</td>
      <td>28.476</td>
      <td>28.242</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190401_QE8_nLC14_RS_QC_MNT_Hela_50cm_01</th>
      <th>YMACCLLYR</th>
      <td>28.353</td>
      <td>29.209</td>
      <td>29.174</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190411_QE3_nLC5_DS_QC_MNT_HeLa_02</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLER</th>
      <td>31.321</td>
      <td>31.189</td>
      <td>30.922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190423_QE3_nLC5_DS_QC_MNT_HeLa_01</th>
      <th>LMIEMDGTENK</th>
      <td>31.220</td>
      <td>32.581</td>
      <td>32.507</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190426_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_02</th>
      <th>VTVLFAGQHIAK</th>
      <td>31.642</td>
      <td>30.289</td>
      <td>29.886</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190509_QE4_LC12_AS_QC_MNT_HeLa_01</th>
      <th>DNSTMGYMMAK</th>
      <td>27.222</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190515_QX6_ChDe_MA_HeLa_Br14_500ng_LC09</th>
      <th>LTPEEEEILNK</th>
      <td>32.239</td>
      <td>30.491</td>
      <td>30.655</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_01</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <td>32.676</td>
      <td>34.261</td>
      <td>33.494</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX0_MePh_MA_HeLa_500ng_LC07_4</th>
      <th>IGEEQSAEDAEDGPPELLFIHGGHTAK</th>
      <td>31.001</td>
      <td>28.731</td>
      <td>28.746</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX4_JiYu_MA_HeLa_500ng</th>
      <th>GLGTDEDSLIEIICSR</th>
      <td>28.753</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190611_QX7_IgPa_MA_HeLa_Br14_500ng_190618134442</th>
      <th>DPVQEAWAEDVDLR</th>
      <td>30.474</td>
      <td>28.674</td>
      <td>29.262</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190617_QX4_JiYu_MA_HeLa_500ng</th>
      <th>HSQFIGYPITLFVEK</th>
      <td>29.815</td>
      <td>30.257</td>
      <td>30.142</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190620_QE2_NLC1_GP_QC_MNT_HELA_01</th>
      <th>LGGSAVISLEGKPL</th>
      <td>31.463</td>
      <td>30.984</td>
      <td>30.939</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_01</th>
      <th>AQIFANTVDNAR</th>
      <td>30.510</td>
      <td>30.806</td>
      <td>30.784</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_02</th>
      <th>LSFQHDPETSVLVLR</th>
      <td>28.231</td>
      <td>29.035</td>
      <td>29.152</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03</th>
      <th>LSFQHDPETSVLVLR</th>
      <td>28.030</td>
      <td>29.035</td>
      <td>29.152</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QX2_SeVW_MA_HeLa_500ng_LC05</th>
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>31.861</td>
      <td>30.578</td>
      <td>30.484</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190621_QX3_MePh_MA_Hela_500ng_LC15</th>
      <th>AMVSEFLK</th>
      <td>30.040</td>
      <td>29.298</td>
      <td>29.089</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QE9_nLC0_RG_MNT_Hela_MUC_50cm_2</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>24.734</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190625_QX0_MaPe_MA_HeLa_500ng_LC07_1</th>
      <th>AMVSEFLK</th>
      <td>30.221</td>
      <td>29.298</td>
      <td>29.089</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190628_QE1_nLC13_ANHO_QC_MNT_HELA_01</th>
      <th>EAPPMEKPEVVK</th>
      <td>29.664</td>
      <td>30.081</td>
      <td>30.135</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED</th>
      <th>IVLLDSSLEYK</th>
      <td>31.686</td>
      <td>29.078</td>
      <td>29.219</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190630_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1</th>
      <th>FWEVISDEHGIDPTGTYHGDSDLQLER</th>
      <td>30.189</td>
      <td>31.189</td>
      <td>30.922</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QE1_nLC13_ANHO_QC_MNT_HELA_02</th>
      <th>FDASFFGVHPK</th>
      <td>28.072</td>
      <td>28.749</td>
      <td>28.809</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03</th>
      <th>AQIFANTVDNAR</th>
      <td>30.447</td>
      <td>30.806</td>
      <td>30.784</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190701_QE8_nLC14_GP_QC_MNT_15cm_Hela_01</th>
      <th>ILNIFGVIK</th>
      <td>30.027</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190702_QX0_AnBr_MA_HeLa_500ng_LC07_01_190702180001</th>
      <th>SEHPGLSIGDTAK</th>
      <td>30.195</td>
      <td>28.873</td>
      <td>28.684</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190715_QE10_nLC0_LiNi_QC_MNT_15cm_HeLa_MUC_01</th>
      <th>ESYSVYVYK</th>
      <td>32.039</td>
      <td>33.046</td>
      <td>32.929</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190715_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <th>EGIPPDQQR</th>
      <td>30.043</td>
      <td>30.294</td>
      <td>30.066</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190718_QE8_nLC14_RG_QC_HeLa_MUC_50cm_1</th>
      <th>SLESLHSFVAAATK</th>
      <td>26.932</td>
      <td>28.385</td>
      <td>28.175</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190724_QX3_MiWi_MA_Hela_500ng_LC15</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>30.051</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01</th>
      <th>ILNIFGVIK</th>
      <td>29.043</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190730_QE1_nLC2_GP_MNT_HELA_01</th>
      <th>AMVSEFLK</th>
      <td>29.997</td>
      <td>29.298</td>
      <td>29.089</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QE3_nLC3_DBJ_AMV_QC_MNT_HELA_01</th>
      <th>ILNIFGVIK</th>
      <td>29.129</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190802_QX7_AlRe_MA_HeLa_Br14_500ng</th>
      <th>MPSLPSYK</th>
      <td>31.669</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_350ng</th>
      <th>GLGTDEDSLIEIICSR</th>
      <td>31.817</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng</th>
      <th>GLGTDEDSLIEIICSR</th>
      <td>31.400</td>
      <td>31.824</td>
      <td>31.328</td>
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
      <td>AFDSGIIPMEFVNK</td>
      <td>28.969</td>
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
      <td>AQIFANTVDNAR</td>
      <td>29.798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DMFQETMEAMR</td>
      <td>27.327</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>DNSTMGYMMAK</td>
      <td>27.949</td>
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
      <td>ADLLLSTQPGREEGSPLELER</td>
      <td>29.346</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>AVCMLSNTTAVAEAWAR</td>
      <td>27.524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>NALESYAFNMK</td>
      <td>28.866</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181219_QE3_nLC3_DS_QC_MNT_HeLa_02</td>
      <td>TGTAEMSSILEER</td>
      <td>29.238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</td>
      <td>AFDSGIIPMEFVNK</td>
      <td>28.794</td>
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
      <td>20190330_QE1_nLC2_GP_MNT_QC_hela_01</td>
      <td>DPFAHLPK</td>
      <td>26.413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190204_QE6_nLC6_MPL_QC_MNT_HeLa_01</td>
      <td>GLGTDEDSLIEIICSR</td>
      <td>32.533</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190410_QE8_nLC14_ASD_QC_MNT_HELA_01</td>
      <td>KYEDICPSTHNMDVPNIK</td>
      <td>32.690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190802_QX7_AlRe_MA_HeLa_Br14_500ng</td>
      <td>TLNDELEIIEGMK</td>
      <td>33.760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190627_QX8_AnPi_MA_HeLa_BR14_500ng</td>
      <td>LSFQHDPETSVLVLR</td>
      <td>27.494</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_3</td>
      <td>HSQFIGYPITLFVEK</td>
      <td>29.591</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190403_QE1_nLC2_GP_MNT_QC_hela_01</td>
      <td>RAPFDLFENR</td>
      <td>25.619</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190619_QE7_nLC7_AP_QC_MNT_HeLa_01</td>
      <td>YMACCLLYR</td>
      <td>28.966</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190117_QE1_nLC2_ANHO_QC_MNT_HELA_01</td>
      <td>GLGTDEDSLIEIICSR</td>
      <td>30.669</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190601_QX1_JoMu_MA_HeLa_DMSO_500ng_LC14</td>
      <td>TLFGLHLSQK</td>
      <td>29.428</td>
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
      <td>20190612_QX3_JoMu_MA_HeLa_500ng_LC15_uPAC200cm</td>
      <td>KYEDICPSTHNMDVPNIK</td>
      <td>32.785</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190225_QE10_PhGe_Evosep_88min_HeLa_4</td>
      <td>YMACCLLYR</td>
      <td>29.107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190328_QE1_nLC2_GP_MNT_QC_hela_01</td>
      <td>TSIAIDTIINQK</td>
      <td>29.315</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190712_QE1_nLC13_ANHO_QC_MNT_HELA_02</td>
      <td>DMFQETMEAMR</td>
      <td>28.493</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190606_QE4_LC12_JE_QC_MNT_HeLa_02b</td>
      <td>MAPYQGPDAVPGALDYK</td>
      <td>28.801</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20190515_QX4_JiYu_MA_HeLa_500ng_BR14</td>
      <td>LTPEEEEILNK</td>
      <td>32.775</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20190626_QX6_ChDe_MA_HeLa_500ng_LC09</td>
      <td>KLEEEQIILEDQNCK</td>
      <td>30.646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20190226_QE10_PhGe_Evosep_176min-30cmCol-HeLa_3</td>
      <td>SEHPGLSIGDTAK</td>
      <td>26.896</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20190803_QE9_nLC13_RG_SA_HeLa_50cm_300ng</td>
      <td>TGTAEMSSILEER</td>
      <td>30.153</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20190527_QX4_IgPa_MA_HeLa_500ng</td>
      <td>EGIPPDQQR</td>
      <td>30.982</td>
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
    
    Optimizer used: <function Adam at 0x0000028849CB6040>
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
      <td>2.033101</td>
      <td>1.752467</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.784413</td>
      <td>0.879928</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.737645</td>
      <td>0.723913</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.679207</td>
      <td>0.694870</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.637846</td>
      <td>0.692560</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.656754</td>
      <td>0.665406</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.641832</td>
      <td>0.651786</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.581681</td>
      <td>0.641705</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.600012</td>
      <td>0.638422</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.612242</td>
      <td>0.638136</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_58_1.png)
    


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
      <th>3,255</th>
      <td>667</td>
      <td>27</td>
      <td>32.785</td>
    </tr>
    <tr>
      <th>1,047</th>
      <td>214</td>
      <td>50</td>
      <td>29.107</td>
    </tr>
    <tr>
      <th>1,599</th>
      <td>325</td>
      <td>46</td>
      <td>29.315</td>
    </tr>
    <tr>
      <th>4,157</th>
      <td>850</td>
      <td>6</td>
      <td>28.493</td>
    </tr>
    <tr>
      <th>3,099</th>
      <td>633</td>
      <td>34</td>
      <td>28.801</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3,906</th>
      <td>797</td>
      <td>40</td>
      <td>28.745</td>
    </tr>
    <tr>
      <th>601</th>
      <td>130</td>
      <td>28</td>
      <td>30.400</td>
    </tr>
    <tr>
      <th>3,500</th>
      <td>718</td>
      <td>40</td>
      <td>29.535</td>
    </tr>
    <tr>
      <th>4,219</th>
      <td>860</td>
      <td>4</td>
      <td>30.477</td>
    </tr>
    <tr>
      <th>1,541</th>
      <td>313</td>
      <td>2</td>
      <td>29.770</td>
    </tr>
  </tbody>
</table>
<p>4926 rows × 3 columns</p>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>29.346</td>
      <td>30.578</td>
      <td>30.484</td>
      <td>29.659</td>
      <td>29.710</td>
    </tr>
    <tr>
      <th>AVCMLSNTTAVAEAWAR</th>
      <td>27.524</td>
      <td>28.473</td>
      <td>28.625</td>
      <td>27.414</td>
      <td>27.817</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>28.866</td>
      <td>30.433</td>
      <td>30.561</td>
      <td>29.379</td>
      <td>29.883</td>
    </tr>
    <tr>
      <th>TGTAEMSSILEER</th>
      <td>29.238</td>
      <td>30.100</td>
      <td>30.101</td>
      <td>29.453</td>
      <td>29.384</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>28.794</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>29.345</td>
      <td>28.673</td>
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
      <th>DNSTMGYMMAK</th>
      <td>26.222</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>27.218</td>
      <td>26.860</td>
    </tr>
    <tr>
      <th>GLGTDEDSLIEIICSR</th>
      <td>32.466</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>32.584</td>
      <td>32.082</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>29.717</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>29.779</td>
      <td>29.331</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>30.843</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>30.904</td>
      <td>30.710</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
      <td>29.968</td>
      <td>29.761</td>
      <td>30.305</td>
      <td>30.027</td>
    </tr>
  </tbody>
</table>
<p>4926 rows × 5 columns</p>
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
    20181219_QE3_nLC3_DS_QC_MNT_HeLa_02     0.138
    20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01    0.129
    20181221_QE8_nLC0_NHS_MNT_HeLa_01       0.213
    20181222_QE9_nLC9_QC_50CM_HeLa1         0.144
    20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01   0.223
    dtype: float32




```python
fig, ax = plt.subplots(figsize=(15, 15))
ax = collab.biases.sample.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_samples',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_samples
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_65_1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 15))
_ = collab.biases.peptide.sort_values().plot(kind='line', rot=90, title='Sample biases', ax=ax)
vaep.io_images._savefig(fig, name='collab_bias_peptides',
                        folder=folder)
```

    vaep.io_images - INFO     Saved Figures to runs\2D\feat_0050_epochs_010\collab_bias_peptides
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_66_1.png)
    



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
      <td>0.123</td>
      <td>-0.106</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.195</td>
      <td>-0.069</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>0.061</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.191</td>
      <td>-0.169</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>0.131</td>
      <td>-0.042</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_68_1.png)
    



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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_69_1.png)
    


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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR</th>
      <th>SLESLHSFVAAATK</th>
      <th>TGTAEMSSILEER</th>
      <th>TLFGLHLSQK</th>
      <th>TLNDELEIIEGMK</th>
      <th>TSIAIDTIINQK</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <th>VTVLFAGQHIAK</th>
      <th>YLTVAAVFR</th>
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
      <td>29.346</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.524</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.238</td>
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
      <td>28.794</td>
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
      <td>33.504</td>
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
      <td>30.519</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.126</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.682</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.364</td>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>SLESLHSFVAAATK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLFGLHLSQK_na</th>
      <th>TLNDELEIIEGMK_na</th>
      <th>TSIAIDTIINQK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VTVLFAGQHIAK_na</th>
      <th>YLTVAAVFR_na</th>
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
      <td>0.063</td>
      <td>-0.304</td>
      <td>-0.847</td>
      <td>-0.912</td>
      <td>-0.095</td>
      <td>-0.815</td>
      <td>-0.201</td>
      <td>0.119</td>
      <td>0.117</td>
      <td>-1.044</td>
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
      <td>-1.044</td>
      <td>0.311</td>
      <td>-1.066</td>
      <td>-0.856</td>
      <td>-0.679</td>
      <td>-1.129</td>
      <td>0.107</td>
      <td>0.143</td>
      <td>-0.191</td>
      <td>-0.834</td>
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
      <td>-0.194</td>
      <td>0.235</td>
      <td>-1.085</td>
      <td>-0.321</td>
      <td>-0.989</td>
      <td>-0.433</td>
      <td>-0.885</td>
      <td>-0.577</td>
      <td>-0.405</td>
      <td>0.004</td>
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
      <td>-0.430</td>
      <td>-0.009</td>
      <td>-0.795</td>
      <td>0.018</td>
      <td>-0.942</td>
      <td>-0.929</td>
      <td>0.065</td>
      <td>0.030</td>
      <td>0.198</td>
      <td>-0.037</td>
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
      <td>-0.666</td>
      <td>0.700</td>
      <td>0.538</td>
      <td>-0.625</td>
      <td>-0.095</td>
      <td>-0.577</td>
      <td>0.079</td>
      <td>-0.193</td>
      <td>-0.273</td>
      <td>-1.270</td>
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
      <td>0.810</td>
      <td>0.185</td>
      <td>0.252</td>
      <td>0.824</td>
      <td>1.184</td>
      <td>-0.631</td>
      <td>1.208</td>
      <td>1.282</td>
      <td>1.255</td>
      <td>0.704</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.543</td>
      <td>0.509</td>
      <td>-0.057</td>
      <td>0.060</td>
      <td>-0.198</td>
      <td>0.682</td>
      <td>-0.463</td>
      <td>-0.346</td>
      <td>-0.675</td>
      <td>0.091</td>
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
      <td>-0.038</td>
      <td>-1.502</td>
      <td>-0.082</td>
      <td>-0.112</td>
      <td>-0.315</td>
      <td>0.850</td>
      <td>-0.493</td>
      <td>-0.316</td>
      <td>-0.449</td>
      <td>0.065</td>
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
      <td>-0.261</td>
      <td>0.818</td>
      <td>-2.977</td>
      <td>0.018</td>
      <td>0.341</td>
      <td>0.806</td>
      <td>-0.374</td>
      <td>-0.415</td>
      <td>-0.764</td>
      <td>-0.132</td>
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
      <td>-0.132</td>
      <td>-1.507</td>
      <td>0.031</td>
      <td>0.018</td>
      <td>-2.930</td>
      <td>0.475</td>
      <td>-0.374</td>
      <td>-0.336</td>
      <td>-0.889</td>
      <td>-0.009</td>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>SLESLHSFVAAATK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLFGLHLSQK_na</th>
      <th>TLNDELEIIEGMK_na</th>
      <th>TSIAIDTIINQK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VTVLFAGQHIAK_na</th>
      <th>YLTVAAVFR_na</th>
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
      <td>0.067</td>
      <td>-0.241</td>
      <td>-0.780</td>
      <td>-0.859</td>
      <td>-0.100</td>
      <td>-0.741</td>
      <td>-0.252</td>
      <td>0.066</td>
      <td>0.077</td>
      <td>-0.986</td>
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
      <td>-0.982</td>
      <td>0.336</td>
      <td>-0.986</td>
      <td>-0.806</td>
      <td>-0.654</td>
      <td>-1.038</td>
      <td>0.035</td>
      <td>0.089</td>
      <td>-0.213</td>
      <td>-0.788</td>
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
      <td>-0.177</td>
      <td>0.265</td>
      <td>-1.005</td>
      <td>-0.301</td>
      <td>-0.947</td>
      <td>-0.380</td>
      <td>-0.889</td>
      <td>-0.597</td>
      <td>-0.416</td>
      <td>-0.001</td>
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
      <td>-0.400</td>
      <td>0.036</td>
      <td>-0.732</td>
      <td>0.019</td>
      <td>-0.902</td>
      <td>-0.849</td>
      <td>-0.005</td>
      <td>-0.019</td>
      <td>0.154</td>
      <td>-0.040</td>
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
      <td>-0.624</td>
      <td>0.701</td>
      <td>0.525</td>
      <td>-0.589</td>
      <td>-0.100</td>
      <td>-0.516</td>
      <td>0.008</td>
      <td>-0.231</td>
      <td>-0.291</td>
      <td>-1.198</td>
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
      <td>0.774</td>
      <td>0.218</td>
      <td>0.255</td>
      <td>0.780</td>
      <td>1.110</td>
      <td>-0.567</td>
      <td>1.058</td>
      <td>1.175</td>
      <td>1.154</td>
      <td>0.657</td>
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
      <th>20190805_QE1_nLC2_AB_MNT_HELA_01</th>
      <td>-0.508</td>
      <td>0.522</td>
      <td>-0.036</td>
      <td>0.059</td>
      <td>-0.198</td>
      <td>0.676</td>
      <td>-0.496</td>
      <td>-0.377</td>
      <td>-0.672</td>
      <td>0.081</td>
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
      <td>-0.029</td>
      <td>-1.365</td>
      <td>-0.059</td>
      <td>-0.103</td>
      <td>-0.308</td>
      <td>0.835</td>
      <td>-0.524</td>
      <td>-0.349</td>
      <td>-0.458</td>
      <td>0.056</td>
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
      <td>-0.240</td>
      <td>0.812</td>
      <td>-2.787</td>
      <td>0.019</td>
      <td>0.312</td>
      <td>0.793</td>
      <td>-0.413</td>
      <td>-0.443</td>
      <td>-0.756</td>
      <td>-0.128</td>
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
      <td>-0.118</td>
      <td>-1.370</td>
      <td>0.047</td>
      <td>0.019</td>
      <td>-2.783</td>
      <td>0.480</td>
      <td>-0.413</td>
      <td>-0.368</td>
      <td>-0.874</td>
      <td>-0.013</td>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
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
      <td>0.007</td>
      <td>0.045</td>
      <td>0.018</td>
      <td>0.002</td>
      <td>-0.011</td>
      <td>0.030</td>
      <td>-0.065</td>
      <td>-0.048</td>
      <td>-0.033</td>
      <td>-0.005</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.948</td>
      <td>0.939</td>
      <td>0.943</td>
      <td>0.945</td>
      <td>0.947</td>
      <td>0.947</td>
      <td>0.931</td>
      <td>0.954</td>
      <td>0.947</td>
      <td>0.940</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.392</td>
      <td>-4.099</td>
      <td>-5.093</td>
      <td>-6.945</td>
      <td>-3.731</td>
      <td>-4.474</td>
      <td>-2.536</td>
      <td>-2.644</td>
      <td>-3.156</td>
      <td>-5.316</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.386</td>
      <td>-0.212</td>
      <td>-0.330</td>
      <td>-0.381</td>
      <td>-0.556</td>
      <td>-0.318</td>
      <td>-0.654</td>
      <td>-0.691</td>
      <td>-0.619</td>
      <td>-0.374</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.067</td>
      <td>0.336</td>
      <td>0.156</td>
      <td>0.019</td>
      <td>-0.100</td>
      <td>0.272</td>
      <td>-0.413</td>
      <td>-0.434</td>
      <td>-0.291</td>
      <td>-0.040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.499</td>
      <td>0.624</td>
      <td>0.550</td>
      <td>0.510</td>
      <td>0.492</td>
      <td>0.643</td>
      <td>0.715</td>
      <td>1.036</td>
      <td>0.732</td>
      <td>0.388</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.951</td>
      <td>1.725</td>
      <td>2.658</td>
      <td>1.938</td>
      <td>2.396</td>
      <td>1.553</td>
      <td>1.903</td>
      <td>1.858</td>
      <td>2.070</td>
      <td>2.078</td>
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




    ((#50) ['ADLLLSTQPGREEGSPLELER','AFDSGIIPMEFVNK','AMVSEFLK','AQIFANTVDNAR','AVCMLSNTTAVAEAWAR','DMFQETMEAMR','DNSTMGYMMAK','DPFAHLPK','DPVQEAWAEDVDLR','EAPPMEKPEVVK'...],
     (#50) ['ADLLLSTQPGREEGSPLELER_na','AFDSGIIPMEFVNK_na','AMVSEFLK_na','AQIFANTVDNAR_na','AVCMLSNTTAVAEAWAR_na','DMFQETMEAMR_na','DNSTMGYMMAK_na','DPFAHLPK_na','DPVQEAWAEDVDLR_na','EAPPMEKPEVVK_na'...])




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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>96.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>94.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.138</td>
      <td>0.011</td>
      <td>-0.099</td>
      <td>-0.074</td>
      <td>-0.090</td>
      <td>0.072</td>
      <td>-0.027</td>
      <td>0.169</td>
      <td>0.100</td>
      <td>-0.032</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.925</td>
      <td>1.045</td>
      <td>1.028</td>
      <td>0.934</td>
      <td>0.971</td>
      <td>0.957</td>
      <td>1.004</td>
      <td>1.041</td>
      <td>0.932</td>
      <td>1.260</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.782</td>
      <td>-3.189</td>
      <td>-3.275</td>
      <td>-4.366</td>
      <td>-3.293</td>
      <td>-3.757</td>
      <td>-1.780</td>
      <td>-2.077</td>
      <td>-2.079</td>
      <td>-5.495</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.377</td>
      <td>-0.287</td>
      <td>-0.655</td>
      <td>-0.643</td>
      <td>-0.609</td>
      <td>-0.244</td>
      <td>-0.771</td>
      <td>-0.650</td>
      <td>-0.451</td>
      <td>-0.552</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.180</td>
      <td>0.347</td>
      <td>0.168</td>
      <td>-0.119</td>
      <td>-0.090</td>
      <td>0.298</td>
      <td>-0.509</td>
      <td>-0.342</td>
      <td>-0.156</td>
      <td>-0.170</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.860</td>
      <td>0.622</td>
      <td>0.675</td>
      <td>0.448</td>
      <td>0.404</td>
      <td>0.676</td>
      <td>1.161</td>
      <td>1.296</td>
      <td>0.944</td>
      <td>0.815</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.788</td>
      <td>1.526</td>
      <td>1.805</td>
      <td>1.612</td>
      <td>2.257</td>
      <td>1.576</td>
      <td>1.776</td>
      <td>1.715</td>
      <td>1.989</td>
      <td>2.068</td>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>SLESLHSFVAAATK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLFGLHLSQK_val</th>
      <th>TLNDELEIIEGMK_val</th>
      <th>TSIAIDTIINQK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VTVLFAGQHIAK_val</th>
      <th>YLTVAAVFR_val</th>
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
      <td>0.067</td>
      <td>-0.241</td>
      <td>-0.780</td>
      <td>-0.859</td>
      <td>-0.100</td>
      <td>-0.741</td>
      <td>-0.252</td>
      <td>0.066</td>
      <td>0.077</td>
      <td>-0.986</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.771</td>
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
      <td>-0.982</td>
      <td>0.336</td>
      <td>-0.986</td>
      <td>-0.806</td>
      <td>-0.654</td>
      <td>-1.038</td>
      <td>0.035</td>
      <td>0.089</td>
      <td>-0.213</td>
      <td>-0.788</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.177</td>
      <td>0.265</td>
      <td>-1.005</td>
      <td>-0.301</td>
      <td>-0.947</td>
      <td>-0.380</td>
      <td>-0.889</td>
      <td>-0.597</td>
      <td>-0.416</td>
      <td>-0.001</td>
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
      <td>-0.400</td>
      <td>0.036</td>
      <td>-0.732</td>
      <td>0.019</td>
      <td>-0.902</td>
      <td>-0.849</td>
      <td>-0.005</td>
      <td>-0.019</td>
      <td>0.154</td>
      <td>-0.040</td>
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
      <td>-0.624</td>
      <td>0.701</td>
      <td>0.525</td>
      <td>-0.589</td>
      <td>-0.100</td>
      <td>-0.516</td>
      <td>0.008</td>
      <td>-0.231</td>
      <td>-0.291</td>
      <td>-1.198</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.853</td>
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
      <td>0.774</td>
      <td>0.218</td>
      <td>0.255</td>
      <td>0.780</td>
      <td>1.110</td>
      <td>-0.567</td>
      <td>1.058</td>
      <td>1.175</td>
      <td>1.154</td>
      <td>0.657</td>
      <td>...</td>
      <td>0.696</td>
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
      <td>-0.508</td>
      <td>0.522</td>
      <td>-0.036</td>
      <td>0.059</td>
      <td>-0.198</td>
      <td>0.676</td>
      <td>-0.496</td>
      <td>-0.377</td>
      <td>-0.672</td>
      <td>0.081</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.969</td>
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
      <td>-0.029</td>
      <td>-1.365</td>
      <td>-0.059</td>
      <td>-0.103</td>
      <td>-0.308</td>
      <td>0.835</td>
      <td>-0.524</td>
      <td>-0.349</td>
      <td>-0.458</td>
      <td>0.056</td>
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
      <td>-0.240</td>
      <td>0.812</td>
      <td>-2.787</td>
      <td>0.019</td>
      <td>0.312</td>
      <td>0.793</td>
      <td>-0.413</td>
      <td>-0.443</td>
      <td>-0.756</td>
      <td>-0.128</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.495</td>
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
      <td>-0.118</td>
      <td>-1.370</td>
      <td>0.047</td>
      <td>0.019</td>
      <td>-2.783</td>
      <td>0.480</td>
      <td>-0.413</td>
      <td>-0.368</td>
      <td>-0.874</td>
      <td>-0.013</td>
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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>SLESLHSFVAAATK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLFGLHLSQK_val</th>
      <th>TLNDELEIIEGMK_val</th>
      <th>TSIAIDTIINQK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VTVLFAGQHIAK_val</th>
      <th>YLTVAAVFR_val</th>
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
      <td>0.067</td>
      <td>-0.241</td>
      <td>-0.780</td>
      <td>-0.859</td>
      <td>-0.100</td>
      <td>-0.741</td>
      <td>-0.252</td>
      <td>0.066</td>
      <td>0.077</td>
      <td>-0.986</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.771</td>
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
      <td>-0.982</td>
      <td>0.336</td>
      <td>-0.986</td>
      <td>-0.806</td>
      <td>-0.654</td>
      <td>-1.038</td>
      <td>0.035</td>
      <td>0.089</td>
      <td>-0.213</td>
      <td>-0.788</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.177</td>
      <td>0.265</td>
      <td>-1.005</td>
      <td>-0.301</td>
      <td>-0.947</td>
      <td>-0.380</td>
      <td>-0.889</td>
      <td>-0.597</td>
      <td>-0.416</td>
      <td>-0.001</td>
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
      <td>-0.400</td>
      <td>0.036</td>
      <td>-0.732</td>
      <td>0.019</td>
      <td>-0.902</td>
      <td>-0.849</td>
      <td>-0.005</td>
      <td>-0.019</td>
      <td>0.154</td>
      <td>-0.040</td>
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
      <td>-0.624</td>
      <td>0.701</td>
      <td>0.525</td>
      <td>-0.589</td>
      <td>-0.100</td>
      <td>-0.516</td>
      <td>0.008</td>
      <td>-0.231</td>
      <td>-0.291</td>
      <td>-1.198</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.853</td>
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
      <td>0.774</td>
      <td>0.218</td>
      <td>0.255</td>
      <td>0.780</td>
      <td>1.110</td>
      <td>-0.567</td>
      <td>1.058</td>
      <td>1.175</td>
      <td>1.154</td>
      <td>0.657</td>
      <td>...</td>
      <td>0.696</td>
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
      <td>-0.508</td>
      <td>0.522</td>
      <td>-0.036</td>
      <td>0.059</td>
      <td>-0.198</td>
      <td>0.676</td>
      <td>-0.496</td>
      <td>-0.377</td>
      <td>-0.672</td>
      <td>0.081</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.969</td>
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
      <td>-0.029</td>
      <td>-1.365</td>
      <td>-0.059</td>
      <td>-0.103</td>
      <td>-0.308</td>
      <td>0.835</td>
      <td>-0.524</td>
      <td>-0.349</td>
      <td>-0.458</td>
      <td>0.056</td>
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
      <td>-0.240</td>
      <td>0.812</td>
      <td>-2.787</td>
      <td>0.019</td>
      <td>0.312</td>
      <td>0.793</td>
      <td>-0.413</td>
      <td>-0.443</td>
      <td>-0.756</td>
      <td>-0.128</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.495</td>
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
      <td>-0.118</td>
      <td>-1.370</td>
      <td>0.047</td>
      <td>0.019</td>
      <td>-2.783</td>
      <td>0.480</td>
      <td>-0.413</td>
      <td>-0.368</td>
      <td>-0.874</td>
      <td>-0.013</td>
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
      <th>ADLLLSTQPGREEGSPLELER_val</th>
      <th>AFDSGIIPMEFVNK_val</th>
      <th>AMVSEFLK_val</th>
      <th>AQIFANTVDNAR_val</th>
      <th>AVCMLSNTTAVAEAWAR_val</th>
      <th>DMFQETMEAMR_val</th>
      <th>DNSTMGYMMAK_val</th>
      <th>DPFAHLPK_val</th>
      <th>DPVQEAWAEDVDLR_val</th>
      <th>EAPPMEKPEVVK_val</th>
      <th>...</th>
      <th>SLEDQVEMLR_val</th>
      <th>SLESLHSFVAAATK_val</th>
      <th>TGTAEMSSILEER_val</th>
      <th>TLFGLHLSQK_val</th>
      <th>TLNDELEIIEGMK_val</th>
      <th>TSIAIDTIINQK_val</th>
      <th>VAPEEHPVLLTEAPLNPK_val</th>
      <th>VTVLFAGQHIAK_val</th>
      <th>YLTVAAVFR_val</th>
      <th>YMACCLLYR_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000</td>
      <td>96.000</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>94.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>98.000</td>
      <td>...</td>
      <td>98.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>100.000</td>
      <td>99.000</td>
      <td>99.000</td>
      <td>96.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.138</td>
      <td>0.011</td>
      <td>-0.099</td>
      <td>-0.074</td>
      <td>-0.090</td>
      <td>0.072</td>
      <td>-0.027</td>
      <td>0.169</td>
      <td>0.100</td>
      <td>-0.032</td>
      <td>...</td>
      <td>0.070</td>
      <td>0.061</td>
      <td>-0.087</td>
      <td>-0.121</td>
      <td>-0.093</td>
      <td>-0.099</td>
      <td>-0.153</td>
      <td>0.209</td>
      <td>-0.209</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.925</td>
      <td>1.045</td>
      <td>1.028</td>
      <td>0.934</td>
      <td>0.971</td>
      <td>0.957</td>
      <td>1.004</td>
      <td>1.041</td>
      <td>0.932</td>
      <td>1.260</td>
      <td>...</td>
      <td>0.885</td>
      <td>1.038</td>
      <td>1.111</td>
      <td>1.044</td>
      <td>0.995</td>
      <td>1.191</td>
      <td>1.064</td>
      <td>1.021</td>
      <td>1.105</td>
      <td>0.862</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.782</td>
      <td>-3.189</td>
      <td>-3.275</td>
      <td>-4.366</td>
      <td>-3.293</td>
      <td>-3.757</td>
      <td>-1.780</td>
      <td>-2.077</td>
      <td>-2.079</td>
      <td>-5.495</td>
      <td>...</td>
      <td>-3.539</td>
      <td>-2.928</td>
      <td>-4.450</td>
      <td>-3.676</td>
      <td>-3.375</td>
      <td>-4.366</td>
      <td>-3.455</td>
      <td>-3.061</td>
      <td>-4.076</td>
      <td>-2.629</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.377</td>
      <td>-0.287</td>
      <td>-0.655</td>
      <td>-0.643</td>
      <td>-0.609</td>
      <td>-0.244</td>
      <td>-0.771</td>
      <td>-0.650</td>
      <td>-0.451</td>
      <td>-0.552</td>
      <td>...</td>
      <td>-0.357</td>
      <td>-0.310</td>
      <td>-0.666</td>
      <td>-0.524</td>
      <td>-0.399</td>
      <td>-0.495</td>
      <td>-0.312</td>
      <td>-0.310</td>
      <td>-0.902</td>
      <td>-0.428</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.180</td>
      <td>0.347</td>
      <td>0.168</td>
      <td>-0.119</td>
      <td>-0.090</td>
      <td>0.298</td>
      <td>-0.509</td>
      <td>-0.342</td>
      <td>-0.156</td>
      <td>-0.170</td>
      <td>...</td>
      <td>0.150</td>
      <td>0.415</td>
      <td>-0.070</td>
      <td>0.046</td>
      <td>0.171</td>
      <td>-0.112</td>
      <td>0.297</td>
      <td>0.531</td>
      <td>-0.206</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.860</td>
      <td>0.622</td>
      <td>0.675</td>
      <td>0.448</td>
      <td>0.404</td>
      <td>0.676</td>
      <td>1.161</td>
      <td>1.296</td>
      <td>0.944</td>
      <td>0.815</td>
      <td>...</td>
      <td>0.625</td>
      <td>0.739</td>
      <td>0.400</td>
      <td>0.531</td>
      <td>0.488</td>
      <td>0.347</td>
      <td>0.523</td>
      <td>0.950</td>
      <td>0.289</td>
      <td>0.481</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.788</td>
      <td>1.526</td>
      <td>1.805</td>
      <td>1.612</td>
      <td>2.257</td>
      <td>1.576</td>
      <td>1.776</td>
      <td>1.715</td>
      <td>1.989</td>
      <td>2.068</td>
      <td>...</td>
      <td>1.801</td>
      <td>1.876</td>
      <td>1.935</td>
      <td>2.241</td>
      <td>1.605</td>
      <td>2.208</td>
      <td>1.077</td>
      <td>1.374</td>
      <td>1.763</td>
      <td>1.836</td>
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
      <th>ADLLLSTQPGREEGSPLELER_na</th>
      <th>AFDSGIIPMEFVNK_na</th>
      <th>AMVSEFLK_na</th>
      <th>AQIFANTVDNAR_na</th>
      <th>AVCMLSNTTAVAEAWAR_na</th>
      <th>DMFQETMEAMR_na</th>
      <th>DNSTMGYMMAK_na</th>
      <th>DPFAHLPK_na</th>
      <th>DPVQEAWAEDVDLR_na</th>
      <th>EAPPMEKPEVVK_na</th>
      <th>...</th>
      <th>SLEDQVEMLR_na</th>
      <th>SLESLHSFVAAATK_na</th>
      <th>TGTAEMSSILEER_na</th>
      <th>TLFGLHLSQK_na</th>
      <th>TLNDELEIIEGMK_na</th>
      <th>TSIAIDTIINQK_na</th>
      <th>VAPEEHPVLLTEAPLNPK_na</th>
      <th>VTVLFAGQHIAK_na</th>
      <th>YLTVAAVFR_na</th>
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
    
    Optimizer used: <function Adam at 0x0000028849CB6040>
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




    
![png](latent_2D_25_30_files/latent_2D_25_30_108_2.png)
    


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
      <td>0.970361</td>
      <td>0.746624</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.673113</td>
      <td>0.402326</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.505905</td>
      <td>0.413212</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.430399</td>
      <td>0.368834</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.391830</td>
      <td>0.364377</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.367334</td>
      <td>0.351250</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.349171</td>
      <td>0.346534</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.339219</td>
      <td>0.344253</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.332567</td>
      <td>0.341038</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.327033</td>
      <td>0.340413</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_112_1.png)
    



```python
# L(zip(learn.recorder.iters, learn.recorder.values))
```


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (4926, 4926)



MSE on transformed data is not too interesting for comparision between models if these use different standardizations


```python
learn.loss_func(pred, target)  # MSE in transformed space not too interesting
```




    TensorBase(0.3428)




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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>29.346</td>
      <td>30.578</td>
      <td>30.484</td>
      <td>29.659</td>
      <td>29.710</td>
      <td>29.545</td>
    </tr>
    <tr>
      <th>AVCMLSNTTAVAEAWAR</th>
      <td>27.524</td>
      <td>28.473</td>
      <td>28.625</td>
      <td>27.414</td>
      <td>27.817</td>
      <td>27.773</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>28.866</td>
      <td>30.433</td>
      <td>30.561</td>
      <td>29.379</td>
      <td>29.883</td>
      <td>29.760</td>
    </tr>
    <tr>
      <th>TGTAEMSSILEER</th>
      <td>29.238</td>
      <td>30.100</td>
      <td>30.101</td>
      <td>29.453</td>
      <td>29.384</td>
      <td>29.333</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>28.794</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>29.345</td>
      <td>28.673</td>
      <td>29.097</td>
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
      <th>DNSTMGYMMAK</th>
      <td>26.222</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>27.218</td>
      <td>26.860</td>
      <td>26.906</td>
    </tr>
    <tr>
      <th>GLGTDEDSLIEIICSR</th>
      <td>32.466</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>32.584</td>
      <td>32.082</td>
      <td>32.264</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>29.717</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>29.779</td>
      <td>29.331</td>
      <td>29.345</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>30.843</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>30.904</td>
      <td>30.710</td>
      <td>30.723</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
      <td>29.968</td>
      <td>29.761</td>
      <td>30.305</td>
      <td>30.027</td>
      <td>30.084</td>
    </tr>
  </tbody>
</table>
<p>4926 rows × 6 columns</p>
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
      <td>-0.459</td>
      <td>-0.288</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>-0.696</td>
      <td>-0.548</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.875</td>
      <td>-0.748</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.181</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.653</td>
      <td>-0.642</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_122_1.png)
    



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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_123_1.png)
    


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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <th>AFDSGIIPMEFVNK</th>
      <th>AMVSEFLK</th>
      <th>AQIFANTVDNAR</th>
      <th>AVCMLSNTTAVAEAWAR</th>
      <th>DMFQETMEAMR</th>
      <th>DNSTMGYMMAK</th>
      <th>DPFAHLPK</th>
      <th>DPVQEAWAEDVDLR</th>
      <th>EAPPMEKPEVVK</th>
      <th>...</th>
      <th>SLEDQVEMLR</th>
      <th>SLESLHSFVAAATK</th>
      <th>TGTAEMSSILEER</th>
      <th>TLFGLHLSQK</th>
      <th>TLNDELEIIEGMK</th>
      <th>TSIAIDTIINQK</th>
      <th>VAPEEHPVLLTEAPLNPK</th>
      <th>VTVLFAGQHIAK</th>
      <th>YLTVAAVFR</th>
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
      <td>0.625</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.490</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.580</td>
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
      <td>0.642</td>
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
      <td>0.774</td>
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
      <td>0.756</td>
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
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.663</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.644</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.597</td>
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
    
    Optimizer used: <function Adam at 0x0000028849CB6040>
    Loss function: <function loss_fct_vae at 0x0000028849CDD940>
    
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




    
![png](latent_2D_25_30_files/latent_2D_25_30_136_2.png)
    



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
      <td>1986.837158</td>
      <td>215.524643</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1946.551147</td>
      <td>208.605545</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1885.978516</td>
      <td>199.759216</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1836.474243</td>
      <td>195.452835</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1804.492676</td>
      <td>193.951614</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1784.040161</td>
      <td>192.539658</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1769.402954</td>
      <td>192.554794</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1759.691406</td>
      <td>192.706909</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1752.457520</td>
      <td>192.660873</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1746.735718</td>
      <td>192.634033</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_138_1.png)
    


### Evaluation


```python
# reorder True: Only 500 predictions returned
pred, target = learn.get_preds(act=noop, concat_dim=0, reorder=False)
len(pred), len(target)
```








    (3, 4926)




```python
len(pred[0])
```




    4926




```python
learn.loss_func(pred, target)
```




    tensor(3047.3096)




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
      <th>ADLLLSTQPGREEGSPLELER</th>
      <td>29.346</td>
      <td>30.578</td>
      <td>30.484</td>
      <td>29.659</td>
      <td>29.710</td>
      <td>29.545</td>
      <td>30.668</td>
    </tr>
    <tr>
      <th>AVCMLSNTTAVAEAWAR</th>
      <td>27.524</td>
      <td>28.473</td>
      <td>28.625</td>
      <td>27.414</td>
      <td>27.817</td>
      <td>27.773</td>
      <td>28.682</td>
    </tr>
    <tr>
      <th>NALESYAFNMK</th>
      <td>28.866</td>
      <td>30.433</td>
      <td>30.561</td>
      <td>29.379</td>
      <td>29.883</td>
      <td>29.760</td>
      <td>30.682</td>
    </tr>
    <tr>
      <th>TGTAEMSSILEER</th>
      <td>29.238</td>
      <td>30.100</td>
      <td>30.101</td>
      <td>29.453</td>
      <td>29.384</td>
      <td>29.333</td>
      <td>30.211</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <th>AFDSGIIPMEFVNK</th>
      <td>28.794</td>
      <td>29.828</td>
      <td>29.328</td>
      <td>29.345</td>
      <td>28.673</td>
      <td>29.097</td>
      <td>29.447</td>
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
      <th>DNSTMGYMMAK</th>
      <td>26.222</td>
      <td>27.516</td>
      <td>28.627</td>
      <td>27.218</td>
      <td>26.860</td>
      <td>26.906</td>
      <td>28.806</td>
    </tr>
    <tr>
      <th>GLGTDEDSLIEIICSR</th>
      <td>32.466</td>
      <td>31.824</td>
      <td>31.328</td>
      <td>32.584</td>
      <td>32.082</td>
      <td>32.264</td>
      <td>31.485</td>
    </tr>
    <tr>
      <th>ILNIFGVIK</th>
      <td>29.717</td>
      <td>29.339</td>
      <td>29.172</td>
      <td>29.779</td>
      <td>29.331</td>
      <td>29.345</td>
      <td>29.335</td>
    </tr>
    <tr>
      <th>MPSLPSYK</th>
      <td>30.843</td>
      <td>30.903</td>
      <td>30.799</td>
      <td>30.904</td>
      <td>30.710</td>
      <td>30.723</td>
      <td>30.846</td>
    </tr>
    <tr>
      <th>NILFVITKPDVYK</th>
      <td>31.059</td>
      <td>29.968</td>
      <td>29.761</td>
      <td>30.305</td>
      <td>30.027</td>
      <td>30.084</td>
      <td>29.886</td>
    </tr>
  </tbody>
</table>
<p>4926 rows × 7 columns</p>
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
      <td>-0.024</td>
      <td>-0.089</td>
    </tr>
    <tr>
      <th>20181219_QE3_nLC3_TSB_QC_MNT_HeLa_01</th>
      <td>0.107</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>20181221_QE8_nLC0_NHS_MNT_HeLa_01</th>
      <td>-0.077</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>20181222_QE9_nLC9_QC_50CM_HeLa1</th>
      <td>0.137</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>20181223_QE7_nLC7_RJC_MEM_MNT_HeLa_01</th>
      <td>-0.078</td>
      <td>-0.099</td>
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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_146_1.png)
    



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
    


    
![png](latent_2D_25_30_files/latent_2D_25_30_147_1.png)
    


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

    vaep - INFO     Drop indices for replicates: [('20181228_QE5_nLC5_OOE_QC_MNT_HELA_15cm_250ng_RO-007', 'HSQFIGYPITLFVEK'), ('20190107_QE2_NLC10_ANHO_QC_MNT_HELA_02', 'GLGTDEDSLIEIICSR'), ('20190111_QE2_NLC10_ANHO_QC_MNT_HELA_03', 'VAPEEHPVLLTEAPLNPK'), ('20190111_QE8_nLC1_ASD_QC_HeLa_01', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190118_QE9_nLC9_NHS_MNT_HELA_50cm_04', 'LTPEEEEILNK'), ('20190121_QE4_LC6_IAH_QC_MNT_HeLa_250ng_03', 'NALESYAFNMK'), ('20190129_QE8_nLC14_FaCo_QC_MNT_50cm_Hela_20190129205246', 'MPSLPSYK'), ('20190201_QE10_nLC0_NHS_MNT_HELA_45cm_01', 'DMFQETMEAMR'), ('20190219_QE2_NLC1_GP_QC_MNT_HELA_01', 'AQIFANTVDNAR'), ('20190221_QE4_LC12_IAH_QC_MNT_HeLa_02', 'FVMQEEFSR'), ('20190221_QE8_nLC9_JM_QC_MNT_HeLa_01', 'LGGSAVISLEGKPL'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_5', 'GLGTDEDSLIEIICSR'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_6', 'LAPITSDPTEATAVGAVEASFK'), ('20190225_QE10_PhGe_Evosep_88min_HeLa_8', 'TSIAIDTIINQK'), ('20190226_QE1_nLC2_AB_QC_MNT_HELA_01', 'EAYPGDVFYLHSR'), ('20190226_QE1_nLC2_AB_QC_MNT_HELA_02', 'KFDQLLAEEK'), ('20190305_QE4_LC12_JE-IAH_QC_MNT_HeLa_03', 'TLFGLHLSQK'), ('20190308_QE3_nLC5_MR_QC_MNT_HeLa_Easy_01', 'FDASFFGVHPK'), ('20190308_QE3_nLC5_MR_QC_MNT_HeLa_Easyctcdon_02', 'DPFAHLPK'), ('20190308_QE3_nLC5_MR_QC_MNT_Hela_CPRHEasy_03', 'KFDQLLAEEK'), ('20190308_QE9_nLC0_FaCo_QC_MNT_Hela_50cm_newcol', 'DNSTMGYMMAK'), ('20190315_QE2_NLC1_GP_MNT_HELA_01', 'SLESLHSFVAAATK'), ('20190328_QE1_nLC2_GP_MNT_QC_hela_01', 'HIDSAHLYNNEEQVGLAIR'), ('20190401_QE8_nLC14_RS_QC_MNT_Hela_50cm_01', 'YMACCLLYR'), ('20190411_QE3_nLC5_DS_QC_MNT_HeLa_02', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190423_QE3_nLC5_DS_QC_MNT_HeLa_01', 'LMIEMDGTENK'), ('20190426_QE10_nLC9_LiNi_QC_MNT_15cm_HeLa_02', 'VTVLFAGQHIAK'), ('20190509_QE4_LC12_AS_QC_MNT_HeLa_01', 'DNSTMGYMMAK'), ('20190515_QX6_ChDe_MA_HeLa_Br14_500ng_LC09', 'LTPEEEEILNK'), ('20190604_QE8_nLC14_ASD_QC_MNT_15cm_Hela_01', 'VAPEEHPVLLTEAPLNPK'), ('20190611_QX0_MePh_MA_HeLa_500ng_LC07_4', 'IGEEQSAEDAEDGPPELLFIHGGHTAK'), ('20190611_QX4_JiYu_MA_HeLa_500ng', 'GLGTDEDSLIEIICSR'), ('20190611_QX7_IgPa_MA_HeLa_Br14_500ng_190618134442', 'DPVQEAWAEDVDLR'), ('20190617_QX4_JiYu_MA_HeLa_500ng', 'HSQFIGYPITLFVEK'), ('20190620_QE2_NLC1_GP_QC_MNT_HELA_01', 'LGGSAVISLEGKPL'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_01', 'AQIFANTVDNAR'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_02', 'LSFQHDPETSVLVLR'), ('20190621_QE1_nLC2_ANHO_QC_MNT_HELA_03', 'LSFQHDPETSVLVLR'), ('20190621_QX2_SeVW_MA_HeLa_500ng_LC05', 'ADLLLSTQPGREEGSPLELER'), ('20190621_QX3_MePh_MA_Hela_500ng_LC15', 'AMVSEFLK'), ('20190625_QE9_nLC0_RG_MNT_Hela_MUC_50cm_2', 'AFDSGIIPMEFVNK'), ('20190625_QX0_MaPe_MA_HeLa_500ng_LC07_1', 'AMVSEFLK'), ('20190628_QE1_nLC13_ANHO_QC_MNT_HELA_01', 'EAPPMEKPEVVK'), ('20190629_QX4_JiYu_MA_HeLa_500ng_MAX_ALLOWED', 'IVLLDSSLEYK'), ('20190630_QE9_nLC0_RG_MNT_Hela_MUC_50cm_1', 'FWEVISDEHGIDPTGTYHGDSDLQLER'), ('20190701_QE1_nLC13_ANHO_QC_MNT_HELA_02', 'FDASFFGVHPK'), ('20190701_QE1_nLC13_ANHO_QC_MNT_HELA_03', 'AQIFANTVDNAR'), ('20190701_QE8_nLC14_GP_QC_MNT_15cm_Hela_01', 'ILNIFGVIK'), ('20190702_QX0_AnBr_MA_HeLa_500ng_LC07_01_190702180001', 'SEHPGLSIGDTAK'), ('20190715_QE10_nLC0_LiNi_QC_MNT_15cm_HeLa_MUC_01', 'ESYSVYVYK'), ('20190715_QE4_LC12_IAH_QC_MNT_HeLa_01', 'EGIPPDQQR'), ('20190718_QE8_nLC14_RG_QC_HeLa_MUC_50cm_1', 'SLESLHSFVAAATK'), ('20190724_QX3_MiWi_MA_Hela_500ng_LC15', 'AFDSGIIPMEFVNK'), ('20190725_QE8_nLC14_ASD_QC_MNT_HeLa_01', 'ILNIFGVIK'), ('20190730_QE1_nLC2_GP_MNT_HELA_01', 'AMVSEFLK'), ('20190802_QE3_nLC3_DBJ_AMV_QC_MNT_HELA_01', 'ILNIFGVIK'), ('20190802_QX7_AlRe_MA_HeLa_Br14_500ng', 'MPSLPSYK'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_350ng', 'GLGTDEDSLIEIICSR'), ('20190803_QE9_nLC13_RG_SA_HeLa_50cm_400ng', 'GLGTDEDSLIEIICSR')]
    




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
      <td>0.638</td>
      <td>0.679</td>
      <td>1.706</td>
      <td>2.053</td>
      <td>2.086</td>
      <td>2.232</td>
    </tr>
    <tr>
      <th>MAE</th>
      <td>0.487</td>
      <td>0.509</td>
      <td>0.894</td>
      <td>1.051</td>
      <td>1.067</td>
      <td>1.042</td>
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
