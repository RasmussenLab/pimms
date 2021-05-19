# Analysis of `summaries.txt` information

- number of raw files (no here)
- number of raw files with MQ-Output
- MS1 per file
- MS2 per file


```python
import ipywidgets as widgets

from src.data_objects import MqAllSummaries
import vaep

from src.src.config import FN_ALL_SUMMARIES

mq_all_summaries = MqAllSummaries()
mq_all_summaries.df.describe().T
```

    MqAllSummaries: Load summaries of 9381 folders.
    




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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Enzyme first search</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Enzyme mode first search</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Multi modifications</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Variable modifications first search</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Multiplicity</th>
      <td>9,381.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Max. missed cleavages</th>
      <td>9,381.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Labels0</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Time-dependent recalibration</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>MS</th>
      <td>9,381.0</td>
      <td>11,285.2</td>
      <td>5,682.4</td>
      <td>0.0</td>
      <td>9,004.0</td>
      <td>11,529.0</td>
      <td>12,883.0</td>
      <td>47,267.0</td>
    </tr>
    <tr>
      <th>MS/MS</th>
      <td>9,381.0</td>
      <td>79,293.1</td>
      <td>43,645.3</td>
      <td>0.0</td>
      <td>36,327.0</td>
      <td>84,141.0</td>
      <td>117,083.0</td>
      <td>189,955.0</td>
    </tr>
    <tr>
      <th>MS3</th>
      <td>9,381.0</td>
      <td>46.4</td>
      <td>1,220.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66,980.0</td>
    </tr>
    <tr>
      <th>MS/MS Submitted</th>
      <td>9,381.0</td>
      <td>91,523.3</td>
      <td>49,986.0</td>
      <td>0.0</td>
      <td>41,827.0</td>
      <td>97,355.0</td>
      <td>133,756.0</td>
      <td>230,182.0</td>
    </tr>
    <tr>
      <th>MS/MS Submitted (SIL)</th>
      <td>9,381.0</td>
      <td>67,044.9</td>
      <td>37,788.4</td>
      <td>0.0</td>
      <td>30,534.0</td>
      <td>71,627.0</td>
      <td>100,832.0</td>
      <td>157,459.0</td>
    </tr>
    <tr>
      <th>MS/MS Submitted (ISO)</th>
      <td>9,381.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MS/MS Submitted (PEAK)</th>
      <td>9,381.0</td>
      <td>24,478.4</td>
      <td>15,152.1</td>
      <td>0.0</td>
      <td>12,960.0</td>
      <td>25,388.0</td>
      <td>32,900.0</td>
      <td>126,478.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified</th>
      <td>9,381.0</td>
      <td>31,944.8</td>
      <td>20,572.5</td>
      <td>0.0</td>
      <td>9,697.0</td>
      <td>39,963.0</td>
      <td>49,095.0</td>
      <td>76,652.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (SIL)</th>
      <td>9,381.0</td>
      <td>30,576.1</td>
      <td>19,735.9</td>
      <td>0.0</td>
      <td>9,323.0</td>
      <td>38,040.0</td>
      <td>47,185.0</td>
      <td>74,314.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (ISO)</th>
      <td>9,381.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (PEAK)</th>
      <td>9,381.0</td>
      <td>1,368.7</td>
      <td>997.6</td>
      <td>0.0</td>
      <td>406.0</td>
      <td>1,513.0</td>
      <td>2,047.0</td>
      <td>7,729.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified [%]</th>
      <td>9,381.0</td>
      <td>31.3</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>36.0</td>
      <td>41.0</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (SIL) [%]</th>
      <td>9,381.0</td>
      <td>40.2</td>
      <td>19.2</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>52.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (ISO) [%]</th>
      <td>9,381.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (PEAK) [%]</th>
      <td>9,381.0</td>
      <td>5.8</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.4</td>
      <td>5.7</td>
      <td>7.3</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>Peptide Sequences Identified</th>
      <td>9,381.0</td>
      <td>24,232.4</td>
      <td>15,771.5</td>
      <td>0.0</td>
      <td>7,245.0</td>
      <td>29,946.0</td>
      <td>36,782.0</td>
      <td>54,316.0</td>
    </tr>
    <tr>
      <th>Peaks</th>
      <td>9,381.0</td>
      <td>1,189,631.5</td>
      <td>499,261.2</td>
      <td>0.0</td>
      <td>935,204.0</td>
      <td>1,317,764.0</td>
      <td>1,462,935.0</td>
      <td>5,503,705.0</td>
    </tr>
    <tr>
      <th>Peaks Sequenced</th>
      <td>9,381.0</td>
      <td>73,119.7</td>
      <td>42,111.2</td>
      <td>0.0</td>
      <td>34,084.0</td>
      <td>78,421.0</td>
      <td>109,051.0</td>
      <td>176,131.0</td>
    </tr>
    <tr>
      <th>Peaks Sequenced [%]</th>
      <td>9,378.0</td>
      <td>5.9</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>4.4</td>
      <td>6.4</td>
      <td>7.9</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>Peaks Repeatedly Sequenced</th>
      <td>9,381.0</td>
      <td>3,533.2</td>
      <td>2,913.5</td>
      <td>0.0</td>
      <td>1,442.0</td>
      <td>2,997.0</td>
      <td>4,884.0</td>
      <td>22,374.0</td>
    </tr>
    <tr>
      <th>Peaks Repeatedly Sequenced [%]</th>
      <td>9,287.0</td>
      <td>7.6</td>
      <td>11.8</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>4.1</td>
      <td>6.5</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns</th>
      <td>9,381.0</td>
      <td>155,514.4</td>
      <td>66,895.8</td>
      <td>0.0</td>
      <td>107,488.0</td>
      <td>176,627.0</td>
      <td>202,098.0</td>
      <td>1,013,471.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Sequenced</th>
      <td>9,381.0</td>
      <td>58,522.6</td>
      <td>33,454.2</td>
      <td>0.0</td>
      <td>27,623.0</td>
      <td>64,766.0</td>
      <td>87,417.0</td>
      <td>123,727.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Sequenced (z&gt;1)</th>
      <td>9,381.0</td>
      <td>57,494.2</td>
      <td>32,957.6</td>
      <td>0.0</td>
      <td>27,193.0</td>
      <td>63,874.0</td>
      <td>85,933.0</td>
      <td>121,666.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Sequenced [%]</th>
      <td>9,378.0</td>
      <td>34.4</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Sequenced (z&gt;1) [%]</th>
      <td>9,378.0</td>
      <td>38.1</td>
      <td>14.6</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Repeatedly Sequenced</th>
      <td>9,381.0</td>
      <td>6,585.2</td>
      <td>4,831.2</td>
      <td>0.0</td>
      <td>2,121.0</td>
      <td>6,367.0</td>
      <td>9,853.0</td>
      <td>28,582.0</td>
    </tr>
    <tr>
      <th>Isotope Patterns Repeatedly Sequenced [%]</th>
      <td>9,281.0</td>
      <td>13.2</td>
      <td>12.4</td>
      <td>0.0</td>
      <td>7.3</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>Av. Absolute Mass Deviation [ppm]</th>
      <td>9,240.0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>Mass Standard Deviation [ppm]</th>
      <td>9,240.0</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>1.1</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>Av. Absolute Mass Deviation [mDa]</th>
      <td>9,240.0</td>
      <td>0.5</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>Mass Standard Deviation [mDa]</th>
      <td>9,240.0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>0.7</td>
      <td>0.7</td>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
</div>



Find unique columns, see [post](https://stackoverflow.com/a/54405767/9684872)


```python
from vaep.pandas import unique_cols
unique_cols(mq_all_summaries.df.Multiplicity), unique_cols(mq_all_summaries.df["Variable modifications first search"]) # int, NA
```




    (True, True)




```python
from vaep.pandas import get_unique_non_unique_columns
columns = get_unique_non_unique_columns(mq_all_summaries.df)
mq_all_summaries.df[columns.unique]
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
      <th>Enzyme</th>
      <th>Enzyme mode</th>
      <th>Enzyme first search</th>
      <th>Enzyme mode first search</th>
      <th>Use enzyme first search</th>
      <th>Variable modifications</th>
      <th>Fixed modifications</th>
      <th>Multi modifications</th>
      <th>Variable modifications first search</th>
      <th>Use variable modifications first search</th>
      <th>Requantify</th>
      <th>Multiplicity</th>
      <th>Max. missed cleavages</th>
      <th>Labels0</th>
      <th>LC-MS run type</th>
      <th>Time-dependent recalibration</th>
      <th>MS/MS Submitted (ISO)</th>
      <th>MS/MS Identified (ISO)</th>
      <th>MS/MS Identified (ISO) [%]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20190819_QX2_SeVW_MA_HeLa_500ng_CTCDoff_LC05</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20200924_EXPL6_nLC09_MBK_QC_MNT_HeLa_42cm_FAIMS_500ng_Short_05</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20170509_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20190624_QE4_nLC12_MM_QC_MNT_HELA_01_20190625144904</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20190110_QE2_NLC10_GP_QC_MNT_HELA_01</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>20160609_QE2_nLC1_BTW_SA_hela_W_proteome_exp2_08</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160304_LUMOS1_nLC9_ChKe_DEV_HeLa_10xPatch_AGC4e4_cutoff5e5_01</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160507_LUMOS1_nLC14_RJC_QC_MNTv3_HeLa_03</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160607_QE1_nlc2_BTW_SA_hela_100pctAc-L_PCA-H_1-10_ACK_01</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160714_QE2_nLC0_SS_SA_hela_L_1Gy_M_10Gy_H_2hrs_400mM_pH11</th>
      <td>Trypsin/P</td>
      <td>Specific</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>Standard</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9381 rows × 19 columns</p>
</div>




```python
mq_all_summaries.df[columns.unique].dtypes
```




    Enzyme                                     category
    Enzyme mode                                category
    Enzyme first search                           Int64
    Enzyme mode first search                      Int64
    Use enzyme first search                     boolean
    Variable modifications                     category
    Fixed modifications                        category
    Multi modifications                           Int64
    Variable modifications first search           Int64
    Use variable modifications first search     boolean
    Requantify                                  boolean
    Multiplicity                                  Int64
    Max. missed cleavages                         Int64
    Labels0                                       Int64
    LC-MS run type                             category
    Time-dependent recalibration                  Int64
    MS/MS Submitted (ISO)                         Int64
    MS/MS Identified (ISO)                        Int64
    MS/MS Identified (ISO) [%]                    Int64
    dtype: object




```python
mq_all_summaries.df[columns.unique].iloc[0,:]
```




    Enzyme                                                                 Trypsin/P
    Enzyme mode                                                             Specific
    Enzyme first search                                                         <NA>
    Enzyme mode first search                                                    <NA>
    Use enzyme first search                                                    False
    Variable modifications                     Oxidation (M);Acetyl (Protein N-term)
    Fixed modifications                                          Carbamidomethyl (C)
    Multi modifications                                                         <NA>
    Variable modifications first search                                         <NA>
    Use variable modifications first search                                    False
    Requantify                                                                 False
    Multiplicity                                                                   1
    Max. missed cleavages                                                          2
    Labels0                                                                     <NA>
    LC-MS run type                                                          Standard
    Time-dependent recalibration                                                <NA>
    MS/MS Submitted (ISO)                                                          0
    MS/MS Identified (ISO)                                                         0
    MS/MS Identified (ISO) [%]                                                     0
    Name: 20190819_QX2_SeVW_MA_HeLa_500ng_CTCDoff_LC05, dtype: object



## Analysis of completeness


```python
class col_summary:
    MS = 'MS'
    MS2 =  'MS/MS Identified'

MS_spectra = mq_all_summaries.df[[col_summary.MS, col_summary.MS2]]
def compute_summary(threshold_ms2_identified):
    mask  = MS_spectra[col_summary.MS2] >= threshold_ms2_identified
    display(MS_spectra.loc[mask].describe())

w_ions_range = widgets.IntSlider(value=0.0, min=.0, max=MS_spectra[col_summary.MS2].max())
display(widgets.interactive(compute_summary, threshold_ms2_identified=w_ions_range))
```


    interactive(children=(IntSlider(value=0, description='threshold_ms2_identified', max=76652), Output()), _dom_c…



```python
mask = (MS_spectra < 1).any(axis=1)
MS_spectra.loc[mask]
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
      <th>MS</th>
      <th>MS/MS Identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20190423_QE7_Evo1_UHG_QC_MNT_HELA_02_20190424195619</th>
      <td>5418</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20131112_QE2_UPLC4_Vyt_MNT_HeLa_1</th>
      <td>21864</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20190906_QE8_nLC14_FM_QC_MNT_HeLa_50cm_test4</th>
      <td>7698</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20191021_QE7_Evo4_QC_MNT_Hela_100ng_21min_03</th>
      <td>5402</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20191029_QE8_nLC01_FaCo_MNT_QC_Hela_15cm</th>
      <td>7604</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20180509_QE1_nLC10_BTW_SA_HeLa_AQUA_Ac_E2_0-1pct_F02</th>
      <td>21573</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20180306_QE8_nLC1_BDA_QC_MNT_HeLa_02</th>
      <td>26506</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160627_LUMOS1_nLC13_ChKe_DEV_HeLa_FullMS_DTSon_01_160628223911</th>
      <td>24087</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160627_LUMOS1_nLC13_ChKe_DEV_HeLa_FullMS_DTSoff_01</th>
      <td>24052</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20160715_LUMOS1_nLC13_ChKe_DEV_HeLa_FullMS_DTSon_01</th>
      <td>24227</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>112 rows × 2 columns</p>
</div>




```python

```
