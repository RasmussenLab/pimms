## Sampling with weights in Pandas

- sampling core utilities is based on numpy (see docstring)
- [file](https://github.com/pandas-dev/pandas/blob/49d371364b734b47c85733aac74b03ac4400c629/pandas/core/sample.py) containing sampling functions

## Some random data


```python
from vaep.utils import create_random_df
X = create_random_df(100, 15, prop_na=0.1).stack().to_frame(
    'intensity').reset_index()

freq = X.peptide.value_counts().sort_index()
freq.name = 'freq'

X = X.set_index(keys=list(X.columns[0:2]))  # to_list as an alternative
freq
```




    feat_00   88
    feat_01   88
    feat_02   94
    feat_03   91
    feat_04   87
    feat_05   88
    feat_06   86
    feat_07   93
    feat_08   87
    feat_09   90
    feat_10   91
    feat_11   90
    feat_12   94
    feat_13   87
    feat_14   87
    Name: freq, dtype: int64




```python
X
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
      <th rowspan="5" valign="top">sample_000</th>
      <th>feat_00</th>
      <td>13.471</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>13.852</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>24.578</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>10.293</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>20.624</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">sample_099</th>
      <th>feat_10</th>
      <td>27.959</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>5.352</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>28.055</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>27.344</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>6.033</td>
    </tr>
  </tbody>
</table>
<p>1341 rows × 1 columns</p>
</div>




```python
print(f"Based on total number of rows, 95% is roughly: {int(len(X) * 0.95)}")
print("Based on each sample's 95% obs, it is roughly: {}".format(
    X.groupby('Sample ID').apply(lambda df: int(len(df) * 0.95)).sum()))
```

    Based on total number of rows, 95% is roughly: 1273
    Based on each sample's 95% obs, it is roughly: 1241
    

## Samling using a column with the weights


```python
X = X.join(freq, on='peptide')
X
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">sample_000</th>
      <th>feat_00</th>
      <td>13.471</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>13.852</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>24.578</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>10.293</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>20.624</td>
      <td>86</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">sample_099</th>
      <th>feat_10</th>
      <td>27.959</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>5.352</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>28.055</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>27.344</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>6.033</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
<p>1341 rows × 2 columns</p>
</div>




```python
t = X.groupby('Sample ID').get_group('sample_003')
t
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="14" valign="top">sample_003</th>
      <th>feat_00</th>
      <td>6.099</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_01</th>
      <td>24.746</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>3.475</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>7.479</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>24.095</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_05</th>
      <td>22.287</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>16.557</td>
      <td>86</td>
    </tr>
    <tr>
      <th>feat_07</th>
      <td>18.809</td>
      <td>93</td>
    </tr>
    <tr>
      <th>feat_08</th>
      <td>13.072</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_10</th>
      <td>26.142</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>27.636</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>19.934</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>26.157</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>20.427</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>




```python
t.sample(frac=0.75, weights='freq')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">sample_003</th>
      <th>feat_02</th>
      <td>3.475</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>26.157</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_01</th>
      <td>24.746</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_05</th>
      <td>22.287</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_08</th>
      <td>13.072</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>7.479</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>27.636</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_00</th>
      <td>6.099</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>20.427</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>24.095</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>



Sampling the entire DataFrame based on the freq will normalize on N of all rows. The normalization leaves relative frequency the same (if no floating point unprecision is reached)


```python
# number of rows not the same as when using groupby (see above)
X.sample(frac=0.95, weights='freq')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Sample ID</th>
      <th>peptide</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_033</th>
      <th>feat_03</th>
      <td>6.949</td>
      <td>91</td>
    </tr>
    <tr>
      <th>sample_015</th>
      <th>feat_02</th>
      <td>3.219</td>
      <td>94</td>
    </tr>
    <tr>
      <th>sample_056</th>
      <th>feat_05</th>
      <td>28.555</td>
      <td>88</td>
    </tr>
    <tr>
      <th>sample_009</th>
      <th>feat_06</th>
      <td>18.601</td>
      <td>86</td>
    </tr>
    <tr>
      <th>sample_058</th>
      <th>feat_08</th>
      <td>16.588</td>
      <td>87</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>sample_059</th>
      <th>feat_03</th>
      <td>18.749</td>
      <td>91</td>
    </tr>
    <tr>
      <th>sample_082</th>
      <th>feat_03</th>
      <td>7.334</td>
      <td>91</td>
    </tr>
    <tr>
      <th>sample_063</th>
      <th>feat_07</th>
      <td>29.337</td>
      <td>93</td>
    </tr>
    <tr>
      <th>sample_047</th>
      <th>feat_08</th>
      <td>23.540</td>
      <td>87</td>
    </tr>
    <tr>
      <th>sample_064</th>
      <th>feat_02</th>
      <td>28.617</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
<p>1274 rows × 2 columns</p>
</div>



### Sampling fails with groupby, reindexing needed

The above is not mapped one to one to the groupby sample method. One needs to apply it to every single df.


```python
# X.groupby('Sample ID').sample(frac=0.95, weights='freq') # does not work
X.groupby('Sample ID').apply(
    lambda df: df.reset_index(0, drop=True).sample(frac=0.95, weights='freq')
).drop('freq', axis=1)
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
      <th rowspan="5" valign="top">sample_000</th>
      <th>feat_04</th>
      <td>10.293</td>
    </tr>
    <tr>
      <th>feat_10</th>
      <td>15.041</td>
    </tr>
    <tr>
      <th>feat_00</th>
      <td>13.471</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>24.578</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>27.786</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">sample_099</th>
      <th>feat_00</th>
      <td>5.455</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>9.720</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>28.055</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>6.033</td>
    </tr>
    <tr>
      <th>feat_09</th>
      <td>20.538</td>
    </tr>
  </tbody>
</table>
<p>1243 rows × 1 columns</p>
</div>



And passing a Series need the original X to be indexed the same (multi-indices are not supported)


```python
# for i, t in X.groupby('Sample ID'):
#     t = t.sample(frac=0.75, weights=freq)
# t
```


```python
X = X.reset_index('Sample ID')
X
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_00</th>
      <td>sample_000</td>
      <td>13.471</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>sample_000</td>
      <td>13.852</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>sample_000</td>
      <td>24.578</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>sample_000</td>
      <td>10.293</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>sample_000</td>
      <td>20.624</td>
      <td>86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>feat_10</th>
      <td>sample_099</td>
      <td>27.959</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>sample_099</td>
      <td>5.352</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>sample_099</td>
      <td>28.055</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>sample_099</td>
      <td>27.344</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_14</th>
      <td>sample_099</td>
      <td>6.033</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
<p>1341 rows × 3 columns</p>
</div>




```python
X.groupby(by='Sample ID').sample(frac=0.95, weights=freq)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_11</th>
      <td>sample_000</td>
      <td>27.786</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>sample_000</td>
      <td>24.578</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>sample_000</td>
      <td>13.852</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>sample_000</td>
      <td>20.624</td>
      <td>86</td>
    </tr>
    <tr>
      <th>feat_07</th>
      <td>sample_000</td>
      <td>7.053</td>
      <td>93</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>feat_07</th>
      <td>sample_099</td>
      <td>14.483</td>
      <td>93</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>sample_099</td>
      <td>9.720</td>
      <td>86</td>
    </tr>
    <tr>
      <th>feat_00</th>
      <td>sample_099</td>
      <td>5.455</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_08</th>
      <td>sample_099</td>
      <td>6.910</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>sample_099</td>
      <td>5.352</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
<p>1243 rows × 3 columns</p>
</div>




```python
X.groupby(by='Sample ID').get_group('sample_002')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample ID</th>
      <th>intensity</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>peptide</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_00</th>
      <td>sample_002</td>
      <td>27.595</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_01</th>
      <td>sample_002</td>
      <td>12.967</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_02</th>
      <td>sample_002</td>
      <td>11.771</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_03</th>
      <td>sample_002</td>
      <td>14.880</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_04</th>
      <td>sample_002</td>
      <td>4.008</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_05</th>
      <td>sample_002</td>
      <td>5.056</td>
      <td>88</td>
    </tr>
    <tr>
      <th>feat_06</th>
      <td>sample_002</td>
      <td>9.375</td>
      <td>86</td>
    </tr>
    <tr>
      <th>feat_07</th>
      <td>sample_002</td>
      <td>25.798</td>
      <td>93</td>
    </tr>
    <tr>
      <th>feat_08</th>
      <td>sample_002</td>
      <td>9.930</td>
      <td>87</td>
    </tr>
    <tr>
      <th>feat_09</th>
      <td>sample_002</td>
      <td>16.754</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_10</th>
      <td>sample_002</td>
      <td>24.120</td>
      <td>91</td>
    </tr>
    <tr>
      <th>feat_11</th>
      <td>sample_002</td>
      <td>4.314</td>
      <td>90</td>
    </tr>
    <tr>
      <th>feat_12</th>
      <td>sample_002</td>
      <td>25.844</td>
      <td>94</td>
    </tr>
    <tr>
      <th>feat_13</th>
      <td>sample_002</td>
      <td>21.958</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>



## Sanity check: Downsampling the first feature


```python
freq.loc['feat_00'] = 1  # none should be selected
```


```python
freq = freq / freq.sum()
freq
```




    feat_00   0.001
    feat_01   0.070
    feat_02   0.075
    feat_03   0.073
    feat_04   0.069
    feat_05   0.070
    feat_06   0.069
    feat_07   0.074
    feat_08   0.069
    feat_09   0.072
    feat_10   0.073
    feat_11   0.072
    feat_12   0.075
    feat_13   0.069
    feat_14   0.069
    Name: freq, dtype: float64




```python
X.groupby(by='Sample ID').sample(
    frac=0.5, weights=freq).sort_index().reset_index().peptide.value_counts()
```




    feat_05   59
    feat_02   54
    feat_12   53
    feat_03   53
    feat_14   50
    feat_11   49
    feat_09   48
    feat_07   47
    feat_13   47
    feat_06   45
    feat_01   42
    feat_04   42
    feat_10   41
    feat_08   36
    Name: peptide, dtype: int64



## Using a series

- in the above approach, sampling weights might be readjusted based on the values present in `sample` as `NAN`s lead to the weights not summing up. Alteratively one could loop through the wide format rows and sample values from these.


```python
freq
```




    feat_00   0.001
    feat_01   0.070
    feat_02   0.075
    feat_03   0.073
    feat_04   0.069
    feat_05   0.070
    feat_06   0.069
    feat_07   0.074
    feat_08   0.069
    feat_09   0.072
    feat_10   0.073
    feat_11   0.072
    feat_12   0.075
    feat_13   0.069
    feat_14   0.069
    Name: freq, dtype: float64




```python
X = X.drop('freq', axis=1).set_index(
    'Sample ID', append=True).squeeze().unstack(0)
X
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>peptide</th>
      <th>feat_00</th>
      <th>feat_01</th>
      <th>feat_02</th>
      <th>feat_03</th>
      <th>feat_04</th>
      <th>feat_05</th>
      <th>feat_06</th>
      <th>feat_07</th>
      <th>feat_08</th>
      <th>feat_09</th>
      <th>feat_10</th>
      <th>feat_11</th>
      <th>feat_12</th>
      <th>feat_13</th>
      <th>feat_14</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_000</th>
      <td>13.471</td>
      <td>NaN</td>
      <td>13.852</td>
      <td>24.578</td>
      <td>10.293</td>
      <td>NaN</td>
      <td>20.624</td>
      <td>7.053</td>
      <td>24.535</td>
      <td>20.481</td>
      <td>15.041</td>
      <td>27.786</td>
      <td>19.603</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sample_001</th>
      <td>26.472</td>
      <td>27.281</td>
      <td>12.528</td>
      <td>26.638</td>
      <td>6.443</td>
      <td>27.948</td>
      <td>13.542</td>
      <td>NaN</td>
      <td>6.119</td>
      <td>17.851</td>
      <td>25.512</td>
      <td>24.466</td>
      <td>26.741</td>
      <td>3.579</td>
      <td>21.051</td>
    </tr>
    <tr>
      <th>sample_002</th>
      <td>27.595</td>
      <td>12.967</td>
      <td>11.771</td>
      <td>14.880</td>
      <td>4.008</td>
      <td>5.056</td>
      <td>9.375</td>
      <td>25.798</td>
      <td>9.930</td>
      <td>16.754</td>
      <td>24.120</td>
      <td>4.314</td>
      <td>25.844</td>
      <td>21.958</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sample_003</th>
      <td>6.099</td>
      <td>24.746</td>
      <td>3.475</td>
      <td>7.479</td>
      <td>24.095</td>
      <td>22.287</td>
      <td>16.557</td>
      <td>18.809</td>
      <td>13.072</td>
      <td>NaN</td>
      <td>26.142</td>
      <td>27.636</td>
      <td>19.934</td>
      <td>26.157</td>
      <td>20.427</td>
    </tr>
    <tr>
      <th>sample_004</th>
      <td>3.276</td>
      <td>22.409</td>
      <td>26.393</td>
      <td>14.989</td>
      <td>27.344</td>
      <td>18.466</td>
      <td>NaN</td>
      <td>19.615</td>
      <td>5.133</td>
      <td>15.270</td>
      <td>19.781</td>
      <td>NaN</td>
      <td>11.369</td>
      <td>18.725</td>
      <td>15.991</td>
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
    </tr>
    <tr>
      <th>sample_095</th>
      <td>NaN</td>
      <td>17.769</td>
      <td>5.093</td>
      <td>5.268</td>
      <td>7.379</td>
      <td>13.006</td>
      <td>5.194</td>
      <td>15.402</td>
      <td>9.701</td>
      <td>20.449</td>
      <td>4.253</td>
      <td>13.477</td>
      <td>11.306</td>
      <td>21.899</td>
      <td>20.080</td>
    </tr>
    <tr>
      <th>sample_096</th>
      <td>24.642</td>
      <td>NaN</td>
      <td>25.374</td>
      <td>22.540</td>
      <td>20.751</td>
      <td>22.650</td>
      <td>4.858</td>
      <td>9.490</td>
      <td>NaN</td>
      <td>15.882</td>
      <td>6.681</td>
      <td>22.297</td>
      <td>26.421</td>
      <td>20.094</td>
      <td>11.791</td>
    </tr>
    <tr>
      <th>sample_097</th>
      <td>NaN</td>
      <td>15.636</td>
      <td>28.461</td>
      <td>19.605</td>
      <td>11.861</td>
      <td>8.055</td>
      <td>26.539</td>
      <td>22.262</td>
      <td>21.054</td>
      <td>26.533</td>
      <td>16.930</td>
      <td>8.532</td>
      <td>13.957</td>
      <td>24.647</td>
      <td>22.729</td>
    </tr>
    <tr>
      <th>sample_098</th>
      <td>8.363</td>
      <td>23.546</td>
      <td>NaN</td>
      <td>11.496</td>
      <td>8.020</td>
      <td>7.415</td>
      <td>8.394</td>
      <td>29.377</td>
      <td>10.098</td>
      <td>11.376</td>
      <td>24.136</td>
      <td>5.800</td>
      <td>20.178</td>
      <td>4.772</td>
      <td>10.712</td>
    </tr>
    <tr>
      <th>sample_099</th>
      <td>5.455</td>
      <td>12.118</td>
      <td>25.230</td>
      <td>12.281</td>
      <td>28.024</td>
      <td>18.249</td>
      <td>9.720</td>
      <td>14.483</td>
      <td>6.910</td>
      <td>20.538</td>
      <td>27.959</td>
      <td>5.352</td>
      <td>28.055</td>
      <td>27.344</td>
      <td>6.033</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>




```python
X.iloc[0].sample(frac=0.8, weights=freq).sort_index()
```




    peptide
    feat_01      NaN
    feat_02   13.852
    feat_03   24.578
    feat_05      NaN
    feat_06   20.624
    feat_07    7.053
    feat_08   24.535
    feat_09   20.481
    feat_10   15.041
    feat_12   19.603
    feat_13      NaN
    feat_14      NaN
    Name: sample_000, dtype: float64



Sampling using the wide format would garuantee that the weights are not adjusted based on missing values, but that instead missing values are sample into on or the other set. Ultimately `NaN`s are dropped also in this approach.


```python
import pandas as pd
data = {}
for row_key in X.index:
    data[row_key] = X.loc[row_key].sample(frac=0.8, weights=freq)
pd.DataFrame(data).stack()
```




    peptide            
    feat_00  sample_004    3.276
             sample_012    8.782
             sample_035    7.149
             sample_044   15.908
             sample_090   29.630
                           ...  
    feat_14  sample_095   20.080
             sample_096   11.791
             sample_097   22.729
             sample_098   10.712
             sample_099    6.033
    Length: 1078, dtype: float64


