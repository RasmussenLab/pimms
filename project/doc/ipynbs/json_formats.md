# Json Formats

- object is loaded with the correct conversions (but this is re-computed)
- can shared information be saved as "meta" information?

- [`pd.json_normalize`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html) should be able to efficiently combine information


```python
import pandas as pd
from vaep.io.data_objects import MqAllSummaries
from vaep.pandas import get_unique_non_unique_columns

mq_all_summaries = MqAllSummaries()
```

    MqAllSummaries: Load summaries of 9381 folders.
    

## summaries.json

### Table format with schema


```python
# json format with categories
columns = get_unique_non_unique_columns(mq_all_summaries.df)
columns.unique[:2]
```




    Index(['Enzyme', 'Enzyme mode'], dtype='object')




```python
mq_all_summaries.df[columns.unique[:3]].dtypes
```




    Enzyme                 category
    Enzyme mode            category
    Enzyme first search       Int64
    dtype: object




```python
type(mq_all_summaries.df.iloc[0,3])
```




    pandas._libs.missing.NAType




```python
meta = mq_all_summaries.df[columns.unique].iloc[0].to_json(indent=4, orient='table')
# print(meta)
```


```python
pd.read_json(meta, orient='table').T.convert_dtypes()
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
  </tbody>
</table>
</div>




```python
pd.read_json(meta, orient='table') # produce errors when having int columns has NaN
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
      <th>20190819_QX2_SeVW_MA_HeLa_500ng_CTCDoff_LC05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Enzyme</th>
      <td>Trypsin/P</td>
    </tr>
    <tr>
      <th>Enzyme mode</th>
      <td>Specific</td>
    </tr>
    <tr>
      <th>Enzyme first search</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Enzyme mode first search</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Use enzyme first search</th>
      <td>False</td>
    </tr>
    <tr>
      <th>Variable modifications</th>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
    </tr>
    <tr>
      <th>Fixed modifications</th>
      <td>Carbamidomethyl (C)</td>
    </tr>
    <tr>
      <th>Multi modifications</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Variable modifications first search</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Use variable modifications first search</th>
      <td>False</td>
    </tr>
    <tr>
      <th>Requantify</th>
      <td>False</td>
    </tr>
    <tr>
      <th>Multiplicity</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Max. missed cleavages</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Labels0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>LC-MS run type</th>
      <td>Standard</td>
    </tr>
    <tr>
      <th>Time-dependent recalibration</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MS/MS Submitted (ISO)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (ISO)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MS/MS Identified (ISO) [%]</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.options.display.max_columns = len(columns.non_unique)
# mq_all_summaries.df[columns.non_unique]
```


```python
data = mq_all_summaries.df[columns.non_unique].iloc[0:3].to_json()
data = pd.read_json(data)
data
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
      <th>MS/MS</th>
      <th>MS3</th>
      <th>MS/MS Submitted</th>
      <th>MS/MS Submitted (SIL)</th>
      <th>MS/MS Submitted (PEAK)</th>
      <th>MS/MS Identified</th>
      <th>MS/MS Identified (SIL)</th>
      <th>MS/MS Identified (PEAK)</th>
      <th>MS/MS Identified [%]</th>
      <th>MS/MS Identified (SIL) [%]</th>
      <th>MS/MS Identified (PEAK) [%]</th>
      <th>Peptide Sequences Identified</th>
      <th>Peaks</th>
      <th>Peaks Sequenced</th>
      <th>Peaks Sequenced [%]</th>
      <th>Peaks Repeatedly Sequenced</th>
      <th>Peaks Repeatedly Sequenced [%]</th>
      <th>Isotope Patterns</th>
      <th>Isotope Patterns Sequenced</th>
      <th>Isotope Patterns Sequenced (z&gt;1)</th>
      <th>Isotope Patterns Sequenced [%]</th>
      <th>Isotope Patterns Sequenced (z&gt;1) [%]</th>
      <th>Isotope Patterns Repeatedly Sequenced</th>
      <th>Isotope Patterns Repeatedly Sequenced [%]</th>
      <th>Recalibrated</th>
      <th>Av. Absolute Mass Deviation [ppm]</th>
      <th>Mass Standard Deviation [ppm]</th>
      <th>Av. Absolute Mass Deviation [mDa]</th>
      <th>Mass Standard Deviation [mDa]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20190819_QX2_SeVW_MA_HeLa_500ng_CTCDoff_LC05</th>
      <td>12,136</td>
      <td>123,187</td>
      <td>0</td>
      <td>138,029</td>
      <td>108,345</td>
      <td>29,684</td>
      <td>55,341</td>
      <td>53,679</td>
      <td>1,662</td>
      <td>40</td>
      <td>50</td>
      <td>5.6</td>
      <td>46,896</td>
      <td>1,323,164</td>
      <td>121,024</td>
      <td>9.1</td>
      <td>1,420</td>
      <td>1.2</td>
      <td>224,886</td>
      <td>102,240</td>
      <td>101,142</td>
      <td>45</td>
      <td>50</td>
      <td>5,586</td>
      <td>5.5</td>
      <td>+</td>
      <td>0.7</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>20200924_EXPL6_nLC09_MBK_QC_MNT_HeLa_42cm_FAIMS_500ng_Short_05</th>
      <td>20,253</td>
      <td>18,887</td>
      <td>0</td>
      <td>23,175</td>
      <td>14,599</td>
      <td>8,576</td>
      <td>5,030</td>
      <td>4,899</td>
      <td>131</td>
      <td>22</td>
      <td>34</td>
      <td>1.5</td>
      <td>4,005</td>
      <td>597,954</td>
      <td>17,071</td>
      <td>2.9</td>
      <td>462</td>
      <td>2.7</td>
      <td>56,374</td>
      <td>13,006</td>
      <td>12,895</td>
      <td>23</td>
      <td>25</td>
      <td>1,372</td>
      <td>11.0</td>
      <td>+</td>
      <td>1.1</td>
      <td>1.4</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20170509_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <td>14,786</td>
      <td>81,890</td>
      <td>0</td>
      <td>92,462</td>
      <td>71,318</td>
      <td>21,144</td>
      <td>51,464</td>
      <td>48,930</td>
      <td>2,534</td>
      <td>56</td>
      <td>69</td>
      <td>12.0</td>
      <td>39,011</td>
      <td>1,480,448</td>
      <td>76,823</td>
      <td>5.2</td>
      <td>2,657</td>
      <td>3.5</td>
      <td>202,663</td>
      <td>65,457</td>
      <td>64,657</td>
      <td>32</td>
      <td>35</td>
      <td>5,010</td>
      <td>7.7</td>
      <td>+</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
mq_all_summaries.fp_summaries.parent /  mq_all_summaries.fp_summaries.stem / '_meta.json'
```




    WindowsPath('data/processed/all_summaries/_meta.json')




```python
meta = mq_all_summaries.df[columns.unique].iloc[0].to_json(indent=4)
meta = pd.read_json(meta, typ='series')
meta
```




    Enzyme                                                                 Trypsin/P
    Enzyme mode                                                             Specific
    Enzyme first search                                                         None
    Enzyme mode first search                                                    None
    Use enzyme first search                                                    False
    Variable modifications                     Oxidation (M);Acetyl (Protein N-term)
    Fixed modifications                                          Carbamidomethyl (C)
    Multi modifications                                                         None
    Variable modifications first search                                         None
    Use variable modifications first search                                    False
    Requantify                                                                 False
    Multiplicity                                                                   1
    Max. missed cleavages                                                          2
    Labels0                                                                     None
    LC-MS run type                                                          Standard
    Time-dependent recalibration                                                None
    MS/MS Submitted (ISO)                                                          0
    MS/MS Identified (ISO)                                                         0
    MS/MS Identified (ISO) [%]                                                     0
    dtype: object




```python
for col, value in meta.items():
    data[col] = value    
```


```python
data
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
      <th>MS/MS</th>
      <th>MS3</th>
      <th>MS/MS Submitted</th>
      <th>MS/MS Submitted (SIL)</th>
      <th>MS/MS Submitted (PEAK)</th>
      <th>MS/MS Identified</th>
      <th>MS/MS Identified (SIL)</th>
      <th>MS/MS Identified (PEAK)</th>
      <th>MS/MS Identified [%]</th>
      <th>MS/MS Identified (SIL) [%]</th>
      <th>MS/MS Identified (PEAK) [%]</th>
      <th>Peptide Sequences Identified</th>
      <th>Peaks</th>
      <th>Peaks Sequenced</th>
      <th>...</th>
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
      <td>12,136</td>
      <td>123,187</td>
      <td>0</td>
      <td>138,029</td>
      <td>108,345</td>
      <td>29,684</td>
      <td>55,341</td>
      <td>53,679</td>
      <td>1,662</td>
      <td>40</td>
      <td>50</td>
      <td>5.6</td>
      <td>46,896</td>
      <td>1,323,164</td>
      <td>121,024</td>
      <td>...</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>None</td>
      <td>Standard</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20200924_EXPL6_nLC09_MBK_QC_MNT_HeLa_42cm_FAIMS_500ng_Short_05</th>
      <td>20,253</td>
      <td>18,887</td>
      <td>0</td>
      <td>23,175</td>
      <td>14,599</td>
      <td>8,576</td>
      <td>5,030</td>
      <td>4,899</td>
      <td>131</td>
      <td>22</td>
      <td>34</td>
      <td>1.5</td>
      <td>4,005</td>
      <td>597,954</td>
      <td>17,071</td>
      <td>...</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>None</td>
      <td>Standard</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20170509_QE4_LC12_IAH_QC_MNT_HeLa_01</th>
      <td>14,786</td>
      <td>81,890</td>
      <td>0</td>
      <td>92,462</td>
      <td>71,318</td>
      <td>21,144</td>
      <td>51,464</td>
      <td>48,930</td>
      <td>2,534</td>
      <td>56</td>
      <td>69</td>
      <td>12.0</td>
      <td>39,011</td>
      <td>1,480,448</td>
      <td>76,823</td>
      <td>...</td>
      <td>False</td>
      <td>Oxidation (M);Acetyl (Protein N-term)</td>
      <td>Carbamidomethyl (C)</td>
      <td>None</td>
      <td>None</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>2</td>
      <td>None</td>
      <td>Standard</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 49 columns</p>
</div>



## Table schema bug

- filed bug report on pandas [#40255](https://github.com/pandas-dev/pandas/issues/40255)


```python
pd.show_versions()
```

    
    INSTALLED VERSIONS
    ------------------
    commit           : f2c8480af2f25efdbd803218b9d87980f416563e
    python           : 3.8.5.final.0
    python-bits      : 64
    OS               : Windows
    OS-release       : 10
    Version          : 10.0.19041
    machine          : AMD64
    processor        : Intel64 Family 6 Model 165 Stepping 2, GenuineIntel
    byteorder        : little
    LC_ALL           : None
    LANG             : None
    LOCALE           : Danish_Denmark.1252
    
    pandas           : 1.2.3
    numpy            : 1.18.5
    pytz             : 2020.1
    dateutil         : 2.8.1
    pip              : 20.2.4
    setuptools       : 50.3.1.post20201107
    Cython           : None
    pytest           : None
    hypothesis       : None
    sphinx           : 3.5.1
    blosc            : None
    feather          : None
    xlsxwriter       : None
    lxml.etree       : None
    html5lib         : None
    pymysql          : None
    psycopg2         : None
    jinja2           : 2.11.2
    IPython          : 7.19.0
    pandas_datareader: None
    bs4              : None
    bottleneck       : None
    fsspec           : None
    fastparquet      : None
    gcsfs            : None
    matplotlib       : 3.3.2
    numexpr          : None
    odfpy            : None
    openpyxl         : 3.0.5
    pandas_gbq       : None
    pyarrow          : None
    pyxlsb           : None
    s3fs             : None
    scipy            : 1.5.2
    sqlalchemy       : None
    tables           : None
    tabulate         : None
    xarray           : None
    xlrd             : None
    xlwt             : None
    numba            : None
    


```python
pd.__version__
```




    '1.2.3'




```python
import pandas 
data = {'A' : [1, 2, 2, pd.NA, 4, 8, 8, 8, 8, 9],
 'B': [pd.NA] * 10}
data = pd.DataFrame(data)
data = data.astype(pd.Int64Dtype()) # in my example I get this from data.convert_dtypes()
data_json = data.to_json(orient='table', indent=4)
pd.read_json(data_json, orient='table') #ValueError: Cannot convert non-finite values (NA or inf) to integer
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-16-6ce0e7bf93a6> in <module>
          5 data = data.astype(pd.Int64Dtype()) # in my example I get this from data.convert_dtypes()
          6 data_json = data.to_json(orient='table', indent=4)
    ----> 7 pd.read_json(data_json, orient='table') #ValueError: Cannot convert non-finite values (NA or inf) to integer
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\util\_decorators.py in wrapper(*args, **kwargs)
        197                 else:
        198                     kwargs[new_arg_name] = new_arg_value
    --> 199             return func(*args, **kwargs)
        200 
        201         return cast(F, wrapper)
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\util\_decorators.py in wrapper(*args, **kwargs)
        297                 )
        298                 warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
    --> 299             return func(*args, **kwargs)
        300 
        301         return wrapper
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_json.py in read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates, keep_default_dates, numpy, precise_float, date_unit, encoding, lines, chunksize, compression, nrows, storage_options)
        561 
        562     with json_reader:
    --> 563         return json_reader.read()
        564 
        565 
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_json.py in read(self)
        692                 obj = self._get_object_parser(self._combine_lines(data_lines))
        693         else:
    --> 694             obj = self._get_object_parser(self.data)
        695         self.close()
        696         return obj
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_json.py in _get_object_parser(self, json)
        714         obj = None
        715         if typ == "frame":
    --> 716             obj = FrameParser(json, **kwargs).parse()
        717 
        718         if typ == "series" or obj is None:
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_json.py in parse(self)
        829 
        830         else:
    --> 831             self._parse_no_numpy()
        832 
        833         if self.obj is None:
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_json.py in _parse_no_numpy(self)
       1093             )
       1094         elif orient == "table":
    -> 1095             self.obj = parse_table_schema(json, precise_float=self.precise_float)
       1096         else:
       1097             self.obj = DataFrame(
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\io\json\_table_schema.py in parse_table_schema(json, precise_float)
        330         )
        331 
    --> 332     df = df.astype(dtypes)
        333 
        334     if "primaryKey" in table["schema"]:
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5860                 if col_name in dtype:
       5861                     results.append(
    -> 5862                         col.astype(dtype=dtype[col_name], copy=copy, errors=errors)
       5863                     )
       5864                 else:
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5875         else:
       5876             # else, only a single dtype is given
    -> 5877             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
       5878             return self._constructor(new_data).__finalize__(self, method="astype")
       5879 
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        629         self, dtype, copy: bool = False, errors: str = "raise"
        630     ) -> "BlockManager":
    --> 631         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        632 
        633     def convert(
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, align_keys, ignore_failures, **kwargs)
        425                     applied = b.apply(f, **kwargs)
        426                 else:
    --> 427                     applied = getattr(b, f)(**kwargs)
        428             except (TypeError, NotImplementedError):
        429                 if not ignore_failures:
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        671             vals1d = values.ravel()
        672             try:
    --> 673                 values = astype_nansafe(vals1d, dtype, copy=True)
        674             except (ValueError, TypeError):
        675                 # e.g. astype_nansafe can fail on object-dtype of strings
    

    ~\Anaconda3\envs\vaep\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
       1066 
       1067         if not np.isfinite(arr).all():
    -> 1068             raise ValueError("Cannot convert non-finite values (NA or inf) to integer")
       1069 
       1070     elif is_object_dtype(arr):
    

    ValueError: Cannot convert non-finite values (NA or inf) to integer



```python
print(data.to_string())
```

          A     B
    0     1  <NA>
    1     2  <NA>
    2     2  <NA>
    3  <NA>  <NA>
    4     4  <NA>
    5     8  <NA>
    6     8  <NA>
    7     8  <NA>
    8     8  <NA>
    9     9  <NA>
    


```python
N = 3
meta = mq_all_summaries.df[columns.unique[:N]].iloc[0:2].reset_index(drop=True)
meta.to_dict()
```




    {'Enzyme': {0: 'Trypsin/P', 1: 'Trypsin/P'},
     'Enzyme mode': {0: 'Specific', 1: 'Specific'},
     'Enzyme first search': {0: <NA>, 1: <NA>}}


