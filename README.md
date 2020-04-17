# Variational Autoencoder for Proteomics 


## Setup
Clone repository and install package in editable mode:

```
pip install -e /path/to/cloned/folder 
```

For a detailed description of a setup using conda, see [docs]()

> Currently there are only notebooks and scripts under `project`, 
> but shared functionality will be added under `vaep` folder-package: This can 
> then be imported using `import vaep`. See `vaep/README.md`

## Overview vaep package
- imputation of data is done based on the standard variation or KNN imputation. Adapted scripts from Annelaura are under `vaep/imputation.py`
- transformation of intensity data is under `vaep/transfrom.py`


## Tools for working with proteomics data
> Enumeration indicates software used and ranked.

### Peptide Idenfication and Quantifications

1. [MaxQuant](http://coxdocs.org/doku.php?id=maxquant:start) (MQ), [MaxQuant-Pipeline](https://github.com/FredHutch/maxquant-pipeline) and ecosystem by Cox's group
    - MaxQuant itself yields peptide intensities
        - `mqpar.xml`: Parameters for running MQ
        - `summary.txt`: Overview of results
        - `peptides.txt`: Peptide information
        - `proteinGroups.txt`: Grouped Protein Information
    - [MaxLFQ]() for Label-free quantification between runs
        - assumes that 90% of the proteins do not vary and can be used for normalization between replicates
    - [Perseus Platform](https://maxquant.net/perseus/)
        - Analyze MaxQuant output
- [PyOpenMS](https://pyopenms.readthedocs.io/en/latest/getting_started.html): Python Client for OpenMS C++ library from [Kohlbacher group](https://kohlbacherlab.org/) in Tübingen, 
    Aebersold group at ETH and [Reinert group](reinert-lab.de) at FU Berlin

- [MSFragger](https://github.com/Nesvilab/MSFragger)
- [MASCOT](http://www.matrixscience.com/blog.html): Properitary software, 
    buy who the `mgf` -fileformat (mascot general fileformat) was initially defined
- [Morpheus](https://cwenger.github.io/Morpheus/) (faster than MaxQuant)
- [Proline](http://www.profiproteomics.fr/proline/) from ProFi in Toulouse. 
- [AlphaPept](https://eugeniavoytik.github.io/), see [talk](https://www.youtube.com/watch?v=bMTNx_4nZlQ&list=PLxWLmFvQ1Jz9Ev_vp6WVuwtaz7qkPLM1x&index=12) by [Maximilian T. Strauss](https://straussmaximilian.github.io/), also [Eugenia Voytik](https://github.com/EugeniaVoytik)


### Fileformat transformation

Thermo-Fisher or Bruker machines output their own file-format (`.raw` and `.d`). These binaries need to be transfered into text based files for protability, 
as `mzML` or `mgf`.
    - [MGF format](http://www.matrixscience.com/help/data_file_help.html) for spectra

> The [HUPO Proteomics Standardiation Initiative](http://www.psidev.info/)
> defines community standards for data representation in proteomics and 
> interactomics to facilitate data comparison, exchange and verification.

1. [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)


### Unsorted
    
Available software
- An overview is provided by Roestlab as [Python for Proteomics](https://github.com/Roestlab/PythonProteomics)
- Pyteomics 4.0 developed by Moskau based group: [BitBucket](https://bitbucket.org/levitsky/pyteomics/src/default/), 
    [Paper](https://pubs.acs.org/doi/10.1021/acs.jproteome.8b00717), 
    [Tutorial](https://pyteomics.readthedocs.io/en/latest/)
    > `Pyteomics.mass`: A subpackage implementing mass and m/z calculations, including specific fragment ion types and modified peptides, isotope abundance calculation, and modification handling. The module offers two interfaces to the Unimod database: One (mass.Unimod) reads the Unimod archive file and stores it as a list of Python dictionaries, and the other (mass.unimod.Unimod) fully represents Unimod as a relational database using the SQLAlchemy engine. Both interfaces were added after the original publication. Additionally, the module now allows us to generate all (significant) isotopic compositions of a molecule.
- [pyproteome](https://github.com/white-lab/pyproteome) by [White lab](http://white-lab.mit.edu/)
- [colims](https://github.com/compomics/colims) by a [Lennart Martens'](https://www.compomics.be/people/lennart-martens/) group. "System for end-to-end mass spectrometry based proteomics identification pipelines."
   - [Computational and Statistical Methods for Protein Quantification by Mass Spectrometry](https://www.wiley.com/en-gb/Computational+and+Statistical+Methods+for+Protein+Quantification+by+Mass+Spectrometry-p-9781119964001)
   - [Computational Methods for Mass Spectrometry Proteomics](https://www.wiley.com/en-us/Computational+Methods+for+Mass+Spectrometry+Proteomics-p-9780470512975)
   - [Handbook of Proteins: Structure, Function and Methods, 2 Volume Set, 2 Volume Set](https://www.wiley.com/en-us/Handbook+of+Proteins%3A+Structure%2C+Function+and+Methods%2C+2+Volume+Set%2C+2+Volume+Set-p-9780470060988)
   - Lennart Marten's [lecutre](https://www.youtube.com/playlist?list=PLXxp6nsBenSX_W8DiOocKJ0laNauYNdYl) and [tutorials](https://www.compomics.com/bioinformatics-for-proteomics/) 

- [spectrum_utils](https://github.com/bittremieux/spectrum_utils)  from [Wout Bittremieux](https://bittremieux.be/)

- [Spectronaut](https://biognosys.com/shop/spectronaut), commercial, for DIA
- [Trans-Proteomic Pipeline](https://moritz.isbscience.org/resources/software/)

- Prosit vs MS2PIP for peptide spectrum verification/prediction
- [OpenSWATH](http://www.openswath.org/en/latest/): DIA workflow using OpenMS, PyProphet, etc.

- [Comet](http://comet-ms.sourceforge.net/), former SEQUEST, is open-source 
- [ProteinProspector](http://prospector.ucsf.edu/prospector/mshome.htm) - looks old-fashioned, but is updated





MS-tech  | Pipelines for processing
-------- | -----------------------
DIA      | OpenSWATH, Spectronaut
DDA      | 
SWATH-MS | OpenSWATH

## Data Repositories

- [MassIVE](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp)
    - [Massive.quant](https://massive.ucsd.edu/ProteoSAFe/static/massive-quant.jsp)
- [Pride](https://www.ebi.ac.uk/pride/archive/)
- [ProteomeXchange](http://www.proteomexchange.org/)

### PRIDE
Using [bioservices](https://bioservices.readthedocs.io/en/master/) to access PRIDE 
dataset [RESTful API](https://www.ebi.ac.uk/pride/ws/archive/#!/project) from the command line.

Alternatives are Downloading project files using 
- an [FTP server](ftp://ftp.pride.ebi.ac.uk/pride/data/archive). Submitted data is ordered 
    by `<year>/<month>/<project-id>`, 
    e.g. `ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2019/11/PXD012110/`. 
    Accessible using [scp]() or 
    python [`ftplib`](https://docs.python.org/3.7/library/ftplib.html) - module


## Hela Cellline
Hela cellline is derived from cervical tissue. 

Identifying a common space using Hela Celllines (wide-spread cells)
```python
from bioservices.pride import PRIDE
client = PRIDE(verbose=True, cache=False)
client.get_project_count(query='Hela') #278
```

- Protein profile of Hela celllines under different conditions. What could these conditions be?
    - check assumptions on similarity of profiles for a given cellline (different conditions change celluar state)
- Common representation of protein profiles over different cell-lines (from the same tissue?)

Variable        | Description
-----------     | ---------------------
index           | artifical index containing several information
Date            | Measurement Date
MS_instrument   | MassSpec Instrument
LC              | 
PID             | Person Identifier
ColumnLength    | Length of mass/charge column


## NextUp - Ideas - Open Topics
- create matched peptides intensities and compare sparseness
    - research available methods for matching

- [VAE in PyTorch](https://github.com/pytorch/examples/tree/master/vae)
- Clustering on cell-lines not on meta-data
    - Confirm that clustering using the original data is based on meta-data
    - Relative abundances per sample (per run): Standardization per sample

- use Wout Bettremieuxs [spetrum_utils](https://github.com/bittremieux/spectrum_utils) library,
  [pyOpenMS-FileReader](https://pyopenms.readthedocs.io/en/latest/file_handling.html)

> Is it possible to use deep learning to remove technical and experimental bias from proteomics data

This raises the question 
    - Are there any clusters (related to biases) in the data?
        - Annelaura found so far only biological bias based on the Hela cellline change in April 2019

### FDR  
Spectral libray identification: Why are some proteins only identified in one single sample?
- benjamini hochberg correction: [FDR Tutorial](http://www.bioinfor.com/fdr-tutorial/)
- [Mann-Witney-U-Test for ranks](https://de.wikipedia.org/wiki/Wilcoxon-Mann-Whitney-Test), [Rank-Biseral-Correlation](https://www.statisticshowto.datasciencecentral.com/rank-biserial-correlation/)


### Relative vs Absolute Quantification
- [Hamid Hamzeiy MQSS2018](https://www.youtube.com/watch?v=3bNaQxRL_10)
    - Spectral Counting as good as MaxLFQ
    - Performance is motivated by eyebowling rather than Metrics
    - Relative LFQ: [MaxLFQ]()
    - Absolute LFQ: [Proteomic Ruler]() (histone based)

### Peptide vs Protein Level
- razor peptide assignment: non-unique peptides are mapped to only one protein 