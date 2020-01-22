# Variational Autoencoder for Proteomics 

## Setup
Clone repository and install package in editable mode:

```
pip install -e /path/to/cloned/folder 
```

> Currently there are only notebooks and script under `bin`, but shared functionality will be added under `vaep` folder-package: This can then be imported using `import vaep`.

## PRIDE
Using [bioservices](https://bioservices.readthedocs.io/en/master/) to access PRIDE dataset [RESTful API](https://www.ebi.ac.uk/pride/ws/archive/#!/project) from the command line.

Alternatives are Downloading project files using 
- an [FTP server](ftp://ftp.pride.ebi.ac.uk/pride/data/archive). Submitted data is ordered by `<year>/<month>/<project-id>`, e.g. `ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2019/11/PXD012110/`. Accessible using [scp]() or python [`ftplib`](https://docs.python.org/3.7/library/ftplib.html) - module


## Standardising Proteomics data

- [HUPO Proteomics Standardiation Initiative](http://www.psidev.info/)
> The HUPO Proteomics Standards Initiative defines community standards for data representation in proteomics and interactomics to facilitate data comparison, exchange and verification.
     - [MGF format](http://www.matrixscience.com/help/data_file_help.html) for spectra
Available software
- An overview is provided by Roestlab as [Python for Proteomics](https://github.com/Roestlab/PythonProteomics)
- Pyteomics 4.0: [BitBucket](https://bitbucket.org/levitsky/pyteomics/src/default/), [Paper](https://pubs.acs.org/doi/10.1021/acs.jproteome.8b00717), [Tutorial](https://pyteomics.readthedocs.io/en/latest/)
    > `Pyteomics.mass`: A subpackage implementing mass and m/z calculations, including specific fragment ion types and modified peptides, isotope abundance calculation, and modification handling. The module offers two interfaces to the Unimod database: One (mass.Unimod) reads the Unimod archive file and stores it as a list of Python dictionaries, and the other (mass.unimod.Unimod) fully represents Unimod as a relational database using the SQLAlchemy engine. Both interfaces were added after the original publication. Additionally, the module now allows us to generate all (significant) isotopic compositions of a molecule.
- [pyproteome](https://github.com/white-lab/pyproteome) by [White lab](http://white-lab.mit.edu/)
- [colims](https://github.com/compomics/colims) by a Ghent based group. "System for end-to-end mass spectrometry based proteomics identification pipelines."
   - [Computational and Statistical Methods for Protein Quantification by Mass Spectrometry](https://www.wiley.com/en-gb/Computational+and+Statistical+Methods+for+Protein+Quantification+by+Mass+Spectrometry-p-9781119964001)
   - [Computational Methods for Mass Spectrometry Proteomics](https://www.wiley.com/en-us/Computational+Methods+for+Mass+Spectrometry+Proteomics-p-9780470512975)
   - [Handbook of Proteins: Structure, Function and Methods, 2 Volume Set, 2 Volume Set](https://www.wiley.com/en-us/Handbook+of+Proteins%3A+Structure%2C+Function+and+Methods%2C+2+Volume+Set%2C+2+Volume+Set-p-9780470060988)
- [MaxQuant](http://coxdocs.org/doku.php?id=maxquant:start), [MaxQuant-Pipeline](https://github.com/FredHutch/maxquant-pipeline)
    - [MaxLFQ]() for Label-free quantification
        - assumes that 90% of the proteins do not vary and can be used for normalization
- [spectrum_utils](https://github.com/bittremieux/spectrum_utils)  from [Wout Bittremieux](https://bittremieux.be/)

- [Perseus Platform](https://maxquant.net/perseus/)
    - Analyze MaxQuant output

- [Morpheus](https://cwenger.github.io/Morpheus/) (faster than MaxQuant)
- [Cytoscape](https://cytoscape.org/) for network visualization and analysis
- [Trans-Proteomic Pipeline](https://moritz.isbscience.org/resources/software/)
- 
## Lili's data

### AFLD project
- roughly 500 AFLD patients (to varying degrees) and 200 controls


### PRIDE Archives from Lili's paper on NAFLD:

- [PXD011839](https://www.ebi.ac.uk/pride/archive/projects/PXD011839) (human)
- [PXD012056](https://www.ebi.ac.uk/pride/archive/projects/PXD012056) (mice)

```
from bioservices.pride import PRIDE
client = PRIDE(verbose=True, cache=False)
```

### Imputation
- imputation of data is done based on the standard variation: Random samples are drawn from a (standard?) normal distribution with mean mu=mean_protein - sd_protein
- there are some custom scripts floating around
## People
- [Prof. Dr. Lennart Martens](https://www.compomics.be/people/lennart-martens/)

## Specialized Journals
- [proteome](https://pubs.acs.org/journal/jprobs)

## Educational resources
- Lennart Marten's [lecutre](https://www.youtube.com/playlist?list=PLXxp6nsBenSX_W8DiOocKJ0laNauYNdYl) and [tutorials](https://www.compomics.com/bioinformatics-for-proteomics/)
- [FDR Tutorial](http://www.bioinfor.com/fdr-tutorial/)


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

- get tour trough the lab (Jeppe)


## Start with VAE
- [VAE in PyTorch](https://github.com/pytorch/examples/tree/master/vae)
- clustering on cell-lines not on meta-data
    1. Confirm that clustering using the original data is based on meta-data

    - Relative abundances per sample (per run): Standardization per sample

### Idea:
> Is it possible to use deep learning to remove technical and experimental bias from proteomics data

This raises the question 
    - Are there any clusters (related to biases) in the data?
        - Annelaura found so far only biological bias based on the Hela cellline change in April 2019
    - 


## FDR  
Spectral libray identification: Why are some proteins only identified in one single sample?
- benjamini hochberg correction
- [Mann-Witney-U-Test for ranks](https://de.wikipedia.org/wiki/Wilcoxon-Mann-Whitney-Test), [Rank-Biseral-Correlation](https://www.statisticshowto.datasciencecentral.com/rank-biserial-correlation/)
