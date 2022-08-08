# Metadata workflow

Get metadata from ThermoFischer proteomics raw files using
[`ThermoRawFileParser`](https://github.com/compomics/ThermoRawFileParser)

## Output

- both json and txt data format into `jsons` and `txts` folder
- create combined `rawfile_metadata.json` (needs to be deleted if files are added)

## Configfile

add a `config/files.yaml` in [config](config):

```yaml
remote_in: erda:folder/path
out_folder: metadata
thermo_raw_file_parser_exe: mono path/to/ThermoRawFileParser/ThermoRawFileParser.exe
files:
- remote/path/file1.raw
- remote/path/file2.raw
```

The list of files is fetched from [`project/04_all_raw_files.ipynb`](../../project/04_all_raw_files.ipynb) notebook.


Then invoke the workflow with the list of config files

```bash
# dry-run
snakemake --configfiles config/ald_study/config.yaml config/ald_study/excluded.yaml -p -n
```


### Excluded files

Some files might be corrupted and not be processed by `ThermoRawFileParser`. These can be
excluded based on the `tmp` folder

```bash 
# check files
echo 'excluded:' > config/excluded_$(date +"%Y%m%d").yaml
find  tmp -name '*.raw*' | awk 'sub(/^.{4}/," ? ")' >> config/excluded_$(date +"%Y%m%d").yaml

# potentially add these to the workflow exclusion files:
find  tmp -name '*.raw*' | awk 'sub(/^.{4}/," ? ")' >> config/excluded.yaml
# rm -r tmp/* # remove excluded files
```

these files are ignored in the workflow (configured as a python set).

## Setup

- download and unzip [`ThermoRawFileParser`](https://github.com/compomics/ThermoRawFileParser)
- add path to `exe` to config

```bash
# sudo apt-get update
sudo apt install mono-complete
conda activate vaep # actually only snakemake needed
snakemake -n  # see job listing
```

## zip outputs


```bash
# could be part of snakemake process
zip -r metadata.zip txt jsons
```