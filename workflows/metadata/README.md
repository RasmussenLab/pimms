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

## Setup

- download and unzip [`ThermoRawFileParser`](https://github.com/compomics/ThermoRawFileParser)
- add path to `exe` to config

```bash
# sudo apt-get update
sudo apt install mono-complete
conda activate vaep # actually only snakemake needed
snakemake -n  # see job listing
```