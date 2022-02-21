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