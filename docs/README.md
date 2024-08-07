# Docs

In order to build the docs you need to 

  1. install sphinx and additional support packages
  2. build the package reference files
  3. run sphinx to create a local html version

Command to be run from `path/to/pimms/docs`, i.e. from within the `docs` package folder: 

Install pimms-learn with docs option locally

```bash
# pwd: ./pimms
pip install .[docs]
```

## Build docs

Using Sphinx command line tools. 

Options:
  - `--separate` to build separate pages for each (sub-)module

```bash	
# pwd: ./pimms/docs
# apidoc
sphinx-apidoc --force --implicit-namespaces --module-first -o reference ../pimmslearn
# build docs
sphinx-build -n -W --keep-going -b html ./ ./_build/
```