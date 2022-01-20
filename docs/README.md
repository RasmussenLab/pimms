# Docs

In order to build the docs you need to 

  1. install sphinx and additional support packages
  2. build the package reference files
  3. run sphinx to create a local html version

Command to be run from `path/to/vaep/docs`, i.e. from within the `docs` package folder: 

```cmd
# pwd: ./vaep/docs
conda env update -f environment.yml
sphinx-apidoc -o source ../vaep
make html
```