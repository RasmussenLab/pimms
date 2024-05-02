The index.rst file has been created in project/runs/dev_dataset_small/proteinGroups_N50:
```bash
project/runs/dev_dataset_small/proteinGroups_N50/index.rst
```
The conf.py file has been created in project/runs/dev_dataset_small/proteinGroups_N50:
```bash
project/runs/dev_dataset_small/proteinGroups_N50/conf.py
```

The dependencies for the website can be installed using pip

```bash
pip install pimms-learn[docs]
```

To create the html website run the following command in the terminal:

```bash
cd project/runs/dev_dataset_small/proteinGroups_N50
sphinx-build -n -W --keep-going -b html ./ ./_build/
```

This will build a website in the _build folder in the project/runs/dev_dataset_small/proteinGroups_N50 directory.

Open the `index.html` file in the `_build` folder to view the website.

Find these instructions in the README.md file in the project/runs/dev_dataset_small/proteinGroups_N50 directory:

project/runs/dev_dataset_small/proteinGroups_N50/README.md
