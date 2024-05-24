The index.rst file has been created in project/runs/alzheimer_study:
```bash
project/runs/alzheimer_study/index.rst
```
The conf.py file has been created in project/runs/alzheimer_study:
```bash
project/runs/alzheimer_study/conf.py
```

The dependencies for the website can be installed using pip

```bash
pip install pimms-learn[docs]
```

To create the html website run the following command in the terminal:

```bash
cd project/runs/alzheimer_study
sphinx-build -n -W --keep-going -b html ./ ./_build/
```

This will build a website in the _build folder in the project/runs/alzheimer_study directory.

Open the `index.html` file in the `_build` folder to view the website.

Find these instructions in the README.md file in the project/runs/alzheimer_study directory:

project/runs/alzheimer_study/README.md
