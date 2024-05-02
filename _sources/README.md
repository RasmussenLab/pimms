The index.rst file has been created in project/runs/example:
project/runs/example/index.rst
The conf.py file has been created in project/runs/example:
project/runs/example/conf.py

The dependencies for the website can be installed using pip

pip install pimms-learn[docs]

To create the html website run the following command in the terminal:

cd project/runs/example
sphinx-build -n -W --keep-going -b html ./ ./_build/

This will build a website in the _build folder in the project/runs/example directory.

Open the index.html file in the _build folder to view the website.

Find these instructions in the README.md file in the project/runs/example directory:
project/runs/example/README.md
