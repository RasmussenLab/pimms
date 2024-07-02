"""Console script to create index.rst and conf.py for static website of
the imputation comparison workflow."""
import argparse
import textwrap
from collections import defaultdict
from pathlib import Path


def split_nb_name(nb: str) -> list:
    return nb.split('.')[0].split('_')


INDEX_RST = textwrap.dedent("""\
    Comparison Workflow Notebooks
    ================================

    Inspect the notebooks associated with the imputation workflow.

    .. toctree::
        :maxdepth: 2
        :caption: Split creation and data handling

    {nb_0}

    .. toctree::
        :maxdepth: 2
        :caption: PIMMS models

    {nb_1_PIMMS}

    .. toctree::
        :maxdepth: 2
        :caption: R models

    {nb_1_NAGuideR}

    .. toctree::
        :maxdepth: 2
        :caption: Imputation model comparison

    {nb_2}
    """)

CONF_PY = textwrap.dedent("""\
    # Configuration file for the Sphinx documentation builder.
    # (Build by PIMMS workflow for the imputation comparison study)
    # This file only contains a selection of the most common options. For a full
    # list see the documentation:
    # https://www.sphinx-doc.org/en/master/usage/configuration.html

    # -- Project information -----------------------------------------------------
    from importlib import metadata

    PACKAGE_VERSION = metadata.version('pimms-learn')

    project = 'pimms_workflow'
    copyright = '2023, Henry Webel'
    author = 'PIMMS'
    version = PACKAGE_VERSION
    release = PACKAGE_VERSION


    # -- General configuration ---------------------------------------------------

    # Add any Sphinx extension module names here, as strings. They can be
    # extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
    # ones.
    extensions = [
        'myst_nb',
        'sphinx_new_tab_link',
    ]

    #  https://myst-nb.readthedocs.io/en/latest/computation/execute.html
    nb_execution_mode = "off"

    myst_enable_extensions = ["dollarmath", "amsmath"]

    # Plolty support through require javascript library
    # https://myst-nb.readthedocs.io/en/latest/render/interactive.html#plotly
    html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]

    # https://myst-nb.readthedocs.io/en/latest/configuration.html
    # Execution
    nb_execution_raise_on_error = True
    # Rendering
    nb_merge_streams = True

    # Add any paths that contain templates here, relative to this directory.
    templates_path = ['_templates']

    # List of patterns, relative to source directory, that match files and
    # directories to ignore when looking for source files.
    # This pattern also affects html_static_path and html_extra_path.
    exclude_patterns = ['_build', 'jupyter_execute', 'figures',
                        'Thumbs.db', '.DS_Store']

    # -- Options for HTML output -------------------------------------------------

    # The theme to use for HTML and HTML Help pages.  See the documentation for
    # a list of builtin themes.
    # See:
    # https://github.com/executablebooks/MyST-NB/blob/master/docs/conf.py
    # html_title = ""
    html_theme = "sphinx_book_theme"
    # html_logo = "_static/logo-wide.svg"
    # html_favicon = "_static/logo-square.svg"
    html_theme_options = {
        "home_page_in_toc": True,
        "use_download_button": True,
        "launch_buttons": {
        },
        "navigation_with_keys": False,
    }
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Create index.rst and conf.py for static website '
                    'of the imputation comparison workflow.')
    parser.add_argument('--folder', '-f',
                        type=str,
                        help='Path to the folder',
                        required=True)
    args = parser.parse_args()

    folder_experiment = args.folder

    folder_experiment = Path(folder_experiment)
    nbs = [_f.name for _f in folder_experiment.iterdir() if _f.suffix == '.ipynb']
    nbs

    groups = defaultdict(list)
    for nb in nbs:
        _group = nb.split('_')[1]
        groups[_group].append(nb)
    groups = dict(groups)
    groups

    # Parse notebooks present in imputation workflow

    nb_0 = ''
    for nb in groups['0']:
        nb_0 += " " * 4 + f"{nb}\n"

    nb_1_PIMMS = ''
    for nb in groups['1']:
        if '_NAGuideR_' not in nb:
            nb_1_PIMMS += " " * 4 + split_nb_name(nb)[-1] + f" <{nb}>\n"

    nb_1_NAGuideR = ''
    for nb in groups['1']:
        if '_NAGuideR_' in nb:
            _model = split_nb_name(nb)[-1]
            if _model.isupper():
                nb_1_NAGuideR += " " * 4 + _model + f" <{nb}>\n"
            else:
                nb_1_NAGuideR += " " * 4 + ' '.join(split_nb_name(nb[5:])) + f" <{nb}>\n"

    nb_2 = ''
    for nb in groups['2']:
        nb_2 += " " * 4 + ' '.join(split_nb_name(nb[5:])) + f" <{nb}>\n"

    index_rst = INDEX_RST.format(nb_0=nb_0,
                                 nb_1_PIMMS=nb_1_PIMMS,
                                 nb_1_NAGuideR=nb_1_NAGuideR,
                                 nb_2=nb_2)
    # write to file and print further instructions
    with open(folder_experiment / 'index.rst', 'w') as f:
        f.write(index_rst)
    with open(folder_experiment / 'conf.py', 'w') as f:
        f.write(CONF_PY)

    msg = f"""\
    The index.rst file has been created in {folder_experiment}:
    ```bash
    {folder_experiment / 'index.rst'}
    ```
    The conf.py file has been created in {folder_experiment}:
    ```bash
    {folder_experiment / 'conf.py'}
    ```

    The dependencies for the website can be installed using pip

    ```bash
    pip install pimms-learn[docs]
    ```

    To create the html website run the following command in the terminal:

    ```bash
    cd {folder_experiment}
    sphinx-build -n -W --keep-going -b html ./ ./_build/
    ```

    This will build a website in the _build folder in the {folder_experiment} directory.

    Open the `index.html` file in the `_build` folder to view the website.

    Find these instructions in the README.md file in the {folder_experiment} directory:

    {folder_experiment / 'README.md'}
    """

    msg = textwrap.dedent(msg)
    print(msg)

    with open(folder_experiment / 'README.md', 'w') as f:
        f.write(msg)


if __name__ == '__main__':
    main()
