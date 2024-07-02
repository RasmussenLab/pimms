"""Console script to create or append index.rst for static website of differential analysis workflow."""
import argparse
import textwrap
from collections import defaultdict
from pathlib import Path


def split_nb_name(nb: str) -> list:
    return nb.split('.')[0].split('_')


INDEX_RST = textwrap.dedent("""\
    Differential Analysis Notebooks
    -------------------------------

    Inspect the notebooks associated with the differential analysis workflow.

    .. toctree::
        :maxdepth: 2
        :caption: Differential analysis (ANCOVA)

    {nb_1}

    .. toctree::
        :maxdepth: 2
        :caption: Compare ANCOVAs

    {nb_2}

    .. toctree::
        :maxdepth: 2
        :caption: Compare single differential analysis

    {nb_4}

    .. toctree::
        :maxdepth: 2
        :caption: Logistic regression models

    {nb_3}
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Create or append index.rst for static website '
                    'displaying differential analysis notebooks.')
    parser.add_argument('--folder', '-f',
                        type=str,
                        help='Path to the folder',
                        required=True)
    parser.add_argument('--subfolder_comparision', '-sf_cp',
                        type=str,
                        help='Subfolder for comparison',
                        required=True)
    args = parser.parse_args()

    folder_experiment = args.folder

    folder_experiment = Path(folder_experiment)
    subfolder_comparison = Path(args.subfolder_comparision)
    nbs = [_f.relative_to(folder_experiment) for _f in subfolder_comparison.glob('**/*.ipynb') if _f.is_file()]
    nbs

    groups = defaultdict(list)
    for nb in nbs:
        _group = nb.name.split('_')[1]
        groups[_group].append(nb)
    groups = dict(groups)
    groups

    # Parse notebooks present in imputation workflow

    nb_1 = ''
    for nb in groups['1']:
        nb_1 += " " * 4 + split_nb_name(nb.name)[-1] + f" <{nb.as_posix()}>\n"

    nb_2 = ''
    for nb in groups['2']:
        nb_2 += " " * 4 + ' '.join(nb.parent.name.split('_')) + f" <{nb.as_posix()}>\n"

    nb_3 = ''
    for nb in groups['3']:
        nb_3 += " " * 4 + ' '.join(nb.parent.name.split('_')) + f" <{nb.as_posix()}>\n"
    print(nb_3)

    nb_4 = groups['4'][0]
    nb_4 = " " * 4 + "Compare single features" + f" <{nb_4.as_posix()}>\n"

    index_rst = INDEX_RST.format(nb_1=nb_1,
                                 nb_2=nb_2,
                                 nb_3=nb_3,
                                 nb_4=nb_4)
    # append to index.rst
    with open(folder_experiment / 'index.rst', 'a') as f:
        f.write(index_rst)

    msg = f"""\
    The index.rst file has been created or extended in {folder_experiment}:
    ```bash
    {folder_experiment / 'index.rst'}
    ```
    """

    msg = textwrap.dedent(msg)
    print(msg)


if __name__ == '__main__':
    main()
