__doc__ = """
Source: 
- blog: https://ogden.eu/run-notebooks,
- gist: https://gist.github.com/tpogden/ec79f2ebe2baf45655445b575dc7f540

jupyter nbconvert: https://nbconvert.readthedocs.io/en/latest/execute_api.html#executing-notebooks-using-the-python-api-interface
"""


# ! python
# coding: utf-8

import os
from pathlib import Path
import argparse
import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# Parse args
parser = argparse.ArgumentParser(description="Runs a set of Jupyter \
                                              notebooks.")
file_text = """ Notebook file(s) to be run, e.g. '*.ipynb' (default),
'my_nb1.ipynb', 'my_nb1.ipynb my_nb2.ipynb', 'my_dir/*.ipynb'
"""
parser.add_argument('file_list', metavar='F', type=str, nargs='*', 
    help=file_text)
parser.add_argument('-k', '--kernel',
help='kernel name to use', default='vaep', 
    required=False)
parser.add_argument('-t', '--timeout', help='Length of time (in secs) a cell \
    can run before raising TimeoutError (default 600).', default=600, 
    required=False)
parser.add_argument('-p', '--run-path', help='The path the notebook will be \
    run from (default pwd).', default='.', required=False)
parser.add_argument('-o', '--output-folder', 
                    help='outputfolder', default='_out_ipynb', required=False)

args = parser.parse_args()
print('args passed:', args)
if not args.file_list: # Default file_list
    args.file_list = glob.glob('*.ipynb')


args.output_folder = Path(args.output_folder)
args.output_folder.mkdir(exist_ok=True)
# Check list of notebooks
notebooks = []
print('Notebooks to run:')
for f in args.file_list:
    if f.endswith('.ipynb'): # and not f.endswith('_out.ipynb'): # Find notebooks but not notebooks previously output from this script
        print(f[:-6])
        notebooks.append(f[:-6]) # Want the filename without '.ipynb'



# Execute notebooks and output
num_notebooks = len(notebooks)
print('*****')
for i, fpath in enumerate(notebooks):
    fpath_out = fpath + '_out'
    with open(fpath + '.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=int(args.timeout), kernel_name=args.kernel)
        try:
            print('Running', fpath, ':', i, '/', num_notebooks)
            out = ep.preprocess(nb, {'metadata': {'path': Path(args.run_path)}})
        except CellExecutionError:
            out = None
            msg = f'Error executing the notebook "{fpath}".\n'
            msg += 'See notebook "%s" for the traceback.' % fpath_out
            print(msg)
        except TimeoutError:
            msg = f'Timeout executing the notebook "{fpath}".\n'
            print(msg)
        finally:
            # Write output file
            with open(args.output_folder /  f'{fpath_out}.ipynb', mode='wt') as f:
                nbformat.write(nb, f)
print(f"Running completed, see results in: {args.output_folder.absolute()}.")



