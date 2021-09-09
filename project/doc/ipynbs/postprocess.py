import re
import argparse
from pathlib import Path, PosixPath

parser = argparse.ArgumentParser(description='Postprocess markdown files from ipynb for github. '
                                 'Removes style objects attached to pandas.DataFrame html tables.')

parser.add_argument('--file', '-i', type=str,
                    help='path to markdown input file.', required=True)
parser.add_argument('--overwrite',
                    help='Overwrite input file?', required=False, default=False, action='store_true')

args = parser.parse_args()
print("Arguments provided: ", args)


file = Path(args.file)

assert file.exists(), f"Missing File: {file}"

with open(file) as f:
    file_str = f.read()

# replaces these kind of strings:
regex = """<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>"""
# # short alternative
print('Remove style scope(s), similar to:', regex, sep='\n')


regex = '<style scoped>\n(.*\n)*?</style>'

file_str, n = re.subn(regex, '', file_str)


print(f'Removed {n = } times.')


# put into list of regexes
regex = '</style><table'  # check if this needs to be replaced by
print('Remove style scope(s), similar to:', regex, sep='\n')
file_str, n = re.subn(regex, '</style>\n<table', file_str)
print(f'Removed {n = } times.')



if not args.overwrite:
    file = file.parent / f"{file.stem}_replaced{file.suffix}"
    
with open(file, 'w') as f:
    f.write(file_str)
print(f"write results to new file {file.absolute()}")

# # does match between first <style scoped> and last </style>
# regex = '<style scoped>.*</style>'
# match = re.search(regex, file_str, re.DOTALL)

# # does find the first correctly
# regex = '<style scoped>\n(.*\n)*?</style>\n'
# res = re.search(regex, file_str)
# res.group(0)

# # why does this not work for all then?
# matches = re.findall(regex, file_str)
# matches # find only: '    }\n'
