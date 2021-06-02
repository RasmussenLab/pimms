import re
import argparse
from pathlib import Path, PosixPath

parser = argparse.ArgumentParser(description='Postprocess markdown files from ipynb for github. '
                                 'Removes style objects attached to pandas.DataFrame html tables.')

parser.add_argument('--file', '-i', type=str,
                    help='path to markdown input file.', required=True)

args = parser.parse_args()

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
# regex = '<style scoped>\n(.*\n)*?</style>'

res, n = re.subn(regex, '', file_str)

print('Remove string:', regex, sep='\n')
print(f'Removed {n = } times.')

regex2 = '</style><table'  # check if this needs to be replaced by
# '</style>\n<table'

with open(file.parent / f"{file.stem}_replaced{file.suffix}", 'w') as f:
    f.write(res)

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
