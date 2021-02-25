'''
Create a copy of a file with duplicates removed.

## hints
- exclude failed and excluded files
'''
import argparse
from pathlib import Path

__author__ = 'Henry Webel'

parser = argparse.ArgumentParser(
    prog='Remove duplicate lines in file',
    description='Remove duplicated lines and save a copy.')

parser.add_argument('-f', '--file',
                    help='File with duplicates', required=True)
parser.add_argument('-v', '--verbose', help='Prinbt additional information',
                    action='count')
parser.add_argument('-o', '--outfile', required=False)


args = parser.parse_args()

file = Path(args.file)

try:
    unique = set()
    with open(file) as f:
        for line in f:
            unique.update((line.strip(),))
    unique = list(unique)
    unique.sort()
except FileNotFoundError:
    raise FileNotFoundError(f"No such file: {file} relative to {file.cwd()}.")

out_file = args.outfile

if not out_file:
    out_file = file.parent / f"{file.stem}_wo_duplicates{file.suffix}"

with open(out_file, 'w') as f:
    f.writelines([f'{line}\n' for line in unique])

