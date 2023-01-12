'''
Create a diff view between files on server and files which should be completed.
The comparison is only based on folders.

## hints
- reads config.yaml
- compare files on erda.io.dk with FILES
- exclude failed and excluded files
- outputs a file with diffs.
'''
import argparse
import yaml

__author__ = 'Henry Webel'

parser = argparse.ArgumentParser(
    prog='FileCheckOnErda',
    description='Check if current set of files which did not fail are uploaded to erda. '
                'Does not check for empty folders.')

parser.add_argument('-f', '--files_on_erda',
                    help='List of folders on erda.', required=True)
parser.add_argument('-v', '--verbose', help='Prinbt additional information',
                    action='count')
parser.add_argument('-o', '--outfile', default='current_files_to_do.txt')

args = parser.parse_args()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

if args.verbose:
    print(f"Load file: {config['FILES']}")

with open(config['FILES'], encoding='utf-8') as f:
    FILES = set(line.strip().split('.raw')[0] for line in f)

if args.verbose:
    print(f"Load file: {config['FILES_EXCLUDED']}")

with open(config['FILES_EXCLUDED'], encoding='utf-8') as f:
    FILES_EXCLUDED = set(line.strip().split('.raw')[0] for line in f)

if args.verbose:
    print(f"Load file: {config['FILES_FAILED']}")

with open(config['FILES_FAILED'], encoding='utf-8') as f:
    FILES_FAILED = set(line.strip().split('.raw')[0] for line in f)

with open(args.files_on_erda, encoding='utf-8') as f:
    # although no `.raw` ending, keep the default loading syntax
    FILES_ON_ERDA = set(line.strip().split('.raw')[0] for line in f)

files_to_do = FILES - FILES_EXCLUDED - FILES_FAILED - FILES_ON_ERDA
files_to_do = [f"{file}.raw" for file in files_to_do]

if args.verbose:
    print(f"In total {len(files_to_do)} are not processed:\n\t",
          "\n\t".join(file for file in files_to_do))

with open(args.outfile, mode='w') as f:
    f.writelines('\n'.join(file for file in files_to_do))

print(f'Saved difference between server and completed files: {args.outfile}')
