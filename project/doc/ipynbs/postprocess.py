import re
import argparse
from pathlib import Path, PosixPath

collection_regex = {}

l_regexes = [
    '<style scoped>\n(.*\n)*?</style>',
    '</style><table'
]


def main(file_str: str, urls_fpath='', verbose=False, l_regexes=l_regexes):
    for regex in l_regexes:
        file_str, n = re.subn(regex, '', file_str)

        if verbose:
            print('Remove style scope(s), similar to:', regex, sep='\n')
            print(f'Removed {n = } times.')

    if urls_fpath:
        urls_fpath = Path(urls_fpath)
        assert urls_fpath.exists(), f"File does not exist: {urls_fpath}"
        with open(urls_fpath) as f:
            png_links_to_use = f.readlines()
            png_links_to_use = [s.strip() for s in png_links_to_use]
        if verbose:
            print("Got links:\n - {}".format("\n - ".join(s for s in png_links_to_use)))

        regex = '\!\[png\]\(.*.png\)'

        png_path_to_replace = re.findall(regex, file_str)
        if verbose:
            print("Found png images to replace:\n - {}".format(
                "\n - ".join(s for s in png_path_to_replace)))
        assert len(png_links_to_use) == len(
            png_path_to_replace), "Number on links does not match number of png images."

        for png_link, url_link in zip(png_path_to_replace, png_links_to_use):
            file_str = file_str.replace(png_link, url_link)  # returns a copie

    return file_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess markdown files from ipynb for github. '
                                     'Removes style objects attached to pandas.DataFrame html tables.')

    parser.add_argument('--file', '-i', type=str,
                        help='path to markdown input file.', required=True)
    parser.add_argument('--overwrite',
                        help='Overwrite input file?', required=False, default=False,
                        action='store_true')
    parser.add_argument('--image-links',
                        help='file with image links to use', type=str, required=False, default='')
    parser.add_argument('--verbose', '-v', required=False, default=False,
                        action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print("Arguments provided: ", args)

    file = Path(args.file)
    assert file.exists(), f"Missing File: {file}"
    with open(file) as f:
        file_str = f.read()

    file_str = main(file_str=file_str,
                    urls_fpath=args.image_links,
                    verbose=args.verbose)

    if not args.overwrite:
        file = file.parent / f"{file.stem}_replaced{file.suffix}"

    with open(file, 'w') as f:
        f.write(file_str)
    print(f"write results to new file {file.absolute()}")
