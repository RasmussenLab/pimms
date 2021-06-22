from typing import Iterable
import re
import logging

logger = logging.getLogger('vaep')

# from collections import namedtuple
# columns = 'date ms_instrument lc_instrument researcher rest'.split()
# RunMetaData = namedtuple('RunMetaData', columns)

regex_researcher = '[A-Z][a-z][A-Z][a-zA-Z]'

assert re.search(regex_researcher, 'HeWe_').group() == 'HeWe'
assert re.search(regex_researcher, '_HeWe_').group() == 'HeWe'
assert re.search(regex_researcher, 'HeWE_').group() == 'HeWE'
assert re.search(regex_researcher, '_HeWE_').group() == 'HeWE'

regex_lc_instrument = '[nN]*((lc)|(LC)|([eE]vo))[a-zA-Z0-9]*'

assert re.search(regex_lc_instrument, 'nlc1_').group() == 'nlc1'
assert re.search(regex_lc_instrument, 'Evo_').group() == 'Evo'


regex_hela = '[Hh]e[Ll]a'

assert re.search(regex_hela, 'HeLa').group() == 'HeLa'
assert re.search(regex_hela, 'Hela').group() == 'Hela'
assert re.search(regex_hela, 'hela').group() == 'hela'


def get_metadata_from_filenames(selected: Iterable):
    data_meta = {}
    for filename in selected:
        # The first two fields are in order, the rest needs matching.
        _entry = {}
        _entry['date'], _entry['ms_instrument'], _rest_filename = filename.split(
            '_', maxsplit=2)
        _entry['ms_instrument'] = _entry['ms_instrument'].upper()
        try:
            _entry['researcher'] = re.search(
                regex_researcher, _rest_filename).group()
            if re.search(regex_hela, _entry['researcher']):
                _cleaned_filename = _rest_filename.replace(
                    _entry['researcher'], '').replace('__', '_')
                _entry['researcher'] = re.search(
                    regex_researcher, _cleaned_filename).group()
            _rest_filename = _rest_filename.replace(
                _entry['researcher'], '').replace('__', '_')
        except AttributeError:
            try:
                _entry['researcher'] = re.search(
                    '[A-Z][a-zA-Z]*[-]*[A-Z][a-zA-Z]*_', _rest_filename).group()[:-1]
                logger.debug(
                    f"Found irregular researcher ID: {_entry['researcher']} (from: {filename})")
                _rest_filename = _rest_filename.replace(
                    _entry['researcher']+'_', '').replace('__', '_')
            except AttributeError:
                logger.critical(f'Found no researcher ID: {filename}')
                _entry['researcher'] = None
        try:
            _entry['lc_instrument'] = re.search(
                regex_lc_instrument, _rest_filename).group()
            _rest_filename = _rest_filename.replace(
                _entry['lc_instrument']+'_', '').replace('__', '_')
        except AttributeError:
            try:
                _entry['lc_instrument'] = re.search(
                    '[Bb][Rr][0-9]+', _rest_filename).group()
                _rest_filename = _rest_filename.replace(
                    _entry['lc_instrument']+'_', '').replace('__', '_')
            except AttributeError:
                _entry['lc_instrument'] = None
                logger.error(f'Could not find LC instrument in {filename}')
        _entry['rest'] = _rest_filename
        data_meta[filename] = _entry
    return data_meta
