from typing import Iterable
import re
import logging

logger = logging.getLogger('vaep')

# from collections import namedtuple
# columns = 'date ms_instrument lc_instrument researcher rest'.split()
# RunMetaData = namedtuple('RunMetaData', columns)

regex_researcher = '[_]*[A-Z][a-z]*[-]*[A-Z][a-zA-Z]*[_]*'

assert re.search(regex_researcher, 'HeWe_').group() == 'HeWe_'
assert re.search(regex_researcher, '_HeWe_').group() == '_HeWe_'
assert re.search(regex_researcher, 'HeWE_').group() == 'HeWE_'
assert re.search(regex_researcher, '_HeWE_').group() == '_HeWE_'

regex_lc_instrument = '[_]*([nN]|UP)*((lc)|(LC)|(CL)|([eE][vV][oO]))[a-zA-Z0-9]*[_]*'

assert re.search(regex_lc_instrument, '_nlc1_').group() == '_nlc1_'
assert re.search(regex_lc_instrument, '_LC6_').group() == '_LC6_'
assert re.search(regex_lc_instrument, '_nLC6_').group() == '_nLC6_'
assert re.search(regex_lc_instrument, 'nLC02_').group() == 'nLC02_'
assert re.search(regex_lc_instrument, '_UPCL_').group() == '_UPCL_'
assert re.search(regex_lc_instrument, '_Evo_').group() == '_Evo_'
assert re.search(regex_lc_instrument, '_EvO_').group() == '_EvO_'


# check not HeLa, HeLA, ON, OFF MNT, MA, QC, ALL
regex_not_researcher = '[Hh][eE][Ll][aA]|ON|OFF|MNT|MA|QC|ALL'

assert re.search(regex_not_researcher, 'HeLa').group() == 'HeLa'
assert re.search(regex_not_researcher, 'Hela').group() == 'Hela'
assert re.search(regex_not_researcher, 'hela').group() == 'hela'
assert re.search(regex_not_researcher, 'MNT').group() == 'MNT'
assert re.search(regex_not_researcher, 'MA').group() == 'MA'
assert re.search(regex_not_researcher, 'QC').group() == 'QC'
assert re.search(regex_not_researcher, 'MA_OFF').group() == 'MA'
assert re.search(regex_not_researcher, '_LiNi_') == None


type_run = {'MA': 'MNT',
            'MNT': 'MNT',
            'QC': 'QC'}

# based on hints from core facility
ms_instrument_mapping = {
    'LUMOS1': 'LUMOS',
    'ECPL0': 'EXPL0',
    'Q10': 'QE10'

}

lc_instrument_mapping = {
    f'LC{i}': f'LC{i:02}' for i in range(10)
}


def get_metadata_from_filenames(selected: Iterable, apply_cleaning=False):
    data_meta = {}
    for filename in selected:
        # The first two fields are in order, the rest needs matching.
        _entry = {}
        _entry['date'], _entry['ms_instrument'], _rest_filename = filename.split(
            '_', maxsplit=2)

        _entry['ms_instrument'] = _entry['ms_instrument'].upper()
        if apply_cleaning and _entry['ms_instrument'] in ms_instrument_mapping:
            _entry['ms_instrument'] = ms_instrument_mapping[_entry['ms_instrument']]

        _entry['lc_instrument'] = None
        try:
            for regex in [regex_lc_instrument, '[_]*[Bb][Rr][0-9]+[_]*']:
                try:
                    _entry['lc_instrument'] = re.search(
                        regex, _rest_filename).group().strip('_')
                    break
                except AttributeError:
                    pass
        finally:
            if _entry['lc_instrument']:
                _rest_filename = _rest_filename.replace(
                    _entry['lc_instrument'], '').replace('__', '_')
                _entry['lc_instrument'] = _entry['lc_instrument'].upper()
                if _entry['lc_instrument'][0] == 'N':
                    if apply_cleaning:
                        _entry['lc_instrument'] = f"{_entry['lc_instrument'][1:]}"
                    else:
                        _entry['lc_instrument'] = f"n{_entry['lc_instrument'][1:]}"
                if apply_cleaning and _entry['lc_instrument'] in lc_instrument_mapping:
                    _entry['lc_instrument'] = lc_instrument_mapping[_entry['lc_instrument']]
            else:
                # try rare cases: "20191216_QE4_nL4_MM_QC_MNT_HELA_01
                logger.error(f'Could not find LC instrument in {filename}')
        # researcher after LC instrument
        try:
            _entry['researcher'] = re.search(
                regex_researcher, _rest_filename).group().strip('_')
            _cleaned_filename = _rest_filename.replace(
                _entry['researcher'], '').replace('__', '_')
            while re.search(regex_not_researcher, _entry['researcher']):
                _entry['researcher'] = re.search(
                    regex_researcher, _cleaned_filename)
                if _entry['researcher']:
                    _entry['researcher'] = _entry['researcher'].group().strip('_')
                else:
                    raise AttributeError
                _cleaned_filename = _cleaned_filename.replace(
                    _entry['researcher'], '').replace('__', '_')
            _rest_filename = _rest_filename.replace(
                _entry['researcher'], '').replace('__', '_')
        except AttributeError:
            logger.critical(f'Found no researcher ID: {filename}')
            _entry['researcher'] = None

        _entry['rest'] = _rest_filename
        data_meta[filename] = _entry
    return data_meta


test_cases = ['20131014_QE5_UPLC9_ALL_MNT_HELA_01']

assert get_metadata_from_filenames(test_cases) == {
    '20131014_QE5_UPLC9_ALL_MNT_HELA_01': {'date': '20131014',
                                           'ms_instrument': 'QE5',
                                           'lc_instrument': 'UPLC9',
                                           'researcher': None,
                                           'rest': '_ALL_MNT_HELA_01'},
}
