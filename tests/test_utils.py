import pathlib
from vaep.utils import append_to_filepath


def test_append_to_filepath():
    fp = pathlib.Path('data/experiment_data.csv')

    fp_new = pathlib.Path('data/experiment_data_processed.csv')
    assert append_to_filepath(filepath=fp, to_append='processed') == fp_new

    fp_new = pathlib.Path('data/experiment_data_processed.pkl')
    assert append_to_filepath(
        filepath=fp, to_append='processed', new_suffix='pkl') == fp_new
