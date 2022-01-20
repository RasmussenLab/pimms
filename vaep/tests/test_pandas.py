from numpy import nan
import pandas as pd
import vaep.pandas


def test_interpolate():
    test_data = {
        "pep1": {0: nan,  1: 27.8, 2: 28.9, 3: nan,  4: 28.7},
        "pep2": {0: 29.1, 1: nan,  2: 27.6, 3: 29.1, 4: nan},
        "pep3": {0: nan,  1: nan,  2: 23.6, 3: nan,  4: nan}, # 4 values replace based on one (edge case)
        "pep4": {0: nan,  1: nan,  2: nan,  3: nan,  4: nan},
        "pep5": {0: 26.0, 1: 27.0, 2: nan,  3: nan,  4: nan},
    }

    df_test_data = pd.DataFrame(test_data)

    mask = df_test_data.isna()

    # floating point problem: numbers are not treated as decimals
    expected = {
        (0, 'pep1'): (27.8 + 28.9) / 2,
        (0, 'pep3'): 23.6,
        (1, "pep2"): (29.1 + 27.6) / 2,
        (1, "pep3"): 23.6,
        (3, "pep1"): (28.9 + 28.7) / 2,
        (3, "pep3"): 23.6,
        (4, "pep2"): (27.6 + 29.1) / 2,
        (4, "pep3"): 23.6,
        (2, "pep5"): 27.0,
        # (3, "pep5"): nan, # dropped
        # (4, "pep5"): nan, # dropped
        # all peptides from pep4 dropped as expected
    }

    actual = vaep.pandas.interpolate(df_test_data).to_dict()

    assert actual == expected
    assert df_test_data.equals(pd.DataFrame(test_data))


def test_flatten_dict_of_dicts():
    expected = {('a', 'a1', 'a2'): 1,
                ('a', 'a1', 'a3'): 2,
                ('b', 'b1', 'b2'): 3,
                ('b', 'b1', 'b3'): 4}
    data = {
        "a": {'a1': {'a2': 1, 'a3': 2}},
        "b": {'b1': {'b2': 3, 'b3': 4}}
    }
    actual = vaep.pandas.flatten_dict_of_dicts(data)

    assert expected == actual
