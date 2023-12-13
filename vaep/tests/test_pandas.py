from numpy import nan
import pandas as pd
import vaep.pandas


def test_interpolate():
    test_data = {
        "pep1": {0: nan, 1: 27.8, 2: 28.9, 3: nan, 4: 28.7},
        "pep2": {0: 29.1, 1: nan, 2: 27.6, 3: 29.1, 4: nan},
        # 4 values replace based on one (edge case):
        "pep3": {0: nan, 1: nan, 2: 23.6, 3: nan, 4: nan},
        "pep4": {0: nan, 1: nan, 2: nan, 3: nan, 4: nan},
        "pep5": {0: 26.0, 1: 27.0, 2: nan, 3: nan, 4: nan},
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


def test_key_map():
    # Build a schema of dicts
    d = {'one': {'alpha': {'a': 0.5, 'b': 0.3}},
         'two': {'beta': {'a': 0.7, 'b': 0.5},
                 'gamma': {'a': 0.8, 'b': 0.9}},
         'three': {'alpha': {'a': 0.4, 'b': 0.4},
                   'beta': {'a': 0.6, 'b': 0.5},
                   'gamma': {'a': 0.7, 'b': 0.6},
                   'delta': {'a': 0.2, 'b': 0.8}}
         }
    expected = {'one': {'alpha': ('a', 'b')},
                'two': {'beta': ('a', 'b'),
                        'gamma': ('a', 'b')},
                'three': {'alpha': ('a', 'b'),
                          'beta': ('a', 'b'),
                          'gamma': ('a', 'b'),
                          'delta': ('a', 'b')}}
    actual = vaep.pandas.key_map(d)
    assert expected == actual

    d = {'one': {'alpha': {'a': 0.5, 'b': 0.3}},
         'two': {'beta': {'a': 0.7, 'b': 0.5},
                 'gamma': {'a': 0.8, 'b': 0.9}},
         'three': {'alpha': {'a': 0.4, 'b': 0.4},
                   'beta': {'a': 0.6, 'b': 0.5},
                   'gamma': {'a': 0.7, 'b': 0.6},
                   'delta': 3}
         }
    expected = {'one': {'alpha': ('a', 'b')},
                'two': {'beta': ('a', 'b'),
                        'gamma': ('a', 'b')},
                'three': {'alpha': ('a', 'b'),
                          'beta': ('a', 'b'),
                          'gamma': ('a', 'b'),
                          'delta': None}}
    actual = vaep.pandas.key_map(d)
    assert expected == actual
