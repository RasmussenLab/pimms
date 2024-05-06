from pathlib import Path

import vaep.io


def test_relative_to():
    fpath = Path('project/runs/experiment_name/run')
    pwd = 'project/runs/'  # per defaut '.' (the current working directory)
    expected = Path('experiment_name/run')
    acutal = vaep.io.resolve_path(fpath, pwd)
    assert expected == acutal

    # # no solution yet, expect chaning notebook pwd
    # fpath = Path('data/file')
    # # pwd is different subfolder
    # pwd  = 'root/home/project/runs/' # per defaut '.' (the current working directory)
    # expected =  Path('root/home/project/data/file')
    # acutal = vaep.io.resolve_path(fpath, pwd)
    # assert expected == acutal
