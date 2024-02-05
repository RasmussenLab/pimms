from vaep.plotting.defaults import assign_colors


def test_assign_colors():
    expected = [(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                (0.21044753832183283, 0.6773105080456748, 0.6433941168468681)]
    assigned = assign_colors(['DAE', 'CF', 'Test'])
    assert assigned == expected
