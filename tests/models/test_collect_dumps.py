from pimmslearn.models.collect_dumps import select_content


def test_select_content():
    test_cases = ['model_metrics_HL_1024_512_256_dae',
                  'model_metrics_HL_1024_512_vae',
                  'model_metrics_collab']
    expected = ['HL_1024_512_256',
                'HL_1024_512',
                'collab']
    for test_case, v in zip(test_cases, expected):
        assert select_content(test_case, first_split='metrics_') == v
