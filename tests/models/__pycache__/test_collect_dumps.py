from vaep.models.collect_dumps import select_content


def test_select_content():
    test_cases = ['model_metrics_HL_1024_512_256_dae',
                  'model_metrics_HL_1024_512_vae',
                  'model_metrics_collab']
    for test_case in test_cases:
        assert select_content(test_case, first_split='metrics_') == test_case.split('metrics_')[1]
