from vaep.tf_board import TensorboardModelNamer

def test_TensorboardModelNamer():
    expected = 'model_hl01_12_13_14_scaler'

    tensorboard_model_namer = TensorboardModelNamer(prefix_folder='experiment')

    assert tensorboard_model_namer.get_model_name(
        hidden_layers=1, neurons=[12, 13, 14], scaler='scaler') == expected
    assert tensorboard_model_namer.get_model_name(
        hidden_layers=1, neurons='12 13 14', scaler='scaler') == expected
    assert tensorboard_model_namer.get_model_name(
        hidden_layers=1, neurons='12_13_14', scaler='scaler') == expected
    assert tensorboard_model_namer.get_model_name(
        hidden_layers=1, neurons='12_13_14', scaler=scaler) == 'model_hl01_12_13_14_StandardScaler()'
    # assert get_writer(hidden_layers=1, neurons=1, scaler=scaler) == TypeError