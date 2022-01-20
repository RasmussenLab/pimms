from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorboardModelNamer():
    """PyTorch SummaryWriter helper class for experiments.
    
    Creates new SummaryWriter for an experiment
    """
    def __init__(self, prefix_folder, root_dir=Path('runs')):
        """[summary]

        Parameters
        ----------
        prefix_folder : str
            Experiment folder-name. All new setups will be written to new summary files.
        root_dir : Path, optional
            Root directory to store experiments, by default Path('runs')
        """
        self.prefix_folder = prefix_folder
        self.root_logdir = Path(root_dir)
        self.folder = (self.root_logdir /
                       f'{self.prefix_folder}_{format(datetime.now(), "%y%m%d_%H%M")}')

    def get_model_name(self, hidden_layers: int,
                       neurons: list,
                       scaler: str,
                       ):
        name = 'model_'
        name += f'hl{hidden_layers:02d}'

        if type(neurons) == str:
            neurons = neurons.split()
        elif not type(neurons) in [list, tuple]:
            raise TypeError(
                "Provide expected format for neurons: [12, 13, 14], '12 13 14' or '12_13_14'")

        for x in neurons:
            name += f'_{x}'

        if type(scaler) == str:
            name += f'_{scaler}'
        else:
            name += f'_{scaler!r}'
        return name

    def get_writer(self, hidden_layers: int,
                   neurons: list,
                   scaler: str,
                   ):
        """Return a new SummaryWriter instance for one setup in an experiment."""
        model_name = self.get_model_name(hidden_layers=hidden_layers,
                                         neurons=neurons,
                                         scaler=scaler)
        return SummaryWriter(log_dir=self.folder / model_name)
