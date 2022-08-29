from pathlib import Path
from pprint import pformat
import yaml

import vaep.io

import logging
logger = logging.getLogger()


class Config():
    """Config class with a setter enforcing that config entries cannot 
    be overwritten.


    Can contain configs, which are itself configs:
    keys, paths,
    
    """
    def __setattr__(self, entry, value):
        """Set if attribute not in instance."""
        if hasattr(self, entry) and getattr(self, entry) != value:
            raise AttributeError(
                f'{entry} already set to {getattr(self, entry)}')
        super().__setattr__(entry, value)

    def __repr__(self):
        return pformat(vars(self))  # does not work in Jupyter?

    def overwrite_entry(self, entry, value):
        """Explicitly overwrite a given value."""
        super().__setattr__(entry, value)

    def dump(self, fname=None):
        if fname is None:
            try:
                fname = self.out_folder
                fname = Path(fname) / 'model_config.yml'
            except AttributeError:
                raise AttributeError(
                    'Specify fname or set "out_folder" attribute.')
        d = vaep.io.parse_dict(input_dict=self.__dict__)
        with open(fname, 'w') as f:
            yaml.dump(d, f)
        logger.info(f"Dumped config to: {fname}")

    @classmethod
    def from_dict(cls, d:dict):
        cfg = cls()
        for k, v in d.items():
            setattr(cfg, k, v)
        return cfg

    def update_from_dict(self, params: dict):
        for k, v in params.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                logger.info(f"Already set attribute: {k} has value {v}")
    
    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

def get_params(args:dict.keys, globals, remove=True) -> dict:
    params = {k: v for k, v in globals.items() if k not in args and k[0] != '_'}
    if not remove:
        return params
    for k in params.keys():
        try:
            del globals[k]
            logger.info(f"Removed from global namespace: {k}")
        except KeyError:
            pass
    return params


def add_default_paths(cfg: Config, folder_data='', out_root=None):
    """Add default paths to config."""
    if out_root:
        cfg.out_folder = Path(out_root)
    else:
        cfg.out_folder = cfg.folder_experiment
    if folder_data:
        cfg.data = Path(folder_data)
    else:
        cfg.data = cfg.folder_experiment / 'data'
    assert cfg.data.exists(), f"Directory not found: {cfg.data}"
    del folder_data
    cfg.out_figures = cfg.folder_experiment / 'figures'
    cfg.out_figures.mkdir(exist_ok=True)
    cfg.out_metrics = cfg.folder_experiment / 'metrics'
    cfg.out_metrics.mkdir(exist_ok=True)
    cfg.out_models = cfg.folder_experiment / 'models'
    cfg.out_models.mkdir(exist_ok=True)
    cfg.out_preds = cfg.folder_experiment / 'preds'
    cfg.out_preds.mkdir(exist_ok=True)
    return cfg
