from functools import partial, update_wrapper
import logging
from pathlib import Path
import json
import yaml
from typing import Iterable, Callable
import vaep.pandas

logger = logging.getLogger('vaep')


def select_content(s: str, first_split):
    s = s.split(first_split)[1]
    assert isinstance(s, str), f"More than one split: {s}"
    entries = s.split('_')
    if len(entries) > 1:
        s = '_'.join(entries[:-1])
    return s


def load_config_file(fname: Path, first_split='config_') -> dict:
    with open(fname) as f:
        loaded = yaml.safe_load(f)
    key = f"{fname.parents[1].name}_{select_content(fname.stem, first_split=first_split)}"
    return key, loaded


def load_metric_file(fname: Path, first_split='metrics_') -> dict:
    with open(fname) as f:
        loaded = json.load(f)
    loaded = vaep.pandas.flatten_dict_of_dicts(loaded)
    key = f"{fname.parents[1].name}_{select_content(fname.stem, first_split=first_split)}"
    return key, loaded


def collect(paths: Iterable,
            load_fn: Callable[[Path], dict],
            ) -> dict:
    all_metrics = {}
    for fname in paths:
        fname = Path(fname)
        key, loaded = load_fn(fname)
        print(f"{key = }")
        if key not in all_metrics:
            all_metrics[key] = loaded
            continue
        for k, v in loaded.items():
            if k in all_metrics[key]:
                if not all_metrics[key][k] == v:
                    logger.info("({key}) Diverging values for {k}: {v1} vs {v2}".format(
                        key=key,
                        k=k,
                        v1=all_metrics[key][k],
                        v2=v))
            else:
                all_metrics[key][k] = v
    return all_metrics


collect_metrics = partial(collect,
                          load_fn=load_metric_file,
                          #   select_content=partial(
                          #       select_content, first_split='metrics_')
                          )
# update_wrapper(collect_metrics, collect)
collect_configs = partial(collect,
                          load_fn=load_config_file,
                          )
