from types import SimpleNamespace

from . import compare_predictions, diff_analysis

__all__ = ['diff_analysis', 'compare_predictions', 'Analysis']


class Analysis(SimpleNamespace):
    pass
