from types import SimpleNamespace

from . import diff_analysis 
from . import compare_predictions

__all__ = ['diff_analysis', 'compare_predictions', 'Analysis']

class Analysis(SimpleNamespace):
    pass