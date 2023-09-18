import vaep.transform
import torch.nn
import fastai.data.core
import fastai.learner

from vaep.analyzers import Analysis


class ModelAnalysis(Analysis):
    """Class describing what an ModelAnalysis
    is supposed to have as attributes."""
    model: torch.nn.Module
    dls: fastai.data.core.DataLoaders
    learn: fastai.learner.Learner
    params: dict
    transform: vaep.transform.VaepPipeline
