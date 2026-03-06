"""Optimizers module with variance tracking support."""

from .base import OptimizerWithVariance
from .sgd import SGD
from .adam import Adam
from .svrg import SVRG
from .scheduler import (
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ConstantLR
)

__all__ = [
    'OptimizerWithVariance',
    'SGD',
    'Adam',
    'SVRG',
    'LRScheduler',
    'StepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ConstantLR'
]
