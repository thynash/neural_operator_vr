"""Dynamical system data generators for neural operator learning."""

from .base import DynamicalSystem
from .logistic_map import LogisticMapDataset
from .lorenz_system import LorenzSystemDataset
from .burgers_equation import BurgersEquationDataset
from .data_manager import (
    OperatorDataset,
    LazyOperatorDataset,
    create_train_val_split,
    create_dataloaders,
    save_dataset,
    load_dataset
)

__all__ = [
    'DynamicalSystem',
    'LogisticMapDataset',
    'LorenzSystemDataset',
    'BurgersEquationDataset',
    'OperatorDataset',
    'LazyOperatorDataset',
    'create_train_val_split',
    'create_dataloaders',
    'save_dataset',
    'load_dataset'
]
