"""Abstract base class for neural operator architectures."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn


class NeuralOperator(nn.Module, ABC):
    """
    Abstract base class for neural operator architectures.
    
    Neural operators learn mappings between infinite-dimensional function spaces.
    Concrete implementations include DeepONet and FNO architectures.
    """
    
    @abstractmethod
    def forward(
        self, 
        input_functions: torch.Tensor, 
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate operator on input functions at query points.
        
        Parameters
        ----------
        input_functions : torch.Tensor
            Discretized input functions with shape [batch, input_dim, num_sensors]
        query_points : torch.Tensor
            Evaluation locations with shape [batch, output_dim, num_queries]
        
        Returns
        -------
        torch.Tensor
            Operator output with shape [batch, output_dim, num_queries]
        """
        pass
    
    @abstractmethod
    def get_parameter_count(self) -> int:
        """
        Return total number of trainable parameters.
        
        Returns
        -------
        int
            Total count of trainable parameters in the model
        """
        pass
