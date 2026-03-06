"""Abstract base class for optimizers with gradient variance tracking."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class OptimizerWithVariance(ABC):
    """Abstract base for optimizers with gradient variance tracking.
    
    This base class defines the interface for optimization algorithms that support
    gradient variance computation, which is essential for comparing variance reduction
    methods like SVRG against standard optimizers like SGD and Adam.
    """
    
    def __init__(self, params, defaults: Dict[str, Any]):
        """Initialize optimizer with parameters and default settings.
        
        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize or dicts defining parameter groups
        defaults : Dict[str, Any]
            Dictionary containing default values of optimization options
        """
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = {}
        self.param_groups = []
        
        # Handle parameter groups
        # Convert generator/iterator to list first
        if not isinstance(params, (list, tuple)):
            params = list(params)
        
        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if not isinstance(params[0], dict):
                params = [{'params': params}]
        
        for param_group in params:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer.
        
        Parameters
        ----------
        param_group : Dict[str, Any]
            Dictionary specifying parameters and optimization options
        """
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections')
        else:
            param_group['params'] = list(params)
        
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors")
        
        # Add defaults
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        
        self.param_groups.append(param_group)
    
    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.
        
        Parameters
        ----------
        closure : Optional[Callable[[], float]]
            A closure that reevaluates the model and returns the loss
        
        Returns
        -------
        Optional[float]
            The loss value if closure is provided, None otherwise
        """
        pass
    
    @abstractmethod
    def compute_gradient_variance(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_samples: int = 10
    ) -> float:
        """Compute empirical gradient variance.
        
        This method samples multiple mini-batches and computes the variance of
        gradient estimates, which is crucial for analyzing optimizer behavior.
        
        Parameters
        ----------
        model : nn.Module
            Neural network model
        data_loader : DataLoader
            DataLoader for sampling mini-batches
        loss_fn : Callable
            Loss function that takes (predictions, targets) and returns scalar loss
        num_samples : int, default=10
            Number of mini-batch samples for variance estimation
        
        Returns
        -------
        float
            Mean squared deviation from mean gradient (gradient variance)
        """
        pass
    
    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer state including parameter groups and state
        """
        pass
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint.
        
        Parameters
        ----------
        state_dict : Dict[str, Any]
            Dictionary containing optimizer state
        """
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
