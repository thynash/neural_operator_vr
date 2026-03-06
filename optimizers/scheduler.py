"""Learning rate schedulers for optimizers."""

from typing import Optional
import math


class LRScheduler:
    """Base class for learning rate schedulers.
    
    Parameters
    ----------
    optimizer : OptimizerWithVariance
        Optimizer to schedule learning rate for
    """
    
    def __init__(self, optimizer):
        """Initialize scheduler."""
        self.optimizer = optimizer
        self.last_epoch = 0
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate.
        
        Parameters
        ----------
        epoch : Optional[int]
            Current epoch number. If None, increments last_epoch by 1
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group in self.optimizer.param_groups:
            param_group['learning_rate'] = self.get_lr(epoch, param_group)
    
    def get_lr(self, epoch: int, param_group: dict) -> float:
        """Compute learning rate for given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        param_group : dict
            Parameter group dictionary
        
        Returns
        -------
        float
            Learning rate for this epoch
        """
        raise NotImplementedError


class StepLR(LRScheduler):
    """Step decay learning rate scheduler.
    
    Decays learning rate by gamma every step_size epochs:
        lr = initial_lr * gamma^(epoch // step_size)
    
    Parameters
    ----------
    optimizer : OptimizerWithVariance
        Optimizer to schedule learning rate for
    step_size : int
        Number of epochs between learning rate decay
    gamma : float, default=0.1
        Multiplicative factor of learning rate decay
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        """Initialize step decay scheduler."""
        if step_size <= 0:
            raise ValueError(f"Invalid step_size: {step_size}")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lrs = [group['learning_rate'] for group in optimizer.param_groups]
        super().__init__(optimizer)
    
    def get_lr(self, epoch: int, param_group: dict) -> float:
        """Compute learning rate for given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        param_group : dict
            Parameter group dictionary
        
        Returns
        -------
        float
            Learning rate for this epoch
        """
        # Get initial learning rate for this param group
        group_idx = self.optimizer.param_groups.index(param_group)
        initial_lr = self.initial_lrs[group_idx]
        
        # Compute decay factor
        decay_factor = self.gamma ** (epoch // self.step_size)
        
        return initial_lr * decay_factor


class ExponentialLR(LRScheduler):
    """Exponential decay learning rate scheduler.
    
    Decays learning rate exponentially:
        lr = initial_lr * gamma^epoch
    
    Parameters
    ----------
    optimizer : OptimizerWithVariance
        Optimizer to schedule learning rate for
    gamma : float
        Multiplicative factor of learning rate decay per epoch
    """
    
    def __init__(self, optimizer, gamma: float):
        """Initialize exponential decay scheduler."""
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        
        self.gamma = gamma
        self.initial_lrs = [group['learning_rate'] for group in optimizer.param_groups]
        super().__init__(optimizer)
    
    def get_lr(self, epoch: int, param_group: dict) -> float:
        """Compute learning rate for given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        param_group : dict
            Parameter group dictionary
        
        Returns
        -------
        float
            Learning rate for this epoch
        """
        # Get initial learning rate for this param group
        group_idx = self.optimizer.param_groups.index(param_group)
        initial_lr = self.initial_lrs[group_idx]
        
        return initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.
    
    Anneals learning rate using cosine function:
        lr = eta_min + (initial_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    
    Parameters
    ----------
    optimizer : OptimizerWithVariance
        Optimizer to schedule learning rate for
    T_max : int
        Maximum number of epochs (period of cosine annealing)
    eta_min : float, default=0.0
        Minimum learning rate
    """
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        """Initialize cosine annealing scheduler."""
        if T_max <= 0:
            raise ValueError(f"Invalid T_max: {T_max}")
        if eta_min < 0.0:
            raise ValueError(f"Invalid eta_min: {eta_min}")
        
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lrs = [group['learning_rate'] for group in optimizer.param_groups]
        super().__init__(optimizer)
    
    def get_lr(self, epoch: int, param_group: dict) -> float:
        """Compute learning rate for given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        param_group : dict
            Parameter group dictionary
        
        Returns
        -------
        float
            Learning rate for this epoch
        """
        # Get initial learning rate for this param group
        group_idx = self.optimizer.param_groups.index(param_group)
        initial_lr = self.initial_lrs[group_idx]
        
        # Compute cosine annealing
        cosine_factor = (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        lr = self.eta_min + (initial_lr - self.eta_min) * cosine_factor
        
        return lr


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler (no decay).
    
    Keeps learning rate constant throughout training.
    
    Parameters
    ----------
    optimizer : OptimizerWithVariance
        Optimizer to schedule learning rate for
    """
    
    def __init__(self, optimizer):
        """Initialize constant scheduler."""
        self.initial_lrs = [group['learning_rate'] for group in optimizer.param_groups]
        super().__init__(optimizer)
    
    def get_lr(self, epoch: int, param_group: dict) -> float:
        """Compute learning rate for given epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        param_group : dict
            Parameter group dictionary
        
        Returns
        -------
        float
            Learning rate for this epoch (constant)
        """
        # Get initial learning rate for this param group
        group_idx = self.optimizer.param_groups.index(param_group)
        return self.initial_lrs[group_idx]
