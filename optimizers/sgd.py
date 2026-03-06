"""Stochastic Gradient Descent optimizer with variance tracking."""

from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import OptimizerWithVariance


class SGD(OptimizerWithVariance):
    """Stochastic Gradient Descent optimizer with momentum and variance tracking.
    
    Implements the SGD update rule:
        θ_{t+1} = θ_t - η * g_t
    
    With momentum:
        v_t = μ * v_{t-1} + g_t
        θ_{t+1} = θ_t - η * v_t
    
    With Nesterov momentum:
        v_t = μ * v_{t-1} + g_t
        θ_{t+1} = θ_t - η * (g_t + μ * v_t)
    
    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    learning_rate : float
        Learning rate (step size)
    momentum : float, default=0.0
        Momentum coefficient
    weight_decay : float, default=0.0
        L2 regularization coefficient
    nesterov : bool, default=False
        Whether to use Nesterov momentum
    """
    
    def __init__(
        self,
        params,
        learning_rate: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        """Initialize SGD optimizer."""
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            'learning_rate': learning_rate,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'nesterov': nesterov
        }
        super().__init__(params, defaults)
    
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
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            learning_rate = group['learning_rate']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state.setdefault(id(p), {})
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-learning_rate)
        
        return loss
    
    def compute_gradient_variance(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_samples: int = 10
    ) -> float:
        """Compute empirical gradient variance.
        
        Samples multiple mini-batches and computes variance as:
            Var = (1/n) Σᵢ ||gᵢ - ḡ||²
        where gᵢ are mini-batch gradients and ḡ is their mean.
        
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
        model.eval()
        
        # Collect gradients from multiple mini-batches
        gradients = []
        data_iter = iter(data_loader)
        
        for _ in range(num_samples):
            try:
                batch_data = next(data_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                data_iter = iter(data_loader)
                batch_data = next(data_iter)
            
            # Move to same device as model
            device = next(model.parameters()).device
            
            # Handle different batch formats
            if len(batch_data) == 3:
                # DeepONet format: (input_functions, query_points, targets)
                input_functions, query_points, targets = batch_data
                input_functions = input_functions.to(device)
                query_points = query_points.to(device)
                targets = targets.to(device)
            elif len(batch_data) == 2:
                # Simple format: (inputs, targets)
                inputs, targets = batch_data
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Reshape for neural operator format
                input_functions = inputs.permute(0, 2, 1)
                query_points = targets.permute(0, 2, 1)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            # Compute gradient for this mini-batch
            model.zero_grad()
            outputs = model(input_functions, query_points)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            # Collect gradients
            grad_list = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_list.append(p.grad.data.clone().flatten())
            
            if len(grad_list) > 0:
                gradients.append(torch.cat(grad_list))
        
        if len(gradients) == 0:
            return 0.0
        
        # Stack gradients: [num_samples, total_params]
        gradients = torch.stack(gradients)
        
        # Compute mean gradient
        mean_grad = gradients.mean(dim=0)
        
        # Compute variance: (1/n) Σᵢ ||gᵢ - ḡ||²
        variance = ((gradients - mean_grad) ** 2).sum(dim=1).mean().item()
        
        model.train()
        return variance
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer state including parameter groups and state
        """
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }
