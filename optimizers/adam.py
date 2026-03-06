"""Adam optimizer with variance tracking."""

from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import OptimizerWithVariance


class Adam(OptimizerWithVariance):
    """Adam optimizer with adaptive learning rates and variance tracking.
    
    Implements the Adam update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)
    
    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    learning_rate : float, default=0.001
        Learning rate (step size)
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    epsilon : float, default=1e-8
        Term added to denominator for numerical stability
    weight_decay : float, default=0.0
        L2 regularization coefficient
    """
    
    def __init__(
        self,
        params,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """Initialize Adam optimizer."""
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'weight_decay': weight_decay
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
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                param_state = self.state.setdefault(id(p), {})
                
                # Initialize state if needed
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                param_state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                
                # Compute bias-corrected second moment estimate
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(epsilon)
                
                # Compute step size
                step_size = learning_rate / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    
    def compute_gradient_variance(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_samples: int = 10
    ) -> float:
        """Compute empirical gradient variance for raw gradients.
        
        Note: This computes variance of raw gradients, not the adaptive gradients
        used by Adam. This allows fair comparison with SGD and SVRG.
        
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
