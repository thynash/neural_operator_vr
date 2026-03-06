"""Metrics computation for training, validation, and convergence analysis."""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_training_metrics(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    loss: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute training metrics for a single iteration.
    
    Args:
        model: Neural operator model
        batch: Tuple of (input, target) tensors
        loss: Computed loss value
        device: Device for computation
    
    Returns:
        Dictionary containing:
            - train_loss: MSE loss value
            - train_grad_norm: L2 norm of gradient vector
            - iteration_time: Time taken for this iteration (placeholder, set externally)
    
    Validates: Requirements 6.1, 8.1
    """
    # Compute gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** 0.5
    
    metrics = {
        'train_loss': loss.item(),
        'train_grad_norm': grad_norm,
    }
    
    return metrics


def compute_validation_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    denormalize_fn: Optional[callable] = None
) -> Dict[str, float]:
    """
    Compute validation metrics on the validation set.
    
    Args:
        model: Neural operator model
        val_loader: DataLoader for validation data
        device: Device for computation
        denormalize_fn: Optional function to denormalize predictions and targets
    
    Returns:
        Dictionary containing:
            - val_loss: Mean squared error on validation set
            - val_relative_error: Relative L2 error normalized by true solution norm
            - val_max_error: Maximum pointwise error
            - val_mean_absolute_error: Mean absolute error
    
    Validates: Requirements 6.2, 8.2
    """
    model.eval()
    
    total_loss = 0.0
    total_relative_error = 0.0
    max_error = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
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
                # inputs: [batch, input_horizon, state_dim]
                # Neural operators expect: input_functions [batch, state_dim, num_sensors]
                #                          query_points [batch, state_dim, num_queries]
                input_functions = inputs.permute(0, 2, 1)  # [batch, state_dim, input_horizon]
                query_points = targets.permute(0, 2, 1)  # [batch, state_dim, output_horizon]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            # Forward pass
            outputs = model(input_functions, query_points)
            
            # Reshape targets to match predictions if needed
            if targets.shape != outputs.shape:
                # targets: [batch, output_horizon, state_dim]
                # outputs: [batch, state_dim, output_horizon]
                if len(targets.shape) == 3 and len(outputs.shape) == 3:
                    if targets.shape[1] != outputs.shape[1]:
                        targets = targets.permute(0, 2, 1)
            
            # Denormalize if function provided
            if denormalize_fn is not None:
                outputs_denorm = denormalize_fn(outputs)
                targets_denorm = denormalize_fn(targets)
            else:
                outputs_denorm = outputs
                targets_denorm = targets
            
            # Compute MSE loss
            mse = torch.mean((outputs_denorm - targets_denorm) ** 2)
            total_loss += mse.item()
            
            # Compute relative L2 error: ||y_pred - y_true|| / ||y_true||
            diff_norm = torch.norm(outputs_denorm - targets_denorm)
            target_norm = torch.norm(targets_denorm)
            if target_norm > 1e-10:  # Avoid division by zero
                relative_error = (diff_norm / target_norm).item()
            else:
                relative_error = 0.0
            total_relative_error += relative_error
            
            # Compute max error
            batch_max_error = torch.max(torch.abs(outputs_denorm - targets_denorm)).item()
            max_error = max(max_error, batch_max_error)
            
            # Compute mean absolute error
            mae = torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()
            total_mae += mae
            
            num_batches += 1
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_relative_error': total_relative_error / num_batches,
        'val_max_error': max_error,
        'val_mean_absolute_error': total_mae / num_batches,
    }
    
    model.train()
    return metrics


def compute_long_horizon_metrics(
    model: nn.Module,
    initial_state: torch.Tensor,
    true_trajectory: torch.Tensor,
    num_steps: int,
    device: torch.device,
    divergence_threshold: float = 1e3
) -> Dict[str, float]:
    """
    Compute long-horizon prediction metrics via autoregressive rollout.
    
    Args:
        model: Neural operator model
        initial_state: Initial state for rollout [batch, input_horizon, state_dim]
        true_trajectory: True trajectory [batch, num_steps, state_dim]
        num_steps: Number of autoregressive steps
        device: Device for computation
        divergence_threshold: Threshold for detecting divergence
    
    Returns:
        Dictionary containing:
            - long_horizon_mse: Mean squared error over prediction horizon
            - long_horizon_steps: Number of steps before divergence
    
    Validates: Requirements 6.3, 6.4, 8.3
    """
    model.eval()
    
    with torch.no_grad():
        current_state = initial_state.to(device)  # [batch, input_horizon, state_dim]
        predictions = []
        
        # Autoregressive rollout
        for step in range(num_steps):
            # Prepare inputs for neural operator
            # current_state: [batch, input_horizon, state_dim]
            # Neural operators expect: input_functions [batch, state_dim, num_sensors]
            #                          query_points [batch, state_dim, num_queries]
            input_functions = current_state.permute(0, 2, 1)  # [batch, state_dim, input_horizon]
            
            # For query points, use the same shape as we want 1 output step
            # Create query points with shape [batch, state_dim, 1]
            query_points = torch.zeros(
                current_state.shape[0], 
                current_state.shape[2], 
                1, 
                device=device
            )
            
            # Predict next state
            next_state_output = model(input_functions, query_points)  # [batch, state_dim, 1]
            next_state = next_state_output.permute(0, 2, 1)  # [batch, 1, state_dim]
            
            predictions.append(next_state.squeeze(1))  # [batch, state_dim]
            
            # Check for divergence
            if torch.max(torch.abs(next_state)) > divergence_threshold:
                # Divergence detected
                steps_before_divergence = step + 1
                break
            
            # Update current state: shift window and append new prediction
            # Keep last (input_horizon - 1) steps and append new prediction
            current_state = torch.cat([
                current_state[:, 1:, :],  # Drop first step
                next_state  # Append new prediction
            ], dim=1)
        else:
            # No divergence
            steps_before_divergence = num_steps
        
        # Stack predictions
        if len(predictions) > 0:
            predictions = torch.stack(predictions, dim=1)  # [batch, steps, state_dim]
            
            # Compute MSE over available steps
            available_steps = predictions.shape[1]
            true_traj_subset = true_trajectory[:, :available_steps, :].to(device)
            mse = torch.mean((predictions - true_traj_subset) ** 2).item()
        else:
            mse = float('inf')
    
    metrics = {
        'long_horizon_mse': mse,
        'long_horizon_steps': steps_before_divergence,
    }
    
    model.train()
    return metrics


def compute_convergence_metrics(
    training_history: Dict[str, List[Tuple[int, float]]],
    target_loss: float
) -> Dict[str, Any]:
    """
    Compute convergence metrics from training history.
    
    Args:
        training_history: Dictionary containing metric histories with (iteration, value) tuples
        target_loss: Target validation loss threshold
    
    Returns:
        Dictionary containing:
            - iterations_to_target: Iterations to reach target loss (or None)
            - time_to_target: Wall-clock time to reach target loss (or None)
            - gradient_evals_to_target: Gradient evaluations to reach target (or None)
            - final_val_loss: Final validation loss achieved
            - min_val_loss: Minimum validation loss achieved
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4
    """
    val_loss_history = training_history.get('val_loss', [])
    iteration_time_history = training_history.get('iteration_time', [])
    
    if not val_loss_history:
        return {
            'iterations_to_target': None,
            'time_to_target': None,
            'gradient_evals_to_target': None,
            'final_val_loss': None,
            'min_val_loss': None,
        }
    
    # Find minimum validation loss
    min_val_loss = min(loss for _, loss in val_loss_history)
    final_val_loss = val_loss_history[-1][1]
    
    # Find first iteration where target loss is reached
    iterations_to_target = None
    time_to_target = None
    gradient_evals_to_target = None
    
    for iteration, loss in val_loss_history:
        if loss < target_loss:
            iterations_to_target = iteration
            
            # Compute wall-clock time up to this iteration
            time_to_target = sum(
                time_val for iter_val, time_val in iteration_time_history
                if iter_val <= iteration
            )
            
            # Gradient evaluations equal iterations (one gradient per iteration)
            gradient_evals_to_target = iteration
            break
    
    metrics = {
        'iterations_to_target': iterations_to_target,
        'time_to_target': time_to_target,
        'gradient_evals_to_target': gradient_evals_to_target,
        'final_val_loss': final_val_loss,
        'min_val_loss': min_val_loss,
    }
    
    return metrics
