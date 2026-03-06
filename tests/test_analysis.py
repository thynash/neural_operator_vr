"""Tests for analysis module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.metrics import (
    compute_training_metrics,
    compute_validation_metrics,
    compute_long_horizon_metrics,
    compute_convergence_metrics,
)
from analysis.spectral import (
    compute_operator_eigenvalues,
    compute_spectral_radius,
    compute_eigenvalue_error,
    track_eigenvalue_evolution,
)
from analysis.baseline import (
    compute_theoretical_sgd_convergence_rate,
    compute_variance_reduction_factor,
    compute_spectral_approximation_quality,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=10, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def test_compute_training_metrics():
    """Test training metrics computation."""
    model = SimpleModel()
    
    # Create dummy batch
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 10)
    batch = (inputs, targets)
    
    # Forward pass
    outputs = model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Compute metrics
    metrics = compute_training_metrics(model, batch, loss, torch.device('cpu'))
    
    assert 'train_loss' in metrics
    assert 'train_grad_norm' in metrics
    assert metrics['train_loss'] > 0
    assert metrics['train_grad_norm'] >= 0


def test_compute_validation_metrics():
    """Test validation metrics computation."""
    model = SimpleModel()
    
    # Create dummy validation data
    inputs = torch.randn(20, 10)
    targets = torch.randn(20, 10)
    dataset = TensorDataset(inputs, targets)
    val_loader = DataLoader(dataset, batch_size=4)
    
    # Compute metrics
    metrics = compute_validation_metrics(model, val_loader, torch.device('cpu'))
    
    assert 'val_loss' in metrics
    assert 'val_relative_error' in metrics
    assert 'val_max_error' in metrics
    assert 'val_mean_absolute_error' in metrics
    assert all(v >= 0 for v in metrics.values())


def test_compute_long_horizon_metrics():
    """Test long-horizon metrics computation."""
    model = SimpleModel()
    
    # Create dummy trajectory
    initial_state = torch.randn(2, 10)
    true_trajectory = torch.randn(2, 20, 10)
    
    # Compute metrics
    metrics = compute_long_horizon_metrics(
        model, initial_state, true_trajectory, num_steps=20, device=torch.device('cpu')
    )
    
    assert 'long_horizon_mse' in metrics
    assert 'long_horizon_steps' in metrics
    assert metrics['long_horizon_mse'] >= 0
    assert metrics['long_horizon_steps'] > 0


def test_compute_convergence_metrics():
    """Test convergence metrics computation."""
    # Create dummy training history
    training_history = {
        'val_loss': [(100, 1.0), (200, 0.5), (300, 0.2), (400, 0.05)],
        'iteration_time': [(100, 0.1), (200, 0.1), (300, 0.1), (400, 0.1)],
    }
    
    # Compute metrics
    metrics = compute_convergence_metrics(training_history, target_loss=0.1)
    
    assert 'iterations_to_target' in metrics
    assert 'time_to_target' in metrics
    assert 'gradient_evals_to_target' in metrics
    assert 'final_val_loss' in metrics
    assert 'min_val_loss' in metrics
    
    assert metrics['iterations_to_target'] == 400
    assert metrics['min_val_loss'] == 0.05


def test_compute_operator_eigenvalues():
    """Test eigenvalue computation."""
    model = SimpleModel(input_dim=5, output_dim=5)
    state_point = torch.randn(5)
    
    # Compute eigenvalues
    eigenvalues = compute_operator_eigenvalues(model, state_point, method='eig')
    
    assert eigenvalues.shape[0] == 5
    assert eigenvalues.dtype == torch.complex64 or eigenvalues.dtype == torch.complex128


def test_compute_spectral_radius():
    """Test spectral radius computation."""
    # Create dummy eigenvalues
    eigenvalues = torch.tensor([1.0 + 0.5j, -0.5 + 0.8j, 0.3 - 0.2j])
    
    # Compute spectral radius
    spectral_radius = compute_spectral_radius(eigenvalues)
    
    # Should be max(|1.0 + 0.5j|, |-0.5 + 0.8j|, |0.3 - 0.2j|)
    expected = max(abs(1.0 + 0.5j), abs(-0.5 + 0.8j), abs(0.3 - 0.2j))
    
    assert abs(spectral_radius - expected) < 1e-5


def test_compute_eigenvalue_error():
    """Test eigenvalue error computation."""
    learned = torch.tensor([1.0 + 0.1j, 0.5 + 0.2j, -0.3 + 0.1j])
    true = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j, -0.3 + 0.0j])
    
    # Compute error
    error = compute_eigenvalue_error(learned, true, method='nearest')
    
    assert error >= 0
    assert error < 1.0  # Should be small since eigenvalues are close


def test_track_eigenvalue_evolution():
    """Test eigenvalue evolution tracking."""
    model = SimpleModel(input_dim=5, output_dim=5)
    state_point = torch.randn(5)
    training_history = {}
    
    # Track eigenvalues
    updated_history = track_eigenvalue_evolution(
        model, state_point, training_history, iteration=100, num_eigenvalues=5
    )
    
    assert 'spectral_radius' in updated_history
    assert 'eigenvalues' in updated_history
    assert len(updated_history['spectral_radius']) == 1
    assert len(updated_history['eigenvalues']) == 1


def test_compute_theoretical_sgd_convergence_rate():
    """Test theoretical SGD convergence rate."""
    rate = compute_theoretical_sgd_convergence_rate(
        learning_rate=0.01,
        strong_convexity=0.1,
        smoothness=1.0,
        gradient_variance=0.5,
        num_iterations=100
    )
    
    assert rate > 0
    assert rate < 1  # Should converge


def test_compute_variance_reduction_factor():
    """Test variance reduction factor computation."""
    sgd_variance = 1.0
    svrg_variance = 0.1
    
    factor = compute_variance_reduction_factor(sgd_variance, svrg_variance)
    
    assert factor == 10.0  # 1.0 / 0.1


def test_compute_spectral_approximation_quality():
    """Test spectral approximation quality."""
    learned = torch.tensor([1.0 + 0.1j, 0.5 + 0.2j, -0.3 + 0.1j])
    true = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j, -0.3 + 0.0j])
    
    # Compute quality metrics
    quality = compute_spectral_approximation_quality(learned, true)
    
    assert 'mean_absolute_error' in quality
    assert 'max_absolute_error' in quality
    assert 'relative_error' in quality
    assert 'spectral_radius_error' in quality
    assert all(v >= 0 for v in quality.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
