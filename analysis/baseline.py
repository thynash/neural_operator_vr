"""Baseline comparison utilities for theoretical analysis."""

from typing import Dict, Optional
import torch
import numpy as np


def compute_theoretical_sgd_convergence_rate(
    learning_rate: float,
    strong_convexity: float,
    smoothness: float,
    gradient_variance: float,
    num_iterations: int
) -> float:
    """
    Compute theoretical convergence rate for SGD on strongly convex problems.
    
    For strongly convex functions with smoothness constant L and strong convexity μ,
    SGD with learning rate η converges as:
    E[f(x_t) - f(x*)] ≤ (1 - ημ)^t * [f(x_0) - f(x*)] + η*σ²/(2μ)
    
    where σ² is the gradient variance.
    
    Args:
        learning_rate: Step size η
        strong_convexity: Strong convexity parameter μ
        smoothness: Smoothness constant L
        gradient_variance: Gradient variance σ²
        num_iterations: Number of iterations t
    
    Returns:
        Theoretical convergence rate (expected suboptimality)
    
    Validates: Requirements 18.1
    """
    # Convergence rate: (1 - η*μ)^t
    convergence_factor = (1 - learning_rate * strong_convexity) ** num_iterations
    
    # Asymptotic error: η*σ²/(2μ)
    asymptotic_error = learning_rate * gradient_variance / (2 * strong_convexity)
    
    # Total expected suboptimality (assuming f(x_0) - f(x*) = 1 for normalization)
    expected_suboptimality = convergence_factor + asymptotic_error
    
    return expected_suboptimality


def compute_variance_reduction_factor(
    sgd_variance: float,
    svrg_variance: float
) -> float:
    """
    Compute variance reduction factor achieved by SVRG relative to SGD.
    
    Variance reduction factor = σ²_SGD / σ²_SVRG
    
    A factor > 1 indicates SVRG reduces variance compared to SGD.
    
    Args:
        sgd_variance: Gradient variance for SGD (σ²_SGD)
        svrg_variance: Gradient variance for SVRG (σ²_SVRG)
    
    Returns:
        Variance reduction factor
    
    Validates: Requirements 18.2
    """
    if svrg_variance < 1e-10:
        # SVRG variance is essentially zero (perfect variance reduction)
        return float('inf')
    
    reduction_factor = sgd_variance / svrg_variance
    
    return reduction_factor


def compute_spectral_approximation_quality(
    learned_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor
) -> Dict[str, float]:
    """
    Compute spectral approximation quality metrics.
    
    Compares learned operator eigenvalues to true dynamical system eigenvalues.
    
    Args:
        learned_eigenvalues: Eigenvalues from learned operator [n]
        true_eigenvalues: True eigenvalues from dynamical system [m]
    
    Returns:
        Dictionary containing:
            - mean_absolute_error: Mean absolute error between matched eigenvalues
            - max_absolute_error: Maximum absolute error
            - relative_error: Relative error normalized by true eigenvalue magnitudes
            - spectral_radius_error: Error in spectral radius
    
    Validates: Requirements 18.3
    """
    from analysis.spectral import compute_eigenvalue_error, compute_spectral_radius
    
    # Convert to numpy for easier manipulation
    learned = learned_eigenvalues.detach().cpu().numpy()
    true = true_eigenvalues.detach().cpu().numpy()
    
    # Compute mean absolute error using optimal matching
    mean_abs_error = compute_eigenvalue_error(
        learned_eigenvalues, true_eigenvalues, method='hungarian'
    )
    
    # Compute maximum absolute error
    from scipy.optimize import linear_sum_assignment
    
    n_learned = len(learned)
    n_true = len(true)
    
    # Cost matrix
    cost_matrix = np.zeros((n_learned, n_true))
    for i in range(n_learned):
        for j in range(n_true):
            cost_matrix[i, j] = np.abs(learned[i] - true[j])
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Maximum error
    max_abs_error = cost_matrix[row_ind, col_ind].max()
    
    # Relative error (normalized by true eigenvalue magnitudes)
    true_magnitudes = np.abs(true[col_ind])
    relative_errors = cost_matrix[row_ind, col_ind] / (true_magnitudes + 1e-10)
    mean_relative_error = relative_errors.mean()
    
    # Spectral radius error
    learned_spectral_radius = compute_spectral_radius(learned_eigenvalues)
    true_spectral_radius = compute_spectral_radius(true_eigenvalues)
    spectral_radius_error = abs(learned_spectral_radius - true_spectral_radius)
    
    quality_metrics = {
        'mean_absolute_error': float(mean_abs_error),
        'max_absolute_error': float(max_abs_error),
        'relative_error': float(mean_relative_error),
        'spectral_radius_error': float(spectral_radius_error),
    }
    
    return quality_metrics


def compute_theoretical_svrg_convergence_rate(
    learning_rate: float,
    strong_convexity: float,
    smoothness: float,
    gradient_variance: float,
    inner_loop_length: int,
    num_epochs: int
) -> float:
    """
    Compute theoretical convergence rate for SVRG on strongly convex problems.
    
    SVRG achieves linear convergence with reduced variance:
    E[f(x_t) - f(x*)] ≤ ρ^t * [f(x_0) - f(x*)]
    
    where ρ < 1 depends on learning rate, strong convexity, and inner loop length.
    
    Args:
        learning_rate: Step size η
        strong_convexity: Strong convexity parameter μ
        smoothness: Smoothness constant L
        gradient_variance: Initial gradient variance σ²
        inner_loop_length: Inner loop length m
        num_epochs: Number of outer loop epochs
    
    Returns:
        Theoretical convergence rate (expected suboptimality)
    
    Validates: Requirements 18.1
    """
    # SVRG convergence rate (simplified)
    # ρ ≈ 1 - μ*η*m / (1 + 2*L*η*m)
    
    numerator = strong_convexity * learning_rate * inner_loop_length
    denominator = 1 + 2 * smoothness * learning_rate * inner_loop_length
    
    convergence_rate = 1 - numerator / denominator
    
    # Expected suboptimality after num_epochs
    expected_suboptimality = convergence_rate ** num_epochs
    
    return expected_suboptimality


def compare_optimizer_efficiency(
    sgd_history: Dict,
    adam_history: Dict,
    svrg_history: Dict,
    target_loss: float
) -> Dict[str, Dict[str, float]]:
    """
    Compare optimizer efficiency across multiple metrics.
    
    Args:
        sgd_history: Training history for SGD
        adam_history: Training history for Adam
        svrg_history: Training history for SVRG
        target_loss: Target validation loss for convergence
    
    Returns:
        Dictionary with efficiency metrics for each optimizer
    
    Validates: Requirements 18.1, 18.2
    """
    from analysis.metrics import compute_convergence_metrics
    
    optimizers = {
        'SGD': sgd_history,
        'Adam': adam_history,
        'SVRG': svrg_history,
    }
    
    efficiency_comparison = {}
    
    for opt_name, history in optimizers.items():
        # Compute convergence metrics
        convergence = compute_convergence_metrics(history, target_loss)
        
        # Extract final gradient variance
        grad_var_history = history.get('train_grad_variance', [])
        final_grad_variance = grad_var_history[-1][1] if grad_var_history else None
        
        efficiency_comparison[opt_name] = {
            'iterations_to_target': convergence['iterations_to_target'],
            'time_to_target': convergence['time_to_target'],
            'gradient_evals_to_target': convergence['gradient_evals_to_target'],
            'final_val_loss': convergence['final_val_loss'],
            'min_val_loss': convergence['min_val_loss'],
            'final_grad_variance': final_grad_variance,
        }
    
    # Compute variance reduction factors
    sgd_var = efficiency_comparison['SGD']['final_grad_variance']
    svrg_var = efficiency_comparison['SVRG']['final_grad_variance']
    
    if sgd_var is not None and svrg_var is not None:
        efficiency_comparison['variance_reduction_factor'] = compute_variance_reduction_factor(
            sgd_var, svrg_var
        )
    
    return efficiency_comparison
