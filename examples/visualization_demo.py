"""Demonstration of visualization module capabilities.

This script shows how to use the visualization module to create
publication-quality plots for neural operator experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path

from visualization import (
    plot_training_curves,
    plot_gradient_variance,
    plot_validation_error,
    plot_long_horizon_predictions,
    plot_eigenvalue_comparison,
    plot_cost_vs_accuracy,
    plot_burgers_spatiotemporal,
    save_all_plots
)


def generate_mock_histories():
    """Generate mock training histories for demonstration."""
    
    # Generate training data for three optimizers
    histories = {}
    
    # SGD: slower convergence, higher variance
    sgd_train_loss = [(i, 1.0 / (i + 1) + 0.1 * np.random.rand()) for i in range(1000)]
    sgd_grad_var = [(i * 10, 0.5 / (i + 1) + 0.05 * np.random.rand()) for i in range(100)]
    sgd_val_error = [(i * 10, 0.8 / (i + 1) + 0.05 * np.random.rand()) for i in range(100)]
    sgd_iter_time = [(i, 0.1 + 0.01 * np.random.rand()) for i in range(1000)]
    
    histories['sgd'] = {
        'train_loss': sgd_train_loss,
        'train_grad_variance': sgd_grad_var,
        'val_relative_error': sgd_val_error,
        'iteration_time': sgd_iter_time
    }
    
    # Adam: faster convergence, moderate variance
    adam_train_loss = [(i, 0.8 / (i + 1) + 0.08 * np.random.rand()) for i in range(1000)]
    adam_grad_var = [(i * 10, 0.4 / (i + 1) + 0.04 * np.random.rand()) for i in range(100)]
    adam_val_error = [(i * 10, 0.6 / (i + 1) + 0.04 * np.random.rand()) for i in range(100)]
    adam_iter_time = [(i, 0.12 + 0.01 * np.random.rand()) for i in range(1000)]
    
    histories['adam'] = {
        'train_loss': adam_train_loss,
        'train_grad_variance': adam_grad_var,
        'val_relative_error': adam_val_error,
        'iteration_time': adam_iter_time
    }
    
    # SVRG: best convergence, lowest variance
    svrg_train_loss = [(i, 0.7 / (i + 1) + 0.05 * np.random.rand()) for i in range(1000)]
    svrg_grad_var = [(i * 10, 0.2 / (i + 1) + 0.02 * np.random.rand()) for i in range(100)]
    svrg_val_error = [(i * 10, 0.5 / (i + 1) + 0.03 * np.random.rand()) for i in range(100)]
    svrg_iter_time = [(i, 0.15 + 0.01 * np.random.rand()) for i in range(1000)]
    
    histories['svrg'] = {
        'train_loss': svrg_train_loss,
        'train_grad_variance': svrg_grad_var,
        'val_relative_error': svrg_val_error,
        'iteration_time': svrg_iter_time
    }
    
    return histories


def generate_mock_trajectories():
    """Generate mock trajectories for demonstration."""
    
    # Logistic Map
    logistic_true = np.zeros(100)
    logistic_true[0] = 0.5
    r = 3.8
    for i in range(1, 100):
        logistic_true[i] = r * logistic_true[i-1] * (1 - logistic_true[i-1])
    
    logistic_pred = logistic_true + np.random.randn(100) * 0.05
    
    # Lorenz System (simplified)
    t = np.linspace(0, 10, 200)
    lorenz_true = np.zeros((200, 3))
    lorenz_true[:, 0] = 10 * np.sin(t)
    lorenz_true[:, 1] = 10 * np.cos(t)
    lorenz_true[:, 2] = 20 + 5 * np.sin(2 * t)
    
    lorenz_pred = lorenz_true + np.random.randn(200, 3) * 0.5
    
    # Burgers Equation
    x = np.linspace(0, 2 * np.pi, 64)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    burgers_true = np.sin(X) * np.exp(-0.1 * T)
    burgers_pred = burgers_true + np.random.randn(100, 64) * 0.05
    
    return {
        'logistic': (logistic_true, logistic_pred),
        'lorenz': (lorenz_true, lorenz_pred),
        'burgers': (burgers_true, burgers_pred)
    }


def generate_mock_eigenvalues():
    """Generate mock eigenvalues for demonstration."""
    
    # True eigenvalues on unit circle
    angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    radii = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])
    
    true_eigs = radii * np.exp(1j * angles)
    
    # Learned eigenvalues with some error
    learned_eigs = true_eigs + (np.random.randn(8) + 1j * np.random.randn(8)) * 0.05
    
    return true_eigs, learned_eigs


def main():
    """Run visualization demonstration."""
    
    print("Generating mock data...")
    histories = generate_mock_histories()
    trajectories = generate_mock_trajectories()
    true_eigs, learned_eigs = generate_mock_eigenvalues()
    
    # Create output directory
    output_dir = Path('visualization_output')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}/...")
    
    # Individual plots
    print("  - Training curves...")
    plot_training_curves(
        histories,
        ['sgd', 'adam', 'svrg'],
        output_path=output_dir / 'training_curves.pdf'
    )
    
    print("  - Gradient variance...")
    plot_gradient_variance(
        histories,
        ['sgd', 'adam', 'svrg'],
        output_path=output_dir / 'gradient_variance.pdf'
    )
    
    print("  - Validation error...")
    plot_validation_error(
        histories,
        ['sgd', 'adam', 'svrg'],
        output_path=output_dir / 'validation_error.pdf'
    )
    
    print("  - Computational cost...")
    plot_cost_vs_accuracy(
        histories,
        ['sgd', 'adam', 'svrg'],
        output_path=output_dir / 'cost_vs_accuracy.pdf'
    )
    
    print("  - Logistic Map predictions...")
    plot_long_horizon_predictions(
        trajectories['logistic'][0],
        trajectories['logistic'][1],
        'logistic',
        output_path=output_dir / 'logistic_predictions.pdf'
    )
    
    print("  - Lorenz System predictions...")
    plot_long_horizon_predictions(
        trajectories['lorenz'][0],
        trajectories['lorenz'][1],
        'lorenz',
        output_path=output_dir / 'lorenz_predictions.pdf'
    )
    
    print("  - Burgers Equation predictions...")
    plot_long_horizon_predictions(
        trajectories['burgers'][0],
        trajectories['burgers'][1],
        'burgers',
        output_path=output_dir / 'burgers_predictions.pdf'
    )
    
    print("  - Eigenvalue comparison...")
    plot_eigenvalue_comparison(
        true_eigs,
        learned_eigs,
        output_path=output_dir / 'eigenvalue_comparison.pdf'
    )
    
    print("  - Burgers spatiotemporal...")
    plot_burgers_spatiotemporal(
        trajectories['burgers'][0],
        output_path=output_dir / 'burgers_spatiotemporal.pdf'
    )
    
    # Save all plots at once
    print("\nGenerating all plots using save_all_plots()...")
    save_all_plots(
        histories,
        ['sgd', 'adam', 'svrg'],
        output_dir / 'all_plots',
        true_trajectory=trajectories['logistic'][0],
        predicted_trajectory=trajectories['logistic'][1],
        system_name='logistic',
        true_eigenvalues=true_eigs,
        learned_eigenvalues=learned_eigs,
        burgers_solution=trajectories['burgers'][0]
    )
    
    print(f"\n✓ All plots generated successfully in {output_dir}/")
    print("\nGenerated files:")
    for pdf_file in sorted(output_dir.rglob('*.pdf')):
        print(f"  - {pdf_file.relative_to(output_dir)}")


if __name__ == '__main__':
    main()
