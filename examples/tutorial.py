"""
Neural Operator Variance Reduction Framework Tutorial

This tutorial demonstrates the end-to-end workflow for training neural operators
with variance-reduced optimization.

Topics covered:
1. Generate training data from a dynamical system
2. Train a neural operator (DeepONet) using different optimizers
3. Compare gradient variance across optimizers
4. Analyze spectral properties of learned operators
5. Generate publication-quality visualizations
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import framework modules
from datasets.logistic_map import LogisticMapDataset
from datasets.data_manager import create_dataloaders
from models.deeponet import DeepONet
from optimizers.sgd import SGD
from optimizers.adam import Adam
from optimizers.svrg import SVRG
from training.training_loop import TrainingLoop
from analysis.spectral import SpectralAnalyzer
from visualization.plots import (
    plot_training_curves,
    plot_gradient_variance,
    plot_eigenvalue_comparison,
    configure_publication_style
)
from utils.seed import set_random_seeds
from utils.device import get_device
from utils.logger import MetricsLogger


def main():
    """Run the complete tutorial workflow."""
    
    print("=" * 80)
    print("Neural Operator Variance Reduction Framework Tutorial")
    print("=" * 80)
    
    # Setup
    set_random_seeds(42, deterministic=True)
    device = get_device('cuda')
    print(f'\nUsing device: {device}')
    
    # Configure publication-quality plots
    configure_publication_style()
    
    # =========================================================================
    # Step 1: Generate Training Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Generating Training Data from Logistic Map")
    print("=" * 80)
    
    # Create dataset
    # The Logistic Map: x_{n+1} = r * x_n * (1 - x_n)
    # For r=3.8, the system exhibits chaotic behavior
    dataset = LogisticMapDataset(
        r=3.8,
        initial_condition=0.5,
        trajectory_length=1000
    )
    
    print("\nGenerating training data...")
    train_data = dataset.create_operator_dataset(
        num_trajectories=800,
        trajectory_length=1000,
        input_horizon=5,
        output_horizon=1
    )
    
    print("Generating validation data...")
    val_data = dataset.create_operator_dataset(
        num_trajectories=200,
        trajectory_length=1000,
        input_horizon=5,
        output_horizon=1
    )
    
    print(f'\nTraining data shape: {train_data[0].shape}')
    print(f'Validation data shape: {val_data[0].shape}')
    
    # Visualize sample trajectory
    print("\nGenerating sample trajectory visualization...")
    sample_trajectory = dataset.generate_trajectory(
        initial_condition=np.array([0.5]),
        length=100
    )
    
    plt.figure(figsize=(10, 4))
    plt.plot(sample_trajectory, 'b-', linewidth=1.5)
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Logistic Map Trajectory (r=3.8)')
    plt.grid(True, alpha=0.3)
    plt.savefig('tutorial_output/logistic_trajectory.pdf', bbox_inches='tight')
    print("Saved: tutorial_output/logistic_trajectory.pdf")
    
    # =========================================================================
    # Step 2: Create Data Loaders
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Creating PyTorch Data Loaders")
    print("=" * 80)
    
    train_loader, val_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=64,
        shuffle=True,
        device=device
    )
    
    print(f'\nNumber of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')
    
    # =========================================================================
    # Step 3: Create Neural Operator Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Creating DeepONet Model")
    print("=" * 80)
    
    print("\nDeepONet Architecture:")
    print("  - Branch Network: Processes input function samples")
    print("  - Trunk Network: Processes query locations")
    print("  - Combination: Inner product of branch and trunk outputs")
    
    model = DeepONet(
        branch_layers=[64, 64, 32],
        trunk_layers=[64, 64, 32],
        basis_dim=32,
        activation='tanh',
        use_bias=True
    ).to(device)
    
    param_count = model.get_parameter_count()
    print(f'\nModel created with {param_count:,} parameters')
    
    # =========================================================================
    # Step 4: Train with Different Optimizers
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training with Different Optimizers")
    print("=" * 80)
    
    histories = {}
    
    # Configuration for training
    from experiments.config_schema import ExperimentConfig
    config = ExperimentConfig.from_dict({
        'training': {
            'validation_interval': 50,
            'variance_interval': 200,
            'checkpoint_interval': 1000,
            'early_stopping_patience': 20,
            'target_loss': 0.0001
        }
    })
    
    # 4.1 Train with SGD
    print("\n--- Training with SGD ---")
    model_sgd = DeepONet(
        branch_layers=[64, 64, 32],
        trunk_layers=[64, 64, 32],
        basis_dim=32,
        activation='tanh',
        use_bias=True
    ).to(device)
    
    optimizer_sgd = SGD(
        model_sgd.parameters(),
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    )
    
    logger_sgd = MetricsLogger('./tutorial_output/logs', 'sgd_experiment')
    
    training_loop_sgd = TrainingLoop(
        model=model_sgd,
        optimizer=optimizer_sgd,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics_logger=logger_sgd,
        config=config
    )
    
    print("Training for 30 epochs...")
    histories['SGD'] = training_loop_sgd.run(num_epochs=30)
    print("SGD training complete!")
    
    # 4.2 Train with Adam
    print("\n--- Training with Adam ---")
    model_adam = DeepONet(
        branch_layers=[64, 64, 32],
        trunk_layers=[64, 64, 32],
        basis_dim=32,
        activation='tanh',
        use_bias=True
    ).to(device)
    
    optimizer_adam = Adam(
        model_adam.parameters(),
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999
    )
    
    logger_adam = MetricsLogger('./tutorial_output/logs', 'adam_experiment')
    
    training_loop_adam = TrainingLoop(
        model=model_adam,
        optimizer=optimizer_adam,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics_logger=logger_adam,
        config=config
    )
    
    print("Training for 30 epochs...")
    histories['Adam'] = training_loop_adam.run(num_epochs=30)
    print("Adam training complete!")
    
    # 4.3 Train with SVRG
    print("\n--- Training with SVRG ---")
    model_svrg = DeepONet(
        branch_layers=[64, 64, 32],
        trunk_layers=[64, 64, 32],
        basis_dim=32,
        activation='tanh',
        use_bias=True
    ).to(device)
    
    optimizer_svrg = SVRG(
        model_svrg.parameters(),
        train_loader=train_loader,
        learning_rate=0.001,
        inner_loop_length=50
    )
    
    logger_svrg = MetricsLogger('./tutorial_output/logs', 'svrg_experiment')
    
    training_loop_svrg = TrainingLoop(
        model=model_svrg,
        optimizer=optimizer_svrg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics_logger=logger_svrg,
        config=config
    )
    
    print("Training for 30 epochs...")
    histories['SVRG'] = training_loop_svrg.run(num_epochs=30)
    print("SVRG training complete!")
    
    # =========================================================================
    # Step 5: Compare Gradient Variance
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Comparing Gradient Variance Across Optimizers")
    print("=" * 80)
    
    print("\nGenerating gradient variance comparison plot...")
    plot_gradient_variance(
        histories=histories,
        optimizer_names=['SGD', 'Adam', 'SVRG'],
        output_path='tutorial_output/gradient_variance.pdf'
    )
    print("Saved: tutorial_output/gradient_variance.pdf")
    
    print("\nKey Observation:")
    print("  SVRG should show lower gradient variance compared to SGD,")
    print("  demonstrating the variance reduction mechanism.")
    
    # =========================================================================
    # Step 6: Analyze Training Convergence
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Analyzing Training Convergence")
    print("=" * 80)
    
    print("\nGenerating training curves...")
    plot_training_curves(
        histories=histories,
        optimizer_names=['SGD', 'Adam', 'SVRG'],
        output_path='tutorial_output/training_curves.pdf'
    )
    print("Saved: tutorial_output/training_curves.pdf")
    
    # Print final losses
    print("\nFinal Validation Losses:")
    for opt_name, history in histories.items():
        if 'val_loss' in history and len(history['val_loss']) > 0:
            final_loss = history['val_loss'][-1][1]
            print(f"  {opt_name}: {final_loss:.6f}")
    
    # =========================================================================
    # Step 7: Spectral Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Spectral Analysis of Learned Operators")
    print("=" * 80)
    
    print("\nComputing eigenvalues of learned operators...")
    
    # Create spectral analyzer
    analyzer = SpectralAnalyzer(
        model=model_svrg,  # Use SVRG-trained model
        dataset=dataset,
        device=device
    )
    
    # Get a sample state point
    sample_batch = next(iter(val_loader))
    sample_input = sample_batch[0][0:1]
    
    # Compute learned eigenvalues
    learned_eigenvalues = analyzer.compute_operator_eigenvalues(
        state_point=sample_input,
        num_eigenvalues=5
    )
    
    # Get true eigenvalues
    true_eigenvalues = dataset.get_true_eigenvalues(sample_input.cpu().numpy())
    
    # Compute spectral radius
    spectral_radius = analyzer.compute_spectral_radius(learned_eigenvalues)
    print(f"\nSpectral radius: {spectral_radius:.4f}")
    
    if spectral_radius < 1.0:
        print("✓ Operator is contractive (stable)")
    else:
        print("⚠ Operator may be unstable")
    
    # Compute eigenvalue error
    if true_eigenvalues is not None:
        eigenvalue_error = analyzer.compute_eigenvalue_error(
            learned_eigenvalues=learned_eigenvalues,
            true_eigenvalues=true_eigenvalues
        )
        print(f"Eigenvalue approximation error: {eigenvalue_error:.6f}")
        
        # Plot eigenvalue comparison
        print("\nGenerating eigenvalue comparison plot...")
        plot_eigenvalue_comparison(
            true_eigenvalues=true_eigenvalues,
            learned_eigenvalues=learned_eigenvalues,
            output_path='tutorial_output/eigenvalue_comparison.pdf'
        )
        print("Saved: tutorial_output/eigenvalue_comparison.pdf")
    
    # =========================================================================
    # Step 8: Summary and Next Steps
    # =========================================================================
    print("\n" + "=" * 80)
    print("Tutorial Complete!")
    print("=" * 80)
    
    print("\nWhat you learned:")
    print("  1. How to generate training data from dynamical systems")
    print("  2. How to create and train neural operator models")
    print("  3. How to compare different optimization algorithms")
    print("  4. How to measure and visualize gradient variance")
    print("  5. How to perform spectral analysis of learned operators")
    print("  6. How to generate publication-quality visualizations")
    
    print("\nNext steps:")
    print("  - Try different dynamical systems (Lorenz, Burgers)")
    print("  - Experiment with FNO architecture")
    print("  - Run multi-seed experiments for statistical analysis")
    print("  - Tune hyperparameters for better performance")
    print("  - Explore long-horizon prediction capabilities")
    
    print("\nAll outputs saved to: tutorial_output/")
    print("\nFor more examples, see:")
    print("  - examples/run_experiment.py")
    print("  - examples/statistical_analysis_demo.py")
    print("  - examples/visualization_demo.py")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Create output directory
    Path('tutorial_output').mkdir(exist_ok=True)
    Path('tutorial_output/logs').mkdir(exist_ok=True, parents=True)
    
    # Run tutorial
    main()
