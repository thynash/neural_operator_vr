# Neural Operator Variance Reduction Framework

A research-grade experimental framework for investigating variance-reduced stochastic optimization in neural operator learning. This framework enables rigorous comparison of SVRG (Stochastic Variance Reduced Gradient) against SGD and Adam optimizers when training neural operators (DeepONet and FNO) for dynamical system modeling.

## Features

- **Neural Operator Architectures**: DeepONet and Fourier Neural Operator (FNO) implementations
- **Optimization Algorithms**: SGD, Adam, and SVRG with gradient variance tracking
- **Benchmark Dynamical Systems**: Logistic Map, Lorenz System, and Burgers Equation
- **Comprehensive Analysis**: Spectral analysis, convergence metrics, and long-horizon prediction evaluation
- **Publication-Quality Visualizations**: Automated generation of plots suitable for academic papers
- **Reproducibility**: Full experiment configuration management with deterministic seeding
- **Statistical Analysis**: Multi-seed experiment support with statistical significance testing

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for large-scale experiments)

### Install Dependencies

```bash
pip install torch numpy scipy matplotlib pyyaml hypothesis pytest
```

Or install from the project directory:

```bash
pip install -e .
```

### Verify Installation

Run the test suite to verify everything is working:

```bash
pytest tests/
```

## Quick Start

### 1. Run a Simple Experiment

Train a DeepONet on the Logistic Map using SVRG:

```bash
python examples/run_experiment.py --config examples/config_deeponet_logistic_svrg.yaml
```

### 2. Compare Optimizers

Run experiments with different optimizers on the same problem:

```bash
# SGD
python examples/run_experiment.py --config examples/config_deeponet_logistic_sgd.yaml

# Adam
python examples/run_experiment.py --config examples/config_deeponet_logistic_adam.yaml

# SVRG
python examples/run_experiment.py --config examples/config_deeponet_logistic_svrg.yaml
```

### 3. Analyze Results

Generate statistical analysis and comparison plots:

```bash
python examples/statistical_analysis_demo.py
```

### 4. Visualize Training

Create publication-quality visualizations:

```bash
python examples/visualization_demo.py
```

## Usage Examples

### Basic Training

```python
from experiments.experiment_runner import ExperimentRunner

# Load configuration and run experiment
runner = ExperimentRunner("examples/config_deeponet_logistic_svrg.yaml")
results = runner.run()

print(f"Final validation loss: {results['final_val_loss']:.6f}")
print(f"Iterations to target: {results['iterations_to_target']}")
```

### Multi-Seed Experiments

```python
from experiments.experiment_runner import ExperimentRunner

# Run with multiple random seeds for statistical analysis
runner = ExperimentRunner("examples/config_deeponet_logistic_svrg.yaml")
aggregated_results = runner.run_multiple_seeds(seeds=[42, 43, 44, 45, 46])

print(f"Mean validation loss: {aggregated_results['mean_val_loss']:.6f} ± {aggregated_results['std_val_loss']:.6f}")
```

### Custom Configuration

```python
from experiments.config_parser import load_config
from experiments.config_serializer import save_config

# Load and modify configuration
config = load_config("examples/config_deeponet_logistic_svrg.yaml")
config['optimizer']['params']['learning_rate'] = 0.005
config['training']['num_epochs'] = 300

# Save modified configuration
save_config(config, "my_custom_config.yaml")
```

### Gradient Variance Analysis

```python
from training.training_loop import TrainingLoop
from optimizers.svrg import SVRG

# During training, gradient variance is automatically computed
# Access variance history from training results
variance_history = results['metrics']['train_grad_variance']

# Plot variance evolution
import matplotlib.pyplot as plt
iterations, variances = zip(*variance_history)
plt.plot(iterations, variances)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Gradient Variance')
plt.savefig('gradient_variance.pdf')
```

### Spectral Analysis

```python
from analysis.spectral import compute_operator_eigenvalues, compute_spectral_radius

# Compute eigenvalues of learned operator
eigenvalues = compute_operator_eigenvalues(
    model=trained_model,
    state_point=test_state,
    num_eigenvalues=10
)

# Compute spectral radius
spectral_radius = compute_spectral_radius(eigenvalues)
print(f"Spectral radius: {spectral_radius:.4f}")

# Check stability (spectral radius < 1 implies contraction)
if spectral_radius < 1.0:
    print("Operator is contractive (stable)")
else:
    print("Operator may be unstable")
```

## Configuration

Experiments are configured using YAML files. See `examples/` directory for complete examples.

### Configuration Structure

```yaml
experiment:
  name: "experiment_name"
  seed: 42
  device: "cuda"  # or "cpu"
  deterministic: true

dataset:
  type: "logistic"  # or "lorenz", "burgers"
  params:
    # System-specific parameters
  num_train_trajectories: 800
  num_val_trajectories: 200
  input_horizon: 5
  output_horizon: 1

model:
  type: "deeponet"  # or "fno"
  params:
    # Architecture-specific parameters

optimizer:
  type: "svrg"  # or "sgd", "adam"
  params:
    learning_rate: 0.001
    # Optimizer-specific parameters

training:
  num_epochs: 150
  batch_size: 64
  validation_interval: 100
  variance_interval: 500
  checkpoint_interval: 1000

analysis:
  compute_spectral_radius: true
  spectral_interval: 1000
  long_horizon_steps: 200

visualization:
  generate_plots: true
  plot_format: "pdf"
  dpi: 300
```

### Available Configurations

- **DeepONet on Logistic Map**: `config_deeponet_logistic_{sgd,adam,svrg}.yaml`
- **DeepONet on Lorenz System**: `config_deeponet_lorenz_svrg.yaml`
- **FNO on Burgers Equation**: `config_fno_burgers_{sgd,adam,svrg}.yaml`

## Project Structure

```
neural_operator_vr/
├── datasets/          # Dynamical system data generators
│   ├── base.py
│   ├── logistic_map.py
│   ├── lorenz_system.py
│   └── burgers_equation.py
├── models/            # Neural operator architectures
│   ├── base.py
│   ├── deeponet.py
│   └── fno.py
├── optimizers/        # Optimization algorithms
│   ├── base.py
│   ├── sgd.py
│   ├── adam.py
│   └── svrg.py
├── training/          # Training loop and checkpointing
│   ├── training_loop.py
│   └── checkpoint_manager.py
├── analysis/          # Metrics and spectral analysis
│   ├── metrics.py
│   ├── spectral.py
│   ├── baseline.py
│   └── statistics.py
├── visualization/     # Publication-quality plotting
│   └── plots.py
├── experiments/       # Configuration and orchestration
│   ├── config_parser.py
│   ├── config_validator.py
│   └── experiment_runner.py
├── utils/             # Utilities
│   ├── logger.py
│   ├── seed.py
│   └── device.py
├── examples/          # Example scripts and configs
└── tests/             # Test suite
```

## Dynamical Systems

### Logistic Map

Discrete-time chaotic map: `x_{n+1} = r * x_n * (1 - x_n)`

- **Parameters**: Growth rate `r` (typically 3.8 for chaotic behavior)
- **Use Case**: Simple 1D discrete-time system for rapid prototyping
- **Operator Task**: Predict next k steps given previous m steps

### Lorenz System

Continuous-time chaotic system modeling atmospheric convection:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

- **Parameters**: σ=10, ρ=28, β=8/3 (standard chaotic regime)
- **Use Case**: 3D continuous-time system with rich dynamics
- **Operator Task**: Predict future state given current state and time interval

### Burgers Equation

1D viscous fluid dynamics PDE: `∂u/∂t + u∂u/∂x = ν∂²u/∂x²`

- **Parameters**: Viscosity ν (typically 0.01)
- **Use Case**: Spatiotemporal PDE with shock formation
- **Operator Task**: Predict solution at t+Δt given solution at t

## Optimizers

### SGD (Stochastic Gradient Descent)

Standard mini-batch gradient descent with momentum:
- **Pros**: Simple, well-understood, memory-efficient
- **Cons**: High gradient variance, slower convergence
- **Best for**: Baseline comparisons

### Adam (Adaptive Moment Estimation)

Adaptive learning rate optimizer with momentum:
- **Pros**: Fast convergence, adaptive per-parameter learning rates
- **Cons**: May not converge to optimal solution, high memory usage
- **Best for**: Quick prototyping, complex loss landscapes

### SVRG (Stochastic Variance Reduced Gradient)

Variance-reduced gradient method with periodic snapshots:
- **Pros**: Reduced gradient variance, faster convergence, theoretical guarantees
- **Cons**: Requires full gradient computation, more complex implementation
- **Best for**: Large datasets where variance reduction matters

## Metrics and Analysis

### Training Metrics

- **Training Loss**: MSE on training batches
- **Gradient Norm**: L2 norm of gradient vector
- **Gradient Variance**: Empirical variance of stochastic gradients
- **Iteration Time**: Wall-clock time per iteration

### Validation Metrics

- **Validation Loss**: MSE on validation set
- **Relative L2 Error**: Normalized prediction error
- **Max Error**: Maximum pointwise error
- **Mean Absolute Error**: Average absolute error

### Convergence Metrics

- **Iterations to Target**: Number of iterations to reach target loss
- **Time to Target**: Wall-clock time to reach target loss
- **Gradient Evaluations**: Total gradient computations to convergence

### Spectral Analysis

- **Eigenvalues**: Eigenvalues of operator Jacobian
- **Spectral Radius**: Maximum absolute eigenvalue (stability indicator)
- **Eigenvalue Error**: Approximation error vs true eigenvalues

## Visualization

The framework generates publication-quality plots:

- **Training Curves**: Loss vs iteration for multiple optimizers
- **Gradient Variance**: Variance evolution showing SVRG reduction
- **Validation Error**: Error curves with confidence intervals
- **Long-Horizon Predictions**: True vs predicted trajectories
- **Spectral Analysis**: Eigenvalue scatter plots
- **Computational Cost**: Time vs accuracy trade-offs

All plots use:
- Vector format (PDF) for scalability
- Colorblind-friendly palettes
- Consistent styling (fonts, sizes, colors)
- 300 DPI for high-quality output

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_models.py -v
pytest tests/test_datasets.py -v
pytest tests/test_analysis.py -v
```

Check code coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

## Reproducibility

The framework ensures full reproducibility:

1. **Deterministic Seeding**: All random number generators seeded
2. **Configuration Tracking**: Complete config saved with results
3. **System Info Logging**: Hardware and software versions recorded
4. **Checkpoint Management**: Full state saving for exact continuation
5. **Deterministic CUDA**: Optional deterministic GPU operations

To ensure reproducibility:
- Set `deterministic: true` in config
- Use the same random seed
- Use the same hardware (for bit-exact reproducibility)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{neural_operator_variance_reduction,
  title={Neural Operator Variance Reduction Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/neural-operator-vr}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation in `docs/`
- See example notebooks in `examples/`

## Acknowledgments

This framework builds on research in:
- Neural operators (DeepONet, FNO)
- Variance-reduced optimization (SVRG)
- Dynamical systems modeling
- Scientific machine learning
