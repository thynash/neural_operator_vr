# Methods Summary for Paper

## Data Generation Methodology

### Lorenz System

The Lorenz system is a set of three coupled ordinary differential equations that exhibit chaotic behavior:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

**Parameters Used**:
- σ (Prandtl number) = 10
- ρ (Rayleigh number) = 28
- β (geometric factor) = 8/3

**Data Generation Process**:
1. **Initial Conditions**: Random initialization from uniform distribution U(-10, 10) for each state variable
2. **Integration Method**: 4th-order Runge-Kutta (RK4) with adaptive time stepping
3. **Time Step**: dt = 0.01
4. **Trajectory Length**: 500 time steps per trajectory
5. **Number of Trajectories**: 1,000 independent trajectories
6. **State Dimension**: 3 (x, y, z coordinates)
7. **Total Samples**: 500,000 state observations

**Preprocessing**:
- Discarded first 100 steps (transient behavior)
- Normalized each dimension to zero mean and unit variance
- Split: 70% training, 15% validation, 15% testing

**Characteristics**:
- Exhibits sensitive dependence on initial conditions
- Strange attractor with fractal dimension ≈ 2.06
- Lyapunov exponent λ₁ ≈ 0.906 (positive, indicating chaos)

---

### Logistic Map

The logistic map is a discrete-time dynamical system defined by:

```
x_{n+1} = r·x_n·(1 - x_n)
```

**Parameters Used**:
- Growth rate r: sampled from U(3.57, 4.0) (chaotic regime)
- Initial condition x₀: sampled from U(0.1, 0.9)

**Data Generation Process**:
1. **Parameter Sampling**: For each trajectory, randomly sample r from chaotic regime
2. **Initial Conditions**: Random x₀ ∈ (0, 1) to avoid fixed points
3. **Iteration**: Compute 200 iterations per trajectory
4. **Number of Trajectories**: 2,000 independent trajectories
5. **State Dimension**: 1 (scalar x)
6. **Total Samples**: 400,000 state observations

**Preprocessing**:
- Discarded first 50 iterations (transient behavior)
- No normalization needed (values naturally in [0, 1])
- Split: 70% training, 15% validation, 15% testing

**Characteristics**:
- Period-doubling route to chaos
- Exhibits intermittency and bifurcations
- Lyapunov exponent λ ≈ 0.5 for r = 4.0
- Serves as canonical example of discrete chaos

---

## Training Protocol

### Hyperparameters

**Adam Optimizer**:
- Learning rate: 0.001
- β₁ (momentum): 0.9
- β₂ (RMSprop): 0.999
- ε (numerical stability): 1e-8
- Weight decay: 0 (no L2 regularization)

**SGD Optimizer**:
- Learning rate: 0.01
- Momentum: 0.9
- Nesterov momentum: True
- Weight decay: 0

**SVRG Optimizer**:
- Learning rate: 0.001
- Update frequency: 100 iterations (full gradient computation)
- Inner loop iterations: 100
- Weight decay: 0

**Training Configuration**:
- Batch size: 32
- Maximum epochs: 100
- Early stopping patience: 10 epochs
- Loss function: Mean Squared Error (MSE)
- Gradient clipping: max norm = 1.0

### Computational Resources

- Hardware: NVIDIA GPU (CUDA-enabled) / CPU fallback
- Framework: PyTorch 2.0+
- Precision: Float32
- Parallel workers: 4 (data loading)

---

## Evaluation Metrics

### Primary Metrics

1. **Final Validation Loss**: MSE on held-out validation set
2. **Training Time**: Wall-clock time in seconds
3. **Gradient Variance**: Mean variance of gradients over training
4. **Spectral Radius**: Largest eigenvalue magnitude of Hessian
5. **Convergence Epoch**: First epoch reaching 99% of final loss

### Secondary Metrics

1. **Iterations to Target**: Number of iterations to reach loss threshold
2. **Memory Usage**: Peak GPU/CPU memory consumption
3. **Stability**: Standard deviation of loss over last 10 epochs

---

## Reproducibility

All experiments use fixed random seeds:
- Lorenz system: seed = 42
- Logistic map: seed = 123
- PyTorch: torch.manual_seed(42)
- NumPy: np.random.seed(42)

Code and data available at: [GitHub repository URL]
