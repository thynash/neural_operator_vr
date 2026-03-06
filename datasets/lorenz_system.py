"""Lorenz System dynamical system dataset."""

from typing import Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp
from .base import DynamicalSystem


class LorenzSystemDataset(DynamicalSystem):
    """Lorenz System continuous-time dynamical system.
    
    The Lorenz system is defined by:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    where σ (sigma), ρ (rho), and β (beta) are system parameters.
    This is a classic example of a chaotic attractor.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize LorenzSystemDataset.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def _lorenz_ode(
        self,
        t: float,
        state: np.ndarray,
        sigma: float,
        rho: float,
        beta: float
    ) -> np.ndarray:
        """Lorenz system ODE right-hand side.
        
        Args:
            t: Time (unused, system is autonomous)
            state: Current state [x, y, z]
            sigma: Prandtl number
            rho: Rayleigh number
            beta: Geometric parameter
        
        Returns:
            derivatives: Time derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])
    
    def generate_trajectory(
        self,
        initial_condition: np.ndarray,
        length: int,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
        **params
    ) -> np.ndarray:
        """Generate a single trajectory from initial condition.
        
        Uses scipy's solve_ivp with RK45 method for numerical integration.
        
        Args:
            initial_condition: Starting state [x, y, z]
            length: Number of time steps to generate
            sigma: Prandtl number (default: 10.0)
            rho: Rayleigh number (default: 28.0)
            beta: Geometric parameter (default: 8/3)
            dt: Time step size (default: 0.01)
            **params: Additional parameters (ignored)
        
        Returns:
            trajectory: Array of shape (length, 3) containing the trajectory
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if sigma <= 0:
            raise ValueError(f"Prandtl number sigma must be positive, got sigma={sigma}")
        
        if rho <= 0:
            raise ValueError(f"Rayleigh number rho must be positive, got rho={rho}")
        
        if beta <= 0:
            raise ValueError(f"Geometric parameter beta must be positive, got beta={beta}")
        
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got dt={dt}")
        
        if length <= 0:
            raise ValueError(f"Trajectory length must be positive, got length={length}")
        
        # Ensure initial condition is 3D
        if isinstance(initial_condition, np.ndarray):
            ic = initial_condition.flatten()[:3]
        else:
            ic = np.array(initial_condition).flatten()[:3]
        
        if len(ic) != 3:
            raise ValueError(f"Initial condition must be 3D, got shape {ic.shape}")
        
        # Time span
        t_span = (0, (length - 1) * dt)
        t_eval = np.linspace(0, (length - 1) * dt, length)
        
        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self._lorenz_ode(t, y, sigma, rho, beta),
            t_span=t_span,
            y0=ic,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        if not solution.success:
            raise RuntimeError(f"ODE integration failed: {solution.message}")
        
        # Transpose to get (length, 3) shape
        trajectory = solution.y.T
        
        return trajectory
    
    def create_operator_dataset(
        self,
        num_trajectories: int,
        trajectory_length: int,
        input_horizon: int,
        output_horizon: int,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
        initial_condition_range: Tuple[Tuple[float, float], ...] = (
            (-10.0, 10.0),
            (-10.0, 10.0),
            (0.0, 40.0)
        ),
        **params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output pairs for operator learning.
        
        Generates multiple trajectories with random initial conditions and
        creates sliding window input-output pairs for operator learning.
        
        Args:
            num_trajectories: Number of trajectories to generate
            trajectory_length: Length of each trajectory
            input_horizon: Number of time steps in input
            output_horizon: Number of time steps in output
            sigma: Prandtl number
            rho: Rayleigh number
            beta: Geometric parameter
            dt: Time step size
            initial_condition_range: Tuple of (min, max) ranges for each dimension
            **params: Additional parameters (ignored)
        
        Returns:
            inputs: Array of shape (num_samples, input_horizon, 3)
            outputs: Array of shape (num_samples, output_horizon, 3)
        """
        # Set seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Calculate number of samples per trajectory
        samples_per_trajectory = trajectory_length - input_horizon - output_horizon + 1
        if samples_per_trajectory <= 0:
            raise ValueError(
                f"trajectory_length ({trajectory_length}) must be >= "
                f"input_horizon ({input_horizon}) + output_horizon ({output_horizon})"
            )
        
        total_samples = num_trajectories * samples_per_trajectory
        
        # Preallocate arrays
        inputs = np.zeros((total_samples, input_horizon, 3))
        outputs = np.zeros((total_samples, output_horizon, 3))
        
        sample_idx = 0
        for _ in range(num_trajectories):
            # Random initial condition
            ic = np.array([
                np.random.uniform(initial_condition_range[0][0], initial_condition_range[0][1]),
                np.random.uniform(initial_condition_range[1][0], initial_condition_range[1][1]),
                np.random.uniform(initial_condition_range[2][0], initial_condition_range[2][1])
            ])
            
            # Generate trajectory
            trajectory = self.generate_trajectory(
                ic, trajectory_length, sigma=sigma, rho=rho, beta=beta, dt=dt
            )
            
            # Create sliding window samples
            for t in range(samples_per_trajectory):
                inputs[sample_idx] = trajectory[t:t + input_horizon]
                outputs[sample_idx] = trajectory[
                    t + input_horizon:t + input_horizon + output_horizon
                ]
                sample_idx += 1
        
        return inputs, outputs
    
    def get_true_eigenvalues(
        self,
        state: np.ndarray,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        **params
    ) -> Optional[np.ndarray]:
        """Return true eigenvalues at given state.
        
        Computes eigenvalues of the Jacobian matrix at the specified state.
        The Jacobian of the Lorenz system is:
            J = [[-σ,    σ,   0  ],
                 [ρ-z,  -1,  -x  ],
                 [y,     x,  -β  ]]
        
        Args:
            state: State point [x, y, z]
            sigma: Prandtl number
            rho: Rayleigh number
            beta: Geometric parameter
            **params: Additional parameters (ignored)
        
        Returns:
            eigenvalues: Array of complex eigenvalues
        """
        # Extract state components
        if isinstance(state, np.ndarray):
            state = state.flatten()[:3]
        else:
            state = np.array(state).flatten()[:3]
        
        if len(state) != 3:
            raise ValueError(f"State must be 3D, got shape {state.shape}")
        
        x, y, z = state
        
        # Construct Jacobian matrix
        jacobian = np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian)
        
        # Ensure complex dtype for consistency
        return eigenvalues.astype(np.complex128)
