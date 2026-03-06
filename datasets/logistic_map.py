"""Logistic Map dynamical system dataset."""

from typing import Tuple, Optional
import numpy as np
from .base import DynamicalSystem


class LogisticMapDataset(DynamicalSystem):
    """Logistic Map discrete-time dynamical system.
    
    The logistic map is defined by: x_{n+1} = r * x_n * (1 - x_n)
    where r is the growth rate parameter and x is in [0, 1].
    
    This is a classic example of a chaotic dynamical system that exhibits
    complex behavior including period-doubling bifurcations and chaos.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize LogisticMapDataset.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_trajectory(
        self,
        initial_condition: np.ndarray,
        length: int,
        r: float = 3.9,
        **params
    ) -> np.ndarray:
        """Generate a single trajectory from initial condition.
        
        Args:
            initial_condition: Starting state (scalar or 1D array)
            length: Number of time steps to generate
            r: Growth rate parameter (typically in [0, 4])
            **params: Additional parameters (ignored)
        
        Returns:
            trajectory: Array of shape (length, 1) containing the trajectory
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if r < 0 or r > 4:
            raise ValueError(
                f"Growth rate parameter r must be in [0, 4], got r={r}. "
                f"Values outside this range lead to unbounded dynamics."
            )
        
        if length <= 0:
            raise ValueError(f"Trajectory length must be positive, got length={length}")
        
        # Ensure initial condition is a scalar
        if isinstance(initial_condition, np.ndarray):
            x = float(initial_condition.flatten()[0])
        else:
            x = float(initial_condition)
        
        if x < 0 or x > 1:
            raise ValueError(
                f"Initial condition must be in [0, 1] for logistic map, got x={x}"
            )
        
        # Generate trajectory
        trajectory = np.zeros((length, 1))
        trajectory[0, 0] = x
        
        for i in range(1, length):
            x = r * x * (1 - x)
            trajectory[i, 0] = x
        
        return trajectory
    
    def create_operator_dataset(
        self,
        num_trajectories: int,
        trajectory_length: int,
        input_horizon: int,
        output_horizon: int,
        r: float = 3.9,
        initial_condition_range: Tuple[float, float] = (0.1, 0.9),
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
            r: Growth rate parameter
            initial_condition_range: Range for random initial conditions
            **params: Additional parameters (ignored)
        
        Returns:
            inputs: Array of shape (num_samples, input_horizon, 1)
            outputs: Array of shape (num_samples, output_horizon, 1)
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
        inputs = np.zeros((total_samples, input_horizon, 1))
        outputs = np.zeros((total_samples, output_horizon, 1))
        
        sample_idx = 0
        for _ in range(num_trajectories):
            # Random initial condition
            ic = np.random.uniform(
                initial_condition_range[0],
                initial_condition_range[1]
            )
            
            # Generate trajectory
            trajectory = self.generate_trajectory(ic, trajectory_length, r=r)
            
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
        r: float = 3.9,
        **params
    ) -> Optional[np.ndarray]:
        """Return true eigenvalues at given state.
        
        For the logistic map, the eigenvalue at a state x is the derivative:
        λ = r * (1 - 2*x)
        
        Args:
            state: State point (scalar or 1D array)
            r: Growth rate parameter
            **params: Additional parameters (ignored)
        
        Returns:
            eigenvalues: Array containing single eigenvalue
        """
        # Extract scalar state
        if isinstance(state, np.ndarray):
            x = float(state.flatten()[0])
        else:
            x = float(state)
        
        # Compute derivative (eigenvalue for 1D system)
        eigenvalue = r * (1 - 2 * x)
        
        return np.array([eigenvalue], dtype=np.complex128)
