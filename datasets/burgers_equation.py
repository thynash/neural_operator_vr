"""Burgers Equation dynamical system dataset."""

from typing import Tuple, Optional
import numpy as np
from .base import DynamicalSystem


class BurgersEquationDataset(DynamicalSystem):
    """Burgers Equation spatiotemporal dynamical system.
    
    The 1D Burgers equation is defined by:
        ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
    
    where u(x,t) is the velocity field, ν is the viscosity coefficient,
    and the equation is solved on a periodic domain [0, 2π].
    
    This is a fundamental PDE in fluid dynamics that exhibits both
    convection (nonlinear term) and diffusion (viscous term).
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize BurgersEquationDataset.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def _compute_spatial_derivative(
        self,
        u: np.ndarray,
        dx: float,
        order: int = 1
    ) -> np.ndarray:
        """Compute spatial derivative using finite differences.
        
        Uses central differences with periodic boundary conditions.
        
        Args:
            u: Field values at spatial points
            dx: Spatial grid spacing
            order: Derivative order (1 or 2)
        
        Returns:
            derivative: Spatial derivative of u
        """
        if order == 1:
            # Central difference: du/dx ≈ (u[i+1] - u[i-1]) / (2*dx)
            du = np.roll(u, -1) - np.roll(u, 1)
            return du / (2 * dx)
        elif order == 2:
            # Central difference: d²u/dx² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²
            d2u = np.roll(u, -1) - 2 * u + np.roll(u, 1)
            return d2u / (dx ** 2)
        else:
            raise ValueError(f"Unsupported derivative order: {order}")
    
    def _burgers_rhs(
        self,
        u: np.ndarray,
        dx: float,
        viscosity: float
    ) -> np.ndarray:
        """Compute right-hand side of Burgers equation.
        
        RHS = -u * ∂u/∂x + ν * ∂²u/∂x²
        
        Args:
            u: Current velocity field
            dx: Spatial grid spacing
            viscosity: Viscosity coefficient ν
        
        Returns:
            rhs: Time derivative ∂u/∂t
        """
        # Convection term: -u * ∂u/∂x
        du_dx = self._compute_spatial_derivative(u, dx, order=1)
        convection = -u * du_dx
        
        # Diffusion term: ν * ∂²u/∂x²
        d2u_dx2 = self._compute_spatial_derivative(u, dx, order=2)
        diffusion = viscosity * d2u_dx2
        
        return convection + diffusion
    
    def generate_trajectory(
        self,
        initial_condition: np.ndarray,
        length: int,
        viscosity: float = 0.01,
        spatial_resolution: int = 256,
        dt: float = 0.001,
        **params
    ) -> np.ndarray:
        """Generate a single trajectory from initial condition.
        
        Uses finite difference method with forward Euler time stepping
        and periodic boundary conditions.
        
        Args:
            initial_condition: Initial velocity field (1D array)
            length: Number of time steps to generate
            viscosity: Viscosity coefficient ν (default: 0.01)
            spatial_resolution: Number of spatial grid points (default: 256)
            dt: Time step size (default: 0.001)
            **params: Additional parameters (ignored)
        
        Returns:
            trajectory: Array of shape (length, spatial_resolution)
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if viscosity < 0:
            raise ValueError(
                f"Viscosity must be non-negative, got viscosity={viscosity}"
            )
        
        if spatial_resolution <= 0:
            raise ValueError(
                f"Spatial resolution must be positive, got spatial_resolution={spatial_resolution}"
            )
        
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got dt={dt}")
        
        if length <= 0:
            raise ValueError(f"Trajectory length must be positive, got length={length}")
        
        # Ensure initial condition has correct shape
        if isinstance(initial_condition, np.ndarray):
            u = initial_condition.flatten()
        else:
            u = np.array(initial_condition).flatten()
        
        # If initial condition doesn't match spatial resolution, interpolate or pad
        if len(u) != spatial_resolution:
            if len(u) < spatial_resolution:
                # Pad with zeros
                u_new = np.zeros(spatial_resolution)
                u_new[:len(u)] = u
                u = u_new
            else:
                # Truncate
                u = u[:spatial_resolution]
        
        # Spatial grid
        L = 2 * np.pi  # Domain length
        dx = L / spatial_resolution
        
        # Check CFL condition for stability
        cfl = np.max(np.abs(u)) * dt / dx
        diffusion_number = viscosity * dt / (dx ** 2)
        if cfl > 1.0 or diffusion_number > 0.5:
            import warnings
            warnings.warn(
                f"Stability conditions may be violated: CFL={cfl:.3f}, "
                f"diffusion_number={diffusion_number:.3f}. "
                f"Consider reducing dt or increasing spatial_resolution."
            )
        
        # Preallocate trajectory
        trajectory = np.zeros((length, spatial_resolution))
        trajectory[0] = u
        
        # Time integration using forward Euler
        for i in range(1, length):
            rhs = self._burgers_rhs(u, dx, viscosity)
            u = u + dt * rhs
            trajectory[i] = u
        
        return trajectory
    
    def create_operator_dataset(
        self,
        num_trajectories: int,
        trajectory_length: int,
        input_horizon: int,
        output_horizon: int,
        viscosity: float = 0.01,
        spatial_resolution: int = 256,
        dt: float = 0.001,
        initial_condition_type: str = "random_fourier",
        num_modes: int = 5,
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
            viscosity: Viscosity coefficient ν
            spatial_resolution: Number of spatial grid points
            dt: Time step size
            initial_condition_type: Type of initial condition ("random_fourier", "gaussian_bump")
            num_modes: Number of Fourier modes for random initial conditions
            **params: Additional parameters (ignored)
        
        Returns:
            inputs: Array of shape (num_samples, input_horizon, spatial_resolution)
            outputs: Array of shape (num_samples, output_horizon, spatial_resolution)
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
        inputs = np.zeros((total_samples, input_horizon, spatial_resolution))
        outputs = np.zeros((total_samples, output_horizon, spatial_resolution))
        
        # Spatial grid
        x = np.linspace(0, 2 * np.pi, spatial_resolution, endpoint=False)
        
        sample_idx = 0
        for _ in range(num_trajectories):
            # Generate random initial condition
            if initial_condition_type == "random_fourier":
                # Random Fourier series
                ic = np.zeros(spatial_resolution)
                for k in range(1, num_modes + 1):
                    amplitude = np.random.uniform(-1, 1)
                    phase = np.random.uniform(0, 2 * np.pi)
                    ic += amplitude * np.sin(k * x + phase)
            elif initial_condition_type == "gaussian_bump":
                # Random Gaussian bump
                center = np.random.uniform(0, 2 * np.pi)
                width = np.random.uniform(0.5, 1.5)
                amplitude = np.random.uniform(0.5, 2.0)
                ic = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            else:
                raise ValueError(f"Unknown initial condition type: {initial_condition_type}")
            
            # Generate trajectory
            trajectory = self.generate_trajectory(
                ic, trajectory_length,
                viscosity=viscosity,
                spatial_resolution=spatial_resolution,
                dt=dt
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
        viscosity: float = 0.01,
        spatial_resolution: int = 256,
        num_eigenvalues: int = 10,
        **params
    ) -> Optional[np.ndarray]:
        """Return true eigenvalues using spectral method.
        
        Computes eigenvalues of the linearized Burgers operator around
        the given state using Fourier spectral method.
        
        The linearized operator is: L[v] = -u * ∂v/∂x - v * ∂u/∂x + ν * ∂²v/∂x²
        
        Args:
            state: Velocity field at which to linearize
            viscosity: Viscosity coefficient ν
            spatial_resolution: Number of spatial grid points
            num_eigenvalues: Number of eigenvalues to compute
            **params: Additional parameters (ignored)
        
        Returns:
            eigenvalues: Array of complex eigenvalues (largest magnitude)
        """
        # Ensure state has correct shape
        if isinstance(state, np.ndarray):
            u = state.flatten()
        else:
            u = np.array(state).flatten()
        
        if len(u) != spatial_resolution:
            raise ValueError(
                f"State length ({len(u)}) must match spatial_resolution ({spatial_resolution})"
            )
        
        # Spatial grid
        L = 2 * np.pi
        dx = L / spatial_resolution
        
        # Construct linearized operator matrix using finite differences
        # This is a sparse matrix, but we'll use dense for simplicity
        operator = np.zeros((spatial_resolution, spatial_resolution))
        
        for i in range(spatial_resolution):
            # Indices with periodic boundary conditions
            i_plus = (i + 1) % spatial_resolution
            i_minus = (i - 1) % spatial_resolution
            
            # Convection terms: -u[i] * ∂v/∂x - v * ∂u/∂x
            # ∂v/∂x at i: (v[i+1] - v[i-1]) / (2*dx)
            operator[i, i_plus] += -u[i] / (2 * dx)
            operator[i, i_minus] += u[i] / (2 * dx)
            
            # -v * ∂u/∂x at i
            du_dx_i = (u[i_plus] - u[i_minus]) / (2 * dx)
            operator[i, i] += -du_dx_i
            
            # Diffusion term: ν * ∂²v/∂x²
            # ∂²v/∂x² at i: (v[i+1] - 2*v[i] + v[i-1]) / dx²
            operator[i, i_plus] += viscosity / (dx ** 2)
            operator[i, i] += -2 * viscosity / (dx ** 2)
            operator[i, i_minus] += viscosity / (dx ** 2)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(operator)
        
        # Sort by magnitude and return largest
        eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
        
        # Ensure complex dtype for consistency
        return eigenvalues[:num_eigenvalues].astype(np.complex128)
