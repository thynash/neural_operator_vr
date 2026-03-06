"""Abstract base class for dynamical system data generators."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class DynamicalSystem(ABC):
    """Abstract base for dynamical system data generators.
    
    This class defines the interface for generating training data from
    dynamical systems for neural operator learning. Concrete implementations
    should provide specific system dynamics (Logistic Map, Lorenz System, etc.).
    """
    
    @abstractmethod
    def generate_trajectory(
        self,
        initial_condition: np.ndarray,
        length: int,
        **params: Any
    ) -> np.ndarray:
        """Generate a single trajectory from initial condition.
        
        Args:
            initial_condition: Starting state of the system
            length: Number of time steps to generate
            **params: System-specific parameters (e.g., r for logistic map)
        
        Returns:
            trajectory: Array of shape (length, state_dim) containing the trajectory
        """
        pass
    
    @abstractmethod
    def create_operator_dataset(
        self,
        num_trajectories: int,
        trajectory_length: int,
        input_horizon: int,
        output_horizon: int,
        **params: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output pairs for operator learning.
        
        Generates multiple trajectories and creates input-output pairs suitable
        for training neural operators. Input consists of the system state over
        input_horizon time steps, and output is the state over output_horizon
        future time steps.
        
        Args:
            num_trajectories: Number of trajectories to generate
            trajectory_length: Length of each trajectory
            input_horizon: Number of time steps in input
            output_horizon: Number of time steps in output
            **params: System-specific parameters
        
        Returns:
            inputs: Array of shape (num_samples, input_horizon, state_dim)
            outputs: Array of shape (num_samples, output_horizon, state_dim)
        """
        pass
    
    @abstractmethod
    def get_true_eigenvalues(
        self,
        state: np.ndarray,
        **params: Any
    ) -> Optional[np.ndarray]:
        """Return true eigenvalues at given state (if analytically known).
        
        Computes the eigenvalues of the system's Jacobian at the specified state.
        This is used for spectral analysis and validation of learned operators.
        
        Args:
            state: State point at which to compute eigenvalues
            **params: System-specific parameters
        
        Returns:
            eigenvalues: Array of complex eigenvalues, or None if not analytically available
        """
        pass
