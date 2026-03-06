"""Spectral analysis for neural operators."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


def compute_operator_eigenvalues(
    model: nn.Module,
    state_point: torch.Tensor,
    num_eigenvalues: int = 10,
    method: str = 'auto'
) -> torch.Tensor:
    """
    Compute eigenvalues of the learned operator Jacobian at a state point.
    
    Uses automatic differentiation to construct the Jacobian matrix,
    then computes eigenvalues using torch.linalg.eig or power iteration.
    
    Args:
        model: Neural operator model
        state_point: State at which to compute eigenvalues [state_dim] or [batch, state_dim]
        num_eigenvalues: Number of eigenvalues to compute (for power iteration)
        method: 'auto', 'eig', or 'power_iteration'
    
    Returns:
        Eigenvalues as complex tensor [num_eigenvalues] or [state_dim]
    
    Validates: Requirements 7.1
    """
    model.eval()
    
    # Ensure state_point has batch dimension
    if state_point.dim() == 1:
        state_point = state_point.unsqueeze(0)
    elif state_point.dim() == 3:
        # Handle time series data [batch, time, state_dim]
        # Use the last time step
        state_point = state_point[:, -1, :]
    
    batch_size, state_dim = state_point.shape
    
    # Ensure requires_grad is True
    state_point = state_point.detach().requires_grad_(True)
    
    # Prepare inputs for neural operator
    # state_point: [batch, state_dim]
    # Neural operators expect: input_functions [batch, state_dim, num_sensors]
    #                          query_points [batch, state_dim, num_queries]
    # For Jacobian computation, we use state as both input and query
    input_functions = state_point.unsqueeze(2)  # [batch, state_dim, 1]
    query_points = state_point.unsqueeze(2)  # [batch, state_dim, 1]
    
    # Compute Jacobian using automatic differentiation
    jacobian = torch.zeros(state_dim, state_dim, device=state_point.device)
    
    for i in range(state_dim):
        # Zero gradients
        if state_point.grad is not None:
            state_point.grad.zero_()
        
        # Forward pass
        output = model(input_functions, query_points)  # [batch, state_dim, 1]
        
        # Backward pass for i-th output component
        # output shape: [batch, state_dim, 1]
        output_i = output[0, i, 0]
        
        output_i.backward(retain_graph=True)
        
        # Store gradient (i-th row of Jacobian)
        if state_point.grad is not None:
            jacobian[i, :] = state_point.grad[0, :].detach()
    
    # Compute eigenvalues
    if method == 'auto':
        # Use eig for small matrices, power iteration for large
        if state_dim <= 100:
            method = 'eig'
        else:
            method = 'power_iteration'
    
    if method == 'eig':
        # Use torch.linalg.eig for full eigenvalue decomposition
        eigenvalues, _ = torch.linalg.eig(jacobian)
    elif method == 'power_iteration':
        # Use power iteration for dominant eigenvalues
        eigenvalues = _power_iteration_eigenvalues(jacobian, num_eigenvalues)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'eig', or 'power_iteration'")
    
    model.train()
    return eigenvalues


def _power_iteration_eigenvalues(
    matrix: torch.Tensor,
    num_eigenvalues: int,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> torch.Tensor:
    """
    Compute dominant eigenvalues using power iteration.
    
    Args:
        matrix: Square matrix [n, n]
        num_eigenvalues: Number of eigenvalues to compute
        max_iterations: Maximum iterations per eigenvalue
        tolerance: Convergence tolerance
    
    Returns:
        Dominant eigenvalues [num_eigenvalues]
    """
    n = matrix.shape[0]
    eigenvalues = []
    
    # Deflation: compute eigenvalues one by one
    A = matrix.clone()
    
    for k in range(min(num_eigenvalues, n)):
        # Random initialization
        v = torch.randn(n, device=matrix.device)
        v = v / torch.norm(v)
        
        lambda_old = 0.0
        
        for iteration in range(max_iterations):
            # Power iteration step
            v_new = torch.matmul(A, v)
            
            # Rayleigh quotient (eigenvalue estimate)
            lambda_new = torch.dot(v, v_new).item()
            
            # Normalize
            v_new_norm = torch.norm(v_new)
            if v_new_norm > 1e-10:
                v = v_new / v_new_norm
            else:
                # Matrix is nearly singular, stop
                break
            
            # Check convergence
            if abs(lambda_new - lambda_old) < tolerance:
                break
            
            lambda_old = lambda_new
        
        eigenvalues.append(lambda_new)
        
        # Deflation: remove the found eigenvector's contribution
        # A_new = A - λ * v * v^T
        if k < num_eigenvalues - 1:
            A = A - lambda_new * torch.outer(v, v)
    
    # Convert to complex tensor (power iteration gives real eigenvalues)
    eigenvalues_tensor = torch.tensor(eigenvalues, dtype=torch.complex64, device=matrix.device)
    
    return eigenvalues_tensor


def compute_spectral_radius(eigenvalues: torch.Tensor) -> float:
    """
    Compute spectral radius as the maximum absolute eigenvalue.
    
    Args:
        eigenvalues: Complex eigenvalues tensor
    
    Returns:
        Spectral radius (maximum |λ_i|)
    
    Validates: Requirements 7.2
    """
    # Compute absolute values of complex eigenvalues
    abs_eigenvalues = torch.abs(eigenvalues)
    
    # Return maximum
    spectral_radius = torch.max(abs_eigenvalues).item()
    
    return spectral_radius


def compute_eigenvalue_error(
    learned_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor,
    method: str = 'hungarian'
) -> float:
    """
    Compute eigenvalue approximation error.
    
    Matches learned eigenvalues to true eigenvalues and computes error.
    Uses Hungarian algorithm for optimal matching.
    
    Args:
        learned_eigenvalues: Learned eigenvalues [n]
        true_eigenvalues: True eigenvalues [m]
        method: 'hungarian' for optimal matching, 'nearest' for greedy matching
    
    Returns:
        Eigenvalue approximation error
    
    Validates: Requirements 7.3
    """
    # Convert to numpy for easier manipulation
    learned = learned_eigenvalues.detach().cpu().numpy()
    true = true_eigenvalues.detach().cpu().numpy()
    
    n_learned = len(learned)
    n_true = len(true)
    
    if method == 'hungarian':
        # Use Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        
        # Compute cost matrix (distances between eigenvalues)
        cost_matrix = np.zeros((n_learned, n_true))
        for i in range(n_learned):
            for j in range(n_true):
                cost_matrix[i, j] = np.abs(learned[i] - true[j])
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute total error
        total_error = cost_matrix[row_ind, col_ind].sum()
        
        # Normalize by number of matched pairs
        error = total_error / len(row_ind)
        
    elif method == 'nearest':
        # Greedy nearest neighbor matching
        total_error = 0.0
        used_true = set()
        
        for learned_eig in learned:
            # Find nearest unused true eigenvalue
            min_dist = float('inf')
            best_idx = -1
            
            for j, true_eig in enumerate(true):
                if j not in used_true:
                    dist = np.abs(learned_eig - true_eig)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = j
            
            if best_idx >= 0:
                total_error += min_dist
                used_true.add(best_idx)
        
        # Normalize by number of learned eigenvalues
        error = total_error / n_learned
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hungarian' or 'nearest'")
    
    return float(error)


def track_eigenvalue_evolution(
    model: nn.Module,
    state_point: torch.Tensor,
    training_history: Dict[str, List[Tuple[int, float]]],
    iteration: int,
    num_eigenvalues: int = 10
) -> Dict[str, List]:
    """
    Track eigenvalue evolution during training.
    
    Computes eigenvalues and spectral radius at current iteration
    and appends to training history.
    
    Args:
        model: Neural operator model
        state_point: State at which to compute eigenvalues
        training_history: Training history dictionary to update
        iteration: Current training iteration
        num_eigenvalues: Number of eigenvalues to compute
    
    Returns:
        Updated training history
    
    Validates: Requirements 7.4
    """
    # Compute eigenvalues
    eigenvalues = compute_operator_eigenvalues(
        model, state_point, num_eigenvalues=num_eigenvalues
    )
    
    # Compute spectral radius
    spectral_radius = compute_spectral_radius(eigenvalues)
    
    # Initialize history lists if not present
    if 'spectral_radius' not in training_history:
        training_history['spectral_radius'] = []
    if 'eigenvalues' not in training_history:
        training_history['eigenvalues'] = []
    
    # Append to history
    training_history['spectral_radius'].append((iteration, spectral_radius))
    
    # Store eigenvalues as list of (real, imag) pairs
    eigenvalues_list = [
        (eig.real.item(), eig.imag.item()) for eig in eigenvalues
    ]
    training_history['eigenvalues'].append((iteration, eigenvalues_list))
    
    return training_history


def save_eigenvalue_data(
    training_history: Dict[str, List],
    filepath: str
) -> None:
    """
    Save eigenvalue data for post-training analysis.
    
    Args:
        training_history: Training history containing eigenvalue data
        filepath: Path to save eigenvalue data (JSON format)
    
    Validates: Requirements 7.5
    """
    import json
    
    # Extract eigenvalue data
    eigenvalue_data = {
        'spectral_radius': training_history.get('spectral_radius', []),
        'eigenvalues': training_history.get('eigenvalues', []),
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(eigenvalue_data, f, indent=2)



class SpectralAnalyzer:
    """Wrapper class for spectral analysis functions.
    
    This class provides a convenient interface to spectral analysis functions
    for use in the experiment runner.
    """
    
    def __init__(self, model: nn.Module, dataset, device: torch.device):
        """Initialize spectral analyzer.
        
        Args:
            model: Neural operator model
            dataset: Dataset (not used but kept for compatibility)
            device: Device for computation
        """
        self.model = model
        self.dataset = dataset
        self.device = device
    
    def compute_operator_eigenvalues(
        self,
        state_point: torch.Tensor,
        num_eigenvalues: int = 10,
        method: str = 'auto'
    ) -> torch.Tensor:
        """Compute eigenvalues of operator Jacobian at state point."""
        return compute_operator_eigenvalues(
            self.model,
            state_point.to(self.device),
            num_eigenvalues,
            method
        )
    
    def compute_spectral_radius(self, eigenvalues: torch.Tensor) -> float:
        """Compute spectral radius from eigenvalues."""
        return compute_spectral_radius(eigenvalues)
    
    def compute_eigenvalue_error(
        self,
        learned_eigenvalues: torch.Tensor,
        true_eigenvalues: torch.Tensor
    ) -> float:
        """Compute eigenvalue approximation error."""
        return compute_eigenvalue_error(learned_eigenvalues, true_eigenvalues)
