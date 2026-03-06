"""Random seed management for reproducibility."""

import random
import numpy as np
import torch


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    
    Sets random seeds for Python's random module, NumPy, PyTorch CPU and CUDA.
    Optionally enables deterministic algorithms for full reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value to set across all libraries.
    deterministic : bool, optional
        If True, enables PyTorch deterministic algorithms and disables CUDA
        benchmarking for full reproducibility. May impact performance.
        Default is True.
    
    Notes
    -----
    When deterministic=True, some PyTorch operations may be slower and some
    operations may not be available. This ensures bit-exact reproducibility
    across runs with the same seed.
    
    Examples
    --------
    >>> set_random_seeds(42, deterministic=True)
    >>> # All random operations will now be reproducible
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Enable deterministic algorithms if requested
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Disable CUDA benchmarking for deterministic behavior
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
