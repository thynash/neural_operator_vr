"""Device management utilities for CPU/GPU computation."""

from typing import Union, Any, List, Dict
import torch
import torch.nn as nn


def get_device(device_name: str = "auto") -> torch.device:
    """
    Get PyTorch device for computation.
    
    Automatically detects CUDA availability when device_name is "auto",
    or returns the specified device.
    
    Parameters
    ----------
    device_name : str, optional
        Device specification. Options:
        - "auto": Automatically select CUDA if available, otherwise CPU
        - "cuda": Use CUDA (raises error if not available)
        - "cpu": Use CPU
        - "cuda:0", "cuda:1", etc.: Use specific GPU device
        Default is "auto".
    
    Returns
    -------
    torch.device
        PyTorch device object for computation.
    
    Raises
    ------
    ValueError
        If device_name is invalid or CUDA is requested but not available.
    
    Examples
    --------
    >>> device = get_device("auto")
    >>> device = get_device("cuda:0")
    >>> device = get_device("cpu")
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        return torch.device("cpu")
    elif device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device '{device_name}' requested but CUDA is not available. "
                "Please check your PyTorch installation and GPU drivers."
            )
        return torch.device(device_name)
    else:
        raise ValueError(
            f"Invalid device name '{device_name}'. "
            "Valid options: 'auto', 'cpu', 'cuda', 'cuda:0', etc."
        )


def move_to_device(
    obj: Union[torch.Tensor, nn.Module, List, Dict, Any],
    device: torch.device
) -> Union[torch.Tensor, nn.Module, List, Dict, Any]:
    """
    Move tensor, model, or collection to specified device.
    
    Handles tensors, nn.Module objects, lists, dictionaries, and nested
    collections. Non-movable objects are returned unchanged.
    
    Parameters
    ----------
    obj : torch.Tensor, nn.Module, list, dict, or other
        Object to move to device. Can be:
        - torch.Tensor: Moved to device
        - nn.Module: Moved to device
        - list: Each element recursively moved
        - dict: Each value recursively moved
        - other: Returned unchanged
    device : torch.device
        Target device for computation.
    
    Returns
    -------
    Same type as input
        Object moved to the specified device.
    
    Examples
    --------
    >>> device = get_device("cuda")
    >>> tensor = torch.randn(3, 3)
    >>> tensor_gpu = move_to_device(tensor, device)
    >>> 
    >>> model = nn.Linear(10, 5)
    >>> model_gpu = move_to_device(model, device)
    >>> 
    >>> batch = {"input": torch.randn(2, 10), "target": torch.randn(2, 5)}
    >>> batch_gpu = move_to_device(batch, device)
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        # Return unchanged for non-movable objects (int, float, str, etc.)
        return obj
