"""System information logging for reproducibility."""

import sys
import platform
from datetime import datetime
from typing import Dict, Any
import torch


def log_system_info(logger: Any = None) -> Dict[str, Any]:
    """
    Capture and log system and environment information.
    
    Collects comprehensive system information including Python version,
    PyTorch version, CUDA availability and version, GPU details, CPU info,
    operating system, and timestamp. This information is crucial for
    reproducibility and debugging.
    
    Parameters
    ----------
    logger : MetricsLogger or similar, optional
        Logger object with a log_dict method. If provided, system info
        will be logged. If None, only returns the info dictionary.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing system information with keys:
        - python_version: Python version string
        - pytorch_version: PyTorch version string
        - cuda_available: Whether CUDA is available
        - cuda_version: CUDA version string (if available)
        - cudnn_version: cuDNN version (if available)
        - num_gpus: Number of available GPUs
        - gpu_names: List of GPU device names
        - gpu_memory: List of GPU memory capacities in GB
        - cpu_info: CPU model information
        - num_cpu_cores: Number of CPU cores
        - os: Operating system name and version
        - timestamp: ISO format timestamp
    
    Examples
    --------
    >>> from utils.logger import MetricsLogger
    >>> logger = MetricsLogger("./logs", "experiment_1")
    >>> system_info = log_system_info(logger)
    >>> print(system_info["python_version"])
    >>> print(system_info["cuda_available"])
    """
    system_info = {}
    
    # Python version
    system_info["python_version"] = sys.version.split()[0]
    
    # PyTorch version
    system_info["pytorch_version"] = torch.__version__
    
    # CUDA information
    system_info["cuda_available"] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        system_info["cuda_version"] = torch.version.cuda
        system_info["cudnn_version"] = torch.backends.cudnn.version()
        system_info["num_gpus"] = torch.cuda.device_count()
        
        # GPU names and memory
        gpu_names = []
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(i))
            # Get total memory in GB
            total_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_memory.append(round(total_memory / (1024**3), 2))
        
        system_info["gpu_names"] = gpu_names
        system_info["gpu_memory_gb"] = gpu_memory
    else:
        system_info["cuda_version"] = None
        system_info["cudnn_version"] = None
        system_info["num_gpus"] = 0
        system_info["gpu_names"] = []
        system_info["gpu_memory_gb"] = []
    
    # CPU information
    system_info["cpu_info"] = platform.processor() or platform.machine()
    system_info["num_cpu_cores"] = torch.get_num_threads()
    
    # Operating system
    system_info["os"] = f"{platform.system()} {platform.release()}"
    system_info["platform"] = platform.platform()
    
    # Timestamp
    system_info["timestamp"] = datetime.now().isoformat()
    
    # Log to logger if provided
    if logger is not None and hasattr(logger, 'log_dict'):
        # Convert to flat structure for logging
        flat_info = {
            f"system_{key}": value 
            for key, value in system_info.items()
            if not isinstance(value, (list, dict))
        }
        logger.log_dict(flat_info, step=0)
    
    return system_info


def format_system_info(system_info: Dict[str, Any]) -> str:
    """
    Format system information as a human-readable string.
    
    Parameters
    ----------
    system_info : Dict[str, Any]
        System information dictionary from log_system_info().
    
    Returns
    -------
    str
        Formatted multi-line string with system information.
    
    Examples
    --------
    >>> system_info = log_system_info()
    >>> print(format_system_info(system_info))
    """
    lines = [
        "=" * 60,
        "System Information",
        "=" * 60,
        f"Python Version: {system_info['python_version']}",
        f"PyTorch Version: {system_info['pytorch_version']}",
        f"Operating System: {system_info['os']}",
        f"CPU: {system_info['cpu_info']}",
        f"CPU Cores: {system_info['num_cpu_cores']}",
        "",
        f"CUDA Available: {system_info['cuda_available']}",
    ]
    
    if system_info['cuda_available']:
        lines.extend([
            f"CUDA Version: {system_info['cuda_version']}",
            f"cuDNN Version: {system_info['cudnn_version']}",
            f"Number of GPUs: {system_info['num_gpus']}",
        ])
        
        for i, (name, memory) in enumerate(zip(
            system_info['gpu_names'], 
            system_info['gpu_memory_gb']
        )):
            lines.append(f"  GPU {i}: {name} ({memory} GB)")
    
    lines.extend([
        "",
        f"Timestamp: {system_info['timestamp']}",
        "=" * 60,
    ])
    
    return "\n".join(lines)
