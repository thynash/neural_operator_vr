"""Data normalization and management utilities for neural operator datasets."""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OperatorDataset(Dataset):
    """PyTorch Dataset wrapper for operator learning data.
    
    Wraps input-output pairs from dynamical systems into a PyTorch Dataset
    for use with DataLoader. Supports normalization and lazy loading.
    """
    
    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        normalize: bool = True,
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        output_mean: Optional[np.ndarray] = None,
        output_std: Optional[np.ndarray] = None
    ):
        """Initialize OperatorDataset.
        
        Args:
            inputs: Input data array of shape (num_samples, input_horizon, state_dim)
            outputs: Output data array of shape (num_samples, output_horizon, state_dim)
            normalize: Whether to normalize the data
            input_mean: Mean for input normalization (computed if None)
            input_std: Std for input normalization (computed if None)
            output_mean: Mean for output normalization (computed if None)
            output_std: Std for output normalization (computed if None)
        """
        self.inputs = inputs
        self.outputs = outputs
        self.normalize = normalize
        
        # Compute or use provided normalization statistics
        if normalize:
            if input_mean is None:
                self.input_mean = np.mean(inputs, axis=(0, 1))
            else:
                self.input_mean = input_mean
            
            if input_std is None:
                self.input_std = np.std(inputs, axis=(0, 1))
                # Avoid division by zero
                self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
            else:
                self.input_std = input_std
            
            if output_mean is None:
                self.output_mean = np.mean(outputs, axis=(0, 1))
            else:
                self.output_mean = output_mean
            
            if output_std is None:
                self.output_std = np.std(outputs, axis=(0, 1))
                # Avoid division by zero
                self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
            else:
                self.output_std = output_std
        else:
            self.input_mean = None
            self.input_std = None
            self.output_mean = None
            self.output_std = None
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            input_tensor: Normalized input tensor
            output_tensor: Normalized output tensor
        """
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]
        
        # Normalize if enabled
        if self.normalize:
            input_data = (input_data - self.input_mean) / self.input_std
            output_data = (output_data - self.output_mean) / self.output_std
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        return input_tensor, output_tensor
    
    def get_normalization_stats(self) -> Dict[str, Optional[np.ndarray]]:
        """Get normalization statistics.
        
        Returns:
            stats: Dictionary containing mean and std for inputs and outputs
        """
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }
    
    def denormalize_output(self, normalized_output: np.ndarray) -> np.ndarray:
        """Denormalize output data.
        
        Args:
            normalized_output: Normalized output array
        
        Returns:
            denormalized: Original scale output
        """
        if not self.normalize:
            return normalized_output
        
        return normalized_output * self.output_std + self.output_mean
    
    def denormalize_input(self, normalized_input: np.ndarray) -> np.ndarray:
        """Denormalize input data.
        
        Args:
            normalized_input: Normalized input array
        
        Returns:
            denormalized: Original scale input
        """
        if not self.normalize:
            return normalized_input
        
        return normalized_input * self.input_std + self.input_mean


class LazyOperatorDataset(Dataset):
    """Lazy-loading PyTorch Dataset for large operator learning datasets.
    
    Generates data on-the-fly instead of storing all trajectories in memory.
    Useful for very large datasets that don't fit in RAM.
    """
    
    def __init__(
        self,
        dynamical_system: Any,
        num_samples: int,
        input_horizon: int,
        output_horizon: int,
        system_params: Dict[str, Any],
        normalize: bool = True,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        seed: Optional[int] = None
    ):
        """Initialize LazyOperatorDataset.
        
        Args:
            dynamical_system: DynamicalSystem instance for generating data
            num_samples: Total number of samples to generate
            input_horizon: Number of time steps in input
            output_horizon: Number of time steps in output
            system_params: Parameters for the dynamical system
            normalize: Whether to normalize the data
            normalization_stats: Pre-computed normalization statistics
            seed: Random seed for reproducibility
        """
        self.dynamical_system = dynamical_system
        self.num_samples = num_samples
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon
        self.system_params = system_params
        self.normalize = normalize
        self.seed = seed
        
        # Store normalization statistics
        if normalize and normalization_stats is not None:
            self.input_mean = normalization_stats.get('input_mean')
            self.input_std = normalization_stats.get('input_std')
            self.output_mean = normalization_stats.get('output_mean')
            self.output_std = normalization_stats.get('output_std')
        else:
            self.input_mean = None
            self.input_std = None
            self.output_mean = None
            self.output_std = None
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate and return a single sample on-the-fly.
        
        Args:
            idx: Sample index (used as seed modifier)
        
        Returns:
            input_tensor: Input tensor
            output_tensor: Output tensor
        """
        # Set seed based on index for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        
        # Generate a short trajectory
        trajectory_length = self.input_horizon + self.output_horizon
        
        # This is a simplified version - in practice, you'd need to
        # generate initial conditions and call the dynamical system
        # For now, this is a placeholder that would need system-specific logic
        raise NotImplementedError(
            "LazyOperatorDataset requires system-specific implementation"
        )


def create_train_val_split(
    inputs: np.ndarray,
    outputs: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and validation sets.
    
    Args:
        inputs: Input data array
        outputs: Output data array
        train_ratio: Fraction of data to use for training (default: 0.8)
        shuffle: Whether to shuffle before splitting (default: True)
        seed: Random seed for shuffling
    
    Returns:
        train_inputs: Training input data
        train_outputs: Training output data
        val_inputs: Validation input data
        val_outputs: Validation output data
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    
    num_samples = len(inputs)
    train_size = int(train_ratio * num_samples)
    
    # Create indices
    indices = np.arange(num_samples)
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Split data
    train_inputs = inputs[train_indices]
    train_outputs = outputs[train_indices]
    val_inputs = inputs[val_indices]
    val_outputs = outputs[val_indices]
    
    return train_inputs, train_outputs, val_inputs, val_outputs


def create_dataloaders(
    train_inputs: np.ndarray,
    train_outputs: np.ndarray,
    val_inputs: np.ndarray,
    val_outputs: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_inputs: Training input data
        train_outputs: Training output data
        val_inputs: Validation input data
        val_outputs: Validation output data
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize the data
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        normalization_stats: Dictionary of normalization statistics
    """
    # Create training dataset (computes normalization statistics)
    train_dataset = OperatorDataset(
        train_inputs,
        train_outputs,
        normalize=normalize
    )
    
    # Get normalization statistics from training data
    normalization_stats = train_dataset.get_normalization_stats()
    
    # Create validation dataset (uses training statistics)
    val_dataset = OperatorDataset(
        val_inputs,
        val_outputs,
        normalize=normalize,
        input_mean=normalization_stats['input_mean'],
        input_std=normalization_stats['input_std'],
        output_mean=normalization_stats['output_mean'],
        output_std=normalization_stats['output_std']
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, normalization_stats


def save_dataset(
    filepath: str,
    inputs: np.ndarray,
    outputs: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save dataset to disk in NPZ format.
    
    Args:
        filepath: Path to save the dataset (should end in .npz)
        inputs: Input data array
        outputs: Output data array
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'inputs': inputs,
        'outputs': outputs
    }
    
    if metadata is not None:
        # Store metadata as a single pickled object
        save_dict['metadata'] = np.array([metadata], dtype=object)
    
    np.savez_compressed(filepath, **save_dict)


def load_dataset(
    filepath: str
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Load dataset from disk.
    
    Args:
        filepath: Path to the dataset file (.npz)
    
    Returns:
        inputs: Input data array
        outputs: Output data array
        metadata: Metadata dictionary (if available)
    """
    data = np.load(filepath, allow_pickle=True)
    
    inputs = data['inputs']
    outputs = data['outputs']
    
    metadata = None
    if 'metadata' in data:
        metadata = data['metadata'].item()
    
    return inputs, outputs, metadata
