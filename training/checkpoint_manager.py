"""Checkpoint management for saving and loading training state."""

import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import numpy as np

from models.base import NeuralOperator
from optimizers.base import OptimizerWithVariance
from utils.exceptions import CheckpointCompatibilityError


class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.
    
    Handles complete training state persistence including model weights,
    optimizer state, random number generator states, training history,
    and configuration for exact reproducibility.
    
    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory where checkpoints will be saved.
    
    Attributes
    ----------
    checkpoint_dir : Path
        Directory for checkpoint storage.
    
    Examples
    --------
    >>> manager = CheckpointManager("./checkpoints")
    >>> manager.save_checkpoint(model, optimizer, iteration=1000, epoch=10, history={}, config={})
    >>> state = manager.load_checkpoint("checkpoint_iter_1000.pt", model, optimizer)
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize checkpoint manager with output directory."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: NeuralOperator,
        optimizer: OptimizerWithVariance,
        iteration: int,
        epoch: int,
        history: Dict[str, Any],
        config: Dict[str, Any],
        filename: str = None
    ) -> Path:
        """
        Save complete training checkpoint.
        
        Stores model state, optimizer state, training iteration/epoch,
        random number generator states, training history, and configuration.
        
        Parameters
        ----------
        model : NeuralOperator
            Neural operator model to save.
        optimizer : OptimizerWithVariance
            Optimizer to save.
        iteration : int
            Current training iteration number.
        epoch : int
            Current training epoch number.
        history : Dict[str, Any]
            Training history dictionary.
        config : Dict[str, Any]
            Experiment configuration dictionary.
        filename : str, optional
            Name of checkpoint file. If None, uses "checkpoint_iter_{iteration}.pt".
        
        Returns
        -------
        Path
            Path to the saved checkpoint file.
        
        Examples
        --------
        >>> manager.save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     iteration=1000,
        ...     epoch=10,
        ...     history=logger.history,
        ...     config=config,
        ...     filename="checkpoint_1000.pt"
        ... )
        """
        if filename is None:
            filename = f"checkpoint_iter_{iteration}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Collect all RNG states
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        
        # Add CUDA RNG state if available
        if torch.cuda.is_available():
            rng_states['torch_cuda'] = torch.cuda.get_rng_state_all()
        
        # Create checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.get_state_dict(),
            'iteration': iteration,
            'epoch': epoch,
            'rng_states': rng_states,
            'history': history,
            'config': config
        }
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_checkpoint(
        self,
        filename: str,
        model: Optional[NeuralOperator] = None,
        optimizer: Optional[OptimizerWithVariance] = None,
        load_rng_states: bool = True,
        weights_only: bool = False
    ) -> Dict[str, Any]:
        """
        Load training checkpoint and restore states.
        
        Supports two modes:
        1. Full restoration: Restores all states for exact training continuation
        2. Weights only: Loads only model weights for inference
        
        Parameters
        ----------
        filename : str
            Name of checkpoint file to load.
        model : NeuralOperator, optional
            Model to load weights into. Required unless weights_only=False.
        optimizer : OptimizerWithVariance, optional
            Optimizer to load state into. Ignored if weights_only=True.
        load_rng_states : bool, default=True
            Whether to restore random number generator states.
        weights_only : bool, default=False
            If True, only load model weights (for inference).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing checkpoint data including iteration, epoch, history, config.
        
        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        ValueError
            If model is None when trying to load weights.
        CheckpointCompatibilityError
            If checkpoint is incompatible with current model architecture.
        
        Examples
        --------
        >>> # Full restoration for training continuation
        >>> state = manager.load_checkpoint("checkpoint_1000.pt", model, optimizer)
        >>> iteration = state['iteration']
        >>> 
        >>> # Weights only for inference
        >>> state = manager.load_checkpoint("final_checkpoint.pt", model, weights_only=True)
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Validate checkpoint compatibility if model provided
        if model is not None:
            self._validate_checkpoint_compatibility(checkpoint, model)
        
        # Load model weights
        if model is not None:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # Provide detailed error information
                checkpoint_info = self._get_model_info_from_state_dict(checkpoint['model_state_dict'])
                model_info = self._get_model_info_from_model(model)
                
                raise CheckpointCompatibilityError(
                    f"Failed to load checkpoint weights: {str(e)}",
                    checkpoint_info=checkpoint_info,
                    model_info=model_info
                ) from e
        elif not weights_only:
            raise ValueError("Model must be provided to load checkpoint")
        
        # If weights_only mode, return early
        if weights_only:
            return {
                'iteration': checkpoint.get('iteration', 0),
                'epoch': checkpoint.get('epoch', 0),
                'config': checkpoint.get('config', {})
            }
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load optimizer state: {str(e)}")
        
        # Restore RNG states
        if load_rng_states and 'rng_states' in checkpoint:
            rng_states = checkpoint['rng_states']
            
            # Restore Python RNG state
            if 'python' in rng_states:
                random.setstate(rng_states['python'])
            
            # Restore NumPy RNG state
            if 'numpy' in rng_states:
                np.random.set_state(rng_states['numpy'])
            
            # Restore PyTorch RNG state
            if 'torch' in rng_states:
                torch.set_rng_state(rng_states['torch'])
            
            # Restore CUDA RNG state
            if 'torch_cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
        
        return {
            'iteration': checkpoint.get('iteration', 0),
            'epoch': checkpoint.get('epoch', 0),
            'history': checkpoint.get('history', {}),
            'config': checkpoint.get('config', {})
        }
    
    def _validate_checkpoint_compatibility(
        self,
        checkpoint: Dict[str, Any],
        model: NeuralOperator
    ) -> None:
        """
        Validate that checkpoint is compatible with current model.
        
        Parameters
        ----------
        checkpoint : Dict[str, Any]
            Loaded checkpoint dictionary.
        model : NeuralOperator
            Current model instance.
        
        Raises
        ------
        CheckpointCompatibilityError
            If checkpoint is incompatible with model.
        """
        checkpoint_state = checkpoint.get('model_state_dict', {})
        model_state = model.state_dict()
        
        # Get parameter information
        checkpoint_info = self._get_model_info_from_state_dict(checkpoint_state)
        model_info = self._get_model_info_from_model(model)
        
        # Check parameter count
        if checkpoint_info['num_parameters'] != model_info['num_parameters']:
            raise CheckpointCompatibilityError(
                "Checkpoint has different number of parameters than current model",
                checkpoint_info=checkpoint_info,
                model_info=model_info
            )
        
        # Check parameter names
        checkpoint_params = set(checkpoint_state.keys())
        model_params = set(model_state.keys())
        
        missing_in_checkpoint = model_params - checkpoint_params
        extra_in_checkpoint = checkpoint_params - model_params
        
        if missing_in_checkpoint or extra_in_checkpoint:
            error_msg = "Checkpoint parameter names don't match current model"
            
            if missing_in_checkpoint:
                checkpoint_info['missing_parameters'] = list(missing_in_checkpoint)[:10]  # Show first 10
            
            if extra_in_checkpoint:
                checkpoint_info['extra_parameters'] = list(extra_in_checkpoint)[:10]  # Show first 10
            
            raise CheckpointCompatibilityError(
                error_msg,
                checkpoint_info=checkpoint_info,
                model_info=model_info
            )
        
        # Check parameter shapes
        shape_mismatches = []
        for name in checkpoint_params:
            checkpoint_shape = checkpoint_state[name].shape
            model_shape = model_state[name].shape
            
            if checkpoint_shape != model_shape:
                shape_mismatches.append({
                    'parameter': name,
                    'checkpoint_shape': checkpoint_shape,
                    'model_shape': model_shape
                })
        
        if shape_mismatches:
            checkpoint_info['shape_mismatches'] = shape_mismatches[:10]  # Show first 10
            
            raise CheckpointCompatibilityError(
                "Checkpoint has parameters with different shapes than current model",
                checkpoint_info=checkpoint_info,
                model_info=model_info
            )
    
    def _get_model_info_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract model information from state dict."""
        num_params = sum(p.numel() for p in state_dict.values())
        param_names = list(state_dict.keys())
        
        return {
            'num_parameters': num_params,
            'num_layers': len(param_names),
            'parameter_names': param_names[:10],  # Show first 10
            'total_parameter_names': len(param_names)
        }
    
    def _get_model_info_from_model(self, model: NeuralOperator) -> Dict[str, Any]:
        """Extract model information from model instance."""
        state_dict = model.state_dict()
        num_params = model.get_parameter_count()
        param_names = list(state_dict.keys())
        
        return {
            'num_parameters': num_params,
            'num_layers': len(param_names),
            'parameter_names': param_names[:10],  # Show first 10
            'total_parameter_names': len(param_names),
            'model_type': type(model).__name__
        }
    
    def list_checkpoints(self) -> list:
        """
        List all checkpoint files in the checkpoint directory.
        
        Returns
        -------
        list
            List of checkpoint filenames sorted by modification time.
        
        Examples
        --------
        >>> checkpoints = manager.list_checkpoints()
        >>> print(f"Found {len(checkpoints)} checkpoints")
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return [cp.name for cp in checkpoints]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the most recently saved checkpoint filename.
        
        Returns
        -------
        Optional[str]
            Filename of the latest checkpoint, or None if no checkpoints exist.
        
        Examples
        --------
        >>> latest = manager.get_latest_checkpoint()
        >>> if latest:
        ...     state = manager.load_checkpoint(latest, model, optimizer)
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def delete_checkpoint(self, filename: str) -> None:
        """
        Delete a checkpoint file.
        
        Parameters
        ----------
        filename : str
            Name of checkpoint file to delete.
        
        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        
        Examples
        --------
        >>> manager.delete_checkpoint("checkpoint_iter_500.pt")
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        filepath.unlink()
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Delete old checkpoints, keeping only the most recent ones.
        
        Parameters
        ----------
        keep_last_n : int, default=5
            Number of most recent checkpoints to keep.
        
        Examples
        --------
        >>> # Keep only the 5 most recent checkpoints
        >>> manager.cleanup_old_checkpoints(keep_last_n=5)
        """
        checkpoints = self.list_checkpoints()
        
        # Keep final checkpoint if it exists
        final_checkpoint = 'final_checkpoint.pt'
        if final_checkpoint in checkpoints:
            checkpoints.remove(final_checkpoint)
        
        # Delete old checkpoints
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                self.delete_checkpoint(checkpoint)
