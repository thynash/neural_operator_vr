"""Training loop implementation for neural operator training."""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import NeuralOperator
from optimizers.base import OptimizerWithVariance
from utils.logger import MetricsLogger
from utils.exceptions import TrainingDivergenceError, GPUMemoryError
from training.checkpoint_manager import CheckpointManager


class TrainingLoop:
    """
    Training loop for neural operator models.
    
    Manages the complete training workflow including training iterations,
    validation, gradient variance computation, checkpointing, and early stopping.
    
    Parameters
    ----------
    model : NeuralOperator
        Neural operator model to train.
    optimizer : OptimizerWithVariance
        Optimizer with gradient variance tracking.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    config : Dict[str, Any]
        Configuration dictionary containing training parameters.
    logger : MetricsLogger, optional
        Metrics logger for tracking training history.
    checkpoint_manager : CheckpointManager, optional
        Manager for saving and loading checkpoints.
    
    Attributes
    ----------
    model : NeuralOperator
        The neural operator being trained.
    optimizer : OptimizerWithVariance
        The optimizer being used.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : Dict[str, Any]
        Training configuration.
    logger : MetricsLogger
        Metrics logger.
    checkpoint_manager : CheckpointManager
        Checkpoint manager.
    device : torch.device
        Device for computation (CPU or CUDA).
    loss_fn : Callable
        Loss function (MSE).
    current_iteration : int
        Current training iteration number.
    current_epoch : int
        Current training epoch number.
    best_val_loss : float
        Best validation loss achieved.
    patience_counter : int
        Counter for early stopping patience.
    
    Examples
    --------
    >>> training_loop = TrainingLoop(model, optimizer, train_loader, val_loader, config)
    >>> training_loop.run(num_epochs=100)
    """
    
    def __init__(
        self,
        model: NeuralOperator,
        optimizer: OptimizerWithVariance,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """Initialize training loop with all components."""
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Get device from config or auto-detect
        device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_name)
        self.model.to(self.device)
        
        # Loss function (MSE)
        self.loss_fn = nn.MSELoss()
        
        # Initialize logger if not provided
        if logger is None:
            log_dir = config.get('log_dir', './logs')
            experiment_name = config.get('experiment_name', 'experiment')
            self.logger = MetricsLogger(log_dir, experiment_name)
        else:
            self.logger = logger
        
        # Initialize checkpoint manager if not provided
        if checkpoint_manager is None:
            checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = checkpoint_manager
        
        # Training state
        self.current_iteration = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Extract training parameters from config
        self.validation_interval = config.get('validation_interval', 100)
        self.variance_interval = config.get('variance_interval', 500)
        self.checkpoint_interval = config.get('checkpoint_interval', 1000)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.num_variance_samples = config.get('num_variance_samples', 10)
        
        # Divergence detection
        self.recent_losses = []
        self.max_recent_losses = 10
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one full pass through training data.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing epoch-level metrics (average loss, etc.).
        
        Examples
        --------
        >>> epoch_metrics = training_loop.train_epoch()
        >>> print(f"Epoch loss: {epoch_metrics['avg_loss']}")
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            try:
                # Handle different batch formats
                if len(batch_data) == 3:
                    # DeepONet format: (input_functions, query_points, targets)
                    input_functions, query_points, targets = batch_data
                    input_functions = input_functions.to(self.device)
                    query_points = query_points.to(self.device)
                    targets = targets.to(self.device)
                elif len(batch_data) == 2:
                    # Simple format: (inputs, targets)
                    # For DeepONet, use inputs as both input_functions and query_points
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Reshape for DeepONet if needed
                    # inputs: [batch, input_horizon, state_dim]
                    # DeepONet expects: input_functions [batch, state_dim, num_sensors]
                    #                   query_points [batch, state_dim, num_queries]
                    input_functions = inputs.permute(0, 2, 1)  # [batch, state_dim, input_horizon]
                    query_points = targets.permute(0, 2, 1)  # [batch, state_dim, output_horizon]
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                
                # Record iteration start time
                iter_start_time = time.time()
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(input_functions, query_points)
                
                # Reshape targets to match predictions if needed
                if targets.shape != predictions.shape:
                    # targets: [batch, output_horizon, state_dim]
                    # predictions: [batch, state_dim, output_horizon]
                    if len(targets.shape) == 3 and len(predictions.shape) == 3:
                        if targets.shape[1] != predictions.shape[1]:
                            targets = targets.permute(0, 2, 1)
                
                loss = self.loss_fn(predictions, targets)
                
                # Check for divergence
                self._check_divergence(loss, predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Compute gradient norm
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # Handle SVRG snapshot if needed
                if hasattr(self.optimizer, 'needs_snapshot') and self.optimizer.needs_snapshot:
                    self.optimizer.take_snapshot(self.model, self.train_loader, self.loss_fn)
                
                # Optimizer step
                self.optimizer.step()
                
                # Record iteration time
                iter_time = time.time() - iter_start_time
                
                # Log metrics
                self.logger.log_dict({
                    'train_loss': loss.item(),
                    'train_grad_norm': grad_norm,
                    'iteration_time': iter_time
                }, step=self.current_iteration)
                
                # Track recent losses for divergence detection
                self.recent_losses.append(loss.item())
                if len(self.recent_losses) > self.max_recent_losses:
                    self.recent_losses.pop(0)
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Validation at specified intervals
                if self.current_iteration % self.validation_interval == 0:
                    val_metrics = self.validate()
                    self.logger.log_dict(val_metrics, step=self.current_iteration)
                    
                    # Check for early stopping
                    if self._check_early_stopping(val_metrics['val_loss']):
                        return {'avg_loss': epoch_loss / num_batches, 'early_stop': True}
                
                # Gradient variance computation at specified intervals
                if self.current_iteration % self.variance_interval == 0:
                    variance = self._compute_variance()
                    self.logger.log_scalar('train_grad_variance', variance, step=self.current_iteration)
                
                # Checkpoint saving at specified intervals
                if self.current_iteration % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                self.current_iteration += 1
            
            except torch.cuda.OutOfMemoryError as e:
                # Handle GPU memory errors
                self._handle_gpu_memory_error(e)
        
        return {'avg_loss': epoch_loss / num_batches, 'early_stop': False}
    
    def validate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing validation metrics (loss, relative error, etc.).
        
        Examples
        --------
        >>> val_metrics = training_loop.validate()
        >>> print(f"Validation loss: {val_metrics['val_loss']}")
        """
        self.model.eval()
        total_loss = 0.0
        total_relative_error = 0.0
        total_max_error = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle different batch formats
                if len(batch_data) == 3:
                    input_functions, query_points, targets = batch_data
                    input_functions = input_functions.to(self.device)
                    query_points = query_points.to(self.device)
                    targets = targets.to(self.device)
                elif len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    input_functions = inputs.permute(0, 2, 1)
                    query_points = targets.permute(0, 2, 1)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                
                # Forward pass
                predictions = self.model(input_functions, query_points)
                
                # Reshape targets to match predictions if needed
                if targets.shape != predictions.shape:
                    if len(targets.shape) == 3 and len(predictions.shape) == 3:
                        if targets.shape[1] != predictions.shape[1]:
                            targets = targets.permute(0, 2, 1)
                
                loss = self.loss_fn(predictions, targets)
                
                # Compute additional metrics
                relative_error = torch.norm(predictions - targets) / (torch.norm(targets) + 1e-8)
                max_error = torch.max(torch.abs(predictions - targets))
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_relative_error += relative_error.item()
                total_max_error += max_error.item()
                total_mae += mae.item()
                num_batches += 1
        
        self.model.train()
        
        return {
            'val_loss': total_loss / num_batches,
            'val_relative_error': total_relative_error / num_batches,
            'val_max_error': total_max_error / num_batches,
            'val_mean_absolute_error': total_mae / num_batches
        }
    
    def run(self, num_epochs: int) -> Dict[str, Any]:
        """
        Execute full training run.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing final training results and history.
        
        Examples
        --------
        >>> results = training_loop.run(num_epochs=100)
        >>> print(f"Final validation loss: {results['final_val_loss']}")
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_parameter_count()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train one epoch
            epoch_metrics = self.train_epoch()
            print(f"  Average training loss: {epoch_metrics['avg_loss']:.6f}")
            
            # Check for early stopping
            if epoch_metrics.get('early_stop', False):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Validate at end of epoch
            val_metrics = self.validate()
            self.logger.log_dict(val_metrics, step=self.current_iteration)
            print(f"  Validation loss: {val_metrics['val_loss']:.6f}")
            print(f"  Validation relative error: {val_metrics['val_relative_error']:.6f}")
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Save training history
        history_path = self.logger.save_history()
        print(f"\nTraining complete. History saved to {history_path}")
        
        # Get final validation metrics
        final_val_metrics = self.validate()
        
        return {
            'final_val_loss': final_val_metrics['val_loss'],
            'final_val_relative_error': final_val_metrics['val_relative_error'],
            'best_val_loss': self.best_val_loss,
            'total_iterations': self.current_iteration,
            'total_epochs': self.current_epoch + 1,
            'history': self.logger.history
        }
    
    def _compute_variance(self) -> float:
        """
        Compute gradient variance at current parameters.
        
        Returns
        -------
        float
            Empirical gradient variance.
        """
        variance = self.optimizer.compute_gradient_variance(
            self.model,
            self.train_loader,
            self.loss_fn,
            num_samples=self.num_variance_samples
        )
        return variance
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Parameters
        ----------
        val_loss : float
            Current validation loss.
        
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                return True
            return False
    
    def _save_checkpoint(self, is_final: bool = False) -> None:
        """
        Save training checkpoint.
        
        Parameters
        ----------
        is_final : bool, default=False
            Whether this is the final checkpoint.
        """
        checkpoint_name = 'final_checkpoint.pt' if is_final else f'checkpoint_iter_{self.current_iteration}.pt'
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=self.current_iteration,
            epoch=self.current_epoch,
            history=self.logger.history,
            config=self.config,
            filename=checkpoint_name
        )
        
        if not is_final:
            print(f"  Checkpoint saved at iteration {self.current_iteration}")
    
    def _check_divergence(
        self,
        loss: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """
        Check for training divergence (NaN or infinite loss).
        
        Parameters
        ----------
        loss : torch.Tensor
            Current loss value.
        predictions : torch.Tensor
            Model predictions.
        targets : torch.Tensor
            Target values.
        
        Raises
        ------
        TrainingDivergenceError
            If loss is NaN or infinite.
        """
        loss_value = loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            # Compute gradient norm for diagnostics
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Get learning rate
            lr = self.optimizer.get_learning_rate() if hasattr(self.optimizer, 'get_learning_rate') else 'unknown'
            
            # Compute batch statistics
            batch_stats = {
                'predictions_mean': predictions.mean().item(),
                'predictions_std': predictions.std().item(),
                'predictions_min': predictions.min().item(),
                'predictions_max': predictions.max().item(),
                'targets_mean': targets.mean().item(),
                'targets_std': targets.std().item(),
                'targets_min': targets.min().item(),
                'targets_max': targets.max().item(),
            }
            
            # Prepare diagnostics
            diagnostics = {
                'iteration': self.current_iteration,
                'epoch': self.current_epoch,
                'loss_value': loss_value,
                'recent_losses': self.recent_losses[-5:] if self.recent_losses else [],
                'gradient_norm': grad_norm,
                'learning_rate': lr,
                'batch_statistics': batch_stats,
            }
            
            # Prepare suggestions
            suggestions = [
                "Reduce the learning rate (try 10x smaller)",
                "Check for data normalization issues",
                "Reduce batch size to improve gradient stability",
                "Add gradient clipping to prevent exploding gradients",
                "Check for numerical instability in the model architecture",
                "Verify that input data doesn't contain NaN or infinite values",
            ]
            
            # Save diagnostic checkpoint
            try:
                self._save_checkpoint(is_final=False)
                diagnostics['checkpoint_saved'] = True
            except Exception as e:
                diagnostics['checkpoint_saved'] = False
                diagnostics['checkpoint_error'] = str(e)
            
            raise TrainingDivergenceError(
                f"Training diverged at iteration {self.current_iteration}: "
                f"loss is {'NaN' if torch.isnan(loss) else 'infinite'}",
                diagnostics=diagnostics,
                suggestions=suggestions
            )
    
    def _handle_gpu_memory_error(self, error: Exception) -> None:
        """
        Handle GPU out of memory errors.
        
        Parameters
        ----------
        error : Exception
            The CUDA out of memory error.
        
        Raises
        ------
        GPUMemoryError
            Wrapped error with memory statistics and suggestions.
        """
        # Get GPU memory statistics
        memory_stats = {}
        if torch.cuda.is_available():
            try:
                memory_stats = {
                    'allocated': f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                    'reserved': f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
                    'max_allocated': f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                    'max_reserved': f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB",
                }
                
                # Get device properties
                device_props = torch.cuda.get_device_properties(self.device)
                memory_stats['total_memory'] = f"{device_props.total_memory / 1e9:.2f} GB"
                memory_stats['device_name'] = device_props.name
            except Exception:
                pass
        
        # Prepare suggestions
        current_batch_size = self.config.get('batch_size', 'unknown')
        suggestions = [
            f"Reduce batch size (current: {current_batch_size}, try half: {current_batch_size // 2 if isinstance(current_batch_size, int) else 'N/A'})",
            "Reduce model size (fewer layers or smaller hidden dimensions)",
            "Use gradient accumulation to simulate larger batches",
            "Enable mixed precision training (torch.cuda.amp)",
            "Clear CUDA cache: torch.cuda.empty_cache()",
        ]
        
        raise GPUMemoryError(
            f"GPU out of memory at iteration {self.current_iteration}",
            memory_stats=memory_stats,
            suggestions=suggestions
        ) from error
