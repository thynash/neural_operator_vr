"""Experiment runner for neural operator variance reduction experiments."""

import os
import json
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

import torch

from experiments.config_schema import Config
from experiments.config_parser import load_config
from experiments.config_validator import ConfigValidator
from experiments.config_serializer import save_config

from datasets import (
    LogisticMapDataset,
    LorenzSystemDataset,
    BurgersEquationDataset,
    create_dataloaders,
)
from models import DeepONet, FNO
from optimizers import SGD, Adam, SVRG, StepLR, ExponentialLR, CosineAnnealingLR
from training import TrainingLoop
from utils import set_random_seeds, get_device, MetricsLogger, log_system_info
from analysis.spectral import SpectralAnalyzer
from visualization.plots import PlotGenerator


class ExperimentRunner:
    """Orchestrates complete neural operator experiments."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        ConfigValidator.validate(self.config)
        
        self.logger = None
        self.device = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_loop = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup Python logging."""
        log_level = getattr(logging, self.config.logging.log_level)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_experiment(self) -> None:
        """Initialize all experiment components."""
        self.logger.info(f"Setting up experiment: {self.config.experiment.name}")
        
        # Set random seeds
        set_random_seeds(
            self.config.experiment.seed,
            deterministic=self.config.experiment.deterministic
        )
        self.logger.info(f"Random seed set to {self.config.experiment.seed}")
        
        # Setup device
        self.device = get_device(self.config.experiment.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Log system info
        log_system_info(self.logger)
        
        # Create dataset
        self._create_dataset()
        
        # Create model
        self._create_model()
        
        # Create optimizer
        self._create_optimizer()
        
        # Create scheduler (if specified)
        if self.config.scheduler is not None:
            self._create_scheduler()
        
        # Create metrics logger
        log_dir = Path(self.config.logging.log_dir) / self.config.experiment.name
        self.metrics_logger = MetricsLogger(
            log_dir=str(log_dir),
            experiment_name=self.config.experiment.name
        )
        
        # Create training loop
        self._create_training_loop()
        
        # Save configuration
        config_save_path = log_dir / "config.yaml"
        save_config(self.config, config_save_path)
        self.logger.info(f"Configuration saved to {config_save_path}")
    
    def _create_dataset(self) -> None:
        """Create dataset and data loaders."""
        self.logger.info(f"Creating {self.config.dataset.type} dataset")
        
        dataset_type = self.config.dataset.type
        params = self.config.dataset.params
        
        # Extract seed from experiment config
        seed = self.config.experiment.seed
        
        if dataset_type == "logistic":
            self.dataset = LogisticMapDataset(seed=seed)
        elif dataset_type == "lorenz":
            self.dataset = LorenzSystemDataset(seed=seed)
        elif dataset_type == "burgers":
            self.dataset = BurgersEquationDataset(seed=seed)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Generate data - pass params to create_operator_dataset
        self.logger.info("Generating training and validation data")
        
        # Prepare params for create_operator_dataset
        dataset_params = dict(params)  # Copy params
        
        train_data = self.dataset.create_operator_dataset(
            num_trajectories=self.config.dataset.num_train_trajectories,
            input_horizon=self.config.dataset.input_horizon,
            output_horizon=self.config.dataset.output_horizon,
            **dataset_params  # Pass all params including trajectory_length
        )
        
        val_data = self.dataset.create_operator_dataset(
            num_trajectories=self.config.dataset.num_val_trajectories,
            input_horizon=self.config.dataset.input_horizon,
            output_horizon=self.config.dataset.output_horizon,
            **dataset_params  # Pass all params including trajectory_length
        )
        
        # Create data loaders
        train_inputs, train_outputs = train_data
        val_inputs, val_outputs = val_data
        
        self.train_loader, self.val_loader, self.normalization_stats = create_dataloaders(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            val_inputs=val_inputs,
            val_outputs=val_outputs,
            batch_size=self.config.training.batch_size,
            shuffle_train=self.config.dataset.shuffle,
        )
        
        self.logger.info(
            f"Created data loaders: {len(self.train_loader)} train batches, "
            f"{len(self.val_loader)} val batches"
        )
    
    def _create_model(self) -> None:
        """Create neural operator model."""
        self.logger.info(f"Creating {self.config.model.type} model")
        
        model_type = self.config.model.type
        params = dict(self.config.model.params)  # Copy params
        
        # Infer input/output dimensions from data
        sample_batch = next(iter(self.train_loader))
        sample_input, sample_output = sample_batch
        
        # For DeepONet: input_dim and output_dim are the feature dimensions
        # For FNO: input_channels and output_channels
        if model_type == "deeponet":
            # Input shape: [batch, input_horizon, state_dim]
            # We need to reshape for DeepONet: [batch, state_dim, input_horizon]
            params['input_dim'] = sample_input.shape[2]  # state_dim
            params['output_dim'] = sample_output.shape[2]  # state_dim
            self.model = DeepONet(**params)
        elif model_type == "fno":
            # FNO expects [batch, channels, spatial_dim]
            params['input_channels'] = sample_input.shape[1]
            params['output_channels'] = sample_output.shape[1]
            self.model = FNO(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        param_count = self.model.get_parameter_count()
        self.logger.info(f"Model created with {param_count:,} parameters")
    
    def _create_optimizer(self) -> None:
        """Create optimizer."""
        self.logger.info(f"Creating {self.config.optimizer.type} optimizer")
        
        opt_type = self.config.optimizer.type
        params = self.config.optimizer.params.copy()
        
        if opt_type == "sgd":
            self.optimizer = SGD(self.model.parameters(), **params)
        elif opt_type == "adam":
            self.optimizer = Adam(self.model.parameters(), **params)
        elif opt_type == "svrg":
            # SVRG doesn't take train_loader in __init__
            self.optimizer = SVRG(self.model.parameters(), **params)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        self.logger.info(f"Optimizer created: {opt_type}")
    
    def _create_scheduler(self) -> None:
        """Create learning rate scheduler."""
        sched_type = self.config.scheduler.type
        params = self.config.scheduler.params
        
        self.logger.info(f"Creating {sched_type} scheduler")
        
        if sched_type == "step":
            self.scheduler = StepLR(self.optimizer, **params)
        elif sched_type == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, **params)
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, **params)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    def _create_training_loop(self) -> None:
        """Create training loop."""
        # Prepare config dict for training loop
        training_config = {
            'device': str(self.device),
            'log_dir': self.config.logging.log_dir,
            'experiment_name': self.config.experiment.name,
            'checkpoint_dir': self.config.logging.log_dir,
            'validation_interval': self.config.training.validation_interval,
            'variance_interval': self.config.training.variance_interval,
            'checkpoint_interval': self.config.training.checkpoint_interval,
            'early_stopping_patience': self.config.training.early_stopping_patience,
        }
        
        self.training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=training_config,
            logger=self.metrics_logger,
        )
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info("Starting experiment execution")
        
        # Setup experiment if not already done
        if self.training_loop is None:
            self.setup_experiment()
        
        # Run training
        self.logger.info("Starting training")
        history = self.training_loop.run(num_epochs=self.config.training.num_epochs)
        
        # Compute analysis metrics
        if self.config.analysis.compute_spectral_radius:
            self.logger.info("Computing spectral analysis")
            self._compute_spectral_analysis(history)
        
        # Generate visualizations
        if self.config.visualization.generate_plots:
            self.logger.info("Generating visualizations")
            self._generate_visualizations(history)
        
        # Save results
        self.logger.info("Saving results")
        results = self._save_results(history)
        
        self.logger.info("Experiment completed successfully")
        return results
    
    def _compute_spectral_analysis(self, history: Dict[str, Any]) -> None:
        """Compute spectral analysis of learned operator."""
        analyzer = SpectralAnalyzer(
            model=self.model,
            dataset=self.dataset,
            device=self.device,
        )
        
        # Get a sample state point
        sample_batch = next(iter(self.val_loader))
        sample_input = sample_batch[0][0:1]  # First sample
        
        # Compute eigenvalues
        eigenvalues = analyzer.compute_operator_eigenvalues(
            state_point=sample_input,
            num_eigenvalues=self.config.analysis.num_eigenvalues,
        )
        
        spectral_radius = analyzer.compute_spectral_radius(eigenvalues)
        
        # Convert complex eigenvalues to JSON-serializable format
        if torch.is_complex(eigenvalues):
            eigenvalues_list = [
                {"real": float(ev.real), "imag": float(ev.imag)}
                for ev in eigenvalues
            ]
        else:
            eigenvalues_list = eigenvalues.tolist() if hasattr(eigenvalues, 'tolist') else eigenvalues
        
        # Store in history
        history["spectral_analysis"] = {
            "eigenvalues": eigenvalues_list,
            "spectral_radius": float(spectral_radius),
        }
        
        # Get true eigenvalues if available
        try:
            true_eigenvalues = self.dataset.get_true_eigenvalues(sample_input)
            if true_eigenvalues is not None:
                eigenvalue_error = analyzer.compute_eigenvalue_error(
                    learned_eigenvalues=eigenvalues,
                    true_eigenvalues=true_eigenvalues,
                )
                history["spectral_analysis"]["true_eigenvalues"] = (
                    true_eigenvalues.tolist() if hasattr(true_eigenvalues, 'tolist')
                    else true_eigenvalues
                )
                history["spectral_analysis"]["eigenvalue_error"] = float(eigenvalue_error)
        except Exception as e:
            self.logger.warning(f"Could not compute true eigenvalues: {e}")
    
    def _generate_visualizations(self, history: Dict[str, Any]) -> None:
        """Generate publication-quality visualizations."""
        output_dir = Path(self.config.visualization.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_gen = PlotGenerator(
            output_dir=str(output_dir),
            plot_format=self.config.visualization.plot_format,
            dpi=self.config.visualization.dpi,
        )
        
        # Training curves
        if "train_loss" in history:
            plot_gen.plot_training_curves(
                histories={"current": history},
                optimizer_names=[self.config.optimizer.type],
            )
        
        # Gradient variance
        if "train_grad_variance" in history:
            plot_gen.plot_gradient_variance(
                histories={"current": history},
                optimizer_names=[self.config.optimizer.type],
            )
        
        # Validation error
        if "val_loss" in history:
            plot_gen.plot_validation_error(
                histories={"current": history},
                optimizer_names=[self.config.optimizer.type],
            )
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def _save_results(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Save experiment results."""
        log_dir = Path(self.config.logging.log_dir) / self.config.experiment.name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        history_path = log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save final model if requested
        if self.config.logging.save_final_model:
            model_path = log_dir / "final_model.pt"
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Final model saved to {model_path}")
        
        # Prepare results summary
        val_loss_list = history.get("val_loss", [])
        final_val_loss = val_loss_list[-1] if val_loss_list else None
        
        results = {
            "experiment_name": self.config.experiment.name,
            "config": self.config.to_dict(),
            "history": history,
            "final_val_loss": final_val_loss,
        }
        
        return results
    
    def run_multiple_seeds(self, seeds: List[int]) -> Dict[str, Any]:
        """
        Run experiment with multiple random seeds for statistical analysis.
        
        Args:
            seeds: List of random seeds to use
            
        Returns:
            Dictionary containing aggregated results across all seeds
        """
        self.logger.info(f"Running experiment with {len(seeds)} different seeds")
        
        all_results = []
        
        for seed in seeds:
            self.logger.info(f"Running with seed {seed}")
            
            # Update config with new seed
            self.config.experiment.seed = seed
            self.config.experiment.name = f"{self.config.experiment.name}_seed{seed}"
            
            # Reset components
            self.dataset = None
            self.model = None
            self.optimizer = None
            self.scheduler = None
            self.training_loop = None
            
            # Run experiment
            results = self.run()
            all_results.append(results)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save aggregated results
        log_dir = Path(self.config.logging.log_dir) / "aggregated"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        agg_path = log_dir / "aggregated_results.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        self.logger.info(f"Aggregated results saved to {agg_path}")
        
        return aggregated
    
    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        import numpy as np
        
        # Extract final validation losses
        final_losses = [
            r["final_val_loss"] for r in all_results
            if r["final_val_loss"] is not None
        ]
        
        aggregated = {
            "num_runs": len(all_results),
            "seeds": [r["config"]["experiment"]["seed"] for r in all_results],
            "final_val_loss": {
                "mean": float(np.mean(final_losses)) if final_losses else None,
                "std": float(np.std(final_losses)) if final_losses else None,
                "min": float(np.min(final_losses)) if final_losses else None,
                "max": float(np.max(final_losses)) if final_losses else None,
            },
            "individual_results": all_results,
        }
        
        return aggregated
