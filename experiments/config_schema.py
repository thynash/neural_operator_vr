"""Configuration schema for neural operator variance reduction experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str
    seed: int = 42
    device: str = "cuda"
    deterministic: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    type: str  # "logistic", "lorenz", "burgers"
    params: Dict[str, Any] = field(default_factory=dict)
    num_train_trajectories: int = 1000
    num_val_trajectories: int = 200
    input_horizon: int = 10
    output_horizon: int = 1
    train_val_split: float = 0.8
    batch_size: int = 32
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str  # "deeponet", "fno"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str  # "sgd", "adam", "svrg"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: Optional[str] = None  # "step", "exponential", "cosine"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training procedure configuration."""
    num_epochs: int = 100
    batch_size: int = 32
    validation_interval: int = 100
    variance_interval: int = 500
    checkpoint_interval: int = 1000
    early_stopping_patience: Optional[int] = 20
    target_loss: Optional[float] = 1e-4


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "./logs"
    save_checkpoints: bool = True
    save_final_model: bool = True
    log_level: str = "INFO"


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    compute_spectral_radius: bool = True
    spectral_interval: int = 1000
    long_horizon_steps: int = 100
    num_eigenvalues: int = 10


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    generate_plots: bool = True
    plot_format: str = "pdf"
    dpi: int = 300
    output_dir: str = "./visualization_output"


@dataclass
class Config:
    """Complete experiment configuration."""
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    logging: LoggingConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    scheduler: Optional[SchedulerConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "experiment": {
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "device": self.experiment.device,
                "deterministic": self.experiment.deterministic,
            },
            "dataset": {
                "type": self.dataset.type,
                "params": self.dataset.params,
                "num_train_trajectories": self.dataset.num_train_trajectories,
                "num_val_trajectories": self.dataset.num_val_trajectories,
                "input_horizon": self.dataset.input_horizon,
                "output_horizon": self.dataset.output_horizon,
                "train_val_split": self.dataset.train_val_split,
                "batch_size": self.dataset.batch_size,
                "shuffle": self.dataset.shuffle,
            },
            "model": {
                "type": self.model.type,
                "params": self.model.params,
            },
            "optimizer": {
                "type": self.optimizer.type,
                "params": self.optimizer.params,
            },
            "training": {
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "validation_interval": self.training.validation_interval,
                "variance_interval": self.training.variance_interval,
                "checkpoint_interval": self.training.checkpoint_interval,
                "early_stopping_patience": self.training.early_stopping_patience,
                "target_loss": self.training.target_loss,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "save_checkpoints": self.logging.save_checkpoints,
                "save_final_model": self.logging.save_final_model,
                "log_level": self.logging.log_level,
            },
            "analysis": {
                "compute_spectral_radius": self.analysis.compute_spectral_radius,
                "spectral_interval": self.analysis.spectral_interval,
                "long_horizon_steps": self.analysis.long_horizon_steps,
                "num_eigenvalues": self.analysis.num_eigenvalues,
            },
            "visualization": {
                "generate_plots": self.visualization.generate_plots,
                "plot_format": self.visualization.plot_format,
                "dpi": self.visualization.dpi,
                "output_dir": self.visualization.output_dir,
            },
        }
        
        if self.scheduler is not None:
            result["scheduler"] = {
                "type": self.scheduler.type,
                "params": self.scheduler.params,
            }
        
        return result
