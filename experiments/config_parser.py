"""Configuration parser for neural operator experiments."""

import json
import yaml
from pathlib import Path
from typing import Union, Dict, Any

from experiments.config_schema import (
    Config,
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    LoggingConfig,
    AnalysisConfig,
    VisualizationConfig,
)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load and parse configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        Config object with parsed configuration
        
    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is invalid with descriptive error message
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load raw configuration
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                raw_config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                raw_config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}. "
                    f"Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        # Extract line number from YAML error if available
        line_info = ""
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            line_info = f" at line {mark.line + 1}, column {mark.column + 1}"
        raise ValueError(f"Failed to parse YAML configuration{line_info}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON configuration at line {e.lineno}: {e.msg}")
    except Exception as e:
        raise ValueError(f"Failed to read configuration file: {e}")
    
    # Parse into configuration objects
    try:
        config = _parse_config_dict(raw_config)
    except KeyError as e:
        raise ValueError(f"Missing required configuration section: {e}")
    except TypeError as e:
        raise ValueError(f"Invalid configuration structure: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse configuration: {e}")
    
    return config


def _parse_config_dict(raw_config: Dict[str, Any]) -> Config:
    """Parse raw configuration dictionary into Config object."""
    
    # Parse experiment section
    if "experiment" not in raw_config:
        raise ValueError("Missing required section: experiment")
    exp_dict = raw_config["experiment"]
    experiment = ExperimentConfig(
        name=exp_dict.get("name", ""),
        seed=exp_dict.get("seed", 42),
        device=exp_dict.get("device", "cuda"),
        deterministic=exp_dict.get("deterministic", True),
    )
    
    # Parse dataset section
    if "dataset" not in raw_config:
        raise ValueError("Missing required section: dataset")
    ds_dict = raw_config["dataset"]
    dataset = DatasetConfig(
        type=ds_dict.get("type", ""),
        params=ds_dict.get("params", {}),
        num_train_trajectories=ds_dict.get("num_train_trajectories", 1000),
        num_val_trajectories=ds_dict.get("num_val_trajectories", 200),
        input_horizon=ds_dict.get("input_horizon", 10),
        output_horizon=ds_dict.get("output_horizon", 1),
        train_val_split=ds_dict.get("train_val_split", 0.8),
        batch_size=ds_dict.get("batch_size", 32),
        shuffle=ds_dict.get("shuffle", True),
    )
    
    # Parse model section
    if "model" not in raw_config:
        raise ValueError("Missing required section: model")
    model_dict = raw_config["model"]
    model = ModelConfig(
        type=model_dict.get("type", ""),
        params=model_dict.get("params", {}),
    )
    
    # Parse optimizer section
    if "optimizer" not in raw_config:
        raise ValueError("Missing required section: optimizer")
    opt_dict = raw_config["optimizer"]
    optimizer = OptimizerConfig(
        type=opt_dict.get("type", ""),
        params=opt_dict.get("params", {}),
    )
    
    # Parse scheduler section (optional)
    scheduler = None
    if "scheduler" in raw_config:
        sched_dict = raw_config["scheduler"]
        scheduler = SchedulerConfig(
            type=sched_dict.get("type"),
            params=sched_dict.get("params", {}),
        )
    
    # Parse training section
    if "training" not in raw_config:
        raise ValueError("Missing required section: training")
    train_dict = raw_config["training"]
    training = TrainingConfig(
        num_epochs=train_dict.get("num_epochs", 100),
        batch_size=train_dict.get("batch_size", 32),
        validation_interval=train_dict.get("validation_interval", 100),
        variance_interval=train_dict.get("variance_interval", 500),
        checkpoint_interval=train_dict.get("checkpoint_interval", 1000),
        early_stopping_patience=train_dict.get("early_stopping_patience", 20),
        target_loss=train_dict.get("target_loss", 1e-4),
    )
    
    # Parse logging section
    if "logging" not in raw_config:
        raise ValueError("Missing required section: logging")
    log_dict = raw_config["logging"]
    logging = LoggingConfig(
        log_dir=log_dict.get("log_dir", "./logs"),
        save_checkpoints=log_dict.get("save_checkpoints", True),
        save_final_model=log_dict.get("save_final_model", True),
        log_level=log_dict.get("log_level", "INFO"),
    )
    
    # Parse analysis section
    if "analysis" not in raw_config:
        raise ValueError("Missing required section: analysis")
    analysis_dict = raw_config["analysis"]
    analysis = AnalysisConfig(
        compute_spectral_radius=analysis_dict.get("compute_spectral_radius", True),
        spectral_interval=analysis_dict.get("spectral_interval", 1000),
        long_horizon_steps=analysis_dict.get("long_horizon_steps", 100),
        num_eigenvalues=analysis_dict.get("num_eigenvalues", 10),
    )
    
    # Parse visualization section
    if "visualization" not in raw_config:
        raise ValueError("Missing required section: visualization")
    vis_dict = raw_config["visualization"]
    visualization = VisualizationConfig(
        generate_plots=vis_dict.get("generate_plots", True),
        plot_format=vis_dict.get("plot_format", "pdf"),
        dpi=vis_dict.get("dpi", 300),
        output_dir=vis_dict.get("output_dir", "./visualization_output"),
    )
    
    return Config(
        experiment=experiment,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        training=training,
        logging=logging,
        analysis=analysis,
        visualization=visualization,
        scheduler=scheduler,
    )
