"""Configuration validation for neural operator experiments."""

from typing import Dict, Any, List
from experiments.config_schema import Config


class ConfigValidator:
    """Validates experiment configurations."""
    
    # Valid values for categorical parameters
    VALID_DATASET_TYPES = {"logistic", "lorenz", "burgers"}
    VALID_MODEL_TYPES = {"deeponet", "fno"}
    VALID_OPTIMIZER_TYPES = {"sgd", "adam", "svrg"}
    VALID_SCHEDULER_TYPES = {"step", "exponential", "cosine", None}
    VALID_DEVICES = {"cpu", "cuda", "auto"}
    VALID_ACTIVATIONS = {"relu", "tanh", "gelu", "silu"}
    
    # Parameter ranges for dynamical systems
    LOGISTIC_PARAM_RANGES = {
        "r": (0.0, 4.0),
        "trajectory_length": (10, 100000),
    }
    
    LORENZ_PARAM_RANGES = {
        "sigma": (0.0, 100.0),
        "rho": (0.0, 100.0),
        "beta": (0.0, 10.0),
        "dt": (1e-5, 1.0),
        "trajectory_length": (10, 100000),
    }
    
    BURGERS_PARAM_RANGES = {
        "viscosity": (1e-5, 1.0),
        "spatial_resolution": (16, 1024),
        "temporal_resolution": (10, 10000),
    }
    
    @staticmethod
    def validate(config: Config) -> None:
        """
        Validate complete configuration.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            ValueError: If configuration is invalid with descriptive message
        """
        ConfigValidator._validate_experiment(config.experiment)
        ConfigValidator._validate_dataset(config.dataset)
        ConfigValidator._validate_model(config.model)
        ConfigValidator._validate_optimizer(config.optimizer)
        ConfigValidator._validate_training(config.training)
        ConfigValidator._validate_logging(config.logging)
        ConfigValidator._validate_analysis(config.analysis)
        ConfigValidator._validate_visualization(config.visualization)
        
        if config.scheduler is not None:
            ConfigValidator._validate_scheduler(config.scheduler)
        
        # Cross-component validation
        ConfigValidator._validate_compatibility(config)
    
    @staticmethod
    def _validate_experiment(exp_config) -> None:
        """Validate experiment configuration."""
        if not exp_config.name:
            raise ValueError("Missing required parameter: name in section experiment")
        
        if exp_config.seed < 0:
            raise ValueError(f"Parameter seed must be non-negative, got {exp_config.seed}")
        
        device = exp_config.device.split(":")[0]  # Handle "cuda:0" format
        if device not in ConfigValidator.VALID_DEVICES:
            raise ValueError(
                f"Parameter device must be one of {ConfigValidator.VALID_DEVICES}, "
                f"got {exp_config.device}"
            )
    
    @staticmethod
    def _validate_dataset(dataset_config) -> None:
        """Validate dataset configuration."""
        if dataset_config.type not in ConfigValidator.VALID_DATASET_TYPES:
            raise ValueError(
                f"Parameter type in section dataset must be one of "
                f"{ConfigValidator.VALID_DATASET_TYPES}, got {dataset_config.type}"
            )
        
        # Validate dataset-specific parameters
        if dataset_config.type == "logistic":
            ConfigValidator._validate_logistic_params(dataset_config.params)
        elif dataset_config.type == "lorenz":
            ConfigValidator._validate_lorenz_params(dataset_config.params)
        elif dataset_config.type == "burgers":
            ConfigValidator._validate_burgers_params(dataset_config.params)
        
        # Validate common dataset parameters
        if dataset_config.num_train_trajectories <= 0:
            raise ValueError(
                f"Parameter num_train_trajectories must be positive, "
                f"got {dataset_config.num_train_trajectories}"
            )
        
        if dataset_config.num_val_trajectories <= 0:
            raise ValueError(
                f"Parameter num_val_trajectories must be positive, "
                f"got {dataset_config.num_val_trajectories}"
            )
        
        if dataset_config.input_horizon <= 0:
            raise ValueError(
                f"Parameter input_horizon must be positive, "
                f"got {dataset_config.input_horizon}"
            )
        
        if dataset_config.output_horizon <= 0:
            raise ValueError(
                f"Parameter output_horizon must be positive, "
                f"got {dataset_config.output_horizon}"
            )
        
        if not 0.0 < dataset_config.train_val_split < 1.0:
            raise ValueError(
                f"Parameter train_val_split must be in (0, 1), "
                f"got {dataset_config.train_val_split}"
            )
        
        if dataset_config.batch_size <= 0:
            raise ValueError(
                f"Parameter batch_size must be positive, "
                f"got {dataset_config.batch_size}"
            )
    
    @staticmethod
    def _validate_logistic_params(params: Dict[str, Any]) -> None:
        """Validate Logistic Map parameters."""
        if "r" in params:
            r_min, r_max = ConfigValidator.LOGISTIC_PARAM_RANGES["r"]
            if not r_min <= params["r"] <= r_max:
                raise ValueError(
                    f"Parameter r must be in [{r_min}, {r_max}], got {params['r']}"
                )
        
        if "trajectory_length" in params:
            t_min, t_max = ConfigValidator.LOGISTIC_PARAM_RANGES["trajectory_length"]
            if not t_min <= params["trajectory_length"] <= t_max:
                raise ValueError(
                    f"Parameter trajectory_length must be in [{t_min}, {t_max}], "
                    f"got {params['trajectory_length']}"
                )
    
    @staticmethod
    def _validate_lorenz_params(params: Dict[str, Any]) -> None:
        """Validate Lorenz System parameters."""
        for param_name, (min_val, max_val) in ConfigValidator.LORENZ_PARAM_RANGES.items():
            if param_name in params:
                value = params[param_name]
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"Parameter {param_name} must be in [{min_val}, {max_val}], "
                        f"got {value}"
                    )
    
    @staticmethod
    def _validate_burgers_params(params: Dict[str, Any]) -> None:
        """Validate Burgers Equation parameters."""
        for param_name, (min_val, max_val) in ConfigValidator.BURGERS_PARAM_RANGES.items():
            if param_name in params:
                value = params[param_name]
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"Parameter {param_name} must be in [{min_val}, {max_val}], "
                        f"got {value}"
                    )
    
    @staticmethod
    def _validate_model(model_config) -> None:
        """Validate model configuration."""
        if model_config.type not in ConfigValidator.VALID_MODEL_TYPES:
            raise ValueError(
                f"Parameter type in section model must be one of "
                f"{ConfigValidator.VALID_MODEL_TYPES}, got {model_config.type}"
            )
        
        # Validate model-specific parameters
        if model_config.type == "deeponet":
            ConfigValidator._validate_deeponet_params(model_config.params)
        elif model_config.type == "fno":
            ConfigValidator._validate_fno_params(model_config.params)
    
    @staticmethod
    def _validate_deeponet_params(params: Dict[str, Any]) -> None:
        """Validate DeepONet parameters."""
        if "branch_layers" in params:
            if not isinstance(params["branch_layers"], list) or not params["branch_layers"]:
                raise ValueError("Parameter branch_layers must be a non-empty list")
            if any(layer <= 0 for layer in params["branch_layers"]):
                raise ValueError("All branch_layers must be positive")
        
        if "trunk_layers" in params:
            if not isinstance(params["trunk_layers"], list) or not params["trunk_layers"]:
                raise ValueError("Parameter trunk_layers must be a non-empty list")
            if any(layer <= 0 for layer in params["trunk_layers"]):
                raise ValueError("All trunk_layers must be positive")
        
        if "basis_dim" in params and params["basis_dim"] <= 0:
            raise ValueError(f"Parameter basis_dim must be positive, got {params['basis_dim']}")
        
        if "activation" in params and params["activation"] not in ConfigValidator.VALID_ACTIVATIONS:
            raise ValueError(
                f"Parameter activation must be one of {ConfigValidator.VALID_ACTIVATIONS}, "
                f"got {params['activation']}"
            )
    
    @staticmethod
    def _validate_fno_params(params: Dict[str, Any]) -> None:
        """Validate FNO parameters."""
        if "modes" in params and params["modes"] <= 0:
            raise ValueError(f"Parameter modes must be positive, got {params['modes']}")
        
        if "width" in params and params["width"] <= 0:
            raise ValueError(f"Parameter width must be positive, got {params['width']}")
        
        if "num_layers" in params and params["num_layers"] <= 0:
            raise ValueError(f"Parameter num_layers must be positive, got {params['num_layers']}")
        
        if "activation" in params and params["activation"] not in ConfigValidator.VALID_ACTIVATIONS:
            raise ValueError(
                f"Parameter activation must be one of {ConfigValidator.VALID_ACTIVATIONS}, "
                f"got {params['activation']}"
            )
    
    @staticmethod
    def _validate_optimizer(opt_config) -> None:
        """Validate optimizer configuration."""
        if opt_config.type not in ConfigValidator.VALID_OPTIMIZER_TYPES:
            raise ValueError(
                f"Parameter type in section optimizer must be one of "
                f"{ConfigValidator.VALID_OPTIMIZER_TYPES}, got {opt_config.type}"
            )
        
        params = opt_config.params
        
        # Validate learning rate (common to all optimizers)
        if "learning_rate" in params:
            if params["learning_rate"] <= 0:
                raise ValueError(
                    f"Parameter learning_rate must be positive, got {params['learning_rate']}"
                )
        
        # Validate optimizer-specific parameters
        if opt_config.type == "sgd":
            if "momentum" in params and not 0 <= params["momentum"] < 1:
                raise ValueError(
                    f"Parameter momentum must be in [0, 1), got {params['momentum']}"
                )
        
        elif opt_config.type == "adam":
            if "beta1" in params and not 0 <= params["beta1"] < 1:
                raise ValueError(
                    f"Parameter beta1 must be in [0, 1), got {params['beta1']}"
                )
            if "beta2" in params and not 0 <= params["beta2"] < 1:
                raise ValueError(
                    f"Parameter beta2 must be in [0, 1), got {params['beta2']}"
                )
            if "epsilon" in params and params["epsilon"] <= 0:
                raise ValueError(
                    f"Parameter epsilon must be positive, got {params['epsilon']}"
                )
        
        elif opt_config.type == "svrg":
            if "inner_loop_length" in params and params["inner_loop_length"] <= 0:
                raise ValueError(
                    f"Parameter inner_loop_length must be positive, "
                    f"got {params['inner_loop_length']}"
                )
        
        # Validate weight decay (common to all)
        if "weight_decay" in params and params["weight_decay"] < 0:
            raise ValueError(
                f"Parameter weight_decay must be non-negative, got {params['weight_decay']}"
            )
    
    @staticmethod
    def _validate_scheduler(sched_config) -> None:
        """Validate scheduler configuration."""
        if sched_config.type not in ConfigValidator.VALID_SCHEDULER_TYPES:
            raise ValueError(
                f"Parameter type in section scheduler must be one of "
                f"{ConfigValidator.VALID_SCHEDULER_TYPES}, got {sched_config.type}"
            )
    
    @staticmethod
    def _validate_training(train_config) -> None:
        """Validate training configuration."""
        if train_config.num_epochs <= 0:
            raise ValueError(
                f"Parameter num_epochs must be positive, got {train_config.num_epochs}"
            )
        
        if train_config.batch_size <= 0:
            raise ValueError(
                f"Parameter batch_size must be positive, got {train_config.batch_size}"
            )
        
        if train_config.validation_interval <= 0:
            raise ValueError(
                f"Parameter validation_interval must be positive, "
                f"got {train_config.validation_interval}"
            )
        
        if train_config.variance_interval <= 0:
            raise ValueError(
                f"Parameter variance_interval must be positive, "
                f"got {train_config.variance_interval}"
            )
        
        if train_config.checkpoint_interval <= 0:
            raise ValueError(
                f"Parameter checkpoint_interval must be positive, "
                f"got {train_config.checkpoint_interval}"
            )
        
        if train_config.early_stopping_patience is not None:
            if train_config.early_stopping_patience <= 0:
                raise ValueError(
                    f"Parameter early_stopping_patience must be positive, "
                    f"got {train_config.early_stopping_patience}"
                )
        
        if train_config.target_loss is not None:
            if train_config.target_loss <= 0:
                raise ValueError(
                    f"Parameter target_loss must be positive, "
                    f"got {train_config.target_loss}"
                )
    
    @staticmethod
    def _validate_logging(log_config) -> None:
        """Validate logging configuration."""
        if not log_config.log_dir:
            raise ValueError("Missing required parameter: log_dir in section logging")
        
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_config.log_level not in valid_log_levels:
            raise ValueError(
                f"Parameter log_level must be one of {valid_log_levels}, "
                f"got {log_config.log_level}"
            )
    
    @staticmethod
    def _validate_analysis(analysis_config) -> None:
        """Validate analysis configuration."""
        if analysis_config.spectral_interval <= 0:
            raise ValueError(
                f"Parameter spectral_interval must be positive, "
                f"got {analysis_config.spectral_interval}"
            )
        
        if analysis_config.long_horizon_steps <= 0:
            raise ValueError(
                f"Parameter long_horizon_steps must be positive, "
                f"got {analysis_config.long_horizon_steps}"
            )
        
        if analysis_config.num_eigenvalues <= 0:
            raise ValueError(
                f"Parameter num_eigenvalues must be positive, "
                f"got {analysis_config.num_eigenvalues}"
            )
    
    @staticmethod
    def _validate_visualization(vis_config) -> None:
        """Validate visualization configuration."""
        valid_formats = {"pdf", "png", "svg", "eps"}
        if vis_config.plot_format not in valid_formats:
            raise ValueError(
                f"Parameter plot_format must be one of {valid_formats}, "
                f"got {vis_config.plot_format}"
            )
        
        if vis_config.dpi <= 0:
            raise ValueError(f"Parameter dpi must be positive, got {vis_config.dpi}")
        
        if not vis_config.output_dir:
            raise ValueError("Missing required parameter: output_dir in section visualization")
    
    @staticmethod
    def _validate_compatibility(config: Config) -> None:
        """Validate cross-component compatibility."""
        # Ensure batch size is consistent
        if config.dataset.batch_size != config.training.batch_size:
            raise ValueError(
                f"Batch size mismatch: dataset.batch_size={config.dataset.batch_size} "
                f"but training.batch_size={config.training.batch_size}"
            )
        
        # Warn if SVRG inner loop length is too large
        if config.optimizer.type == "svrg":
            inner_loop = config.optimizer.params.get("inner_loop_length", 100)
            total_batches = config.dataset.num_train_trajectories // config.training.batch_size
            if inner_loop > total_batches:
                raise ValueError(
                    f"SVRG inner_loop_length ({inner_loop}) exceeds number of training "
                    f"batches ({total_batches}). Consider reducing inner_loop_length."
                )
