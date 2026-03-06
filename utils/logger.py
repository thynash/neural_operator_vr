"""Metrics logging and training history management."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional


class MetricsLogger:
    """
    Logger for training metrics and history management.
    
    Tracks scalar metrics during training and provides serialization
    to JSON format for persistence and analysis.
    
    Parameters
    ----------
    log_dir : str or Path
        Directory where logs and history will be saved.
    experiment_name : str
        Name of the experiment for organizing logs.
    
    Attributes
    ----------
    log_dir : Path
        Directory for log storage.
    experiment_name : str
        Experiment identifier.
    history : Dict[str, List[Tuple[int, float]]]
        Dictionary mapping metric names to lists of (step, value) tuples.
    
    Examples
    --------
    >>> logger = MetricsLogger("./logs", "experiment_1")
    >>> logger.log_scalar("train_loss", 0.5, step=0)
    >>> logger.log_dict({"val_loss": 0.3, "val_acc": 0.9}, step=100)
    >>> logger.save_history()
    """
    
    def __init__(self, log_dir: Union[str, Path], experiment_name: str):
        """Initialize logger with output directory and experiment name."""
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.history: Dict[str, List[Tuple[int, float]]] = {}
        self.timestamps: Dict[str, List[Tuple[int, str]]] = {}  # Track timestamps for each metric
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create structured log file
        self.log_file = self.log_dir / f"{experiment_name}_metrics.log"
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """
        Log a scalar metric value with timestamp.
        
        Parameters
        ----------
        name : str
            Name of the metric (e.g., "train_loss", "val_accuracy").
        value : float
            Metric value to log.
        step : int
            Training step or iteration number.
        
        Examples
        --------
        >>> logger.log_scalar("train_loss", 0.5, step=100)
        """
        if name not in self.history:
            self.history[name] = []
            self.timestamps[name] = []
        
        timestamp = datetime.now().isoformat()
        self.history[name].append((step, float(value)))
        self.timestamps[name].append((step, timestamp))
        
        # Write to structured log file
        self._write_to_log(name, value, step, timestamp)
    
    def log_dict(self, metrics_dict: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics from a dictionary.
        
        Parameters
        ----------
        metrics_dict : Dict[str, float]
            Dictionary mapping metric names to values.
        step : int
            Training step or iteration number.
        
        Examples
        --------
        >>> metrics = {"train_loss": 0.5, "train_grad_norm": 1.2}
        >>> logger.log_dict(metrics, step=100)
        """
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step)
    
    def _write_to_log(self, name: str, value: float, step: int, timestamp: str) -> None:
        """
        Write a metric entry to the structured log file.
        
        Parameters
        ----------
        name : str
            Metric name.
        value : float
            Metric value.
        step : int
            Training step.
        timestamp : str
            ISO format timestamp.
        """
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "metric": name,
            "value": value
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_history(self, filename: str = None) -> Path:
        """
        Save complete training history to JSON file.
        
        Parameters
        ----------
        filename : str, optional
            Name of the output file. If None, uses "{experiment_name}_history.json".
        
        Returns
        -------
        Path
            Path to the saved history file.
        
        Examples
        --------
        >>> logger.save_history()
        >>> logger.save_history("custom_history.json")
        """
        if filename is None:
            filename = f"{self.experiment_name}_history.json"
        
        filepath = self.log_dir / filename
        
        # Convert history to JSON-serializable format
        serializable_history = {
            name: [[step, value] for step, value in values]
            for name, values in self.history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        return filepath
    
    def save_results(
        self,
        config: Optional[Dict[str, Any]] = None,
        system_info: Optional[Dict[str, Any]] = None,
        convergence: Optional[Dict[str, Any]] = None,
        eigenvalues: Optional[Dict[str, Any]] = None,
        filename: str = None
    ) -> Path:
        """
        Save complete training results with full structure.
        
        This method saves training history in the complete format specified
        in the design document, including config, system_info, metrics,
        convergence, and eigenvalues.
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Complete experiment configuration.
        system_info : Dict[str, Any], optional
            System information (Python version, PyTorch version, GPU info, etc.).
        convergence : Dict[str, Any], optional
            Convergence metrics (iterations_to_target, time_to_target, etc.).
        eigenvalues : Dict[str, Any], optional
            Eigenvalue data (true and learned eigenvalues).
        filename : str, optional
            Name of the output file. If None, uses "{experiment_name}_results.json".
        
        Returns
        -------
        Path
            Path to the saved results file.
        
        Examples
        --------
        >>> logger.save_results(
        ...     config={"model": "deeponet", "optimizer": "svrg"},
        ...     system_info={"python_version": "3.9.0"},
        ...     convergence={"iterations_to_target": 1000},
        ...     eigenvalues={"true": [[1.0, 0.0]], "learned": [[0.98, 0.02]]}
        ... )
        """
        if filename is None:
            filename = f"{self.experiment_name}_results.json"
        
        filepath = self.log_dir / filename
        
        # Convert history to JSON-serializable format with timestamps
        metrics = {}
        for name, values in self.history.items():
            metrics[name] = [[step, value] for step, value in values]
        
        # Build complete structure
        results = {
            "config": config or {},
            "system_info": system_info or {},
            "metrics": metrics,
            "convergence": convergence or {},
            "eigenvalues": eigenvalues or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath
    
    def load_history(self, path: Union[str, Path]) -> Dict[str, List[Tuple[int, float]]]:
        """
        Load training history from JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to the history JSON file.
        
        Returns
        -------
        Dict[str, List[Tuple[int, float]]]
            Dictionary mapping metric names to lists of (step, value) tuples.
        
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        
        Examples
        --------
        >>> history = logger.load_history("./logs/experiment_1_history.json")
        >>> train_loss = history["train_loss"]
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {path}")
        
        with open(path, 'r') as f:
            serializable_history = json.load(f)
        
        # Convert back to tuple format
        history = {
            name: [(step, value) for step, value in values]
            for name, values in serializable_history.items()
        }
        
        # Update internal history
        self.history = history
        
        return history
    
    def load_results(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load complete training results for post-hoc analysis.
        
        This method loads the complete results structure saved by save_results(),
        including config, system_info, metrics, convergence, and eigenvalues.
        
        Parameters
        ----------
        path : str or Path
            Path to the results JSON file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - config: Experiment configuration
            - system_info: System information
            - metrics: Training metrics history
            - convergence: Convergence metrics
            - eigenvalues: Eigenvalue data
        
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        KeyError
            If the file is missing required structure fields.
        
        Examples
        --------
        >>> results = logger.load_results("./logs/experiment_1_results.json")
        >>> config = results["config"]
        >>> metrics = results["metrics"]
        >>> train_loss = metrics["train_loss"]
        >>> convergence = results["convergence"]
        >>> iterations_to_target = convergence["iterations_to_target"]
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        
        with open(path, 'r') as f:
            results = json.load(f)
        
        # Validate structure
        required_fields = ["config", "system_info", "metrics", "convergence", "eigenvalues"]
        missing_fields = [field for field in required_fields if field not in results]
        if missing_fields:
            raise KeyError(f"Results file missing required fields: {missing_fields}")
        
        # Convert metrics back to tuple format for internal use
        if results["metrics"]:
            self.history = {
                name: [(step, value) for step, value in values]
                for name, values in results["metrics"].items()
            }
        
        return results
    
    def get_metric_history(self, name: str) -> List[Tuple[int, float]]:
        """
        Get history for a specific metric.
        
        Parameters
        ----------
        name : str
            Name of the metric.
        
        Returns
        -------
        List[Tuple[int, float]]
            List of (step, value) tuples for the metric.
        
        Raises
        ------
        KeyError
            If the metric name is not found in history.
        
        Examples
        --------
        >>> train_loss_history = logger.get_metric_history("train_loss")
        """
        if name not in self.history:
            raise KeyError(f"Metric '{name}' not found in history")
        return self.history[name]
    
    def get_latest_value(self, name: str) -> float:
        """
        Get the most recent value for a metric.
        
        Parameters
        ----------
        name : str
            Name of the metric.
        
        Returns
        -------
        float
            Most recent value logged for the metric.
        
        Raises
        ------
        KeyError
            If the metric name is not found in history.
        ValueError
            If no values have been logged for the metric.
        
        Examples
        --------
        >>> latest_loss = logger.get_latest_value("train_loss")
        """
        if name not in self.history:
            raise KeyError(f"Metric '{name}' not found in history")
        if not self.history[name]:
            raise ValueError(f"No values logged for metric '{name}'")
        return self.history[name][-1][1]
