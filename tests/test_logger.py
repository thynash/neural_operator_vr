"""Tests for metrics logging functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from utils.logger import MetricsLogger


class TestMetricsLogger:
    """Test suite for MetricsLogger class."""
    
    def test_log_scalar_with_timestamp(self):
        """Test that log_scalar logs metrics with timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            # Log some metrics
            logger.log_scalar("train_loss", 0.5, step=0)
            logger.log_scalar("train_loss", 0.3, step=1)
            
            # Check history
            assert "train_loss" in logger.history
            assert len(logger.history["train_loss"]) == 2
            assert logger.history["train_loss"][0] == (0, 0.5)
            assert logger.history["train_loss"][1] == (1, 0.3)
            
            # Check timestamps
            assert "train_loss" in logger.timestamps
            assert len(logger.timestamps["train_loss"]) == 2
    
    def test_log_dict(self):
        """Test that log_dict logs multiple metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            metrics = {
                "train_loss": 0.5,
                "train_grad_norm": 1.2,
                "val_loss": 0.4
            }
            logger.log_dict(metrics, step=0)
            
            assert len(logger.history) == 3
            assert "train_loss" in logger.history
            assert "train_grad_norm" in logger.history
            assert "val_loss" in logger.history
    
    def test_structured_log_file(self):
        """Test that metrics are written to structured log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_scalar("train_loss", 0.5, step=0)
            
            # Check log file exists
            log_file = Path(tmpdir) / "test_exp_metrics.log"
            assert log_file.exists()
            
            # Check log file content
            with open(log_file, 'r') as f:
                line = f.readline()
                entry = json.loads(line)
                assert entry["metric"] == "train_loss"
                assert entry["value"] == 0.5
                assert entry["step"] == 0
                assert "timestamp" in entry
    
    def test_save_results(self):
        """Test saving complete results structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            # Log some metrics
            logger.log_scalar("train_loss", 0.5, step=0)
            logger.log_scalar("val_loss", 0.4, step=0)
            
            # Save results with complete structure
            config = {"model": "deeponet", "optimizer": "svrg"}
            system_info = {"python_version": "3.9.0"}
            convergence = {"iterations_to_target": 1000}
            eigenvalues = {"true": [[1.0, 0.0]], "learned": [[0.98, 0.02]]}
            
            filepath = logger.save_results(
                config=config,
                system_info=system_info,
                convergence=convergence,
                eigenvalues=eigenvalues
            )
            
            # Check file exists
            assert filepath.exists()
            
            # Load and verify structure
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            assert "config" in results
            assert "system_info" in results
            assert "metrics" in results
            assert "convergence" in results
            assert "eigenvalues" in results
            
            assert results["config"] == config
            assert results["system_info"] == system_info
            assert results["convergence"] == convergence
            assert results["eigenvalues"] == eigenvalues
            
            assert "train_loss" in results["metrics"]
            assert "val_loss" in results["metrics"]
    
    def test_load_results(self):
        """Test loading complete results structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            # Log and save
            logger.log_scalar("train_loss", 0.5, step=0)
            config = {"model": "deeponet"}
            system_info = {"python_version": "3.9.0"}
            convergence = {"iterations_to_target": 1000}
            eigenvalues = {"true": [[1.0, 0.0]]}
            
            filepath = logger.save_results(
                config=config,
                system_info=system_info,
                convergence=convergence,
                eigenvalues=eigenvalues
            )
            
            # Load results
            logger2 = MetricsLogger(tmpdir, "test_exp2")
            results = logger2.load_results(filepath)
            
            assert results["config"] == config
            assert results["system_info"] == system_info
            assert results["convergence"] == convergence
            assert results["eigenvalues"] == eigenvalues
            assert "train_loss" in results["metrics"]
            
            # Check that history was updated
            assert "train_loss" in logger2.history
    
    def test_load_results_missing_file(self):
        """Test that load_results raises FileNotFoundError for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            with pytest.raises(FileNotFoundError):
                logger.load_results(Path(tmpdir) / "nonexistent.json")
    
    def test_load_results_invalid_structure(self):
        """Test that load_results raises KeyError for invalid structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid results file
            invalid_file = Path(tmpdir) / "invalid.json"
            with open(invalid_file, 'w') as f:
                json.dump({"config": {}, "metrics": {}}, f)  # Missing required fields
            
            logger = MetricsLogger(tmpdir, "test_exp")
            
            with pytest.raises(KeyError):
                logger.load_results(invalid_file)
    
    def test_save_history_backward_compatibility(self):
        """Test that save_history still works for backward compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_scalar("train_loss", 0.5, step=0)
            logger.log_scalar("train_loss", 0.3, step=1)
            
            filepath = logger.save_history()
            
            assert filepath.exists()
            
            # Load and verify
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            assert "train_loss" in history
            assert history["train_loss"] == [[0, 0.5], [1, 0.3]]
