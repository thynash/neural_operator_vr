"""
Demonstration of the enhanced logging infrastructure.

This script demonstrates the new logging capabilities including:
1. Timestamped metrics logging
2. Structured log files
3. Complete results serialization with config, system_info, metrics, convergence, and eigenvalues
4. Results loading for post-hoc analysis
"""

import tempfile
from pathlib import Path

from utils.logger import MetricsLogger


def main():
    """Demonstrate logging infrastructure features."""
    
    # Create a temporary directory for this demo
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Demo log directory: {tmpdir}\n")
        
        # Initialize logger
        logger = MetricsLogger(tmpdir, "demo_experiment")
        print("✓ Initialized MetricsLogger")
        
        # 1. Log training metrics with timestamps
        print("\n1. Logging training metrics with timestamps...")
        for step in range(5):
            logger.log_scalar("train_loss", 1.0 / (step + 1), step=step)
            logger.log_scalar("train_grad_norm", 0.5 / (step + 1), step=step)
        
        logger.log_dict({
            "val_loss": 0.3,
            "val_relative_error": 0.05,
            "spectral_radius": 0.95
        }, step=4)
        
        print(f"   Logged {len(logger.history)} metrics")
        print(f"   Structured log file: {logger.log_file}")
        
        # 2. Save complete results structure
        print("\n2. Saving complete results structure...")
        
        config = {
            "model": {"type": "deeponet", "basis_dim": 64},
            "optimizer": {"type": "svrg", "learning_rate": 0.001},
            "dataset": {"type": "lorenz", "num_trajectories": 1000}
        }
        
        system_info = {
            "python_version": "3.9.0",
            "pytorch_version": "2.0.0",
            "cuda_available": True,
            "gpu_name": "NVIDIA RTX 3090"
        }
        
        convergence = {
            "iterations_to_target": 1000,
            "time_to_target": 45.2,
            "gradient_evals_to_target": 32000,
            "final_val_loss": 0.0012
        }
        
        eigenvalues = {
            "true": [[1.0, 0.0], [0.8, 0.2], [0.6, -0.1]],
            "learned": [[0.98, 0.02], [0.79, 0.21], [0.61, -0.09]]
        }
        
        results_path = logger.save_results(
            config=config,
            system_info=system_info,
            convergence=convergence,
            eigenvalues=eigenvalues
        )
        
        print(f"   Results saved to: {results_path}")
        
        # 3. Load results for post-hoc analysis
        print("\n3. Loading results for post-hoc analysis...")
        
        # Create a new logger instance to simulate loading in a different session
        analysis_logger = MetricsLogger(tmpdir, "analysis_session")
        results = analysis_logger.load_results(results_path)
        
        print(f"   Loaded results structure:")
        print(f"   - Config: {len(results['config'])} sections")
        print(f"   - System info: {len(results['system_info'])} fields")
        print(f"   - Metrics: {len(results['metrics'])} tracked")
        print(f"   - Convergence: {len(results['convergence'])} metrics")
        print(f"   - Eigenvalues: {len(results['eigenvalues']['true'])} true, "
              f"{len(results['eigenvalues']['learned'])} learned")
        
        # 4. Access specific metrics
        print("\n4. Accessing specific metrics...")
        
        train_loss_history = results["metrics"]["train_loss"]
        print(f"   Train loss history: {train_loss_history}")
        
        iterations_to_target = results["convergence"]["iterations_to_target"]
        print(f"   Iterations to target: {iterations_to_target}")
        
        model_type = results["config"]["model"]["type"]
        print(f"   Model type: {model_type}")
        
        # 5. Demonstrate backward compatibility
        print("\n5. Backward compatibility with save_history()...")
        
        history_path = logger.save_history()
        print(f"   History saved to: {history_path}")
        
        loaded_history = logger.load_history(history_path)
        print(f"   Loaded {len(loaded_history)} metrics from history file")
        
        print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()
