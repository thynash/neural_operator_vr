#!/usr/bin/env python3
"""Monitor the progress of large-scale experiments."""

import json
import time
from pathlib import Path
from datetime import datetime

def check_experiment_progress(exp_name):
    """Check progress of a single experiment."""
    log_dir = Path(f"logs/{exp_name}")
    history_file = log_dir / "training_history.json"
    
    if not history_file.exists():
        return None, "Not started"
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        total_iters = history.get('total_iterations', 0)
        train_losses = history.get('history', {}).get('train_loss', [])
        
        if train_losses:
            current_iter = train_losses[-1][0]
            current_loss = train_losses[-1][1]
            return current_iter, f"Loss: {current_loss:.6f}"
        else:
            return 0, "Starting..."
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    """Monitor all experiments."""
    experiments = [
        "largescale_deeponet_logistic_adam",
        "largescale_deeponet_logistic_sgd",
        "largescale_deeponet_logistic_svrg",
    ]
    
    print("="*80)
    print("EXPERIMENT PROGRESS MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for exp in experiments:
        iter_num, status = check_experiment_progress(exp)
        
        if iter_num is not None:
            progress = f"Iteration {iter_num:5d}"
        else:
            progress = "Not started"
        
        print(f"{exp:40s} {progress:20s} {status}")
    
    print()
    print("Refresh this script to see updated progress.")
    print("Run: python monitor_experiments.py")

if __name__ == "__main__":
    main()
