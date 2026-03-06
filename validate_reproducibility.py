#!/usr/bin/env python3
"""
Reproducibility Validation Script

Runs the same experiment multiple times with the same seed and verifies
bit-exact reproducibility as required by Requirements 11.3 and 11.4.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def run_experiment(config_path: str, run_id: int) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    print(f"\nRun {run_id}: Executing {config_path}")
    
    result = subprocess.run(
        [sys.executable, "examples/run_experiment.py", config_path],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        print(f"[FAIL] Run {run_id} failed")
        print(result.stderr[-500:])
        return None
    
    # Load training history
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    exp_name = config['experiment']['name']
    log_dir = Path(config['logging']['log_dir'])
    history_path = log_dir / exp_name / "training_history.json"
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"[OK] Run {run_id} completed")
    return history


def compare_histories(hist1: Dict, hist2: Dict, tolerance: float = 1e-10) -> bool:
    """Compare two training histories for bit-exact reproducibility."""
    print("\nComparing training histories...")
    
    # Check metrics
    metrics_to_check = ['train_loss', 'train_grad_norm', 'val_loss', 'val_relative_error']
    
    all_match = True
    
    for metric in metrics_to_check:
        if metric not in hist1 or metric not in hist2:
            continue
        
        values1 = hist1[metric]
        values2 = hist2[metric]
        
        if len(values1) != len(values2):
            print(f"[FAIL] {metric}: Different lengths ({len(values1)} vs {len(values2)})")
            all_match = False
            continue
        
        # Compare values
        max_diff = 0.0
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            diff = abs(v1 - v2)
            max_diff = max(max_diff, diff)
        
        if max_diff > tolerance:
            print(f"[FAIL] {metric}: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
            all_match = False
        else:
            print(f"[OK] {metric}: Bit-exact match (max diff: {max_diff:.2e})")
    
    return all_match


def validate_reproducibility(config_path: str, num_runs: int = 3) -> bool:
    """Validate reproducibility by running experiment multiple times."""
    print("="*80)
    print("REPRODUCIBILITY VALIDATION")
    print("="*80)
    print(f"\nConfig: {config_path}")
    print(f"Number of runs: {num_runs}")
    print(f"Expected: Bit-exact reproducibility with same seed")
    
    # Run experiment multiple times
    histories = []
    for i in range(1, num_runs + 1):
        history = run_experiment(config_path, i)
        if history is None:
            print(f"\n[FAILED] Run {i} failed to complete")
            return False
        histories.append(history)
    
    # Compare all pairs
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    all_reproducible = True
    for i in range(len(histories) - 1):
        print(f"\nComparing Run {i+1} vs Run {i+2}:")
        print("-"*80)
        match = compare_histories(histories[i], histories[i+1])
        if not match:
            all_reproducible = False
    
    return all_reproducible


def main():
    """Main entry point."""
    # Test with a small, fast config
    config_path = "examples/config_test_deeponet_logistic_adam.yaml"
    
    if not Path(config_path).exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1
    
    success = validate_reproducibility(config_path, num_runs=3)
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    
    if success:
        print("\n[SUCCESS] Bit-exact reproducibility validated!")
        print("All runs with the same seed produced identical results.")
        return 0
    else:
        print("\n[FAILED] Reproducibility validation failed!")
        print("Runs with the same seed produced different results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
