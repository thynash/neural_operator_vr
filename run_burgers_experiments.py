#!/usr/bin/env python3
"""
Launch large-scale Burgers equation experiments with Adam, SGD, and SVRG.

This script runs all three optimizers on a scaled-up Burgers equation dataset
to validate abstract claims about SVRG performance.
"""

import subprocess
import sys
from pathlib import Path

# Experiment configurations
CONFIGS = [
    "examples/config_largescale_fno_burgers_adam.yaml",
    "examples/config_largescale_fno_burgers_sgd.yaml",
    "examples/config_largescale_fno_burgers_svrg.yaml",
]

def run_experiment(config_path):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {config_path}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "examples/run_experiment.py", config_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Experiment completed: {config_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: {config_path}")
        print(f"Error: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠ Experiment interrupted by user: {config_path}\n")
        return False

def main():
    """Run all experiments sequentially."""
    print("\n" + "="*80)
    print("LARGE-SCALE BURGERS EQUATION EXPERIMENTS")
    print("="*80)
    print("\nThis will run three experiments:")
    print("1. Adam optimizer")
    print("2. SGD optimizer")
    print("3. SVRG optimizer")
    print("\nDataset: 5000 training trajectories, 256 spatial points")
    print("Model: FNO with 4 layers, 64 width")
    print("Training: 50 epochs, batch size 128")
    print("\nEstimated time: 12-24 hours total (4-8 hours each)")
    print("="*80)
    
    # Confirm before starting
    response = input("\nStart experiments? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Experiments cancelled.")
        return
    
    results = {}
    for config in CONFIGS:
        config_path = Path(config)
        if not config_path.exists():
            print(f"\n✗ Config file not found: {config}")
            results[config] = False
            continue
        
        success = run_experiment(config)
        results[config] = success
        
        if not success:
            print(f"\n⚠ Stopping experiments due to failure in {config}")
            break
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for config, success in results.items():
        status = "✓ Complete" if success else "✗ Failed/Skipped"
        print(f"{status}: {config}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✓ All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Run: python generate_burgers_results.py")
        print("2. Review: burgers_results/BURGERS_SUMMARY.md")
        print("3. Compare with logistic map results")
    else:
        print("\n⚠ Some experiments failed or were skipped.")
        print("Check the output above for details.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
