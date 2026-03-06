#!/usr/bin/env python3
"""
Fast confirmation experiment on Lorenz system (chaotic dynamics).
Optimized to complete in ~2-3 hours total.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

CONFIGS = [
    "examples/config_fast_deeponet_lorenz_adam.yaml",
    "examples/config_fast_deeponet_lorenz_sgd.yaml",
    "examples/config_fast_deeponet_lorenz_svrg.yaml",
]

def run_experiment(config_path):
    """Run a single experiment."""
    name = Path(config_path).stem
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "examples/run_experiment.py", config_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed: {name}")
        print(f"Time: {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed: {name}")
        print(f"Time: {elapsed/60:.1f} minutes")
        return False, elapsed
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted: {name}\n")
        return False, 0

def main():
    print("\n" + "="*80)
    print("FAST CONFIRMATION EXPERIMENT - LORENZ SYSTEM")
    print("="*80)
    print("\nConfiguration:")
    print("- Dataset: Lorenz attractor (chaotic dynamics)")
    print("- Trajectories: 1000 train, 200 val")
    print("- Model: DeepONet (smaller, 64-64-32)")
    print("- Training: 20 epochs max, early stopping")
    print("- Batch size: 64 (larger for speed)")
    print("- No checkpoints (speed optimization)")
    print("\nEstimated time: 2-3 hours total (~40-60 min each)")
    print("Expected completion: Before 5 PM")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%H:%M:%S')}")
    
    response = input("\nStart experiments? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    results = {}
    total_time = 0
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/3")
        print(f"{'#'*80}")
        
        success, elapsed = run_experiment(config)
        results[config] = success
        total_time += elapsed
        
        if not success:
            print(f"\n⚠ Stopping due to failure")
            break
        
        # Estimate remaining time
        if i < len(CONFIGS):
            avg_time = total_time / i
            remaining = avg_time * (len(CONFIGS) - i)
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"\nEstimated completion: {eta.strftime('%H:%M:%S')}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for config, success in results.items():
        name = Path(config).stem.replace('config_fast_deeponet_lorenz_', '').upper()
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\nTotal time: {total_duration:.1f} minutes")
    print(f"Completed: {end_time.strftime('%H:%M:%S')}")
    
    if all(results.values()):
        print("\n✓ All experiments completed!")
        print("\nNext step:")
        print("  python generate_comprehensive_analysis.py")
        print("\nThis will update the analysis with Lorenz results.")
    else:
        print("\n⚠ Some experiments failed.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
