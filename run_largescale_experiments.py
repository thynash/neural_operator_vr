#!/usr/bin/env python3
"""
Run Large-Scale Experiments to Validate Abstract Claims

This script runs the experiments needed to demonstrate that SVRG
outperforms SGD and Adam on large-scale neural operator training.
"""

import sys
import subprocess
import time
from pathlib import Path

def run_experiment(config_path, description):
    """Run a single large-scale experiment."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "examples/run_experiment.py", config_path],
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[FAILED] Failed after {elapsed/60:.1f} minutes")
        return False, elapsed
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment interrupted by user")
        sys.exit(1)


def main():
    """Run all large-scale experiments."""
    print("="*80)
    print("LARGE-SCALE EXPERIMENTS FOR ABSTRACT VALIDATION")
    print("="*80)
    print()
    print("These experiments will:")
    print("  - Use 2000 training trajectories (vs 20 in tests)")
    print("  - Train for 100 epochs (vs 2 in tests)")
    print("  - Use larger models (17K parameters vs 3K)")
    print("  - Run SVRG with proper inner loop (50 vs 3)")
    print()
    print("Expected runtime: 4-6 hours total")
    print("  - Each experiment: 60-90 minutes")
    print("  - 3 optimizers: Adam, SGD, SVRG")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    experiments = [
        ("examples/config_largescale_deeponet_logistic_adam.yaml", "DeepONet + Logistic Map + Adam"),
        ("examples/config_largescale_deeponet_logistic_sgd.yaml", "DeepONet + Logistic Map + SGD"),
        ("examples/config_largescale_deeponet_logistic_svrg.yaml", "DeepONet + Logistic Map + SVRG"),
    ]
    
    results = []
    total_start = time.time()
    
    for config, desc in experiments:
        success, elapsed = run_experiment(config, desc)
        results.append((desc, success, elapsed))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUITE SUMMARY")
    print("="*80)
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print()
    
    for desc, success, elapsed in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {desc:50s} {elapsed/60:6.1f} min")
    
    success_count = sum(1 for _, s, _ in results if s)
    print(f"\nCompleted: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n[SUCCESS] All experiments completed!")
        print("\nNext steps:")
        print("  1. Check logs/ for training histories")
        print("  2. Run: python generate_publication_results.py")
        print("  3. Check visualization_output/ for plots")
    else:
        print("\n[WARNING] Some experiments failed. Check logs for details.")


if __name__ == "__main__":
    main()
