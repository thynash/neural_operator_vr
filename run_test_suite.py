#!/usr/bin/env python3
"""
Quick Test Suite Runner

Runs fast experiments to validate the complete framework.
"""

import sys
import subprocess
from pathlib import Path


def run_test_experiment(config_path):
    """Run a single test experiment."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_path}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "examples/run_experiment.py", config_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("[OK] Test passed")
            return True
        else:
            print(f"[FAIL] Test failed")
            print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Test timed out")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False


def main():
    """Run test suite."""
    print("\n" + "="*80)
    print("NEURAL OPERATOR FRAMEWORK - QUICK TEST SUITE")
    print("="*80)
    
    test_configs = [
        "examples/config_test_deeponet_logistic_adam.yaml",
        "examples/config_test_deeponet_logistic_sgd.yaml",
        "examples/config_test_deeponet_logistic_svrg.yaml",
    ]
    
    results = []
    for config in test_configs:
        passed = run_test_experiment(config)
        results.append((config, passed))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}\n")
    
    for config, p in results:
        status = "[OK]" if p else "[FAIL]"
        print(f"{status} {Path(config).stem}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILED] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
