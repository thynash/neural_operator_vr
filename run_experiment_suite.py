#!/usr/bin/env python3
"""
Comprehensive Experiment Suite Runner

Executes all optimizer-system combinations and validates:
- All metrics computed correctly
- All visualizations generated
- Complete experiment workflow
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time


class ExperimentSuiteRunner:
    """Orchestrates running complete experiment suite."""
    
    def __init__(self):
        self.config_files = [
            "examples/config_deeponet_logistic_adam.yaml",
            "examples/config_deeponet_logistic_sgd.yaml",
            "examples/config_deeponet_logistic_svrg.yaml",
            "examples/config_deeponet_lorenz_svrg.yaml",
            "examples/config_fno_burgers_adam.yaml",
            "examples/config_fno_burgers_sgd.yaml",
            "examples/config_fno_burgers_svrg.yaml",
        ]
        self.results = []
        
    def run_single_experiment(self, config_path: str) -> Dict[str, Any]:
        """Run a single experiment and collect results."""
        print(f"\n{'='*80}")
        print(f"Running experiment: {config_path}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # Run experiment
            result = subprocess.run(
                [sys.executable, "examples/run_experiment.py", config_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"[SUCCESS] Experiment completed successfully in {elapsed_time:.2f}s")
                status = "SUCCESS"
                error = None
            else:
                print(f"[FAILED] Experiment failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                status = "FAILED"
                error = result.stderr
                
            return {
                "config": config_path,
                "status": status,
                "elapsed_time": elapsed_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": error
            }
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"[TIMEOUT] Experiment timed out after {elapsed_time:.2f}s")
            return {
                "config": config_path,
                "status": "TIMEOUT",
                "elapsed_time": elapsed_time,
                "error": "Experiment exceeded 1 hour timeout"
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[CRASHED] Experiment crashed: {str(e)}")
            return {
                "config": config_path,
                "status": "CRASHED",
                "elapsed_time": elapsed_time,
                "error": str(e)
            }
    
    def verify_outputs(self, config_path: str) -> Dict[str, bool]:
        """Verify all expected outputs were generated."""
        # Extract experiment name from config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exp_name = config['experiment']['name']
        log_dir = Path(config['logging']['log_dir'])
        viz_dir = Path(config['visualization']['output_dir'])
        
        checks = {}
        
        # Check training history
        history_path = log_dir / exp_name / "training_history.json"
        checks['training_history'] = history_path.exists()
        
        # Check final model
        model_path = log_dir / exp_name / "final_model.pt"
        checks['final_model'] = model_path.exists()
        
        # Check visualizations
        expected_plots = [
            'training_curves.pdf',
            'gradient_variance.pdf',
            'validation_error.pdf',
        ]
        
        for plot in expected_plots:
            plot_path = viz_dir / plot
            checks[f'plot_{plot}'] = plot_path.exists()
        
        # Verify metrics in training history
        if checks['training_history']:
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                required_metrics = [
                    'train_loss',
                    'train_grad_norm',
                    'val_loss',
                    'val_relative_error'
                ]
                
                for metric in required_metrics:
                    checks[f'metric_{metric}'] = metric in history.get('metrics', {})
                    
            except Exception as e:
                checks['history_valid'] = False
                print(f"  Warning: Could not validate history file: {e}")
        
        return checks
    
    def run_suite(self) -> bool:
        """Run complete experiment suite."""
        print("\n" + "="*80)
        print("NEURAL OPERATOR VARIANCE REDUCTION - EXPERIMENT SUITE")
        print("="*80)
        print(f"\nTotal experiments to run: {len(self.config_files)}")
        print("\nExperiments:")
        for i, config in enumerate(self.config_files, 1):
            print(f"  {i}. {config}")
        print()
        
        # Run all experiments
        for config_path in self.config_files:
            result = self.run_single_experiment(config_path)
            self.results.append(result)
            
            # Verify outputs if experiment succeeded
            if result['status'] == 'SUCCESS':
                print("\nVerifying outputs...")
                checks = self.verify_outputs(config_path)
                result['output_checks'] = checks
                
                all_passed = all(checks.values())
                if all_passed:
                    print("[OK] All output checks passed")
                else:
                    print("[WARNING] Some output checks failed:")
                    for check, passed in checks.items():
                        if not passed:
                            print(f"    - {check}: MISSING")
        
        # Generate summary report
        self.generate_report()
        
        # Return overall success
        all_success = all(r['status'] == 'SUCCESS' for r in self.results)
        return all_success
    
    def generate_report(self):
        """Generate summary report of experiment suite."""
        print("\n" + "="*80)
        print("EXPERIMENT SUITE SUMMARY")
        print("="*80 + "\n")
        
        total = len(self.results)
        success = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in self.results if r['status'] == 'FAILED')
        timeout = sum(1 for r in self.results if r['status'] == 'TIMEOUT')
        crashed = sum(1 for r in self.results if r['status'] == 'CRASHED')
        
        print(f"Total experiments: {total}")
        print(f"  [OK] Success: {success}")
        print(f"  [FAIL] Failed: {failed}")
        print(f"  [TIME] Timeout: {timeout}")
        print(f"  [CRASH] Crashed: {crashed}")
        print()
        
        # Detailed results
        print("Detailed Results:")
        print("-" * 80)
        for result in self.results:
            config_name = Path(result['config']).stem
            status_symbol = "[OK]" if result['status'] == 'SUCCESS' else "[FAIL]"
            print(f"{status_symbol} {config_name:40s} {result['status']:10s} {result['elapsed_time']:8.2f}s")
            
            if result['status'] != 'SUCCESS' and result.get('error'):
                print(f"    Error: {result['error'][:100]}")
        
        print()
        
        # Save detailed report
        report_path = Path("experiment_suite_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'success': success,
                    'failed': failed,
                    'timeout': timeout,
                    'crashed': crashed
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"Detailed report saved to: {report_path}")
        print()


def main():
    """Main entry point."""
    runner = ExperimentSuiteRunner()
    success = runner.run_suite()
    
    if success:
        print("\n[SUCCESS] All experiments completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Some experiments failed. Check report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
