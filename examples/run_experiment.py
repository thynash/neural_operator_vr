"""Example script for running neural operator experiments."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import ExperimentRunner


def main():
    """Run a single experiment from configuration file."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run neural operator experiment")
    parser.add_argument("config", nargs='?', default="examples/config_deeponet_lorenz_svrg.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Running experiment: {args.config}")
    print("=" * 80)
    
    runner = ExperimentRunner(args.config)
    results = runner.run()
    
    print("\nExperiment completed!")
    if results.get('final_val_loss') is not None:
        print(f"Final validation loss: {results['final_val_loss']:.6f}")
    else:
        print("No validation loss recorded")


def run_multiple_seeds():
    """Run experiment with multiple seeds for statistical analysis."""
    print("=" * 80)
    print("Running experiment with multiple seeds")
    print("=" * 80)
    
    runner = ExperimentRunner("examples/config_deeponet_lorenz_svrg.yaml")
    
    # Run with 3 different seeds
    seeds = [42, 123, 456]
    aggregated_results = runner.run_multiple_seeds(seeds)
    
    print("\nAll runs completed!")
    print(f"Mean final validation loss: {aggregated_results['final_val_loss']['mean']:.6f}")
    print(f"Std final validation loss: {aggregated_results['final_val_loss']['std']:.6f}")


if __name__ == "__main__":
    # Run single experiment
    main()
    
    # Uncomment to run multiple seeds
    # run_multiple_seeds()
