"""
Demonstration of statistical analysis for multi-seed experiments.

This script shows how to:
1. Aggregate results from multiple experimental runs
2. Generate summary tables with confidence intervals
3. Perform statistical significance testing between optimizers
4. Save results to CSV format
"""

from pathlib import Path
import numpy as np

from analysis.statistics import (
    aggregate_results,
    generate_summary_table,
    compare_optimizers,
    save_aggregated_statistics_csv,
    save_comparison_results_csv,
)


def simulate_experiment_results():
    """
    Simulate experimental results from multiple runs with different optimizers.
    
    In practice, these would come from actual training runs.
    """
    np.random.seed(42)
    
    # Simulate SGD results (slower convergence, higher variance)
    sgd_results = []
    for _ in range(5):
        sgd_results.append({
            'val_loss': np.random.normal(0.12, 0.02),
            'iterations_to_target': int(np.random.normal(1500, 200)),
            'time_to_target': np.random.normal(300, 50),
            'final_spectral_radius': np.random.normal(0.95, 0.05),
        })
    
    # Simulate Adam results (faster convergence, moderate variance)
    adam_results = []
    for _ in range(5):
        adam_results.append({
            'val_loss': np.random.normal(0.09, 0.015),
            'iterations_to_target': int(np.random.normal(1200, 150)),
            'time_to_target': np.random.normal(250, 40),
            'final_spectral_radius': np.random.normal(0.92, 0.04),
        })
    
    # Simulate SVRG results (best convergence, lowest variance)
    svrg_results = []
    for _ in range(5):
        svrg_results.append({
            'val_loss': np.random.normal(0.07, 0.01),
            'iterations_to_target': int(np.random.normal(1000, 100)),
            'time_to_target': np.random.normal(220, 30),
            'final_spectral_radius': np.random.normal(0.90, 0.03),
        })
    
    return {
        'SGD': sgd_results,
        'Adam': adam_results,
        'SVRG': svrg_results,
    }


def main():
    """Run statistical analysis demonstration."""
    print("=" * 80)
    print("Statistical Analysis for Multi-Seed Experiments")
    print("=" * 80)
    print()
    
    # Step 1: Simulate experimental results
    print("Step 1: Simulating experimental results from multiple runs...")
    results_by_optimizer = simulate_experiment_results()
    
    for optimizer_name, results in results_by_optimizer.items():
        print(f"  {optimizer_name}: {len(results)} runs")
    print()
    
    # Step 2: Aggregate results for each optimizer
    print("Step 2: Aggregating results across runs...")
    aggregated_results = {}
    
    for optimizer_name, results in results_by_optimizer.items():
        aggregated = aggregate_results(results)
        aggregated_results[optimizer_name] = aggregated
        print(f"  {optimizer_name}: Aggregated {len(aggregated)} metrics")
    print()
    
    # Step 3: Generate and display summary table
    print("Step 3: Generating summary table with confidence intervals...")
    print()
    summary_table = generate_summary_table(aggregated_results)
    print(summary_table)
    print()
    
    # Step 4: Perform statistical significance testing
    print("Step 4: Performing statistical significance testing...")
    print()
    
    metrics_to_compare = ['val_loss', 'iterations_to_target', 'time_to_target']
    
    for metric_name in metrics_to_compare:
        print(f"Comparing optimizers on metric: {metric_name}")
        print("-" * 80)
        
        comparison = compare_optimizers(
            results_by_optimizer,
            metric_name,
            test_type='auto',
            alpha=0.05
        )
        
        print(f"Test type: {comparison['test_type']}")
        print()
        
        for comp in comparison['comparisons']:
            opt1 = comp['optimizer1']
            opt2 = comp['optimizer2']
            mean1 = comp['mean1']
            mean2 = comp['mean2']
            p_value = comp['p_value']
            significant = comp['significant']
            
            print(f"  {opt1} vs {opt2}:")
            print(f"    Mean {opt1}: {mean1:.6f}")
            print(f"    Mean {opt2}: {mean2:.6f}")
            print(f"    p-value: {p_value:.6f}")
            print(f"    Significant (α=0.05): {significant}")
            
            if 'effect_size' in comp:
                print(f"    Effect size (Cohen's d): {comp['effect_size']:.3f}")
            
            print()
        
        print()
    
    # Step 5: Save results to CSV files
    print("Step 5: Saving results to CSV files...")
    output_dir = Path('statistical_analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save aggregated statistics
    stats_csv_path = output_dir / 'aggregated_statistics.csv'
    save_aggregated_statistics_csv(aggregated_results, stats_csv_path)
    print(f"  Saved aggregated statistics to: {stats_csv_path}")
    
    # Save comparison results for each metric
    for metric_name in metrics_to_compare:
        comparison = compare_optimizers(
            results_by_optimizer,
            metric_name,
            test_type='auto',
            alpha=0.05
        )
        
        comparison_csv_path = output_dir / f'comparison_{metric_name}.csv'
        save_comparison_results_csv(comparison, comparison_csv_path)
        print(f"  Saved {metric_name} comparisons to: {comparison_csv_path}")
    
    # Save summary table
    summary_txt_path = output_dir / 'summary_table.txt'
    generate_summary_table(aggregated_results, output_path=summary_txt_path)
    print(f"  Saved summary table to: {summary_txt_path}")
    
    print()
    print("=" * 80)
    print("Statistical analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
