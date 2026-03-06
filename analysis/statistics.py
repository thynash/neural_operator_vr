"""Statistical analysis for multi-seed experiments."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats


def aggregate_results(
    results_list: List[Dict[str, Any]],
    metrics_to_aggregate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple experimental runs.
    
    Computes mean and standard deviation for each metric across runs.
    
    Args:
        results_list: List of result dictionaries from multiple runs
        metrics_to_aggregate: List of metric names to aggregate. If None, aggregates all numeric metrics.
    
    Returns:
        Dictionary mapping metric names to aggregated statistics:
            {
                'metric_name': {
                    'mean': float,
                    'std': float,
                    'min': float,
                    'max': float,
                    'count': int,
                    'ci_95_lower': float,  # 95% confidence interval lower bound
                    'ci_95_upper': float,  # 95% confidence interval upper bound
                }
            }
    
    Validates: Requirements 15.2, 15.4
    
    Example:
        >>> results = [
        ...     {'val_loss': 0.1, 'iterations_to_target': 1000},
        ...     {'val_loss': 0.12, 'iterations_to_target': 1100},
        ...     {'val_loss': 0.11, 'iterations_to_target': 1050},
        ... ]
        >>> aggregated = aggregate_results(results)
        >>> print(aggregated['val_loss']['mean'])
        0.11
    """
    if not results_list:
        return {}
    
    # Determine which metrics to aggregate
    if metrics_to_aggregate is None:
        # Aggregate all numeric metrics found in the first result
        metrics_to_aggregate = []
        for key, value in results_list[0].items():
            if isinstance(value, (int, float, np.number)):
                metrics_to_aggregate.append(key)
    
    aggregated = {}
    
    for metric_name in metrics_to_aggregate:
        # Collect values for this metric across all runs
        values = []
        for result in results_list:
            if metric_name in result and result[metric_name] is not None:
                values.append(float(result[metric_name]))
        
        if not values:
            continue
        
        # Compute statistics
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array, ddof=1) if len(values) > 1 else 0.0
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        count = len(values)
        
        # Compute 95% confidence interval using t-distribution
        if count > 1:
            confidence_level = 0.95
            degrees_freedom = count - 1
            t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
            margin_error = t_critical * (std_val / np.sqrt(count))
            ci_lower = mean_val - margin_error
            ci_upper = mean_val + margin_error
        else:
            ci_lower = mean_val
            ci_upper = mean_val
        
        aggregated[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val),
            'count': count,
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
        }
    
    return aggregated


def generate_summary_table(
    aggregated_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a formatted summary table from aggregated results.
    
    Args:
        aggregated_results: Dictionary mapping optimizer names to aggregated statistics
            {
                'optimizer_name': {
                    'metric_name': {'mean': ..., 'std': ..., 'ci_95_lower': ..., 'ci_95_upper': ...}
                }
            }
        output_path: Optional path to save the table as text file
    
    Returns:
        Formatted string table
    
    Validates: Requirements 15.4
    
    Example:
        >>> results = {
        ...     'SGD': {'val_loss': {'mean': 0.1, 'std': 0.01, 'ci_95_lower': 0.09, 'ci_95_upper': 0.11}},
        ...     'Adam': {'val_loss': {'mean': 0.08, 'std': 0.015, 'ci_95_lower': 0.065, 'ci_95_upper': 0.095}},
        ... }
        >>> table = generate_summary_table(results)
    """
    if not aggregated_results:
        return "No results to display"
    
    # Collect all unique metrics
    all_metrics = set()
    for optimizer_stats in aggregated_results.values():
        all_metrics.update(optimizer_stats.keys())
    all_metrics = sorted(all_metrics)
    
    # Build table
    lines = []
    lines.append("=" * 100)
    lines.append("Summary Statistics Across Multiple Runs")
    lines.append("=" * 100)
    lines.append("")
    
    for metric_name in all_metrics:
        lines.append(f"Metric: {metric_name}")
        lines.append("-" * 100)
        lines.append(f"{'Optimizer':<15} {'Mean':<12} {'Std Dev':<12} {'95% CI Lower':<15} {'95% CI Upper':<15} {'Count':<8}")
        lines.append("-" * 100)
        
        for optimizer_name, optimizer_stats in aggregated_results.items():
            if metric_name in optimizer_stats:
                stats_dict = optimizer_stats[metric_name]
                mean_val = stats_dict['mean']
                std_val = stats_dict['std']
                ci_lower = stats_dict['ci_95_lower']
                ci_upper = stats_dict['ci_95_upper']
                count = stats_dict['count']
                
                lines.append(
                    f"{optimizer_name:<15} {mean_val:<12.6f} {std_val:<12.6f} "
                    f"{ci_lower:<15.6f} {ci_upper:<15.6f} {count:<8}"
                )
        
        lines.append("")
    
    lines.append("=" * 100)
    
    table_str = "\n".join(lines)
    
    # Save to file if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
    
    return table_str


def compare_optimizers(
    results_by_optimizer: Dict[str, List[Dict[str, Any]]],
    metric_name: str,
    test_type: str = 'auto',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical significance testing comparing optimizer performance.
    
    Args:
        results_by_optimizer: Dictionary mapping optimizer names to lists of results
            {
                'SGD': [result1, result2, ...],
                'Adam': [result1, result2, ...],
                'SVRG': [result1, result2, ...],
            }
        metric_name: Name of the metric to compare (e.g., 'val_loss', 'iterations_to_target')
        test_type: Type of statistical test to perform:
            - 'auto': Automatically choose based on data (default)
            - 't-test': Paired or unpaired t-test
            - 'wilcoxon': Wilcoxon signed-rank test (non-parametric)
            - 'mann-whitney': Mann-Whitney U test (non-parametric)
        alpha: Significance level (default: 0.05)
    
    Returns:
        Dictionary containing:
            - 'comparisons': List of pairwise comparison results
            - 'test_type': Type of test performed
            - 'metric': Metric name
            - 'alpha': Significance level
    
    Validates: Requirements 15.3
    
    Example:
        >>> results = {
        ...     'SGD': [{'val_loss': 0.1}, {'val_loss': 0.12}],
        ...     'Adam': [{'val_loss': 0.08}, {'val_loss': 0.09}],
        ... }
        >>> comparison = compare_optimizers(results, 'val_loss')
        >>> print(comparison['comparisons'][0]['significant'])
    """
    # Extract metric values for each optimizer
    optimizer_values = {}
    for optimizer_name, results_list in results_by_optimizer.items():
        values = []
        for result in results_list:
            if metric_name in result and result[metric_name] is not None:
                values.append(float(result[metric_name]))
        if values:
            optimizer_values[optimizer_name] = np.array(values)
    
    if len(optimizer_values) < 2:
        return {
            'comparisons': [],
            'test_type': test_type,
            'metric': metric_name,
            'alpha': alpha,
            'error': 'Need at least 2 optimizers with valid data for comparison'
        }
    
    # Determine test type
    if test_type == 'auto':
        # Use t-test if sample sizes are reasonable (>= 5), otherwise use non-parametric
        min_sample_size = min(len(values) for values in optimizer_values.values())
        if min_sample_size >= 5:
            test_type = 't-test'
        else:
            test_type = 'mann-whitney'
    
    # Perform pairwise comparisons
    comparisons = []
    optimizer_names = sorted(optimizer_values.keys())
    
    for i in range(len(optimizer_names)):
        for j in range(i + 1, len(optimizer_names)):
            opt1_name = optimizer_names[i]
            opt2_name = optimizer_names[j]
            values1 = optimizer_values[opt1_name]
            values2 = optimizer_values[opt2_name]
            
            # Perform statistical test
            if test_type == 't-test':
                # Independent samples t-test
                statistic, p_value = stats.ttest_ind(values1, values2)
                test_name = "Independent t-test"
            elif test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (requires paired samples)
                if len(values1) == len(values2):
                    statistic, p_value = stats.wilcoxon(values1, values2)
                    test_name = "Wilcoxon signed-rank test"
                else:
                    # Fall back to Mann-Whitney if not paired
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    test_name = "Mann-Whitney U test (fallback)"
            elif test_type == 'mann-whitney':
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Determine significance
            significant = p_value < alpha
            
            # Compute effect size (Cohen's d for t-test)
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            if test_type == 't-test':
                pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                     (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                    (len(values1) + len(values2) - 2))
                effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
            else:
                effect_size = None
            
            comparison = {
                'optimizer1': opt1_name,
                'optimizer2': opt2_name,
                'mean1': float(mean1),
                'mean2': float(mean2),
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'test_name': test_name,
            }
            
            if effect_size is not None:
                comparison['effect_size'] = float(effect_size)
            
            comparisons.append(comparison)
    
    return {
        'comparisons': comparisons,
        'test_type': test_type,
        'metric': metric_name,
        'alpha': alpha,
    }


def save_aggregated_statistics_csv(
    aggregated_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path
) -> None:
    """
    Save aggregated statistics to CSV format.
    
    Args:
        aggregated_results: Dictionary mapping optimizer names to aggregated statistics
            {
                'optimizer_name': {
                    'metric_name': {'mean': ..., 'std': ..., 'ci_95_lower': ..., 'ci_95_upper': ...}
                }
            }
        output_path: Path to save CSV file
    
    Validates: Requirements 15.5
    
    Example:
        >>> results = {
        ...     'SGD': {'val_loss': {'mean': 0.1, 'std': 0.01}},
        ...     'Adam': {'val_loss': {'mean': 0.08, 'std': 0.015}},
        ... }
        >>> save_aggregated_statistics_csv(results, Path('results.csv'))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique metrics
    all_metrics = set()
    for optimizer_stats in aggregated_results.values():
        all_metrics.update(optimizer_stats.keys())
    all_metrics = sorted(all_metrics)
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['optimizer', 'metric', 'mean', 'std', 'min', 'max', 'count', 'ci_95_lower', 'ci_95_upper']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for optimizer_name, optimizer_stats in aggregated_results.items():
            for metric_name in all_metrics:
                if metric_name in optimizer_stats:
                    stats_dict = optimizer_stats[metric_name]
                    row = {
                        'optimizer': optimizer_name,
                        'metric': metric_name,
                        'mean': stats_dict['mean'],
                        'std': stats_dict['std'],
                        'min': stats_dict['min'],
                        'max': stats_dict['max'],
                        'count': stats_dict['count'],
                        'ci_95_lower': stats_dict['ci_95_lower'],
                        'ci_95_upper': stats_dict['ci_95_upper'],
                    }
                    writer.writerow(row)


def save_comparison_results_csv(
    comparison_results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save statistical comparison results to CSV format.
    
    Args:
        comparison_results: Dictionary from compare_optimizers()
        output_path: Path to save CSV file
    
    Example:
        >>> comparison = compare_optimizers(results, 'val_loss')
        >>> save_comparison_results_csv(comparison, Path('comparisons.csv'))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparisons = comparison_results.get('comparisons', [])
    
    if not comparisons:
        # Write empty file with header
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['optimizer1', 'optimizer2', 'mean1', 'mean2', 'statistic', 
                         'p_value', 'significant', 'effect_size', 'test_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return
    
    # Determine fieldnames based on first comparison
    fieldnames = ['optimizer1', 'optimizer2', 'mean1', 'mean2', 'statistic', 
                 'p_value', 'significant', 'test_name']
    if 'effect_size' in comparisons[0]:
        fieldnames.insert(-1, 'effect_size')
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for comparison in comparisons:
            row = {key: comparison.get(key, '') for key in fieldnames}
            writer.writerow(row)
