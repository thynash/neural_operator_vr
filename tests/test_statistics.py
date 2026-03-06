"""Tests for statistical analysis module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import csv

from analysis.statistics import (
    aggregate_results,
    generate_summary_table,
    compare_optimizers,
    save_aggregated_statistics_csv,
    save_comparison_results_csv,
)


class TestAggregateResults:
    """Tests for aggregate_results function."""
    
    def test_basic_aggregation(self):
        """Test basic aggregation with simple data."""
        results = [
            {'val_loss': 0.1, 'iterations_to_target': 1000},
            {'val_loss': 0.12, 'iterations_to_target': 1100},
            {'val_loss': 0.11, 'iterations_to_target': 1050},
        ]
        
        aggregated = aggregate_results(results)
        
        assert 'val_loss' in aggregated
        assert 'iterations_to_target' in aggregated
        
        # Check mean calculation
        assert abs(aggregated['val_loss']['mean'] - 0.11) < 1e-6
        assert abs(aggregated['iterations_to_target']['mean'] - 1050.0) < 1e-6
        
        # Check std calculation
        expected_std_loss = np.std([0.1, 0.12, 0.11], ddof=1)
        assert abs(aggregated['val_loss']['std'] - expected_std_loss) < 1e-6
        
        # Check count
        assert aggregated['val_loss']['count'] == 3
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        results = [
            {'val_loss': 0.1},
            {'val_loss': 0.12},
            {'val_loss': 0.11},
            {'val_loss': 0.13},
            {'val_loss': 0.09},
        ]
        
        aggregated = aggregate_results(results)
        
        # Check that confidence intervals are computed
        assert 'ci_95_lower' in aggregated['val_loss']
        assert 'ci_95_upper' in aggregated['val_loss']
        
        # CI should contain the mean
        mean = aggregated['val_loss']['mean']
        ci_lower = aggregated['val_loss']['ci_95_lower']
        ci_upper = aggregated['val_loss']['ci_95_upper']
        
        assert ci_lower <= mean <= ci_upper
        
        # CI should be wider than zero for multiple samples
        assert ci_upper > ci_lower
    
    def test_single_result(self):
        """Test aggregation with single result."""
        results = [{'val_loss': 0.1}]
        
        aggregated = aggregate_results(results)
        
        assert aggregated['val_loss']['mean'] == 0.1
        assert aggregated['val_loss']['std'] == 0.0
        assert aggregated['val_loss']['count'] == 1
        # CI should equal mean for single sample
        assert aggregated['val_loss']['ci_95_lower'] == 0.1
        assert aggregated['val_loss']['ci_95_upper'] == 0.1
    
    def test_empty_results(self):
        """Test aggregation with empty results list."""
        results = []
        aggregated = aggregate_results(results)
        assert aggregated == {}
    
    def test_none_values(self):
        """Test handling of None values."""
        results = [
            {'val_loss': 0.1, 'iterations_to_target': 1000},
            {'val_loss': 0.12, 'iterations_to_target': None},
            {'val_loss': 0.11, 'iterations_to_target': 1050},
        ]
        
        aggregated = aggregate_results(results)
        
        # val_loss should have 3 samples
        assert aggregated['val_loss']['count'] == 3
        
        # iterations_to_target should have 2 samples (None excluded)
        assert aggregated['iterations_to_target']['count'] == 2
        assert abs(aggregated['iterations_to_target']['mean'] - 1025.0) < 1e-6
    
    def test_specific_metrics(self):
        """Test aggregation of specific metrics only."""
        results = [
            {'val_loss': 0.1, 'train_loss': 0.2, 'iterations': 1000},
            {'val_loss': 0.12, 'train_loss': 0.22, 'iterations': 1100},
        ]
        
        aggregated = aggregate_results(results, metrics_to_aggregate=['val_loss', 'iterations'])
        
        assert 'val_loss' in aggregated
        assert 'iterations' in aggregated
        assert 'train_loss' not in aggregated
    
    def test_min_max_values(self):
        """Test min and max value tracking."""
        results = [
            {'val_loss': 0.1},
            {'val_loss': 0.15},
            {'val_loss': 0.08},
            {'val_loss': 0.12},
        ]
        
        aggregated = aggregate_results(results)
        
        assert aggregated['val_loss']['min'] == 0.08
        assert aggregated['val_loss']['max'] == 0.15


class TestGenerateSummaryTable:
    """Tests for generate_summary_table function."""
    
    def test_basic_table_generation(self):
        """Test basic summary table generation."""
        aggregated_results = {
            'SGD': {
                'val_loss': {'mean': 0.1, 'std': 0.01, 'ci_95_lower': 0.09, 'ci_95_upper': 0.11, 'count': 5}
            },
            'Adam': {
                'val_loss': {'mean': 0.08, 'std': 0.015, 'ci_95_lower': 0.065, 'ci_95_upper': 0.095, 'count': 5}
            },
        }
        
        table = generate_summary_table(aggregated_results)
        
        assert 'Summary Statistics' in table
        assert 'SGD' in table
        assert 'Adam' in table
        assert 'val_loss' in table
        assert '0.1' in table or '0.10' in table
    
    def test_empty_results(self):
        """Test table generation with empty results."""
        table = generate_summary_table({})
        assert 'No results' in table
    
    def test_save_to_file(self):
        """Test saving table to file."""
        aggregated_results = {
            'SGD': {
                'val_loss': {'mean': 0.1, 'std': 0.01, 'ci_95_lower': 0.09, 'ci_95_upper': 0.11, 'count': 5}
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'summary.txt'
            table = generate_summary_table(aggregated_results, output_path=output_path)
            
            assert output_path.exists()
            with open(output_path, 'r') as f:
                content = f.read()
            assert content == table


class TestCompareOptimizers:
    """Tests for compare_optimizers function."""
    
    def test_basic_comparison(self):
        """Test basic optimizer comparison."""
        results = {
            'SGD': [
                {'val_loss': 0.1},
                {'val_loss': 0.12},
                {'val_loss': 0.11},
            ],
            'Adam': [
                {'val_loss': 0.08},
                {'val_loss': 0.09},
                {'val_loss': 0.085},
            ],
        }
        
        comparison = compare_optimizers(results, 'val_loss', test_type='t-test')
        
        assert comparison['metric'] == 'val_loss'
        assert comparison['test_type'] == 't-test'
        assert len(comparison['comparisons']) == 1
        
        comp = comparison['comparisons'][0]
        assert comp['optimizer1'] == 'Adam'
        assert comp['optimizer2'] == 'SGD'
        assert 'p_value' in comp
        assert 'statistic' in comp
        assert 'significant' in comp
    
    def test_auto_test_selection(self):
        """Test automatic test type selection."""
        results = {
            'SGD': [{'val_loss': 0.1}, {'val_loss': 0.12}],
            'Adam': [{'val_loss': 0.08}, {'val_loss': 0.09}],
        }
        
        comparison = compare_optimizers(results, 'val_loss', test_type='auto')
        
        # With small sample size, should use non-parametric test
        assert comparison['test_type'] in ['mann-whitney', 't-test']
    
    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test."""
        results = {
            'SGD': [{'val_loss': 0.1}, {'val_loss': 0.12}, {'val_loss': 0.11}],
            'Adam': [{'val_loss': 0.08}, {'val_loss': 0.09}],
        }
        
        comparison = compare_optimizers(results, 'val_loss', test_type='mann-whitney')
        
        assert comparison['test_type'] == 'mann-whitney'
        assert len(comparison['comparisons']) == 1
    
    def test_multiple_optimizers(self):
        """Test comparison with multiple optimizers."""
        results = {
            'SGD': [{'val_loss': 0.1}, {'val_loss': 0.12}],
            'Adam': [{'val_loss': 0.08}, {'val_loss': 0.09}],
            'SVRG': [{'val_loss': 0.07}, {'val_loss': 0.075}],
        }
        
        comparison = compare_optimizers(results, 'val_loss')
        
        # Should have 3 pairwise comparisons: SGD-Adam, SGD-SVRG, Adam-SVRG
        assert len(comparison['comparisons']) == 3
    
    def test_insufficient_data(self):
        """Test comparison with insufficient data."""
        results = {
            'SGD': [{'val_loss': 0.1}],
        }
        
        comparison = compare_optimizers(results, 'val_loss')
        
        assert 'error' in comparison
        assert len(comparison['comparisons']) == 0
    
    def test_none_values_handling(self):
        """Test handling of None values in comparisons."""
        results = {
            'SGD': [
                {'val_loss': 0.1},
                {'val_loss': None},
                {'val_loss': 0.12},
            ],
            'Adam': [
                {'val_loss': 0.08},
                {'val_loss': 0.09},
            ],
        }
        
        comparison = compare_optimizers(results, 'val_loss')
        
        # Should work with None values excluded
        assert len(comparison['comparisons']) == 1
        comp = comparison['comparisons'][0]
        # SGD should have 2 valid samples
        assert comp['optimizer2'] == 'SGD'
    
    def test_significance_detection(self):
        """Test significance detection with clearly different distributions."""
        # Create clearly different distributions
        np.random.seed(42)
        results = {
            'SGD': [{'val_loss': x} for x in np.random.normal(0.1, 0.01, 10)],
            'Adam': [{'val_loss': x} for x in np.random.normal(0.05, 0.01, 10)],
        }
        
        comparison = compare_optimizers(results, 'val_loss', alpha=0.05)
        
        comp = comparison['comparisons'][0]
        # With such different means, should be significant
        assert comp['significant'] == True
        assert comp['p_value'] < 0.05


class TestSaveAggregatedStatisticsCSV:
    """Tests for save_aggregated_statistics_csv function."""
    
    def test_basic_csv_save(self):
        """Test basic CSV saving."""
        aggregated_results = {
            'SGD': {
                'val_loss': {'mean': 0.1, 'std': 0.01, 'min': 0.09, 'max': 0.11, 
                           'count': 5, 'ci_95_lower': 0.09, 'ci_95_upper': 0.11},
                'iterations': {'mean': 1000, 'std': 50, 'min': 950, 'max': 1050,
                             'count': 5, 'ci_95_lower': 950, 'ci_95_upper': 1050},
            },
            'Adam': {
                'val_loss': {'mean': 0.08, 'std': 0.015, 'min': 0.065, 'max': 0.095,
                           'count': 5, 'ci_95_lower': 0.065, 'ci_95_upper': 0.095},
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'stats.csv'
            save_aggregated_statistics_csv(aggregated_results, output_path)
            
            assert output_path.exists()
            
            # Read and verify CSV content
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Should have 3 rows: SGD val_loss, SGD iterations, Adam val_loss
            assert len(rows) == 3
            
            # Check first row
            assert rows[0]['optimizer'] == 'SGD'
            assert rows[0]['metric'] == 'iterations'
            assert float(rows[0]['mean']) == 1000.0
    
    def test_empty_results(self):
        """Test CSV saving with empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'stats.csv'
            save_aggregated_statistics_csv({}, output_path)
            
            assert output_path.exists()
            
            # Should have header only
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 0


class TestSaveComparisonResultsCSV:
    """Tests for save_comparison_results_csv function."""
    
    def test_basic_comparison_csv_save(self):
        """Test basic comparison CSV saving."""
        comparison_results = {
            'comparisons': [
                {
                    'optimizer1': 'Adam',
                    'optimizer2': 'SGD',
                    'mean1': 0.08,
                    'mean2': 0.1,
                    'statistic': -2.5,
                    'p_value': 0.03,
                    'significant': True,
                    'effect_size': -0.8,
                    'test_name': 'Independent t-test',
                }
            ],
            'test_type': 't-test',
            'metric': 'val_loss',
            'alpha': 0.05,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'comparisons.csv'
            save_comparison_results_csv(comparison_results, output_path)
            
            assert output_path.exists()
            
            # Read and verify CSV content
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 1
            assert rows[0]['optimizer1'] == 'Adam'
            assert rows[0]['optimizer2'] == 'SGD'
            assert float(rows[0]['p_value']) == 0.03
            assert rows[0]['significant'] == 'True'
    
    def test_empty_comparisons(self):
        """Test CSV saving with empty comparisons."""
        comparison_results = {
            'comparisons': [],
            'test_type': 't-test',
            'metric': 'val_loss',
            'alpha': 0.05,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'comparisons.csv'
            save_comparison_results_csv(comparison_results, output_path)
            
            assert output_path.exists()
            
            # Should have header only
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 0


class TestStatisticalFormulas:
    """Tests validating statistical formulas (Property 15)."""
    
    def test_mean_formula(self):
        """Test that mean is computed as (1/n) Σᵢ xᵢ."""
        values = [0.1, 0.12, 0.11, 0.13, 0.09]
        results = [{'val_loss': v} for v in values]
        
        aggregated = aggregate_results(results)
        
        expected_mean = sum(values) / len(values)
        assert abs(aggregated['val_loss']['mean'] - expected_mean) < 1e-10
    
    def test_std_formula(self):
        """Test that std is computed as √[(1/(n-1)) Σᵢ (xᵢ - x̄)²]."""
        values = [0.1, 0.12, 0.11, 0.13, 0.09]
        results = [{'val_loss': v} for v in values]
        
        aggregated = aggregate_results(results)
        
        # Compute expected std with ddof=1
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        expected_std = variance ** 0.5
        
        assert abs(aggregated['val_loss']['std'] - expected_std) < 1e-10
    
    def test_confidence_interval_formula(self):
        """Test confidence interval computation using t-distribution."""
        values = [0.1, 0.12, 0.11, 0.13, 0.09, 0.10, 0.11]
        results = [{'val_loss': v} for v in values]
        
        aggregated = aggregate_results(results)
        
        # Manually compute CI
        from scipy import stats as scipy_stats
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        t_critical = scipy_stats.t.ppf(0.975, n - 1)  # 95% CI
        margin = t_critical * (std / np.sqrt(n))
        expected_lower = mean - margin
        expected_upper = mean + margin
        
        assert abs(aggregated['val_loss']['ci_95_lower'] - expected_lower) < 1e-10
        assert abs(aggregated['val_loss']['ci_95_upper'] - expected_upper) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
