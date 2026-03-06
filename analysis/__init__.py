"""Metrics computation and spectral analysis."""

from analysis.metrics import (
    compute_training_metrics,
    compute_validation_metrics,
    compute_long_horizon_metrics,
    compute_convergence_metrics,
)

from analysis.spectral import (
    compute_operator_eigenvalues,
    compute_spectral_radius,
    compute_eigenvalue_error,
    track_eigenvalue_evolution,
    save_eigenvalue_data,
)

from analysis.baseline import (
    compute_theoretical_sgd_convergence_rate,
    compute_theoretical_svrg_convergence_rate,
    compute_variance_reduction_factor,
    compute_spectral_approximation_quality,
    compare_optimizer_efficiency,
)

from analysis.statistics import (
    aggregate_results,
    generate_summary_table,
    compare_optimizers,
    save_aggregated_statistics_csv,
    save_comparison_results_csv,
)

__all__ = [
    # Metrics
    'compute_training_metrics',
    'compute_validation_metrics',
    'compute_long_horizon_metrics',
    'compute_convergence_metrics',
    # Spectral analysis
    'compute_operator_eigenvalues',
    'compute_spectral_radius',
    'compute_eigenvalue_error',
    'track_eigenvalue_evolution',
    'save_eigenvalue_data',
    # Baseline comparisons
    'compute_theoretical_sgd_convergence_rate',
    'compute_theoretical_svrg_convergence_rate',
    'compute_variance_reduction_factor',
    'compute_spectral_approximation_quality',
    'compare_optimizer_efficiency',
    # Statistical analysis
    'aggregate_results',
    'generate_summary_table',
    'compare_optimizers',
    'save_aggregated_statistics_csv',
    'save_comparison_results_csv',
]
