"""Publication-quality plotting utilities."""

from .plots import (
    configure_publication_style,
    plot_training_curves,
    plot_gradient_variance,
    plot_validation_error,
    plot_long_horizon_predictions,
    plot_eigenvalue_comparison,
    plot_cost_vs_accuracy,
    plot_burgers_spatiotemporal,
    save_all_plots,
    COLORBLIND_COLORS,
    OPTIMIZER_COLORS
)

__all__ = [
    'configure_publication_style',
    'plot_training_curves',
    'plot_gradient_variance',
    'plot_validation_error',
    'plot_long_horizon_predictions',
    'plot_eigenvalue_comparison',
    'plot_cost_vs_accuracy',
    'plot_burgers_spatiotemporal',
    'save_all_plots',
    'COLORBLIND_COLORS',
    'OPTIMIZER_COLORS'
]
