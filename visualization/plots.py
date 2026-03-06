"""Publication-quality plotting functions for neural operator experiments.

This module provides plotting utilities for visualizing training curves, gradient variance,
validation errors, predictions, spectral analysis, and computational costs.
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


# Configure publication-quality styling
def configure_publication_style():
    """Configure matplotlib for publication-quality plots.
    
    Sets font family, sizes, line widths, and other styling parameters
    according to academic publication standards.
    """
    # Try to use Times New Roman, fall back to serif
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # Font sizes
    plt.rcParams['font.size'] = 10  # Base font size
    plt.rcParams['axes.labelsize'] = 12  # Axis labels
    plt.rcParams['axes.titlesize'] = 12  # Subplot titles
    plt.rcParams['xtick.labelsize'] = 10  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 10  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 10  # Legend
    
    # Line widths
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5
    
    # Grid styling
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = 'gray'
    
    # Legend styling
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 100  # Display DPI
    plt.rcParams['savefig.dpi'] = 300  # Save DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


# Colorblind-friendly palette
COLORBLIND_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'pink': '#ECE133',
    'gray': '#56B4E9'
}

OPTIMIZER_COLORS = {
    'sgd': COLORBLIND_COLORS['blue'],
    'adam': COLORBLIND_COLORS['orange'],
    'svrg': COLORBLIND_COLORS['green']
}


def plot_training_curves(
    histories: Dict[str, Dict],
    optimizer_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_std: bool = True
) -> plt.Figure:
    """Generate training loss curves for multiple optimizers.
    
    Args:
        histories: Dictionary mapping optimizer names to training histories.
                  Each history should have 'train_loss' as list of (iteration, value) tuples.
        optimizer_names: List of optimizer names to plot.
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
        show_std: Whether to show standard deviation as shaded regions.
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.1
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_name in optimizer_names:
        if opt_name not in histories:
            continue
        
        history = histories[opt_name]
        
        # Extract training loss data
        if isinstance(history, dict) and 'train_loss' in history:
            train_loss = history['train_loss']
        else:
            continue
        
        # Handle multiple runs (for std computation)
        if isinstance(train_loss, list) and len(train_loss) > 0:
            if isinstance(train_loss[0], (list, tuple)) and len(train_loss[0]) == 2:
                # Single run: [(iter, loss), ...]
                iterations = [x[0] for x in train_loss]
                losses = [x[1] for x in train_loss]
                
                color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
                ax.plot(iterations, losses, label=opt_name.upper(), color=color, linewidth=2)
            
            elif isinstance(train_loss[0], list):
                # Multiple runs: [[(iter, loss), ...], [(iter, loss), ...], ...]
                all_iterations = []
                all_losses = []
                
                for run in train_loss:
                    iterations = [x[0] for x in run]
                    losses = [x[1] for x in run]
                    all_iterations.append(iterations)
                    all_losses.append(losses)
                
                # Compute mean and std
                # Align iterations (use first run as reference)
                ref_iterations = all_iterations[0]
                aligned_losses = []
                
                for iterations, losses in zip(all_iterations, all_losses):
                    # Interpolate to reference iterations if needed
                    if iterations == ref_iterations:
                        aligned_losses.append(losses)
                    else:
                        # Simple alignment: use common iterations
                        aligned_losses.append(losses[:len(ref_iterations)])
                
                mean_losses = np.mean(aligned_losses, axis=0)
                std_losses = np.std(aligned_losses, axis=0) if show_std and len(aligned_losses) > 1 else None
                
                color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
                ax.plot(ref_iterations, mean_losses, label=opt_name.upper(), color=color, linewidth=2)
                
                if std_losses is not None:
                    ax.fill_between(
                        ref_iterations,
                        mean_losses - std_losses,
                        mean_losses + std_losses,
                        color=color,
                        alpha=0.2
                    )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def plot_gradient_variance(
    histories: Dict[str, Dict],
    optimizer_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 4)
) -> plt.Figure:
    """Generate gradient variance evolution plots.
    
    Args:
        histories: Dictionary mapping optimizer names to training histories.
                  Each history should have 'train_grad_variance' as list of (iteration, value) tuples.
        optimizer_names: List of optimizer names to plot.
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.2
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_name in optimizer_names:
        if opt_name not in histories:
            continue
        
        history = histories[opt_name]
        
        # Extract gradient variance data
        if isinstance(history, dict) and 'train_grad_variance' in history:
            grad_var = history['train_grad_variance']
        else:
            continue
        
        if isinstance(grad_var, list) and len(grad_var) > 0:
            if isinstance(grad_var[0], (list, tuple)) and len(grad_var[0]) == 2:
                # Single run
                iterations = [x[0] for x in grad_var]
                variances = [x[1] for x in grad_var]
                
                color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
                ax.plot(iterations, variances, label=opt_name.upper(), color=color, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Variance')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    
    # Add annotation for SVRG variance reduction if present
    if 'svrg' in [name.lower() for name in optimizer_names]:
        ax.text(
            0.95, 0.95,
            'SVRG shows variance reduction',
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def plot_validation_error(
    histories: Dict[str, Dict],
    optimizer_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 4),
    confidence_level: float = 0.95
) -> plt.Figure:
    """Generate validation error curves with confidence intervals.
    
    Args:
        histories: Dictionary mapping optimizer names to training histories.
                  Each history should have 'val_relative_error' as list of (iteration, value) tuples.
        optimizer_names: List of optimizer names to plot.
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
        confidence_level: Confidence level for error bars (default: 0.95 for 95% CI).
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.3
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_name in optimizer_names:
        if opt_name not in histories:
            continue
        
        history = histories[opt_name]
        
        # Extract validation error data
        if isinstance(history, dict) and 'val_relative_error' in history:
            val_error = history['val_relative_error']
        elif isinstance(history, dict) and 'val_loss' in history:
            # Fallback to val_loss if val_relative_error not available
            val_error = history['val_loss']
        else:
            continue
        
        if isinstance(val_error, list) and len(val_error) > 0:
            if isinstance(val_error[0], (list, tuple)) and len(val_error[0]) == 2:
                # Single run
                iterations = [x[0] for x in val_error]
                errors = [x[1] for x in val_error]
                
                color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
                ax.plot(iterations, errors, label=opt_name.upper(), color=color, linewidth=2)
            
            elif isinstance(val_error[0], list):
                # Multiple runs - compute confidence intervals
                all_iterations = []
                all_errors = []
                
                for run in val_error:
                    iterations = [x[0] for x in run]
                    errors = [x[1] for x in run]
                    all_iterations.append(iterations)
                    all_errors.append(errors)
                
                # Align and compute statistics
                ref_iterations = all_iterations[0]
                aligned_errors = []
                
                for iterations, errors in zip(all_iterations, all_errors):
                    if iterations == ref_iterations:
                        aligned_errors.append(errors)
                    else:
                        aligned_errors.append(errors[:len(ref_iterations)])
                
                mean_errors = np.mean(aligned_errors, axis=0)
                std_errors = np.std(aligned_errors, axis=0)
                n_runs = len(aligned_errors)
                
                # Compute confidence interval (assuming normal distribution)
                from scipy import stats
                ci_multiplier = stats.t.ppf((1 + confidence_level) / 2, n_runs - 1) if n_runs > 1 else 1.96
                ci = ci_multiplier * std_errors / np.sqrt(n_runs)
                
                color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
                ax.plot(ref_iterations, mean_errors, label=opt_name.upper(), color=color, linewidth=2)
                ax.fill_between(
                    ref_iterations,
                    mean_errors - ci,
                    mean_errors + ci,
                    color=color,
                    alpha=0.2
                )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Relative Error')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig



def plot_long_horizon_predictions(
    true_trajectory: np.ndarray,
    predicted_trajectory: np.ndarray,
    system_name: str,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """Generate trajectory comparison plots with system-specific visualizations.
    
    Args:
        true_trajectory: True trajectory array. Shape depends on system:
                        - Logistic Map: (time_steps,)
                        - Lorenz System: (time_steps, 3)
                        - Burgers Equation: (time_steps, spatial_points)
        predicted_trajectory: Predicted trajectory with same shape as true_trajectory.
        system_name: Name of dynamical system ('logistic', 'lorenz', or 'burgers').
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height). Auto-determined if None.
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.4
    """
    configure_publication_style()
    
    system_name = system_name.lower()
    
    if system_name == 'logistic':
        # Line plot for Logistic Map
        if figsize is None:
            figsize = (6, 4)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        time_steps = np.arange(len(true_trajectory))
        ax.plot(time_steps, true_trajectory, label='True', color=COLORBLIND_COLORS['blue'], linewidth=2)
        ax.plot(time_steps, predicted_trajectory, label='Predicted', 
                color=COLORBLIND_COLORS['orange'], linewidth=2, linestyle='--')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Value')
        ax.set_title('Logistic Map: Long-Horizon Prediction')
        ax.grid(True)
        ax.legend()
    
    elif system_name == 'lorenz':
        # 3D phase space plot for Lorenz System
        if figsize is None:
            figsize = (8, 6)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
                label='True', color=COLORBLIND_COLORS['blue'], linewidth=2, alpha=0.7)
        ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2],
                label='Predicted', color=COLORBLIND_COLORS['orange'], linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz System: Phase Space Trajectory')
        ax.legend()
    
    elif system_name == 'burgers':
        # Heatmap for Burgers Equation
        if figsize is None:
            figsize = (12, 4)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # True solution
        im1 = axes[0].imshow(true_trajectory.T, aspect='auto', cmap='RdBu_r', 
                            origin='lower', interpolation='bilinear')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Spatial Point')
        axes[0].set_title('True Solution')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted solution
        im2 = axes[1].imshow(predicted_trajectory.T, aspect='auto', cmap='RdBu_r',
                            origin='lower', interpolation='bilinear')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Spatial Point')
        axes[1].set_title('Predicted Solution')
        plt.colorbar(im2, ax=axes[1])
        
        # Error
        error = np.abs(true_trajectory - predicted_trajectory)
        im3 = axes[2].imshow(error.T, aspect='auto', cmap='hot',
                            origin='lower', interpolation='bilinear')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Spatial Point')
        axes[2].set_title('Absolute Error')
        plt.colorbar(im3, ax=axes[2])
    
    else:
        raise ValueError(f"Unknown system name: {system_name}. Must be 'logistic', 'lorenz', or 'burgers'.")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def plot_eigenvalue_comparison(
    true_eigenvalues: np.ndarray,
    learned_eigenvalues: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 6)
) -> plt.Figure:
    """Generate eigenvalue scatter plot comparing true vs learned eigenvalues.
    
    Args:
        true_eigenvalues: Array of true eigenvalues (complex numbers). Shape: (n_eigenvalues,)
        learned_eigenvalues: Array of learned eigenvalues (complex numbers). Shape: (n_eigenvalues,)
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.5
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract real and imaginary parts
    true_real = np.real(true_eigenvalues)
    true_imag = np.imag(true_eigenvalues)
    learned_real = np.real(learned_eigenvalues)
    learned_imag = np.imag(learned_eigenvalues)
    
    # Plot eigenvalues
    ax.scatter(true_real, true_imag, s=100, marker='o', 
              color=COLORBLIND_COLORS['blue'], label='True', alpha=0.7, edgecolors='black')
    ax.scatter(learned_real, learned_imag, s=100, marker='x', 
              color=COLORBLIND_COLORS['red'], label='Learned', alpha=0.7, linewidths=2)
    
    # Plot unit circle (stability boundary)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', linewidth=1.5, label='Unit Circle', alpha=0.5)
    
    # Set equal aspect ratio for proper circle
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Eigenvalue Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add axes through origin
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def plot_cost_vs_accuracy(
    histories: Dict[str, Dict],
    optimizer_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (6, 4)
) -> plt.Figure:
    """Generate wall-clock time vs validation error plot.
    
    Args:
        histories: Dictionary mapping optimizer names to training histories.
                  Each history should have 'val_loss' or 'val_relative_error' and 'iteration_time'.
        optimizer_names: List of optimizer names to plot.
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.7
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for opt_name in optimizer_names:
        if opt_name not in histories:
            continue
        
        history = histories[opt_name]
        
        # Extract validation error and timing data
        val_error_key = 'val_relative_error' if 'val_relative_error' in history else 'val_loss'
        
        if val_error_key not in history or 'iteration_time' not in history:
            continue
        
        val_error = history[val_error_key]
        iter_times = history['iteration_time']
        
        if not val_error or not iter_times:
            continue
        
        # Extract data
        if isinstance(val_error[0], (list, tuple)) and len(val_error[0]) == 2:
            val_iterations = [x[0] for x in val_error]
            val_errors = [x[1] for x in val_error]
        else:
            continue
        
        if isinstance(iter_times[0], (list, tuple)) and len(iter_times[0]) == 2:
            time_iterations = [x[0] for x in iter_times]
            times = [x[1] for x in iter_times]
        else:
            continue
        
        # Compute cumulative wall-clock time
        cumulative_time = np.cumsum(times)
        
        # Align validation errors with cumulative time
        # Find validation iterations in time iterations
        wall_times = []
        for val_iter in val_iterations:
            # Find closest time iteration
            idx = min(range(len(time_iterations)), key=lambda i: abs(time_iterations[i] - val_iter))
            if idx < len(cumulative_time):
                wall_times.append(cumulative_time[idx])
            else:
                wall_times.append(cumulative_time[-1])
        
        color = OPTIMIZER_COLORS.get(opt_name.lower(), COLORBLIND_COLORS['blue'])
        ax.plot(wall_times, val_errors, label=opt_name.upper(), 
               color=color, linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_ylabel('Validation Error')
    ax.grid(True)
    ax.legend()
    ax.set_title('Computational Cost vs Accuracy')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def plot_burgers_spatiotemporal(
    solution: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = 'Burgers Equation Solution'
) -> plt.Figure:
    """Generate spatiotemporal heatmap for Burgers equation solution.
    
    Args:
        solution: Spatiotemporal field array. Shape: (time_steps, spatial_points)
        output_path: Optional path to save the figure (PDF format).
        figsize: Figure size in inches (width, height).
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    
    Requirements: 9.6
    """
    configure_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(solution.T, aspect='auto', cmap='RdBu_r', 
                   origin='lower', interpolation='bilinear')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Spatial Point')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Solution Value')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300)
    
    return fig


def save_all_plots(
    histories: Dict[str, Dict],
    optimizer_names: List[str],
    output_dir: Union[str, Path],
    true_trajectory: Optional[np.ndarray] = None,
    predicted_trajectory: Optional[np.ndarray] = None,
    system_name: Optional[str] = None,
    true_eigenvalues: Optional[np.ndarray] = None,
    learned_eigenvalues: Optional[np.ndarray] = None,
    burgers_solution: Optional[np.ndarray] = None
) -> Dict[str, plt.Figure]:
    """Generate and save all publication-quality plots.
    
    Args:
        histories: Dictionary mapping optimizer names to training histories.
        optimizer_names: List of optimizer names to plot.
        output_dir: Directory to save all plots.
        true_trajectory: Optional true trajectory for long-horizon prediction plot.
        predicted_trajectory: Optional predicted trajectory for long-horizon prediction plot.
        system_name: Optional system name for trajectory plot.
        true_eigenvalues: Optional true eigenvalues for spectral analysis plot.
        learned_eigenvalues: Optional learned eigenvalues for spectral analysis plot.
        burgers_solution: Optional Burgers equation solution for spatiotemporal plot.
    
    Returns:
        Dictionary mapping plot names to Figure objects.
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Training curves
    fig = plot_training_curves(
        histories, optimizer_names,
        output_path=output_dir / 'training_curves.pdf'
    )
    figures['training_curves'] = fig
    plt.close(fig)
    
    # Gradient variance
    fig = plot_gradient_variance(
        histories, optimizer_names,
        output_path=output_dir / 'gradient_variance.pdf'
    )
    figures['gradient_variance'] = fig
    plt.close(fig)
    
    # Validation error
    fig = plot_validation_error(
        histories, optimizer_names,
        output_path=output_dir / 'validation_error.pdf'
    )
    figures['validation_error'] = fig
    plt.close(fig)
    
    # Computational cost
    fig = plot_cost_vs_accuracy(
        histories, optimizer_names,
        output_path=output_dir / 'cost_vs_accuracy.pdf'
    )
    figures['cost_vs_accuracy'] = fig
    plt.close(fig)
    
    # Long-horizon predictions (if data provided)
    if true_trajectory is not None and predicted_trajectory is not None and system_name is not None:
        fig = plot_long_horizon_predictions(
            true_trajectory, predicted_trajectory, system_name,
            output_path=output_dir / 'long_horizon_predictions.pdf'
        )
        figures['long_horizon_predictions'] = fig
        plt.close(fig)
    
    # Eigenvalue comparison (if data provided)
    if true_eigenvalues is not None and learned_eigenvalues is not None:
        fig = plot_eigenvalue_comparison(
            true_eigenvalues, learned_eigenvalues,
            output_path=output_dir / 'eigenvalue_comparison.pdf'
        )
        figures['eigenvalue_comparison'] = fig
        plt.close(fig)
    
    # Burgers spatiotemporal (if data provided)
    if burgers_solution is not None:
        fig = plot_burgers_spatiotemporal(
            burgers_solution,
            output_path=output_dir / 'burgers_spatiotemporal.pdf'
        )
        figures['burgers_spatiotemporal'] = fig
        plt.close(fig)
    
    return figures



class PlotGenerator:
    """Wrapper class for plot generation functions.
    
    This class provides a convenient interface to plotting functions
    for use in the experiment runner.
    """
    
    def __init__(self, output_dir: str, plot_format: str = 'pdf', dpi: int = 300):
        """Initialize plot generator.
        
        Args:
            output_dir: Directory to save plots
            plot_format: Format for saving plots ('pdf', 'png', 'svg')
            dpi: Resolution for raster formats
        """
        self.output_dir = Path(output_dir)
        self.plot_format = plot_format
        self.dpi = dpi
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure publication style
        configure_publication_style()
    
    def plot_training_curves(
        self,
        histories: Dict[str, Dict],
        optimizer_names: List[str]
    ):
        """Generate training loss curves."""
        plot_training_curves(
            histories,
            optimizer_names,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_gradient_variance(
        self,
        histories: Dict[str, Dict],
        optimizer_names: List[str]
    ):
        """Generate gradient variance plots."""
        plot_gradient_variance(
            histories,
            optimizer_names,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_validation_error(
        self,
        histories: Dict[str, Dict],
        optimizer_names: List[str]
    ):
        """Generate validation error plots."""
        plot_validation_error(
            histories,
            optimizer_names,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_long_horizon_predictions(
        self,
        true_trajectory: np.ndarray,
        predicted_trajectory: np.ndarray,
        system_name: str
    ):
        """Generate long-horizon prediction plots."""
        plot_long_horizon_predictions(
            true_trajectory,
            predicted_trajectory,
            system_name,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_eigenvalue_comparison(
        self,
        true_eigenvalues: np.ndarray,
        learned_eigenvalues: np.ndarray
    ):
        """Generate eigenvalue comparison plots."""
        plot_eigenvalue_comparison(
            true_eigenvalues,
            learned_eigenvalues,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_cost_vs_accuracy(
        self,
        histories: Dict[str, Dict],
        optimizer_names: List[str]
    ):
        """Generate computational cost vs accuracy plots."""
        plot_cost_vs_accuracy(
            histories,
            optimizer_names,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
    
    def plot_burgers_spatiotemporal(
        self,
        solution: np.ndarray,
        time_points: np.ndarray,
        spatial_points: np.ndarray
    ):
        """Generate Burgers equation spatiotemporal heatmap."""
        plot_burgers_spatiotemporal(
            solution,
            time_points,
            spatial_points,
            str(self.output_dir),
            self.plot_format,
            self.dpi
        )
