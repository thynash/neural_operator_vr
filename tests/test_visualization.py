"""Tests for visualization module."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from visualization import (
    configure_publication_style,
    plot_training_curves,
    plot_gradient_variance,
    plot_validation_error,
    plot_long_horizon_predictions,
    plot_eigenvalue_comparison,
    plot_cost_vs_accuracy,
    plot_burgers_spatiotemporal,
    save_all_plots
)


class TestPublicationStyle:
    """Tests for publication-quality styling configuration."""
    
    def test_configure_publication_style(self):
        """Test that publication style configuration sets correct parameters."""
        configure_publication_style()
        
        # Check font sizes
        assert plt.rcParams['axes.labelsize'] == 12
        assert plt.rcParams['xtick.labelsize'] == 10
        assert plt.rcParams['ytick.labelsize'] == 10
        
        # Check line widths
        assert plt.rcParams['lines.linewidth'] == 2.0
        
        # Check DPI
        assert plt.rcParams['savefig.dpi'] == 300


class TestTrainingCurvePlotter:
    """Tests for training curve plotting."""
    
    def test_plot_training_curves_single_run(self):
        """Test plotting training curves with single run data."""
        histories = {
            'sgd': {
                'train_loss': [(i, 1.0 / (i + 1)) for i in range(100)]
            },
            'adam': {
                'train_loss': [(i, 0.8 / (i + 1)) for i in range(100)]
            }
        }
        
        fig = plot_training_curves(histories, ['sgd', 'adam'])
        assert fig is not None
        plt.close(fig)
    
    def test_plot_training_curves_with_std(self):
        """Test plotting training curves with multiple runs and std."""
        # Multiple runs
        run1 = [(i, 1.0 / (i + 1)) for i in range(100)]
        run2 = [(i, 1.1 / (i + 1)) for i in range(100)]
        
        histories = {
            'sgd': {
                'train_loss': [run1, run2]
            }
        }
        
        fig = plot_training_curves(histories, ['sgd'], show_std=True)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_training_curves_save(self):
        """Test saving training curves to file."""
        histories = {
            'sgd': {
                'train_loss': [(i, 1.0 / (i + 1)) for i in range(100)]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'training_curves.pdf'
            fig = plot_training_curves(histories, ['sgd'], output_path=output_path)
            assert output_path.exists()
            plt.close(fig)


class TestGradientVariancePlotter:
    """Tests for gradient variance plotting."""
    
    def test_plot_gradient_variance(self):
        """Test plotting gradient variance evolution."""
        histories = {
            'sgd': {
                'train_grad_variance': [(i * 10, 1.0 / (i + 1)) for i in range(50)]
            },
            'svrg': {
                'train_grad_variance': [(i * 10, 0.5 / (i + 1)) for i in range(50)]
            }
        }
        
        fig = plot_gradient_variance(histories, ['sgd', 'svrg'])
        assert fig is not None
        plt.close(fig)


class TestValidationErrorPlotter:
    """Tests for validation error plotting."""
    
    def test_plot_validation_error_single_run(self):
        """Test plotting validation error with single run."""
        histories = {
            'sgd': {
                'val_relative_error': [(i * 10, 0.5 / (i + 1)) for i in range(50)]
            }
        }
        
        fig = plot_validation_error(histories, ['sgd'])
        assert fig is not None
        plt.close(fig)
    
    def test_plot_validation_error_with_confidence(self):
        """Test plotting validation error with confidence intervals."""
        run1 = [(i * 10, 0.5 / (i + 1)) for i in range(50)]
        run2 = [(i * 10, 0.6 / (i + 1)) for i in range(50)]
        
        histories = {
            'sgd': {
                'val_relative_error': [run1, run2]
            }
        }
        
        fig = plot_validation_error(histories, ['sgd'], confidence_level=0.95)
        assert fig is not None
        plt.close(fig)


class TestLongHorizonPredictionPlotter:
    """Tests for long-horizon prediction plotting."""
    
    def test_plot_logistic_map(self):
        """Test plotting Logistic Map predictions."""
        true_traj = np.random.rand(100)
        pred_traj = true_traj + np.random.randn(100) * 0.1
        
        fig = plot_long_horizon_predictions(true_traj, pred_traj, 'logistic')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_lorenz_system(self):
        """Test plotting Lorenz System predictions."""
        true_traj = np.random.rand(100, 3)
        pred_traj = true_traj + np.random.randn(100, 3) * 0.1
        
        fig = plot_long_horizon_predictions(true_traj, pred_traj, 'lorenz')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_burgers_equation(self):
        """Test plotting Burgers Equation predictions."""
        true_traj = np.random.rand(50, 64)
        pred_traj = true_traj + np.random.randn(50, 64) * 0.1
        
        fig = plot_long_horizon_predictions(true_traj, pred_traj, 'burgers')
        assert fig is not None
        plt.close(fig)
    
    def test_plot_invalid_system(self):
        """Test that invalid system name raises error."""
        true_traj = np.random.rand(100)
        pred_traj = np.random.rand(100)
        
        with pytest.raises(ValueError, match="Unknown system name"):
            plot_long_horizon_predictions(true_traj, pred_traj, 'invalid')


class TestEigenvaluePlotter:
    """Tests for eigenvalue comparison plotting."""
    
    def test_plot_eigenvalue_comparison(self):
        """Test plotting eigenvalue comparison."""
        # Create some eigenvalues on and near unit circle
        true_eigs = np.array([0.9 + 0.1j, 0.8 - 0.2j, -0.7 + 0.3j, -0.6 - 0.4j])
        learned_eigs = true_eigs + np.random.randn(4) * 0.05 + 1j * np.random.randn(4) * 0.05
        
        fig = plot_eigenvalue_comparison(true_eigs, learned_eigs)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_eigenvalue_comparison_real(self):
        """Test plotting real eigenvalues."""
        true_eigs = np.array([0.9, 0.5, -0.3, -0.8])
        learned_eigs = true_eigs + np.random.randn(4) * 0.05
        
        fig = plot_eigenvalue_comparison(true_eigs, learned_eigs)
        assert fig is not None
        plt.close(fig)


class TestCostVsAccuracyPlotter:
    """Tests for computational cost plotting."""
    
    def test_plot_cost_vs_accuracy(self):
        """Test plotting cost vs accuracy."""
        histories = {
            'sgd': {
                'val_loss': [(i * 10, 1.0 / (i + 1)) for i in range(50)],
                'iteration_time': [(i, 0.1) for i in range(500)]
            },
            'adam': {
                'val_loss': [(i * 10, 0.8 / (i + 1)) for i in range(50)],
                'iteration_time': [(i, 0.12) for i in range(500)]
            }
        }
        
        fig = plot_cost_vs_accuracy(histories, ['sgd', 'adam'])
        assert fig is not None
        plt.close(fig)


class TestBurgersSpatiotemporalPlotter:
    """Tests for Burgers equation spatiotemporal plotting."""
    
    def test_plot_burgers_spatiotemporal(self):
        """Test plotting Burgers equation spatiotemporal heatmap."""
        solution = np.random.rand(100, 64)
        
        fig = plot_burgers_spatiotemporal(solution)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_burgers_spatiotemporal_custom_title(self):
        """Test plotting with custom title."""
        solution = np.random.rand(100, 64)
        
        fig = plot_burgers_spatiotemporal(solution, title='Custom Title')
        assert fig is not None
        plt.close(fig)


class TestSaveAllPlots:
    """Tests for saving all plots at once."""
    
    def test_save_all_plots_basic(self):
        """Test saving all basic plots."""
        histories = {
            'sgd': {
                'train_loss': [(i, 1.0 / (i + 1)) for i in range(100)],
                'train_grad_variance': [(i * 10, 1.0 / (i + 1)) for i in range(50)],
                'val_relative_error': [(i * 10, 0.5 / (i + 1)) for i in range(50)],
                'iteration_time': [(i, 0.1) for i in range(100)]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = save_all_plots(histories, ['sgd'], tmpdir)
            
            assert 'training_curves' in figures
            assert 'gradient_variance' in figures
            assert 'validation_error' in figures
            assert 'cost_vs_accuracy' in figures
            
            # Check files exist
            assert (Path(tmpdir) / 'training_curves.pdf').exists()
            assert (Path(tmpdir) / 'gradient_variance.pdf').exists()
            assert (Path(tmpdir) / 'validation_error.pdf').exists()
            assert (Path(tmpdir) / 'cost_vs_accuracy.pdf').exists()
    
    def test_save_all_plots_with_optional(self):
        """Test saving all plots including optional ones."""
        histories = {
            'sgd': {
                'train_loss': [(i, 1.0 / (i + 1)) for i in range(100)],
                'train_grad_variance': [(i * 10, 1.0 / (i + 1)) for i in range(50)],
                'val_relative_error': [(i * 10, 0.5 / (i + 1)) for i in range(50)],
                'iteration_time': [(i, 0.1) for i in range(100)]
            }
        }
        
        true_traj = np.random.rand(100)
        pred_traj = true_traj + np.random.randn(100) * 0.1
        true_eigs = np.array([0.9 + 0.1j, 0.8 - 0.2j])
        learned_eigs = true_eigs + np.random.randn(2) * 0.05 + 1j * np.random.randn(2) * 0.05
        burgers_sol = np.random.rand(100, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = save_all_plots(
                histories, ['sgd'], tmpdir,
                true_trajectory=true_traj,
                predicted_trajectory=pred_traj,
                system_name='logistic',
                true_eigenvalues=true_eigs,
                learned_eigenvalues=learned_eigs,
                burgers_solution=burgers_sol
            )
            
            assert 'long_horizon_predictions' in figures
            assert 'eigenvalue_comparison' in figures
            assert 'burgers_spatiotemporal' in figures
            
            # Check files exist
            assert (Path(tmpdir) / 'long_horizon_predictions.pdf').exists()
            assert (Path(tmpdir) / 'eigenvalue_comparison.pdf').exists()
            assert (Path(tmpdir) / 'burgers_spatiotemporal.pdf').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
