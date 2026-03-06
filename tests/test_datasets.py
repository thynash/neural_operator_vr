"""Unit tests for datasets module."""

import numpy as np
import pytest
import torch
from datasets import (
    LogisticMapDataset,
    LorenzSystemDataset,
    BurgersEquationDataset,
    create_train_val_split,
    create_dataloaders,
    OperatorDataset
)


class TestLogisticMapDataset:
    """Tests for LogisticMapDataset."""
    
    def test_generate_trajectory(self):
        """Test trajectory generation."""
        dataset = LogisticMapDataset(seed=42)
        trajectory = dataset.generate_trajectory(
            initial_condition=0.5,
            length=100,
            r=3.9
        )
        
        assert trajectory.shape == (100, 1)
        assert not np.any(np.isnan(trajectory))
        assert not np.any(np.isinf(trajectory))
    
    def test_create_operator_dataset(self):
        """Test operator dataset creation."""
        dataset = LogisticMapDataset(seed=42)
        inputs, outputs = dataset.create_operator_dataset(
            num_trajectories=10,
            trajectory_length=100,
            input_horizon=5,
            output_horizon=1,
            r=3.9
        )
        
        # Check shapes
        expected_samples = 10 * (100 - 5 - 1 + 1)
        assert inputs.shape == (expected_samples, 5, 1)
        assert outputs.shape == (expected_samples, 1, 1)
        
        # Check no NaN or inf
        assert not np.any(np.isnan(inputs))
        assert not np.any(np.isnan(outputs))
    
    def test_get_true_eigenvalues(self):
        """Test eigenvalue computation."""
        dataset = LogisticMapDataset()
        eigenvalues = dataset.get_true_eigenvalues(
            state=np.array([0.5]),
            r=3.9
        )
        
        assert eigenvalues.shape == (1,)
        assert eigenvalues.dtype == np.complex128
        
        # For x=0.5, r=3.9: eigenvalue = 3.9 * (1 - 2*0.5) = 0
        expected = 3.9 * (1 - 2 * 0.5)
        assert np.isclose(eigenvalues[0].real, expected)
    
    def test_deterministic_generation(self):
        """Test that same seed produces same results."""
        dataset1 = LogisticMapDataset(seed=42)
        trajectory1 = dataset1.generate_trajectory(0.5, 100, r=3.9)
        
        dataset2 = LogisticMapDataset(seed=42)
        trajectory2 = dataset2.generate_trajectory(0.5, 100, r=3.9)
        
        np.testing.assert_array_equal(trajectory1, trajectory2)


class TestLorenzSystemDataset:
    """Tests for LorenzSystemDataset."""
    
    def test_generate_trajectory(self):
        """Test trajectory generation."""
        dataset = LorenzSystemDataset(seed=42)
        trajectory = dataset.generate_trajectory(
            initial_condition=np.array([1.0, 1.0, 1.0]),
            length=100,
            sigma=10.0,
            rho=28.0,
            beta=8.0/3.0,
            dt=0.01
        )
        
        assert trajectory.shape == (100, 3)
        assert not np.any(np.isnan(trajectory))
        assert not np.any(np.isinf(trajectory))
    
    def test_create_operator_dataset(self):
        """Test operator dataset creation."""
        dataset = LorenzSystemDataset(seed=42)
        inputs, outputs = dataset.create_operator_dataset(
            num_trajectories=5,
            trajectory_length=50,
            input_horizon=10,
            output_horizon=1,
            sigma=10.0,
            rho=28.0,
            beta=8.0/3.0,
            dt=0.01
        )
        
        # Check shapes
        expected_samples = 5 * (50 - 10 - 1 + 1)
        assert inputs.shape == (expected_samples, 10, 3)
        assert outputs.shape == (expected_samples, 1, 3)
    
    def test_get_true_eigenvalues(self):
        """Test eigenvalue computation."""
        dataset = LorenzSystemDataset()
        eigenvalues = dataset.get_true_eigenvalues(
            state=np.array([0.0, 0.0, 0.0]),
            sigma=10.0,
            rho=28.0,
            beta=8.0/3.0
        )
        
        assert eigenvalues.shape == (3,)
        assert eigenvalues.dtype == np.complex128


class TestBurgersEquationDataset:
    """Tests for BurgersEquationDataset."""
    
    def test_generate_trajectory(self):
        """Test trajectory generation."""
        dataset = BurgersEquationDataset(seed=42)
        
        # Create simple initial condition
        spatial_resolution = 64
        x = np.linspace(0, 2*np.pi, spatial_resolution, endpoint=False)
        ic = np.sin(x)
        
        trajectory = dataset.generate_trajectory(
            initial_condition=ic,
            length=50,
            viscosity=0.01,
            spatial_resolution=spatial_resolution,
            dt=0.001
        )
        
        assert trajectory.shape == (50, spatial_resolution)
        assert not np.any(np.isnan(trajectory))
        assert not np.any(np.isinf(trajectory))
    
    def test_create_operator_dataset(self):
        """Test operator dataset creation."""
        dataset = BurgersEquationDataset(seed=42)
        inputs, outputs = dataset.create_operator_dataset(
            num_trajectories=3,
            trajectory_length=30,
            input_horizon=5,
            output_horizon=1,
            viscosity=0.01,
            spatial_resolution=64,
            dt=0.001,
            initial_condition_type="random_fourier",
            num_modes=3
        )
        
        # Check shapes
        expected_samples = 3 * (30 - 5 - 1 + 1)
        assert inputs.shape == (expected_samples, 5, 64)
        assert outputs.shape == (expected_samples, 1, 64)
    
    def test_get_true_eigenvalues(self):
        """Test eigenvalue computation."""
        dataset = BurgersEquationDataset()
        
        spatial_resolution = 64
        state = np.zeros(spatial_resolution)
        
        eigenvalues = dataset.get_true_eigenvalues(
            state=state,
            viscosity=0.01,
            spatial_resolution=spatial_resolution,
            num_eigenvalues=5
        )
        
        assert eigenvalues.shape == (5,)
        assert eigenvalues.dtype == np.complex128


class TestDataManagement:
    """Tests for data management utilities."""
    
    def test_train_val_split(self):
        """Test train/validation split."""
        inputs = np.random.randn(100, 10, 1)
        outputs = np.random.randn(100, 1, 1)
        
        train_inputs, train_outputs, val_inputs, val_outputs = create_train_val_split(
            inputs, outputs, train_ratio=0.8, shuffle=True, seed=42
        )
        
        assert len(train_inputs) == 80
        assert len(val_inputs) == 20
        assert train_inputs.shape[1:] == inputs.shape[1:]
        assert val_inputs.shape[1:] == inputs.shape[1:]
    
    def test_operator_dataset(self):
        """Test OperatorDataset wrapper."""
        inputs = np.random.randn(50, 10, 1)
        outputs = np.random.randn(50, 1, 1)
        
        dataset = OperatorDataset(inputs, outputs, normalize=True)
        
        assert len(dataset) == 50
        
        # Get a sample
        input_tensor, output_tensor = dataset[0]
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(output_tensor, torch.Tensor)
        assert input_tensor.shape == (10, 1)
        assert output_tensor.shape == (1, 1)
    
    def test_normalization_stats(self):
        """Test normalization statistics computation."""
        inputs = np.random.randn(50, 10, 1) * 10 + 5
        outputs = np.random.randn(50, 1, 1) * 2 + 3
        
        dataset = OperatorDataset(inputs, outputs, normalize=True)
        stats = dataset.get_normalization_stats()
        
        assert 'input_mean' in stats
        assert 'input_std' in stats
        assert 'output_mean' in stats
        assert 'output_std' in stats
        
        # Check that mean is close to expected
        assert np.abs(stats['input_mean'].mean() - 5) < 1.0
        assert np.abs(stats['output_mean'].mean() - 3) < 1.0
    
    def test_create_dataloaders(self):
        """Test DataLoader creation."""
        train_inputs = np.random.randn(80, 10, 1)
        train_outputs = np.random.randn(80, 1, 1)
        val_inputs = np.random.randn(20, 10, 1)
        val_outputs = np.random.randn(20, 1, 1)
        
        train_loader, val_loader, stats = create_dataloaders(
            train_inputs, train_outputs,
            val_inputs, val_outputs,
            batch_size=16,
            normalize=True
        )
        
        assert len(train_loader) == 5  # 80 / 16
        assert len(val_loader) == 2    # 20 / 16 (rounded up)
        
        # Check that we can iterate
        for batch_inputs, batch_outputs in train_loader:
            assert batch_inputs.shape[0] <= 16
            assert batch_outputs.shape[0] <= 16
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
