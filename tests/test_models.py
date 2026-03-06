"""Unit tests for neural operator models."""

import torch
import pytest
from models import DeepONet, FNO


class TestDeepONet:
    """Test suite for DeepONet architecture."""
    
    def test_instantiation(self):
        """Test DeepONet can be instantiated with valid configuration."""
        model = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            basis_dim=32,
            activation='relu',
            use_bias=True
        )
        assert model is not None
        assert isinstance(model, DeepONet)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            basis_dim=32
        )
        
        batch_size = 4
        num_sensors = 50
        num_queries = 50
        
        input_functions = torch.randn(batch_size, 1, num_sensors)
        query_points = torch.randn(batch_size, 1, num_queries)
        
        output = model(input_functions, query_points)
        
        assert output.shape == (batch_size, 1, num_queries)
    
    def test_parameter_count(self):
        """Test parameter counting returns positive integer."""
        model = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            basis_dim=32
        )
        
        param_count = model.get_parameter_count()
        assert isinstance(param_count, int)
        assert param_count > 0
    
    def test_xavier_initialization(self):
        """Test weights are initialized with Xavier uniform."""
        model = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[64],
            trunk_layers=[64],
            basis_dim=32
        )
        
        # Check that weights have reasonable variance (Xavier property)
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Xavier uniform should have mean close to 0
                assert abs(param.mean().item()) < 0.1
                # And reasonable variance
                assert param.std().item() > 0.01
    
    def test_different_activations(self):
        """Test DeepONet works with different activation functions."""
        for activation in ['relu', 'tanh', 'gelu']:
            model = DeepONet(
                input_dim=1,
                output_dim=1,
                branch_layers=[32],
                trunk_layers=[32],
                basis_dim=16,
                activation=activation
            )
            
            input_functions = torch.randn(2, 1, 10)
            query_points = torch.randn(2, 1, 10)
            output = model(input_functions, query_points)
            
            assert output.shape == (2, 1, 10)
    
    def test_invalid_activation(self):
        """Test invalid activation raises ValueError."""
        with pytest.raises(ValueError):
            DeepONet(
                input_dim=1,
                output_dim=1,
                branch_layers=[32],
                trunk_layers=[32],
                basis_dim=16,
                activation='invalid'
            )


class TestFNO:
    """Test suite for FNO architecture."""
    
    def test_instantiation(self):
        """Test FNO can be instantiated with valid configuration."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=12,
            width=64,
            num_layers=4,
            activation='relu'
        )
        assert model is not None
        assert isinstance(model, FNO)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=12,
            width=64,
            num_layers=4
        )
        
        batch_size = 4
        spatial_dim = 64
        
        input_functions = torch.randn(batch_size, 1, spatial_dim)
        query_points = torch.randn(batch_size, 1, spatial_dim)
        
        output = model(input_functions, query_points)
        
        assert output.shape == (batch_size, 1, spatial_dim)
    
    def test_parameter_count(self):
        """Test parameter counting returns positive integer."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=12,
            width=64,
            num_layers=4
        )
        
        param_count = model.get_parameter_count()
        assert isinstance(param_count, int)
        assert param_count > 0
    
    def test_xavier_initialization(self):
        """Test weights are initialized with Xavier uniform."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=8,
            width=32,
            num_layers=2
        )
        
        # Check that linear layer weights have reasonable variance
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Xavier uniform should have mean close to 0
                assert abs(param.mean().item()) < 0.2
                # Spectral conv weights have custom initialization, so check variance is non-zero
                # Linear layers should have reasonable variance
                if 'spectral_conv' not in name:
                    assert param.std().item() > 0.01
                else:
                    # Spectral conv weights are initialized with smaller scale
                    assert param.std().item() > 0.0001
    
    def test_different_activations(self):
        """Test FNO works with different activation functions."""
        for activation in ['relu', 'tanh', 'gelu']:
            model = FNO(
                input_channels=1,
                output_channels=1,
                modes=8,
                width=32,
                num_layers=2,
                activation=activation
            )
            
            input_functions = torch.randn(2, 1, 32)
            query_points = torch.randn(2, 1, 32)
            output = model(input_functions, query_points)
            
            assert output.shape == (2, 1, 32)
    
    def test_invalid_activation(self):
        """Test invalid activation raises ValueError."""
        with pytest.raises(ValueError):
            FNO(
                input_channels=1,
                output_channels=1,
                modes=8,
                width=32,
                num_layers=2,
                activation='invalid'
            )
    
    def test_with_padding(self):
        """Test FNO works with padding."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=8,
            width=32,
            num_layers=2,
            padding=8
        )
        
        input_functions = torch.randn(2, 1, 32)
        query_points = torch.randn(2, 1, 32)
        output = model(input_functions, query_points)
        
        assert output.shape == (2, 1, 32)
    
    def test_different_spatial_dimensions(self):
        """Test FNO handles different input/output spatial dimensions."""
        model = FNO(
            input_channels=1,
            output_channels=1,
            modes=8,
            width=32,
            num_layers=2
        )
        
        input_functions = torch.randn(2, 1, 64)
        query_points = torch.randn(2, 1, 32)  # Different size
        output = model(input_functions, query_points)
        
        assert output.shape == (2, 1, 32)


class TestModelComparison:
    """Test suite comparing DeepONet and FNO."""
    
    def test_both_models_produce_output(self):
        """Test both models can process the same input."""
        batch_size = 2
        spatial_dim = 32
        
        input_functions = torch.randn(batch_size, 1, spatial_dim)
        query_points = torch.randn(batch_size, 1, spatial_dim)
        
        deeponet = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[32],
            trunk_layers=[32],
            basis_dim=16
        )
        
        fno = FNO(
            input_channels=1,
            output_channels=1,
            modes=8,
            width=32,
            num_layers=2
        )
        
        deeponet_output = deeponet(input_functions, query_points)
        fno_output = fno(input_functions, query_points)
        
        assert deeponet_output.shape == fno_output.shape
        assert deeponet_output.shape == (batch_size, 1, spatial_dim)
    
    def test_parameter_counts_are_different(self):
        """Test DeepONet and FNO have different parameter counts."""
        deeponet = DeepONet(
            input_dim=1,
            output_dim=1,
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            basis_dim=32
        )
        
        fno = FNO(
            input_channels=1,
            output_channels=1,
            modes=12,
            width=64,
            num_layers=4
        )
        
        deeponet_params = deeponet.get_parameter_count()
        fno_params = fno.get_parameter_count()
        
        # They should have different architectures
        assert deeponet_params != fno_params
