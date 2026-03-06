"""Fourier Neural Operator (FNO) architecture implementation."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import NeuralOperator
from utils.exceptions import ShapeError


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution layer for FNO.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Multiply low-frequency modes with learned weights
    3. IFFT back to physical space
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes : int
        Number of Fourier modes to keep
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Learnable weights for Fourier modes (complex-valued)
        # Scale by 1/sqrt(in_channels * out_channels) for initialization
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, in_channels, spatial_dim]
        
        Returns
        -------
        torch.Tensor
            Output tensor with shape [batch, out_channels, spatial_dim]
        """
        batch_size = x.shape[0]
        spatial_dim = x.shape[2]
        
        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, dim=2)  # [batch, in_channels, modes_total]
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, 
            self.out_channels, 
            spatial_dim // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )
        
        # Multiply relevant modes with learned weights
        # Only process first 'modes' frequencies
        modes_to_use = min(self.modes, spatial_dim // 2 + 1)
        
        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)  # [in_channels, out_channels, modes]
        
        # Perform multiplication: einsum over input channels
        out_ft[:, :, :modes_to_use] = torch.einsum(
            'bix,iox->box',
            x_ft[:, :, :modes_to_use],
            weights_complex[:, :, :modes_to_use]
        )
        
        # IFFT back to physical space
        x_out = torch.fft.irfft(out_ft, n=spatial_dim, dim=2)
        
        return x_out


class FourierLayer(nn.Module):
    """
    Fourier layer combining spectral convolution and local convolution.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes : int
        Number of Fourier modes to keep
    activation : str
        Activation function name
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        modes: int,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.local_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(activations.keys())}"
            )
        
        return activations[activation.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier layer with residual connection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, in_channels, spatial_dim]
        
        Returns
        -------
        torch.Tensor
            Output tensor with shape [batch, out_channels, spatial_dim]
        """
        # Spectral convolution
        x1 = self.spectral_conv(x)
        
        # Local 1x1 convolution
        x2 = self.local_conv(x)
        
        # Combine and activate
        out = self.activation(x1 + x2)
        
        return out


class FNO(NeuralOperator):
    """
    Fourier Neural Operator architecture.
    
    FNO learns operators using spectral convolutions in Fourier space,
    enabling efficient learning of PDEs and dynamical systems.
    
    Parameters
    ----------
    input_channels : int
        Number of input channels
    output_channels : int
        Number of output channels
    modes : int
        Number of Fourier modes to keep
    width : int
        Hidden channel dimension
    num_layers : int
        Number of Fourier layers
    activation : str, optional
        Activation function name ('relu', 'tanh', 'gelu'), default 'relu'
    padding : int, optional
        Padding for domain extension, default 0
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modes: int,
        width: int,
        num_layers: int,
        activation: str = 'relu',
        padding: int = 0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        
        # Lifting layer: project input to hidden dimension
        self.lifting = nn.Linear(input_channels, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, width, modes, activation)
            for _ in range(num_layers)
        ])
        
        # Projection layer: project back to output dimension
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            self._get_activation(activation),
            nn.Linear(128, output_channels)
        )
        
        # Initialize weights with Xavier uniform
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        
        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(activations.keys())}"
            )
        
        return activations[activation.lower()]
    
    def _initialize_weights(self):
        """Initialize all weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_functions: torch.Tensor, 
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate operator on input functions at query points.
        
        For FNO, we treat the input_functions as spatiotemporal fields.
        
        Parameters
        ----------
        input_functions : torch.Tensor
            Discretized input functions with shape [batch, input_dim, num_sensors]
        query_points : torch.Tensor
            Evaluation locations with shape [batch, output_dim, num_queries]
            (Note: FNO processes entire spatial domain, query_points used for output shape)
        
        Returns
        -------
        torch.Tensor
            Operator output with shape [batch, output_dim, num_queries]
        """
        # Validate input shapes
        if input_functions.ndim != 3:
            raise ShapeError(
                "input_functions must be 3-dimensional",
                expected_shape=(None, self.input_channels, None),
                actual_shape=input_functions.shape,
                tensor_name="input_functions"
            )
        
        if query_points.ndim != 3:
            raise ShapeError(
                "query_points must be 3-dimensional",
                expected_shape=(None, self.output_channels, None),
                actual_shape=query_points.shape,
                tensor_name="query_points"
            )
        
        if input_functions.shape[1] != self.input_channels:
            raise ShapeError(
                f"input_functions dimension 1 must match input_channels={self.input_channels}",
                expected_shape=(None, self.input_channels, None),
                actual_shape=input_functions.shape,
                tensor_name="input_functions"
            )
        
        batch_size = input_functions.shape[0]
        spatial_dim = input_functions.shape[2]
        
        # Permute to [batch, spatial_dim, input_channels] for lifting
        x = input_functions.permute(0, 2, 1)  # [batch, spatial_dim, input_channels]
        
        # Lifting layer
        x = self.lifting(x)  # [batch, spatial_dim, width]
        
        # Permute to [batch, width, spatial_dim] for Fourier layers
        x = x.permute(0, 2, 1)  # [batch, width, spatial_dim]
        
        # Apply padding if specified
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))
        
        # Apply Fourier layers
        for fourier_layer in self.fourier_layers:
            x = fourier_layer(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding]
        
        # Permute back to [batch, spatial_dim, width] for projection
        x = x.permute(0, 2, 1)  # [batch, spatial_dim, width]
        
        # Projection layer
        x = self.projection(x)  # [batch, spatial_dim, output_channels]
        
        # Permute to match expected output shape [batch, output_dim, num_queries]
        x = x.permute(0, 2, 1)  # [batch, output_channels, spatial_dim]
        
        # Interpolate or slice to match query_points shape if needed
        num_queries = query_points.shape[2]
        if spatial_dim != num_queries:
            x = F.interpolate(x, size=num_queries, mode='linear', align_corners=True)
        
        return x
    
    def get_parameter_count(self) -> int:
        """
        Return total number of trainable parameters.
        
        Returns
        -------
        int
            Total count of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
