"""DeepONet (Deep Operator Network) architecture implementation."""

from typing import List, Optional
import torch
import torch.nn as nn
from models.base import NeuralOperator
from utils.exceptions import ShapeError


class DeepONet(NeuralOperator):
    """
    Deep Operator Network architecture.
    
    DeepONet learns operators by decomposing them into branch and trunk networks.
    The branch network processes input function samples, while the trunk network
    processes query locations. The outputs are combined via inner product.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input function values at each sensor
    output_dim : int
        Dimension of output function values at each query point
    branch_layers : List[int]
        Hidden layer sizes for branch network (e.g., [128, 128, 64])
    trunk_layers : List[int]
        Hidden layer sizes for trunk network (e.g., [128, 128, 64])
    basis_dim : int
        Dimension of basis function space
    activation : str, optional
        Activation function name ('relu', 'tanh', 'gelu'), default 'relu'
    use_bias : bool, optional
        Whether to include bias terms in linear layers, default True
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch_layers: List[int],
        trunk_layers: List[int],
        basis_dim: int,
        activation: str = 'relu',
        use_bias: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_dim = basis_dim
        
        # Build branch network
        self.branch_net = self._build_mlp(
            input_dim, 
            branch_layers, 
            basis_dim, 
            activation, 
            use_bias
        )
        
        # Build trunk network
        self.trunk_net = self._build_mlp(
            output_dim, 
            trunk_layers, 
            basis_dim, 
            activation, 
            use_bias
        )
        
        # Initialize weights with Xavier uniform
        self._initialize_weights()
    
    def _build_mlp(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str,
        use_bias: bool
    ) -> nn.Sequential:
        """
        Build a multi-layer perceptron.
        
        Parameters
        ----------
        input_size : int
            Input dimension
        hidden_layers : List[int]
            List of hidden layer sizes
        output_size : int
            Output dimension
        activation : str
            Activation function name
        use_bias : bool
            Whether to use bias terms
        
        Returns
        -------
        nn.Sequential
            MLP module
        """
        layers = []
        
        # Get activation function
        act_fn = self._get_activation(activation)
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            layers.append(act_fn)
            prev_size = hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size, bias=use_bias))
        
        return nn.Sequential(*layers)
    
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
    
    def forward(
        self, 
        input_functions: torch.Tensor, 
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate operator on input functions at query points.
        
        Parameters
        ----------
        input_functions : torch.Tensor
            Discretized input functions with shape [batch, input_dim, num_sensors]
        query_points : torch.Tensor
            Evaluation locations with shape [batch, output_dim, num_queries]
        
        Returns
        -------
        torch.Tensor
            Operator output with shape [batch, output_dim, num_queries]
        """
        # Validate input shapes
        if input_functions.ndim != 3:
            raise ShapeError(
                "input_functions must be 3-dimensional",
                expected_shape=(None, self.input_dim, None),
                actual_shape=input_functions.shape,
                tensor_name="input_functions"
            )
        
        if query_points.ndim != 3:
            raise ShapeError(
                "query_points must be 3-dimensional",
                expected_shape=(None, self.output_dim, None),
                actual_shape=query_points.shape,
                tensor_name="query_points"
            )
        
        if input_functions.shape[1] != self.input_dim:
            raise ShapeError(
                f"input_functions dimension 1 must match input_dim={self.input_dim}",
                expected_shape=(None, self.input_dim, None),
                actual_shape=input_functions.shape,
                tensor_name="input_functions"
            )
        
        if query_points.shape[1] != self.output_dim:
            raise ShapeError(
                f"query_points dimension 1 must match output_dim={self.output_dim}",
                expected_shape=(None, self.output_dim, None),
                actual_shape=query_points.shape,
                tensor_name="query_points"
            )
        
        batch_size = input_functions.shape[0]
        num_sensors = input_functions.shape[2]
        num_queries = query_points.shape[2]
        
        # Branch network: process input function samples
        # Reshape: [batch, input_dim, num_sensors] -> [batch * num_sensors, input_dim]
        branch_input = input_functions.permute(0, 2, 1).reshape(-1, self.input_dim)
        branch_output = self.branch_net(branch_input)  # [batch * num_sensors, basis_dim]
        
        # Average over sensors to get basis coefficients
        branch_output = branch_output.reshape(batch_size, num_sensors, self.basis_dim)
        branch_output = branch_output.mean(dim=1)  # [batch, basis_dim]
        
        # Trunk network: process query locations
        # Reshape: [batch, output_dim, num_queries] -> [batch * num_queries, output_dim]
        trunk_input = query_points.permute(0, 2, 1).reshape(-1, self.output_dim)
        trunk_output = self.trunk_net(trunk_input)  # [batch * num_queries, basis_dim]
        trunk_output = trunk_output.reshape(batch_size, num_queries, self.basis_dim)
        
        # Combine via inner product: branch · trunk
        # branch_output: [batch, basis_dim]
        # trunk_output: [batch, num_queries, basis_dim]
        # Result: [batch, num_queries]
        output = torch.einsum('bd,bqd->bq', branch_output, trunk_output)
        
        # Reshape to match expected output: [batch, output_dim, num_queries]
        # For now, assume output_dim = 1 (scalar output at each query point)
        output = output.unsqueeze(1).expand(-1, self.output_dim, -1)
        
        return output
    
    def get_parameter_count(self) -> int:
        """
        Return total number of trainable parameters.
        
        Returns
        -------
        int
            Total count of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
