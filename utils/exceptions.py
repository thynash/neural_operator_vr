"""Custom exceptions for the neural operator variance reduction framework."""


class TrainingDivergenceError(Exception):
    """Raised when training diverges (NaN or infinite loss).
    
    This exception is raised when the training loop detects numerical
    instability, typically indicated by NaN or infinite loss values.
    
    Attributes
    ----------
    message : str
        Error message describing the divergence.
    diagnostics : dict
        Dictionary containing diagnostic information including:
        - iteration: Current training iteration
        - loss_value: The problematic loss value
        - recent_losses: List of recent loss values
        - gradient_norm: Current gradient norm
        - learning_rate: Current learning rate
        - batch_statistics: Statistics about the current batch
    suggestions : list
        List of suggested remedies.
    """
    
    def __init__(self, message: str, diagnostics: dict = None, suggestions: list = None):
        """Initialize TrainingDivergenceError.
        
        Parameters
        ----------
        message : str
            Error message describing the divergence.
        diagnostics : dict, optional
            Diagnostic information about the divergence.
        suggestions : list, optional
            List of suggested remedies.
        """
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.suggestions = suggestions or []
    
    def __str__(self):
        """Return formatted error message with diagnostics and suggestions."""
        msg = super().__str__()
        
        if self.diagnostics:
            msg += "\n\nDiagnostics:"
            for key, value in self.diagnostics.items():
                msg += f"\n  {key}: {value}"
        
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        
        return msg


class ShapeError(ValueError):
    """Raised when tensor shapes don't match expected dimensions.
    
    This exception is raised when tensor shape validation fails,
    providing detailed information about expected vs actual shapes.
    
    Attributes
    ----------
    message : str
        Error message describing the shape mismatch.
    expected_shape : tuple
        Expected tensor shape.
    actual_shape : tuple
        Actual tensor shape.
    tensor_name : str
        Name or description of the tensor.
    """
    
    def __init__(
        self,
        message: str,
        expected_shape: tuple = None,
        actual_shape: tuple = None,
        tensor_name: str = None
    ):
        """Initialize ShapeError.
        
        Parameters
        ----------
        message : str
            Error message describing the shape mismatch.
        expected_shape : tuple, optional
            Expected tensor shape.
        actual_shape : tuple, optional
            Actual tensor shape.
        tensor_name : str, optional
            Name or description of the tensor.
        """
        super().__init__(message)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.tensor_name = tensor_name
    
    def __str__(self):
        """Return formatted error message with shape information."""
        msg = super().__str__()
        
        if self.tensor_name:
            msg += f"\n  Tensor: {self.tensor_name}"
        
        if self.expected_shape is not None:
            msg += f"\n  Expected shape: {self.expected_shape}"
        
        if self.actual_shape is not None:
            msg += f"\n  Actual shape: {self.actual_shape}"
        
        return msg


class CheckpointCompatibilityError(Exception):
    """Raised when checkpoint is incompatible with current model.
    
    This exception is raised when attempting to load a checkpoint
    that doesn't match the current model architecture.
    
    Attributes
    ----------
    message : str
        Error message describing the incompatibility.
    checkpoint_info : dict
        Information about the checkpoint.
    model_info : dict
        Information about the current model.
    """
    
    def __init__(
        self,
        message: str,
        checkpoint_info: dict = None,
        model_info: dict = None
    ):
        """Initialize CheckpointCompatibilityError.
        
        Parameters
        ----------
        message : str
            Error message describing the incompatibility.
        checkpoint_info : dict, optional
            Information about the checkpoint.
        model_info : dict, optional
            Information about the current model.
        """
        super().__init__(message)
        self.checkpoint_info = checkpoint_info or {}
        self.model_info = model_info or {}
    
    def __str__(self):
        """Return formatted error message with compatibility information."""
        msg = super().__str__()
        
        if self.checkpoint_info:
            msg += "\n\nCheckpoint information:"
            for key, value in self.checkpoint_info.items():
                msg += f"\n  {key}: {value}"
        
        if self.model_info:
            msg += "\n\nCurrent model information:"
            for key, value in self.model_info.items():
                msg += f"\n  {key}: {value}"
        
        return msg


class GPUMemoryError(RuntimeError):
    """Raised when GPU memory is exhausted.
    
    This exception wraps torch.cuda.OutOfMemoryError and provides
    additional context and suggestions for resolution.
    
    Attributes
    ----------
    message : str
        Error message describing the memory issue.
    memory_stats : dict
        GPU memory usage statistics.
    suggestions : list
        List of suggested remedies.
    """
    
    def __init__(
        self,
        message: str,
        memory_stats: dict = None,
        suggestions: list = None
    ):
        """Initialize GPUMemoryError.
        
        Parameters
        ----------
        message : str
            Error message describing the memory issue.
        memory_stats : dict, optional
            GPU memory usage statistics.
        suggestions : list, optional
            List of suggested remedies.
        """
        super().__init__(message)
        self.memory_stats = memory_stats or {}
        self.suggestions = suggestions or []
    
    def __str__(self):
        """Return formatted error message with memory stats and suggestions."""
        msg = super().__str__()
        
        if self.memory_stats:
            msg += "\n\nGPU Memory Statistics:"
            for key, value in self.memory_stats.items():
                msg += f"\n  {key}: {value}"
        
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        
        return msg
