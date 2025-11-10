import torch
import torch.nn as nn
import torch.nn.functional as F
class DenseResidualNet(nn.Module):
    """Dense residual network with configurable activation, batch norm, and GLU options."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list,
                 activation: str = "gelu", batch_norm: bool = False,
                 dropout: float = 0.0, theta_with_glu: bool = False,
                 context_with_glu: bool = False, context_dim: int = 0):
        super().__init__()
        
        self.activation_name = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.theta_with_glu = theta_with_glu
        self.context_with_glu = context_with_glu
        self.context_dim = context_dim
        
        # Get activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # GLU preprocessing layers
        if theta_with_glu:
            theta_input_dim = input_dim - context_dim - 1  # subtract context and time
            self.theta_glu = nn.Linear(theta_input_dim, theta_input_dim)
            
        if context_with_glu:
            self.context_glu = nn.Linear(context_dim, context_dim)
        
        # Build the network layers
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation_layer())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.ModuleList(layers)
        
        # For residual connections
        self.residual_layers = []
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if current_dim == hidden_dim:
                self.residual_layers.append(True)
            else:
                self.residual_layers.append(False)
            current_dim = hidden_dim
    
    def _get_activation_layer(self):
        """Return activation as a layer for ModuleList."""
        class ActivationLayer(nn.Module):
            def __init__(self, activation_fn):
                super().__init__()
                self.activation_fn = activation_fn
            
            def forward(self, x):
                return self.activation_fn(x)
        
        return ActivationLayer(self.activation)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # Apply GLU preprocessing if enabled
        if hasattr(self, 'theta_glu') or hasattr(self, 'context_glu'):
            # Split input: [theta, context, time]
            theta_dim = x.shape[1] - self.context_dim - 1
            theta = x[:, :theta_dim]
            context = x[:, theta_dim:theta_dim + self.context_dim]
            time = x[:, -1:]
            
            if hasattr(self, 'theta_glu'):
                # Apply GLU to theta: split in half, apply sigmoid to second half
                theta_glu_out = self.theta_glu(theta)
                gate_dim = theta_glu_out.shape[1] // 2
                theta = theta_glu_out[:, :gate_dim] * torch.sigmoid(theta_glu_out[:, gate_dim:])
            
            if hasattr(self, 'context_glu'):
                # Apply GLU to context
                context_glu_out = self.context_glu(context)
                gate_dim = context_glu_out.shape[1] // 2
                context = context_glu_out[:, :gate_dim] * torch.sigmoid(context_glu_out[:, gate_dim:])
            
            # Recombine
            x = torch.cat([theta, context, time], dim=1)
        
        # Process through network with residual connections
        layer_idx = 0
        residual_idx = 0
        
        for layer in self.network[:-1]:  # All layers except output
            if isinstance(layer, nn.Linear):
                if residual_idx < len(self.residual_layers) and self.residual_layers[residual_idx]:
                    # Apply residual connection
                    residual = x
                    x = layer(x)
                    x = x + residual
                else:
                    x = layer(x)
                residual_idx += 1
            else:
                x = layer(x)
        
        # Output layer (no residual)
        x = self.network[-1](x)
        return x