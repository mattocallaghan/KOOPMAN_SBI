import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
from torchdiffeq import odeint
from nn import DenseResidualNet


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching model for learning posterior p(theta|x) in simulation based inference."""
    
    def __init__(self, input_dim: int, context_dim: int,
                 posterior_kwargs: dict = None, device: str = "cpu",
                 output_dir: str = "/logs"):
        super().__init__()
        self.input_dim = input_dim  # theta dimension
        self.context_dim = context_dim  # x dimension  
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        assert posterior_kwargs is not None,"posterior_kwargs must be provided."
        self.posterior_kwargs = posterior_kwargs
        self.sigma_min = posterior_kwargs["sigma_min"]
        self.time_prior_exponent = posterior_kwargs["time_prior_exponent"]
        
        # TensorBoard writer (initialized when training starts)
        self.writer = None
        
        # Vector field network: takes (theta, x, t) -> velocity field
        # Input: theta (input_dim) + x (context_dim) + t (1) 
        network_input_dim = input_dim + context_dim + 1  # +1 for time
        
        if posterior_kwargs["type"] == "DenseResidualNet":
            self.vector_field = DenseResidualNet(
                input_dim=network_input_dim,
                output_dim=input_dim,  # Output velocity field
                hidden_dims=posterior_kwargs["hidden_dims"],
                activation=posterior_kwargs["activation"],
                batch_norm=posterior_kwargs["batch_norm"],
                dropout=posterior_kwargs["dropout"],
                theta_with_glu=posterior_kwargs["theta_with_glu"],
                context_with_glu=posterior_kwargs["context_with_glu"],
                context_dim=context_dim
            )
        else:
            raise ValueError("Unsupported posterior network type.")
        
        # Training configuration
        self.optimizer_kwargs = {}
        self.scheduler_kwargs = {}
        self.optimizer = None
        self.scheduler = None
        

    def initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and scheduler from kwargs."""
        if self.optimizer_kwargs:
            optimizer_class = getattr(optim, self.optimizer_kwargs.get('name', 'Adam'))
            optimizer_params = {k: float(v) for k, v in self.optimizer_kwargs.items() if k != 'name'}
            self.optimizer = optimizer_class(self.parameters(), **optimizer_params)
        
        if self.scheduler_kwargs and self.optimizer:
            scheduler_type = self.scheduler_kwargs.get('type', 'step_lr')
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',  # Reduce when metric has stopped decreasing
                    factor=self.scheduler_kwargs['factor'],
                    patience=self.scheduler_kwargs['patience'],
                )
            elif scheduler_type == 'StepLR':
                scheduler_class = getattr(optim.lr_scheduler, 'StepLR')
                scheduler_params = {k: float(v) for k, v in self.scheduler_kwargs.items() 
                                   if k not in ['name', 'type']}
                self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample random times with power law distribution based on time_prior_exponent."""
        if self.time_prior_exponent == 1:
            # Uniform distribution
            return torch.rand(batch_size, device=self.device)
        else:
            # Power law distribution: t ~ U(0,1)^(1/time_prior_exponent)
            u = torch.rand(batch_size, device=self.device)
            return u ** (1.0 / self.time_prior_exponent)
    
    def sample_theta_0(self, batch_size: int) -> torch.Tensor:
        """Sample noise theta_0 ~ N(0, I)."""
        return torch.randn(batch_size, self.input_dim, device=self.device)
    
    def ot_conditional_flow(self, theta_0: torch.Tensor, theta_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Optimal transport conditional flow interpolation."""
        return (1 - (1 - self.sigma_min) * t)[:, None] * theta_0 + t[:, None] * theta_1
    
    def forward(self, t: torch.Tensor, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict velocity field v_t(theta|x). 
        
        Args:
            t: time tensor of shape (batch_size,)
            theta: theta tensor of shape (batch_size, input_dim)
            context: context tensor of shape (batch_size, context_dim)
        """
        # broadcase to match batch size if t is a scalar
        t = t * torch.ones(len(theta), device=theta.device)
        # Ensure time has correct shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Concatenate theta, context, and time
        input_tensor = torch.cat([theta, context, t], dim=-1)
        velocity = self.vector_field(input_tensor)
        return velocity
    
    def compute_flow_matching_loss(self, theta_target: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the flow matching loss for (theta, x) pairs."""
        batch_size = theta_target.shape[0]
        
        # Match the exact mathematical form provided
        mse = nn.MSELoss()
        
        # Sample random times t ~ U(0, 1) 
        t = self.sample_t(batch_size)
        
        # Sample noise theta_0 ~ N(0, I)
        theta_0 = self.sample_theta_0(batch_size)
        
        # theta_1 is the target data
        theta_1 = theta_target
        
        # Compute interpolated path using optimal transport
        theta_t = self.ot_conditional_flow(theta_0, theta_1, t)
        
        # True velocity field: theta_1 - (1 - sigma_min) * theta_0
        true_vf = theta_1 - (1 - self.sigma_min) * theta_0
        
        # Predict velocity field using the network
        predicted_vf = self.forward(t, theta_t, context)
        
        # Flow matching loss: MSE between predicted and true velocity fields
        loss = mse(predicted_vf, true_vf)
        
        return {
            'total_loss': loss,
            'flow_matching_loss': loss
        }
    
    def sample_batch(self, context: torch.Tensor,custom_theta_0=None) -> torch.Tensor:
        """Generate samples using Euler integration of the learned flow."""
        self.eval()
        batch_size = context.shape[0]
        with torch.no_grad():
            if custom_theta_0 is not None:
                assert custom_theta_0.shape[0] == batch_size, "custom_theta_0 batch size must match context batch size."
                theta_0 = custom_theta_0
            else:
                theta_0 = self.sample_theta_0(batch_size)
            _, theta_1 = odeint(
                lambda t, theta_t: self.forward(t, theta_t, context),
                theta_0,
                torch.tensor([0.0, 1.0 - self.sigma_min]).type(torch.float32).to(self.device),
                atol=1e-7,
                rtol=1e-7,
                method="dopri5",
            )

        return theta_1
    
    def log_prob_batch(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for theta given context using continuous normalizing flows."""
        # This would require implementing the log-determinant of the Jacobian
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("Log probability computation requires implementing CNF with log-det Jacobian.")
                
    def sample_and_log_prob_batch(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and their log probabilities."""
        raise NotImplementedError("Sampling and log probability computation requires implementing CNF with log-det Jacobian.")
    
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_losses = {'total_loss': 0.0, 'flow_matching_loss': 0.0}
        num_batches = 0
        for batch in train_loader:
            theta, context = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            losses = self.compute_flow_matching_loss(theta, context)
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_batches
        
        return total_losses
    
    def validation_epoch(self, validation_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                theta, context = batch[0].to(self.device), batch[1].to(self.device)
                losses = self.compute_flow_matching_loss(theta, context)
                total_loss += losses['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader, 
              train_dir: str, epochs: int, early_stopping: bool = True, 
              use_tensorboard: bool = True, patience: int = 10):
        """Train the model."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.to(self.device)
        
        if not self.optimizer:
            self.initialize_optimizer_and_scheduler()
        
        # Initialize TensorBoard writer
        if use_tensorboard:
            tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
        
        best_test_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_losses = self.train_epoch(train_loader)
            validation_loss = self.validation_epoch(validation_loader)
            
            if self.scheduler:
                # ReduceLROnPlateau requires validation loss as input
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_losses['total_loss']:.4f} "
                  f"(Flow Matching: {train_losses['flow_matching_loss']:.4f}), "
                  f"Validation Loss: {validation_loss:.4f}")
            
            # Log to TensorBoard
            if self.writer is not None:
                # Log training losses
                for key, value in train_losses.items():
                    self.writer.add_scalar(f"Train/{key}", value, epoch + 1)
                
                # Log test loss
                self.writer.add_scalar("Validation/total_loss", validation_loss, epoch + 1)
                
                # Log learning rate
                if self.optimizer:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("Training/learning_rate", lr, epoch + 1)
            
            # Save best model
            if validation_loss < best_test_loss:
                best_test_loss = validation_loss
                patience_counter = 0
                self.save(os.path.join(train_dir, "best_model.pt"))
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            self.writer = None
    
    def save(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'context_dim': self.context_dim,
            'posterior_kwargs': self.posterior_kwargs,
            'output_dir': self.output_dir,
            'optimizer_kwargs': self.optimizer_kwargs,
            'scheduler_kwargs': self.scheduler_kwargs,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            context_dim=checkpoint['context_dim'],
            posterior_kwargs=checkpoint.get('posterior_kwargs', None),
            output_dir=checkpoint.get('output_dir', './logs'),
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer_kwargs = checkpoint.get('optimizer_kwargs', {})
        model.scheduler_kwargs = checkpoint.get('scheduler_kwargs', {})
        return model
    




