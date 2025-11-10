import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
from nn import DenseResidualNet


class KoopmanFlow(nn.Module):
    """Koopman-based distillation model for learning posterior p(theta|x) in simulation based inference."""
    
    def __init__(self, input_dim: int, context_dim: int,
                 lifting_dim: int = 256, network_kwargs: dict = None, device: str = "cpu",
                 lambda_rec: float = 1.0, lambda_lat: float = 1.0, lambda_pred: float = 1.0,
                 output_dir: str = "/logs"):
        super().__init__()
        self.input_dim = input_dim  # theta dimension
        self.context_dim = context_dim  # x dimension  
        self.lifting_dim = lifting_dim
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        # Set default network_kwargs if not provided
        assert network_kwargs is not None, "network_kwargs must be provided."

        self.network_kwargs = network_kwargs
        
        # Loss weights
        self.lambda_rec = lambda_rec
        self.lambda_lat = lambda_lat
        self.lambda_pred = lambda_pred
        
        # TensorBoard writer (initialized when training starts)
        self.writer = None
        
        # Encoder for epsilon (noise), conditioned on x, -> lifting space
        encoder_input_dim = input_dim + context_dim
        if network_kwargs["type"]== "DenseResidualNet":
            self.encoder = DenseResidualNet(
                input_dim=encoder_input_dim,
                output_dim=lifting_dim,
                hidden_dims=network_kwargs["hidden_dims"],
                activation=network_kwargs["activation"],
                batch_norm=network_kwargs["batch_norm"],
                dropout=network_kwargs["dropout"],
                theta_with_glu=network_kwargs["theta_with_glu"],
                context_with_glu=network_kwargs["context_with_glu"],
                context_dim=context_dim
            )
        else:
            raise ValueError(f"Unsupported network type: {network_kwargs.get('type')}")
        
        # Context-dependent Koopman operator: maps context x to Koopman matrix parameters
        # Generate the full Koopman matrix from context
        if network_kwargs.get("type") == "DenseResidualNet":
            self.koopman_generator = DenseResidualNet(
                input_dim=context_dim,
                output_dim=lifting_dim * lifting_dim,  # Generate full matrix
                hidden_dims=[64, 128, 256, 128, 64],  # Smaller network for matrix generation
                activation=network_kwargs["activation"],
                batch_norm=network_kwargs["batch_norm"],
                dropout=network_kwargs["dropout"],
                theta_with_glu=False,
                context_with_glu=network_kwargs["context_with_glu"],
                context_dim=context_dim
            )
        else:
            raise ValueError(f"Unsupported network type: {network_kwargs.get('type')}")
        
        # Decoder evolved koopman, conditioned on x, from lifting space back to theta space
        decoder_input_dim = lifting_dim + context_dim
        if network_kwargs.get("type") == "DenseResidualNet":
            # For decoder, reverse the hidden dimensions
            reversed_hidden_dims = network_kwargs["hidden_dims"][::-1]
            self.decoder = DenseResidualNet(
                input_dim=decoder_input_dim,
                output_dim=input_dim,
                hidden_dims=reversed_hidden_dims,
                activation=network_kwargs["activation"],
                batch_norm=network_kwargs["batch_norm"],
                dropout=network_kwargs["dropout"],
                theta_with_glu=False,  # No GLU for decoder output
                context_with_glu=network_kwargs["context_with_glu"],
                context_dim=context_dim
            )
        else:
            raise ValueError(f"Unsupported network type: {network_kwargs.get('type')}")
        
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
            scheduler_type = self.scheduler_kwargs['type']
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',  # Reduce when metric has stopped decreasing
                    factor=self.scheduler_kwargs['factor'],
                    patience=self.scheduler_kwargs['patience']
                )
            elif scheduler_type == 'StepLR':
                # Fallback to StepLR or other schedulers
                scheduler_class = getattr(optim.lr_scheduler, 'StepLR')
                scheduler_params = {k: float(v) for k, v in self.scheduler_kwargs.items() 
                                   if k not in ['name', 'type']}
                self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def forward(self, eps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass: (eps, theta, x) -> theta prediction."""
        batch_size = eps.shape[0]
        
        # lift epsilon conditioned on x
        eps_x_input = torch.cat([eps, context], dim=-1)
        z_lifted = self.encoder(eps_x_input)
        
        # Generate context-dependent Koopman matrix
        koopman_matrix_flat = self.koopman_generator(context)  # (batch_size, lifting_dim^2)
        koopman_matrix = koopman_matrix_flat.view(batch_size, self.lifting_dim, self.lifting_dim)
        
        # Apply context-dependent Koopman operator: z_evolved = K(x) @ z_lifted
        z_evolved = torch.bmm(koopman_matrix, z_lifted.unsqueeze(-1)).squeeze(-1)
        
        # Decode back to theta space, conditioned on x
        z_evolved_x_input = torch.cat([z_evolved, context], dim=-1)
        theta_pred = self.decoder(z_evolved_x_input)
        return theta_pred
    
    
    def compute_koopman_loss(self, eps: torch.Tensor, theta_target: torch.Tensor, 
                           context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the Koopman loss for (eps, theta, x) -> theta mapping."""
        batch_size = eps.shape[0]
        
        # Forward pass through the full pipeline
        eps_x_input = torch.cat([eps, context], dim=-1)
        z_lifted = self.encoder(eps_x_input)
        
        # Generate context-dependent Koopman matrix
        koopman_matrix_flat = self.koopman_generator(context)  # (batch_size, lifting_dim^2)
        koopman_matrix = koopman_matrix_flat.view(batch_size, self.lifting_dim, self.lifting_dim)
        
        # Apply context-dependent Koopman operator: z_evolved = K(x) @ z_lifted
        z_evolved = torch.bmm(koopman_matrix, z_lifted.unsqueeze(-1)).squeeze(-1)
        
        # Decode back to theta space, conditioned on x
        z_evolved_x_input = torch.cat([z_evolved, context], dim=-1)
        theta_pred = self.decoder(z_evolved_x_input)
        
        # Main prediction loss: how well we predict theta from eps
        L_pred = nn.MSELoss()(theta_pred, theta_target)
        
        # Reconstruction loss: ensure encoder-decoder consistency
        # Create a "clean" path by encoding theta directly as eps
        theta_x_input = torch.cat([theta_target, context], dim=-1)
        theta_lifted = self.encoder(theta_x_input)
        theta_lifted_x_input = torch.cat([theta_lifted, context], dim=-1)
        theta_rec = self.decoder(theta_lifted_x_input)
        L_rec = nn.MSELoss()(theta_rec, theta_target)
        
        # Latent dynamics loss: ensure consistency in lifted space
        L_lat = nn.MSELoss()(z_evolved, theta_lifted)
        
        # Total loss
        total_loss = (self.lambda_rec * L_rec + 
                     self.lambda_lat * L_lat + 
                     self.lambda_pred * L_pred)
        
        return {
            'total_loss': total_loss,
            'L_rec': L_rec,
            'L_lat': L_lat,
            'L_pred': L_pred
        }
    
    def sample_batch(self, context: torch.Tensor) -> torch.Tensor:
        """Generate samples by sampling epsilon and transforming via the model."""
        self.eval()
        with torch.no_grad():
            batch_size = context.shape[0]
            
            # Sample epsilon from a standard normal prior
            eps_prior = torch.randn(batch_size, self.input_dim, device=self.device)
            # Forward pass through the full pipeline
            samples = self.forward(eps_prior, context)
            
        return samples
    
    def log_prob_batch(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for theta given context."""
        raise NotImplementedError("Log probability computation is not implemented for KoopmanFlow.")
                
    def sample_and_log_prob_batch(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and their log probabilities."""
        raise NotImplementedError("Sampling and log probability computation is not implemented for KoopmanFlow.")
    
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_losses = {'total_loss': 0.0, 'L_rec': 0.0, 'L_lat': 0.0, 'L_pred': 0.0}
        num_batches = 0
        for batch in train_loader:
            eps, theta, context = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            self.optimizer.zero_grad()
            losses = self.compute_koopman_loss(eps, theta, context)
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
                eps, theta, context = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                losses = self.compute_koopman_loss(eps, theta, context)
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
                  f"(Rec: {train_losses['L_rec']:.4f}, "
                  f"Lat: {train_losses['L_lat']:.4f}, "
                  f"Pred: {train_losses['L_pred']:.4f}), "
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
            'lifting_dim': self.lifting_dim,
            'network_kwargs': self.network_kwargs,
            'output_dir': self.output_dir,
            'optimizer_kwargs': self.optimizer_kwargs,
            'scheduler_kwargs': self.scheduler_kwargs,
            'lambda_rec': self.lambda_rec,
            'lambda_lat': self.lambda_lat,
            'lambda_pred': self.lambda_pred,
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            context_dim=checkpoint['context_dim'],
            lifting_dim=checkpoint['lifting_dim'],
            network_kwargs=checkpoint.get('network_kwargs', None),
            output_dir=checkpoint.get('output_dir', './logs'),
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer_kwargs = checkpoint.get('optimizer_kwargs', {})
        model.scheduler_kwargs = checkpoint.get('scheduler_kwargs', {})
        model.lambda_rec = checkpoint.get('lambda_rec', 1.0)
        model.lambda_lat = checkpoint.get('lambda_lat', 1.0)
        model.lambda_pred = checkpoint.get('lambda_pred', 1.0)
        return model

