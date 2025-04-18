import math
import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion Model for image denoising.
    
    Implements the diffusion process defined in the DDPM paper.
    
    Args:
        model (nn.Module): The model that predicts noise or x0.
        image_size (int): Size of the input images.
        channels (int): Number of channels in the input images.
        timesteps (int): Number of timesteps in the diffusion process.
        loss_type (str): Type of loss to use ('l1', 'l2').
        beta_schedule (str): Schedule for variance ('linear', 'cosine').
    """
    
    def __init__(
        self,
        model,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type='l2',
        beta_schedule='linear'
    ):
        super(GaussianDiffusion, self).__init__()
        
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-calculate diffusion parameters
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                            torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', 
                            torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', 
                            torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 
                            torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', 
                            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', 
                            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process.
        
        Given x_0 and t, sample from q(x_t | x_0).
        
        Args:
            x_0 (Tensor): Initial clean image.
            t (Tensor): Timesteps.
            noise (Tensor, optional): Noise to add. If None, random Gaussian noise is used.
            
        Returns:
            Tensor: Noisy image at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Sample from q(x_t | x_0) = N(x_t; sqrt(alphas_cumprod) * x_0, sqrt(1 - alphas_cumprod) * I)
        sample = (
            self.sqrt_alphas_cumprod[t, None, None, None] * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )
        
        return sample
    
    def p_losses(self, x_0, t, event_features=None, noise=None):
        """Compute training losses.
        
        Args:
            x_0 (Tensor): Clean images.
            t (Tensor): Timesteps.
            event_features (list, optional): Event features at different scales.
            noise (Tensor, optional): Noise to add. If None, random Gaussian noise is used.
            
        Returns:
            Tensor: Loss value.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Generate noisy image at timestep t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict the noise using the model
        if event_features is not None:
            noise_pred = self.model(x_t, t, event_features)
        else:
            noise_pred = self.model(x_t, t)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = torch.abs(noise - noise_pred).mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(self, x_0, event_features=None):
        """Forward pass for training.
        
        Args:
            x_0 (Tensor): Clean images.
            event_features (list, optional): Event features at different scales.
            
        Returns:
            Tensor: Loss value.
        """
        # Sample random timesteps
        b = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_0.device).long()
        
        # Compute loss
        return self.p_losses(x_0, t, event_features)
    
    @torch.no_grad()
    def p_sample(self, x_t, t, event_features=None):
        """Single step of the reverse diffusion sampling process.
        
        Args:
            x_t (Tensor): Noisy image at timestep t.
            t (Tensor): Current timestep.
            event_features (list, optional): Event features at different scales.
            
        Returns:
            Tensor: Less noisy image at timestep t-1.
        """
        # Get model prediction
        if event_features is not None:
            noise_pred = self.model(x_t, t, event_features)
        else:
            noise_pred = self.model(x_t, t)
        
        # Get the posterior mean and variance
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_t +
            self.posterior_mean_coef2[t] * noise_pred
        )
        
        # No noise when t == 0
        if t[0] == 0:
            return posterior_mean
        else:
            posterior_variance = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, event_features=None):
        """Full reverse diffusion sampling loop.
        
        Args:
            shape (tuple): Shape of the target image.
            event_features (list, optional): Event features at different scales.
            
        Returns:
            Tensor: Generated clean image.
        """
        device = self.betas.device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                event_features
            )
        
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=1, event_features=None):
        """Generate samples from the model.
        
        Args:
            batch_size (int): Number of samples to generate.
            event_features (list, optional): Event features at different scales.
            
        Returns:
            Tensor: Generated clean images.
        """
        image_shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(image_shape, event_features)
    
    @torch.no_grad()
    def conditional_sample(self, blur_image, event_encoder, timesteps=20):
        """Generate deblurred image from a blurry image and event data.
        
        Args:
            blur_image (Tensor): Blurry image input.
            event_encoder (nn.Module): Event encoder model.
            timesteps (int): Number of sampling steps (can be less than training steps).
            
        Returns:
            Tensor: Deblurred image.
        """
        device = self.betas.device
        b = blur_image.shape[0]
        
        # Start from the blurry image with some noise
        x = blur_image + 0.1 * torch.randn_like(blur_image)
        
        # Get event features (assuming event data is already processed by caller)
        event_features = event_encoder
        
        # Use a subset of timesteps for faster inference
        step_size = self.timesteps // timesteps
        timestep_seq = list(range(0, self.timesteps, step_size))
        
        # Iteratively denoise
        for i in reversed(timestep_seq):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, event_features)
        
        return x


class EventConditionedDiffusion(nn.Module):
    """Event-conditioned diffusion model for image deblurring.
    
    This model combines an event encoder with a diffusion model for deblurring.
    
    Args:
        diffusion_model (nn.Module): The diffusion model.
        event_encoder (nn.Module): The event encoder.
    """
    
    def __init__(self, diffusion_model, event_encoder):
        super(EventConditionedDiffusion, self).__init__()
        
        self.diffusion = diffusion_model
        self.event_encoder = event_encoder
    
    def forward(self, blur_img, sharp_img, event_data, train=True):
        """Forward pass.
        
        Args:
            blur_img (Tensor): Blurry input image.
            sharp_img (Tensor): Sharp ground truth image (for training).
            event_data (Tensor): Event data.
            train (bool): Whether in training mode.
            
        Returns:
            Tensor or tuple: Loss during training, or deblurred image during inference.
        """
        # Encode event data
        event_features = self.event_encoder(event_data)
        
        if train:
            # Training: compute diffusion loss
            loss = self.diffusion(sharp_img, event_features)
            return loss
        else:
            # Inference: generate deblurred image
            deblurred = self.diffusion.conditional_sample(blur_img, event_features)
            return deblurred
