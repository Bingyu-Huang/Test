import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions.
    
    Args:
        channels (int): Number of channels in the convolutional layers.
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class EventEncoder(nn.Module):
    """Event encoder for event voxel grid data.
    
    This module encodes event voxel grid data into feature representations
    that can be used for conditioning the diffusion model.
    
    Args:
        in_channels (int): Number of input channels of the event voxel grid.
            Default: 10 (5 bins × 2 polarities)
        mid_channels (int): Number of channels in the intermediate layers.
            Default: 64
        out_channels (int): Number of output channels. Default: 64
        n_blocks (int): Number of residual blocks. Default: 4
    """
    
    def __init__(self, in_channels=10, mid_channels=64, out_channels=64, n_blocks=4):
        super(EventEncoder, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(mid_channels) for _ in range(n_blocks)
        ])
        
        # Output convolution
        self.output_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input event voxel grid tensor with shape (B, C, H, W).
                C = bins * polarities, typically 10 (5 bins × 2 polarities)
                
        Returns:
            Tensor: Encoded event features with shape (B, out_channels, H, W)
        """
        # Initial convolution
        out = self.relu(self.initial_conv(x))
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Output convolution
        out = self.output_conv(out)
        
        return out


class DownsampleBlock(nn.Module):
    """Block for spatial downsampling.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).
                
        Returns:
            Tensor: Output tensor with shape (B, C, H/2, W/2).
        """
        return self.relu(self.conv(x))


class EventUNet(nn.Module):
    """U-Net architecture for event feature extraction at multiple scales.
    
    This module extracts event features at multiple scales, which can be used
    to condition the diffusion model at different resolutions.
    
    Args:
        in_channels (int): Number of input channels of the event voxel grid.
            Default: 10 (5 bins × 2 polarities)
        base_channels (int): Number of base channels. Default: 64
        num_scales (int): Number of scales to extract features. Default: 3
    """
    
    def __init__(self, in_channels=10, base_channels=64, num_scales=3):
        super(EventUNet, self).__init__()
        
        self.num_scales = num_scales
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder (downsampling) path
        self.down_blocks = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()
        
        in_ch = base_channels
        for i in range(num_scales):
            out_ch = in_ch * 2
            self.down_blocks.append(DownsampleBlock(in_ch, out_ch))
            self.down_res_blocks.append(ResidualBlock(out_ch))
            in_ch = out_ch
        
        # Feature processing blocks at each scale
        self.feature_blocks = nn.ModuleList()
        
        # Scale 0 (original resolution)
        self.feature_blocks.append(nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ))
        
        # Scales 1 to num_scales (downsampled)
        for i in range(num_scales):
            channels = base_channels * (2 ** (i+1))
            self.feature_blocks.append(nn.Sequential(
                ResidualBlock(channels),
                ResidualBlock(channels)
            ))
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input event voxel grid tensor with shape (B, C, H, W).
                
        Returns:
            list[Tensor]: List of event feature tensors at different scales.
        """
        # Initial convolution
        x = self.relu(self.initial_conv(x))
        
        # Store features at each scale
        features = [self.feature_blocks[0](x)]  # Original resolution
        
        # Downsampling path and feature extraction
        feat = x
        for i in range(self.num_scales):
            # Downsample
            feat = self.down_blocks[i](feat)
            feat = self.down_res_blocks[i](feat)
            
            # Extract features at this scale
            processed_feat = self.feature_blocks[i+1](feat)
            features.append(processed_feat)
        
        return features
