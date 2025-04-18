import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSplicingFusion(nn.Module):
    """Simple feature splicing fusion module.
    
    This module concatenates image features and event features along the channel dimension,
    then uses a 1x1 convolution to fuse them.
    
    Args:
        image_channels (int): Number of channels in the image features.
        event_channels (int): Number of channels in the event features.
        output_channels (int): Number of channels in the output features.
    """
    
    def __init__(self, image_channels, event_channels, output_channels):
        super(FeatureSplicingFusion, self).__init__()
        
        self.fusion_conv = nn.Conv2d(
            image_channels + event_channels, output_channels, kernel_size=1
        )
    
    def forward(self, image_feat, event_feat):
        """Forward function.
        
        Args:
            image_feat (Tensor): Image features with shape (B, image_channels, H, W).
            event_feat (Tensor): Event features with shape (B, event_channels, H, W).
            
        Returns:
            Tensor: Fused features with shape (B, output_channels, H, W).
        """
        # Ensure spatial dimensions match
        if image_feat.shape[2:] != event_feat.shape[2:]:
            event_feat = F.interpolate(
                event_feat, size=image_feat.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Concatenate features
        concat_feat = torch.cat([image_feat, event_feat], dim=1)
        
        # Fuse with 1x1 convolution
        out = self.fusion_conv(concat_feat)
        
        return out


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module.
    
    This module uses cross-attention to fuse image features with event features.
    The image features are treated as queries, while event features are treated as keys and values.
    
    Args:
        image_channels (int): Number of channels in the image features.
        event_channels (int): Number of channels in the event features.
        output_channels (int): Number of channels in the output features.
        num_heads (int): Number of attention heads.
    """
    
    def __init__(self, image_channels, event_channels, output_channels, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = image_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers for queries (from image features)
        self.to_q = nn.Linear(image_channels, image_channels)
        
        # Projection layers for keys and values (from event features)
        self.to_k = nn.Linear(event_channels, image_channels)
        self.to_v = nn.Linear(event_channels, image_channels)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(image_channels, output_channels),
            nn.Dropout(0.1)
        )
        
        # Spatial projections to match dimensions if needed
        self.img_proj = None
        if image_channels != output_channels:
            self.img_proj = nn.Conv2d(image_channels, output_channels, kernel_size=1)
    
    def forward(self, image_feat, event_feat):
        """Forward function.
        
        Args:
            image_feat (Tensor): Image features with shape (B, image_channels, H, W).
            event_feat (Tensor): Event features with shape (B, event_channels, H, W).
            
        Returns:
            Tensor: Fused features with shape (B, output_channels, H, W).
        """
        # Ensure spatial dimensions match
        if image_feat.shape[2:] != event_feat.shape[2:]:
            event_feat = F.interpolate(
                event_feat, size=image_feat.shape[2:], mode='bilinear', align_corners=False
            )
        
        B, C, H, W = image_feat.shape
        
        # Reshape for attention
        image_feat_flat = image_feat.flatten(2).transpose(1, 2)  # B, H*W, C
        event_feat_flat = event_feat.flatten(2).transpose(1, 2)  # B, H*W, C_event
        
        # Project queries, keys, values
        q = self.to_q(image_feat_flat)
        k = self.to_k(event_feat_flat)
        v = self.to_v(event_feat_flat)
        
        # Reshape for multi-head attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B, num_heads, H*W, head_dim
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B, num_heads, H*W, head_dim
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B, num_heads, H*W, head_dim
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # B, num_heads, H*W, H*W
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # B, num_heads, H*W, head_dim
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, H*W, C)  # B, H*W, C
        
        # Output projection
        out = self.to_out(out)  # B, H*W, output_channels
        
        # Reshape to spatial
        out = out.transpose(1, 2).view(B, -1, H, W)  # B, output_channels, H, W
        
        # Skip connection with projection if needed
        if self.img_proj is not None:
            return out + self.img_proj(image_feat)
        else:
            return out + image_feat


class AdaptiveFusionModule(nn.Module):
    """Adaptive fusion module for different fusion strategies.
    
    This module can switch between different fusion strategies based on configuration.
    
    Args:
        image_channels (int): Number of channels in the image features.
        event_channels (int): Number of channels in the event features.
        output_channels (int): Number of channels in the output features.
        fusion_type (str): Type of fusion ('splice', 'attention', 'add').
        num_heads (int): Number of attention heads if using attention fusion.
    """
    
    def __init__(self, image_channels, event_channels, output_channels, 
                 fusion_type='splice', num_heads=8):
        super(AdaptiveFusionModule, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'splice':
            self.fusion = FeatureSplicingFusion(
                image_channels, event_channels, output_channels
            )
        elif fusion_type == 'attention':
            self.fusion = CrossAttentionFusion(
                image_channels, event_channels, output_channels, num_heads
            )
        elif fusion_type == 'add':
            # Simple addition with projection if needed
            self.event_proj = nn.Conv2d(event_channels, output_channels, kernel_size=1)
            self.image_proj = None
            if image_channels != output_channels:
                self.image_proj = nn.Conv2d(image_channels, output_channels, kernel_size=1)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(self, image_feat, event_feat):
        """Forward function.
        
        Args:
            image_feat (Tensor): Image features with shape (B, image_channels, H, W).
            event_feat (Tensor): Event features with shape (B, event_channels, H, W).
            
        Returns:
            Tensor: Fused features with shape (B, output_channels, H, W).
        """
        if self.fusion_type in ['splice', 'attention']:
            return self.fusion(image_feat, event_feat)
        elif self.fusion_type == 'add':
            # Ensure spatial dimensions match
            if image_feat.shape[2:] != event_feat.shape[2:]:
                event_feat = F.interpolate(
                    event_feat, size=image_feat.shape[2:], mode='bilinear', align_corners=False
                )
            
            # Project event features
            event_feat = self.event_proj(event_feat)
            
            # Project image features if needed
            if self.image_proj is not None:
                image_feat = self.image_proj(image_feat)
            
            # Simple addition
            return image_feat + event_feat


class MultiScaleFusion(nn.Module):
    """Multi-scale fusion module.
    
    This module manages fusion at multiple scales, with a separate fusion module for each scale.
    
    Args:
        image_channels_list (list): List of channel counts for image features at each scale.
        event_channels_list (list): List of channel counts for event features at each scale.
        fusion_type (str): Type of fusion ('splice', 'attention', 'add').
    """
    
    def __init__(self, image_channels_list, event_channels_list, fusion_type='splice'):
        super(MultiScaleFusion, self).__init__()
        
        assert len(image_channels_list) == len(event_channels_list), \
            "Image and event channel lists must have the same length"
        
        self.num_scales = len(image_channels_list)
        self.fusion_modules = nn.ModuleList()
        
        for i in range(self.num_scales):
            img_ch = image_channels_list[i]
            evt_ch = event_channels_list[i]
            self.fusion_modules.append(
                AdaptiveFusionModule(img_ch, evt_ch, img_ch, fusion_type)
            )
    
    def forward(self, image_feats, event_feats):
        """Forward function.
        
        Args:
            image_feats (list): List of image feature tensors at different scales.
            event_feats (list): List of event feature tensors at different scales.
            
        Returns:
            list: List of fused feature tensors at different scales.
        """
        assert len(image_feats) == len(event_feats) == self.num_scales, \
            "Number of feature scales must match the number of fusion modules"
        
        fused_feats = []
        for i in range(self.num_scales):
            fused_feat = self.fusion_modules[i](image_feats[i], event_feats[i])
            fused_feats.append(fused_feat)
        
        return fused_feats
