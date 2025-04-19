import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion model timesteps.
    
    Args:
        embedding_dim (int): Dimension of the embedding.
    """
    
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Linear layers for time embedding
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, 4 * embedding_dim)
        
        # Activation
        self.act = nn.SiLU()
    
    def forward(self, t):
        """Forward function.
        
        Args:
            t (Tensor): Timestep tensor of shape [B].
            
        Returns:
            Tensor: Time embedding of shape [B, 4*embedding_dim].
        """
        # Calculate the sinusoidal position embedding
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Map to higher dimension
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)
        
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time conditioning.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_channels (int): Number of channels for time embedding.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.0):
        super(ResidualBlock, self).__init__()
        
        # First conv block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, time_emb):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            time_emb (Tensor): Time embedding of shape [B, time_channels].
            
        Returns:
            Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        # Identity for skip connection
        identity = self.skip_connection(x)
        
        # First conv block
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        
        # Add time embedding
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_out
        
        # Second conv block
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Skip connection
        return out + identity


class AttentionBlock(nn.Module):
    """Self-attention block.
    
    Args:
        channels (int): Number of input channels.
        num_heads (int): Number of attention heads.
    """
    
    def __init__(self, channels, num_heads=8):
        super(AttentionBlock, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Normalization and projections
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            
        Returns:
            Tensor: Output tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Get q, k, v
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 2, 3)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        
        # Attention
        scale = 1 / math.sqrt(C // self.num_heads)
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Projection
        out = self.proj(out)
        
        # Skip connection
        return out + x


class DownBlock(nn.Module):
    """Downsampling block with residual connections and optional attention.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_channels (int): Number of channels for time embedding.
        has_attn (bool): Whether to use attention.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False, dropout=0.0):
        super(DownBlock, self).__init__()
        
        # Residual blocks
        self.res1 = ResidualBlock(in_channels, out_channels, time_channels, dropout)
        self.res2 = ResidualBlock(out_channels, out_channels, time_channels, dropout)
        
        # Attention
        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        
        # Downsampling
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, time_emb):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            time_emb (Tensor): Time embedding of shape [B, time_channels].
            
        Returns:
            tuple: (skip_connection, downsampled_output)
        """
        # First residual block
        h = self.res1(x, time_emb)
        
        # Second residual block
        h = self.res2(h, time_emb)
        
        # Attention if needed
        if self.has_attn:
            h = self.attn(h)
        
        # Return both the current level (for skip connection) and downsampled
        return h, self.downsample(h)


class UpBlock(nn.Module):
    """Upsampling block with residual connections and optional attention.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_channels (int): Number of channels for time embedding.
        has_attn (bool): Whether to use attention.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False, dropout=0.0):
        super(UpBlock, self).__init__()
        
        # Residual blocks
        # Note: in_channels needs to be doubled because of the skip connection
        self.res1 = ResidualBlock(in_channels + out_channels, out_channels, time_channels, dropout)
        self.res2 = ResidualBlock(out_channels, out_channels, time_channels, dropout)
        
        # Attention
        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip, time_emb):
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            skip (Tensor): Skip connection tensor of shape [B, C', H, W].
            time_emb (Tensor): Time embedding of shape [B, time_channels].

        Returns:
            Tensor: Output tensor of shape [B, out_channels, H*2, W*2].
        """
        # Upsample first
        h = self.upsample(x)

        # Ensure spatial dimensions match before concatenating
        if h.shape[2:] != skip.shape[2:]:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)


        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)

        # Apply residual blocks
        h = self.res1(h, time_emb)
        h = self.res2(h, time_emb)

        # Attention if needed
        if self.has_attn:
            h = self.attn(h)


        return h


class MiddleBlock(nn.Module):
    """Middle block with residual connections and attention.
    
    Args:
        channels (int): Number of channels.
        time_channels (int): Number of channels for time embedding.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, channels, time_channels, dropout=0.0):
        super(MiddleBlock, self).__init__()
        
        self.res1 = ResidualBlock(channels, channels, time_channels, dropout)
        self.attn = AttentionBlock(channels)
        self.res2 = ResidualBlock(channels, channels, time_channels, dropout)
    
    def forward(self, x, time_emb):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            time_emb (Tensor): Time embedding of shape [B, time_channels].
            
        Returns:
            Tensor: Output tensor of shape [B, C, H, W].
        """
        h = self.res1(x, time_emb)
        h = self.attn(h)
        h = self.res2(h, time_emb)
        return h


class UNet(nn.Module):
    """U-Net model for diffusion.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB).
        model_channels (int): Base number of channels.
        out_channels (int): Number of output channels (typically same as in_channels).
        num_res_blocks (int): Number of residual blocks per resolution.
        attention_resolutions (tuple): Resolutions at which to apply attention.
        dropout (float): Dropout rate.
        channel_mult (tuple): Channel multiplier for each resolution.
        time_embedding_dim (int): Dimension of time embedding.
    """
    
    def __init__(
        self,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        time_embedding_dim=128,
    ):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        time_embed_dim = time_embedding_dim * 4
        self.time_embed = TimeEmbedding(time_embedding_dim)
        
        # Initial projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling path
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, mult * model_channels, time_embed_dim, dropout
                    )
                ]
                ch = mult * model_channels
                
                # Add attention if needed
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # Add downsampling except for the last level
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    DownBlock(ch, ch, time_embed_dim, False, dropout)
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = MiddleBlock(ch, time_embed_dim, dropout)
        
        # Upsampling path
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout,
                    )
                ]
                ch = model_channels * mult
                
                # Add attention if needed
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                # Add upsampling except for the last block
                if level > 0 and i == num_res_blocks:
                    layers.append(UpBlock(ch, ch, time_embed_dim, False, dropout))
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Final output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            timesteps (Tensor): Timesteps tensor of shape [B].
            
        Returns:
            Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Downsampling path
        h = x
        skips = []
        
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
                skips.append(h)
            elif isinstance(module, DownBlock):
                skip_connection, h = module(h, time_emb)
                skips.append(skip_connection)
            else:  # ModuleList of residual blocks and attention
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
                skips.append(h)
        
        # Middle block
        h = self.middle_block(h, time_emb)
        
        # Upsampling path
        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, UpBlock):
                    h = layer(h, skips[-1], time_emb)
                else:
                    h = layer(h)
        
        # Final output
        return self.out(h)


class ConditionedUNet(nn.Module):
    """U-Net model conditioned on event features.
    
    This extends the base UNet to incorporate event features at multiple scales.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB).
        model_channels (int): Base number of channels.
        out_channels (int): Number of output channels (typically same as in_channels).
        num_res_blocks (int): Number of residual blocks per resolution.
        attention_resolutions (tuple): Resolutions at which to apply attention.
        dropout (float): Dropout rate.
        channel_mult (tuple): Channel multiplier for each resolution.
        time_embedding_dim (int): Dimension of time embedding.
        event_channels (int): Number of channels in event features at base resolution.
    """
    
    def __init__(
        self,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        time_embedding_dim=128,
        event_channels=64,
        fusion_type="splice",
    ):
        super(ConditionedUNet, self).__init__()
        
        # Base UNet
        self.base_unet = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            time_embedding_dim=time_embedding_dim,
        )
        
        # Event feature channels at each scale
        event_ch_list = [event_channels * mult for mult in channel_mult]
        
        # Image feature channels at each scale
        img_ch_list = [model_channels * mult for mult in channel_mult]
        
        # Event fusion modules
        self.fusion_type = fusion_type
        self.fusion_modules = nn.ModuleList()
        
        for i in range(len(channel_mult)):
            # For each resolution level, create a fusion module
            self.fusion_modules.append(
                nn.Conv2d(img_ch_list[i] + event_ch_list[i], img_ch_list[i], kernel_size=1)
                if fusion_type == "splice" else
                nn.Conv2d(event_ch_list[i], img_ch_list[i], kernel_size=1)
            )

    def forward(self, x, timesteps, event_features):
        """Forward function with event conditioning.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            timesteps (Tensor): Timesteps tensor of shape [B].
            event_features (list or Tensor): Event feature tensors.

        Returns:
            Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        # Time embedding
        time_emb = self.base_unet.time_embed(timesteps)

        # Downsampling path
        h = x
        skips = []

        # Keep track of current scale
        scale_idx = 0
        curr_resolution = x.shape[2]  # Height of the input
        target_resolutions = [x.shape[2] // (2 ** i) for i in range(len(self.base_unet.channel_mult))]

        # Handle case where event_features is a single tensor, not a list
        if not isinstance(event_features, list):
            event_features = [event_features]  # Convert to list with single item

        for i, module in enumerate(self.base_unet.input_blocks):
            if isinstance(module, nn.Conv2d):
                h = module(h)
                skips.append(h)
            elif isinstance(module, DownBlock):
                skip_connection, h = module(h, time_emb)

                # Track the shapes for debugging
                skip_shape = skip_connection.shape
                h_shape = h.shape

                # Only try to inject event features if we have them available
                if scale_idx < len(event_features) and scale_idx < len(self.fusion_modules):
                    event_feat = event_features[scale_idx]

                    # Ensure spatial dimensions match
                    if h.shape[2:] != event_feat.shape[2:]:
                        event_feat = F.interpolate(
                            event_feat, size=h.shape[2:], mode='bilinear', align_corners=False
                        )

                    # Apply fusion
                    if self.fusion_type == "splice":
                        h = torch.cat([h, event_feat], dim=1)
                        h = self.fusion_modules[scale_idx](h)
                    else:  # add
                        event_feat = self.fusion_modules[scale_idx](event_feat)
                        h = h + event_feat

                    scale_idx += 1

                skips.append(skip_connection)
            else:  # ModuleList of residual blocks and attention
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
                skips.append(h)

        # Middle block
        h = self.base_unet.middle_block(h, time_emb)

        # Upsampling path - be more careful with skip connections
        for i, module in enumerate(self.base_unet.output_blocks):
            # Get the skip connection and ensure dimensions match
            skip = skips.pop()

            # If dimensions don't match, resize h to match skip's spatial dimensions
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Now concatenate
            h = torch.cat([h, skip], dim=1)

            # Process through layers
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, UpBlock):
                    # Ensure there's still a skip connection available
                    if skips:
                        next_skip = skips[-1]
                        # Ensure the upsampled feature and skip connection have the same spatial dimensions
                        if h.shape[2:] != next_skip.shape[2:]:
                            h = F.interpolate(h, size=next_skip.shape[2:], mode='bilinear', align_corners=False)
                        h = layer(h, next_skip, time_emb)
                    else:
                        # If no skip left, just pass through without skip connection
                        h = layer.upsample(h)
                        h = layer.res1(h, time_emb)
                        h = layer.res2(h, time_emb)
                        if layer.has_attn:
                            h = layer.attn(h)

        # Final output
        return self.base_unet.out(h)