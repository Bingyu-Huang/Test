import os
import sys
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# Add project directory to path
sys.path.insert(0, os.path.abspath('.'))

# Import project modules
from basicsr.data.event_deblur_dataset import EventDeblurDataset
from basicsr.archs.event_encoder_arch import EventEncoder
from basicsr.archs.unet_arch import ConditionedUNet
from basicsr.archs.diffusion_arch import GaussianDiffusion, EventConditionedDiffusion


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """Build model from configuration."""
    # Build event encoder
    encoder_cfg = config['network_g']['event_encoder']
    event_encoder = EventEncoder(
        in_channels=encoder_cfg['in_channels'],
        mid_channels=encoder_cfg['mid_channels'],
        out_channels=encoder_cfg['out_channels'],
        n_blocks=encoder_cfg['n_blocks']
    )
    
    # Build UNet
    unet_cfg = config['network_g']['unet']
    unet = ConditionedUNet(
        in_channels=unet_cfg['in_channels'],
        model_channels=unet_cfg['model_channels'],
        out_channels=unet_cfg['out_channels'],
        num_res_blocks=unet_cfg['num_res_blocks'],
        attention_resolutions=unet_cfg['attention_resolutions'],
        dropout=unet_cfg['dropout'],
        channel_mult=unet_cfg['channel_mult'],
        time_embedding_dim=unet_cfg['time_embedding_dim'],
        event_channels=encoder_cfg['out_channels'],
        fusion_type=unet_cfg['fusion_type']
    )
    
    # Build diffusion model
    diffusion = GaussianDiffusion(
        model=unet,
        image_size=config['datasets']['train']['gt_size'],
        channels=unet_cfg['in_channels'],
        timesteps=config['network_g']['timesteps'],
        loss_type=config['network_g']['loss_type'],
        beta_schedule=config['network_g']['beta_schedule']
    )
    
    # Build event-conditioned diffusion model
    model = EventConditionedDiffusion(
        diffusion_model=diffusion,
        event_encoder=event_encoder
    )
    
    return model


def test_model():
    """Test model components and forward pass."""
    print("Testing EventDeblur model components...")
    
    # Load configuration
    config_path = 'options/train/train_event_diffusion.yml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating config directory...")
        os.makedirs('options/train', exist_ok=True)
        
        # Copy the config from our predefined version
        import shutil
        shutil.copy('train_config.py', config_path)
    
    config = load_config(config_path)
    
    # Update dataroot path if provided as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to GOPRO dataset')
    args = parser.parse_args()
    
    if args.dataroot:
        config['datasets']['train']['dataroot'] = args.dataroot
        config['datasets']['val']['dataroot'] = args.dataroot.replace('train', 'test')
    
    # Create dataset
    print("Creating dataset...")
    dataset_opt = config['datasets']['train']
    dataset = EventDeblurDataset(dataset_opt)
    
    if len(dataset) == 0:
        print("ERROR: No valid samples found in the dataset!")
        return
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0
    )
    
    # Get a batch
    print("Loading a batch...")
    batch = next(iter(dataloader))
    
    blur = batch['blur']
    sharp = batch['sharp']
    event = batch['event']
    
    print(f"Blur shape: {blur.shape}")
    print(f"Sharp shape: {sharp.shape}")
    print(f"Event shape: {event.shape}")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    
    # Test model components
    print("Testing event encoder...")
    event_features = model.event_encoder(event)
    if isinstance(event_features, list):
        print(f"Event features: {[f.shape for f in event_features]}")
    else:
        print(f"Event features shape: {event_features.shape}")
    
    # Test full model forward pass
    print("Testing full model forward pass...")
    try:
        # Set model to training mode
        model.train()
        
        # Forward pass with training=True should return loss
        loss = model(blur, sharp, event, train=True)
        print(f"Training loss: {loss.item()}")
        
        # Set model to eval mode
        model.eval()
        
        # Forward pass with training=False should return deblurred image
        with torch.no_grad():
            output = model(blur, sharp, event, train=False)
        
        print(f"Output shape: {output.shape}")
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Convert tensors to numpy arrays for visualization
        blur_img = blur[0].permute(1, 2, 0).numpy()
        sharp_img = sharp[0].permute(1, 2, 0).numpy()
        output_img = output[0].permute(1, 2, 0).numpy()
        
        # Clip images to valid range
        blur_img = np.clip(blur_img, 0, 1)
        sharp_img = np.clip(sharp_img, 0, 1)
        output_img = np.clip(output_img, 0, 1)
        
        # Plot images
        plt.subplot(1, 3, 1)
        plt.imshow(blur_img)
        plt.title('Blur')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(output_img)
        plt.title('Deblurred')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(sharp_img)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_result.png')
        print("Visualization saved to 'test_result.png'")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()
