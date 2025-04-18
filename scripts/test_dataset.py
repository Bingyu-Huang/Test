import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import importlib.util

# Import the EventDeblurDataset class
spec = importlib.util.spec_from_file_location(
    "event_deblur_dataset", 
    "basicsr/data/event_deblur_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EventDeblurDataset = module.EventDeblurDataset


def main():
    """Test the dataset loader with samples from the GOPRO dataset."""
    # Dataset options - update this path to your GOPRO dataset location
    opt = {
        'dataroot': '/scratch/datasets/GoPro/GOPRO_Large/train',  # MODIFY THIS PATH to your dataset location
        'phase': 'train',
        'gt_size': 256,
        'use_flip': True,
        'use_rot': True
    }
    
    # Create dataset
    dataset = EventDeblurDataset(opt)
    
    if len(dataset) == 0:
        print("ERROR: No valid samples found! Check your dataset path.")
        return
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a single sample
    print("Testing single sample loading...")
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Blur tensor shape:", sample['blur'].shape)
    print("Sharp tensor shape:", sample['sharp'].shape)
    print("Event tensor shape:", sample['event'].shape)
    
    # Create dataloader with batch size 4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Get a batch
    print("Testing batch loading...")
    batch = next(iter(dataloader))
    
    print("Batch keys:", list(batch.keys()))
    print("Blur tensor batch shape:", batch['blur'].shape)
    print("Sharp tensor batch shape:", batch['sharp'].shape)
    print("Event tensor batch shape:", batch['event'].shape)
    
    # Display sample images
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    for i in range(4):
        # Get sample
        blur = batch['blur'][i].permute(1, 2, 0).numpy()
        sharp = batch['sharp'][i].permute(1, 2, 0).numpy()
        event = batch['event'][i]
        
        # Event visualization: combine all bins and both polarities
        event_vis = torch.sum(event.reshape(5, 2, event.shape[1], event.shape[2]), dim=1)
        event_vis = torch.sum(event_vis, dim=0).numpy()
        event_vis = (event_vis - event_vis.min()) / (event_vis.max() - event_vis.min() + 1e-5)
        
        # Plot
        axes[i, 0].imshow(blur)
        axes[i, 0].set_title(f"Blur {i}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sharp)
        axes[i, 1].set_title(f"Sharp {i}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(event_vis, cmap='viridis')
        axes[i, 2].set_title(f"Event {i}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    print("Saved sample visualization to dataset_samples.png")
    
    # Test a few more batches to ensure stability
    print("Testing more batches...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test 2 more batches
                break
            print(f"Batch {i+1} shapes: blur={batch['blur'].shape}, event={batch['event'].shape}")
    except Exception as e:
        print(f"Error during batch loading: {e}")
    
    print("Dataset test completed!")


if __name__ == "__main__":
    main()