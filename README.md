# EventDeblur: Event-based Image Deblurring with Diffusion Models

This repository implements a conditional diffusion model for image deblurring that leverages event data to enhance deblurring performance. The project is built on the BasicSR framework and integrates event data processing with diffusion models.

## Project Structure

```
EventDeblur/
├── basicsr/
│   ├── archs/                        # Model architecture files
│   │   ├── event_encoder_arch.py     # Event data encoder
│   │   ├── diffusion_arch.py         # Conditional diffusion model
│   │   ├── unet_arch.py              # U-Net backbone network
│   │   ├── fusion_module_arch.py     # Feature fusion module
│   ├── data/                         # Data processing files
│   │   ├── event_deblur_dataset.py   # Dataset for event-blur pairs
│   ├── models/                       # Model training/testing files
│   │   ├── event_diffusion_model.py  # Training process implementation
├── options/                          # Configuration files
│   ├── train/                        # Training configurations
│   │   ├── train_event_diffusion.yml # Main training config
│   ├── test/                         # Testing configurations
├── scripts/                          # Utility scripts
│   ├── train.py                      # Training script
│   ├── test_model.py                 # Model testing script
```

## Features

- **Event-conditioned Diffusion Model**: Utilizes event data to guide the deblurring process
- **Flexible Fusion Strategies**: Supports different methods for fusing event and image features (splicing, attention, addition)
- **Multi-scale Processing**: Processes event features at multiple scales for better deblurring
- **WandB Integration**: Built-in logging with Weights & Biases for experiment tracking
- **Distributed Training**: Supports multi-GPU training for faster experimentation

## Requirements

- PyTorch >= 2.0
- BasicSR
- diffusers
- wandb
- CUDA-compatible GPU

## Installation

1. Create a conda environment:

```bash
conda create -n event_deblur python=3.10
conda activate event_deblur
```

2. Install PyTorch with CUDA support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

4. Install BasicSR:

```bash
pip install basicsr
```

## Dataset Preparation

This project works with the GOPRO dataset, which should be organized as follows:

```
GOPRO_Large/
├── train/
│   ├── GOPR0xxx_xx_xx/
│   │   ├── blur_gamma/      # Blurry images with gamma correction
│   │   ├── sharp/           # Sharp ground truth images
│   │   ├── voxel/           # Event data in voxel representation (.npz files)
├── test/
    ├── GOPR0xxx_xx_xx/
        ├── blur_gamma/
        ├── sharp/
        ├── voxel/
```

Each voxel .npz file should contain the key 'voxel_grid' with shape [5, H, W, 2], where the last dimension stores positive and negative polarities.

## Training

1. Modify the configuration file in `options/train/train_event_diffusion.yml` according to your needs.

2. Start training:

```bash
python scripts/train.py --opt options/train/train_event_diffusion.yml --dataroot /path/to/GOPRO_Large/train
```

For distributed training with multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py --opt options/train/train_event_diffusion.yml --launcher pytorch
```

## Testing the Model

To test if the model components work correctly:

```bash
python scripts/test_model.py --dataroot /path/to/GOPRO_Large/train
```

This will run a forward pass through the model and save a visualization of the results.

## Inference

To run inference on test data:

```bash
python scripts/test.py --opt options/test/test_event_diffusion.yml --dataroot /path/to/GOPRO_Large/test
```

## Results

The model generates deblurred images from blurry inputs using the additional information from event data. The diffusion-based approach allows for high-quality deblurring with fewer artifacts compared to traditional methods.

## Acknowledgements

This project builds upon the following frameworks:
- [BasicSR](https://github.com/XPixelGroup/BasicSR): A framework for super-resolution and restoration tasks
- [Diffusers](https://github.com/huggingface/diffusers): A library for state-of-the-art diffusion models

## License

This project is released under the MIT License.
