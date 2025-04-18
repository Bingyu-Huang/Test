import os
import os.path as osp
import random
import numpy as np
import torch
import cv2
import glob
from torch.utils.data import Dataset


class EventDeblurDataset(Dataset):
    """Dataset for GOPRO event-based image deblurring.

    This dataset class loads blur-sharp image pairs and corresponding event voxel data
    from the GOPRO dataset structure.

    Args:
        opt (dict): Config options including:
            dataroot (str): Root directory of the GOPRO dataset
            phase (str): 'train' or 'val'
            gt_size (int): Target size for training patches
            use_flip (bool): Whether to use horizontal flipping for augmentation
            use_rot (bool): Whether to use rotation for augmentation
    """

    def __init__(self, opt):
        super(EventDeblurDataset, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.phase = opt.get('phase', 'train')

        # Get all scene folders
        scene_dirs = []
        for scene in os.listdir(self.dataroot):
            scene_path = osp.join(self.dataroot, scene)
            if osp.isdir(scene_path):
                # Check if it has the expected subfolders
                if (osp.exists(osp.join(scene_path, 'blur_gamma')) and
                        osp.exists(osp.join(scene_path, 'sharp')) and
                        osp.exists(osp.join(scene_path, 'voxel'))):
                    scene_dirs.append(scene_path)

        print(f"Found {len(scene_dirs)} valid scene directories.")

        # Get all image files
        self.samples = []
        for scene_dir in scene_dirs:
            blur_dir = osp.join(scene_dir, 'blur_gamma')
            sharp_dir = osp.join(scene_dir, 'sharp')
            voxel_dir = osp.join(scene_dir, 'voxel')

            # Get all blur images in this scene
            blur_files = sorted(glob.glob(osp.join(blur_dir, '*.png')))

            for blur_file in blur_files:
                base_name = osp.basename(blur_file)
                file_id = osp.splitext(base_name)[0]  # e.g., '000047'

                # Get corresponding sharp and voxel files
                sharp_file = osp.join(sharp_dir, base_name)
                voxel_file = osp.join(voxel_dir, f"{file_id}.npz")

                # Verify all files exist
                if osp.exists(blur_file) and osp.exists(sharp_file) and osp.exists(voxel_file):
                    self.samples.append({
                        'blur': blur_file,
                        'sharp': sharp_file,
                        'voxel': voxel_file,
                        'scene': osp.basename(scene_dir),
                        'id': file_id
                    })

        print(f"Found {len(self.samples)} valid samples.")

        # Training settings
        self.gt_size = opt.get('gt_size', 256)
        self.use_flip = opt.get('use_flip', True) if self.phase == 'train' else False
        self.use_rot = opt.get('use_rot', True) if self.phase == 'train' else False

    def __getitem__(self, index):
        """Get training sample by index.

        Returns a dict containing:
            - blur: Tensor [3, H, W] - Blurred RGB image
            - sharp: Tensor [3, H, W] - Sharp ground truth image
            - event: Tensor [10, H, W] - Event voxel grid (5 bins × 2 polarities)
            - blur_path: Path to blur image file
            - sharp_path: Path to sharp image file
            - voxel_path: Path to voxel data file
            - scene: Scene name
            - id: Image ID
        """
        sample = self.samples[index]

        # Load blur image
        blur_img = cv2.imread(sample['blur'])
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)

        # Load sharp image
        sharp_img = cv2.imread(sample['sharp'])
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        # Load voxel data
        try:
            voxel_data = np.load(sample['voxel'], allow_pickle=True)
            voxel_grid = voxel_data['voxel_grid']  # Expected shape: [5, H, W, 2]
        except Exception as e:
            print(f"Error loading voxel data from {sample['voxel']}: {e}")
            # Create dummy voxel grid with expected shape [5, H, W, 2]
            h, w = blur_img.shape[:2]
            voxel_grid = np.zeros((5, h, w, 2), dtype=np.float32)

        # For training: random crop and augmentations
        if self.phase == 'train':
            # Random crop
            h, w = blur_img.shape[:2]
            if h > self.gt_size and w > self.gt_size:
                # Random crop coordinates
                top = random.randint(0, h - self.gt_size)
                left = random.randint(0, w - self.gt_size)

                # Crop images
                blur_img = blur_img[top:top + self.gt_size, left:left + self.gt_size, :]
                sharp_img = sharp_img[top:top + self.gt_size, left:left + self.gt_size, :]

                # Crop voxel grid
                voxel_grid = voxel_grid[:, top:top + self.gt_size, left:left + self.gt_size, :]
            else:
                # Resize if smaller than gt_size
                blur_img = cv2.resize(blur_img, (self.gt_size, self.gt_size))
                sharp_img = cv2.resize(sharp_img, (self.gt_size, self.gt_size))
                # Resize voxel grid (for each bin and polarity)
                new_voxel = np.zeros((5, self.gt_size, self.gt_size, 2), dtype=np.float32)
                for b in range(5):
                    for p in range(2):
                        new_voxel[b, :, :, p] = cv2.resize(voxel_grid[b, :, :, p], (self.gt_size, self.gt_size))
                voxel_grid = new_voxel

            # Random flip
            if self.use_flip and random.random() < 0.5:
                blur_img = np.flip(blur_img, axis=1)
                sharp_img = np.flip(sharp_img, axis=1)
                voxel_grid = np.flip(voxel_grid, axis=2)  # Flip width dimension

            # Random rotation
            if self.use_rot and random.random() < 0.5:
                k = random.randint(1, 3)  # 1: 90°, 2: 180°, 3: 270°
                blur_img = np.rot90(blur_img, k=k, axes=(0, 1))
                sharp_img = np.rot90(sharp_img, k=k, axes=(0, 1))
                voxel_grid = np.rot90(voxel_grid, k=k, axes=(1, 2))  # Rotate H,W dimensions

        # Convert numpy arrays to tensors
        blur_img = torch.from_numpy(blur_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        sharp_img = torch.from_numpy(sharp_img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Reshape voxel grid: [5, H, W, 2] -> [10, H, W]
        voxel_flat = voxel_grid.reshape(5 * 2, voxel_grid.shape[1], voxel_grid.shape[2])
        voxel_tensor = torch.from_numpy(voxel_flat.astype(np.float32))

        return {
            'blur': blur_img,
            'sharp': sharp_img,
            'event': voxel_tensor,
            'blur_path': sample['blur'],
            'sharp_path': sample['sharp'],
            'voxel_path': sample['voxel'],
            'scene': sample['scene'],
            'id': sample['id']
        }

    def __len__(self):
        return len(self.samples)