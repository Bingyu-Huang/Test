import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import wandb

from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss


@MODEL_REGISTRY.register()
class EventDiffusionModel(BaseModel):
    """Event-conditioned diffusion model for image deblurring task.
    
    This model implements training and testing processes for EventDeblur project.
    """
    
    def __init__(self, opt):
        super(EventDiffusionModel, self).__init__(opt)
        
        # Define network structure
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # Load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))
        
        if self.is_train:
            self.init_training_settings()
    
    def init_training_settings(self):
        """Initialize training settings."""
        self.net_g.train()
        
        # Set up optimizers and schedulers
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
        # Set up losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        # Set up schedulers
        if train_opt.get('scheduler'):
            scheduler_type = train_opt['scheduler'].pop('type')
            self.scheduler_g = self.get_scheduler(scheduler_type, self.optimizer_g, **train_opt['scheduler'])
            self.schedulers.append(self.scheduler_g)
    
    def feed_data(self, data):
        """Feed input data."""
        self.blur = data['blur'].to(self.device)
        self.sharp = data['sharp'].to(self.device)
        self.event = data['event'].to(self.device)
        
        # Store paths for debugging/validation
        self.blur_path = data['blur_path']
        self.sharp_path = data['sharp_path']
    
    def optimize_parameters(self, current_iter):
        """Optimize network parameters."""
        self.optimizer_g.zero_grad()
        
        # Diffusion loss is computed inside the forward pass
        self.loss = self.net_g(self.blur, self.sharp, self.event, train=True)
        
        self.loss.backward()
        self.optimizer_g.step()
        
        # Log to WandB
        if wandb.run:
            wandb.log({
                'train/loss': self.loss.item(),
                'train/lr': self.get_current_learning_rate()
            }, step=current_iter)
    
    def test(self):
        """Testing function."""
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.blur, self.sharp, self.event, train=False)
        self.net_g.train()
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation with distributed training."""
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation without distributed training."""
        pbar = utils.ProgressBar(len(dataloader))
        avg_psnr = 0.0
        avg_ssim = 0.0
        dataset_name = self.opt['datasets']['val']['name']
        
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['blur_path'][0]))[0]
            
            self.feed_data(val_data)
            self.test()
            
            # Calculate metrics
            visuals = self.get_current_visuals()
            result_img = visuals['result']
            gt_img = visuals['sharp']
            
            # Save validation images
            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 
                                         f'{img_name}_{current_iter}.png')
                img_result = tensor2img(result_img)
                utils.mkdir_or_exist(osp.dirname(save_img_path))
                imwrite(img_result, save_img_path)
            
            # Calculate PSNR & SSIM
            psnr = utils.calculate_psnr(result_img, gt_img)
            ssim = utils.calculate_ssim(result_img, gt_img)
            avg_psnr += psnr
            avg_ssim += ssim
            
            pbar.update(f'Test {img_name} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
        
        # Average metrics
        avg_psnr /= len(dataloader)
        avg_ssim /= len(dataloader)
        
        # Log to tensorboard and wandb
        if tb_logger:
            tb_logger.add_scalar('val/psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('val/ssim', avg_ssim, current_iter)
        
        if wandb.run:
            wandb.log({
                'val/psnr': avg_psnr,
                'val/ssim': avg_ssim
            }, step=current_iter)
        
        return avg_psnr, avg_ssim
    
    def get_current_visuals(self):
        """Return visualization images."""
        out_dict = OrderedDict()
        out_dict['blur'] = self.blur.detach().cpu()
        out_dict['sharp'] = self.sharp.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        
        return out_dict
    
    def get_current_learning_rate(self):
        """Return current learning rate."""
        return self.optimizer_g.param_groups[0]['lr']
    
    def get_current_log(self):
        """Return current log information."""
        return dict(loss=self.loss.item())
    
    def save(self, epoch, current_iter):
        """Save models and training state."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
