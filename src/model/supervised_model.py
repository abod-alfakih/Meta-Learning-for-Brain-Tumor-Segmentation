from src.utils.registry import DATASET_REGISTRY, MODEL_REGISTRY
import src.losses.brats21 as losses
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SupervisedTrainer(BaseModel):
    def __init__(self, arch, device, epochs, optim_type='AdamW', lr=1e-3, loss_type="SoftDiceBCEWithLogits", aux_arch=None, mode='train'):
        super.__init__(arch, device, epochs, aux_arch, mode)

        self.arch = arch
        self.device = device
        self.mode = mode
        self.epochs = epochs
        
        if aux_arch:
            # this is for additional networks such as scorer from meta-weight-net or etc.
            self.aux_arch = aux_arch
        
        if loss_type == "SoftDiceBCEWithLogits":
            self.criterion = losses.SoftDiceBCEWithLogitsLoss()
            self.criterion = self.criterion.to(self.device)
        else:
            ValueError(f"loss type: {loss_type} is not available")
            
        self.optimizer = self.get_optimizer(self, optim_type=optim_type, params= self.arch, lr=lr)
        
    def feed_data(self, batch):
        img, label = batch['image'], batch['label']
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label
    
    def train_step(self, batch):
        img, label = self.feed_data(batch)
        pred_seg = self.arch(img)
        
        loss = self.criterion(pred_seg, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_epoch(self, dataloader, current_epoch):
        # running for one epoch
        pass
        
    