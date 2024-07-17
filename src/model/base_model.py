import autorootcwd
import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import src.metrics.brats21 as metrics
import abc

class BaseModel():
    """Base model class for training segmentation models """
    
    def __init__(self, arch, device, epochs, aux_arch=None, mode='train'):
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"
        
        self.arch = arch
        self.device = device
        self.mode = mode
        self.epochs = epochs
        
        if aux_arch:
            # this is for additional networks such as scorer from meta-weight-net or etc.
            self.aux_arch = aux_arch
    
    @abc.abstractmethod
    def feed_data(self, batch):
        # how is input and label data derived from batch
        ''' Example
        img, label = batch['image'], batch['label']
        img = img.to(self.device)
        label = label.to(self.device)
        '''
        pass
    
    @abc.abstractmethod
    def train_step(self, batch):
        """Train step.
        Args:
            batch : batch
        """ 
        pass

    @abc.abstractmethod
    def validation_step(self, batch):
        """Validation step.
        Args:
            batch: batch
        """ 
        pass
    
    @abc.abstractmethod
    def train_epoch(self, dataloader, current_epoch):
        pass

    @abc.abstractmethod
    def val_epoch(self, dataloader, current_epoch):
        pass
    
    @abc.abstractmethod
    def get_current_visuals(self):
        pass
    
    def get_current_log(self):
        pass
    
    @abc.abstractmethod
    def save(self, epoch):
        """Save networks and training state."""
        pass
    
    def compute_metrics(self, seg_map, label):
        dice = metrics.dice(seg_map, label.bool())
        hd95 = metrics.hd95(seg_map, label.bool())
        
        # case by case
        dice_wt = dice[:,1]
        dice_tc = dice[:,0]
        dice_et = dice[:,2]
        
        hd95_wt = hd95[:,1]
        hd95_tc = hd95[:,0]
        hd95_et = hd95[:,2]
        
        dice = dice.mean()
        dice_wt = dice_wt.mean()
        dice_tc = dice_tc.mean()
        dice_et = dice_et.mean()
        
        hd95 = hd95.mean()
        hd95_wt = hd95_wt.mean()
        hd95_tc = hd95_tc.mean()
        hd95_et = hd95_et.mean()
        
        return {
            'dice': dice,
            'dice_wt': dice_wt,
            'dice_tc': dice_tc,
            'dice_et': dice_et,
            'hd95': hd95,
            'hd95_wt': hd95_wt,
            'hd95_tc': hd95_tc,
            'hd95_et': hd95_et
        }


    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer


    def print_model_summary(self):
        print("Model Summary:")
        total_params = 0
        for name, param in self.arch.named_parameters():
            total_params += param.numel()
            print(f"Layer: {name}, Size: {param.size()}")
        print(f"Total Parameters: {total_params}")
        
        if hasattr(self, 'aux_arch'):
            # Add the code you want to run if aux_arch is defined
            print("Auxiliary Model Summary:")
            aux_total_params = 0
            for name, param in self.aux_arch.named_parameters():
                aux_total_params += param.numel()
                print(f"Aux Layer: {name}, Size: {param.size()}")
            print(f"Auxiliary Total Parameters: {aux_total_params}")