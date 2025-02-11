import os
import time
import warnings
from copy import deepcopy
from os.path import join
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torchopt
import functorch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
import torchopt
import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet
from utils.loss import SoftDiceBCEWithLogitsLoss,clean_SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer, get_clean_optimizer
from utils.scheduler import get_scheduler,get_clean_scheduler

def print_model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        print(f"Layer: {name}, Size: {param.size()}")
    print(f"Total Parameters: {total_params}")




def infer(args, epoch, model:nn.Module, infer_loader, writer, logger, mode:str, save_pred:bool=False):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()

    # make save epoch folder
    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.exp_dir, folder_dir)
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            # get data
            image, label = image.cuda(), label.bool().cuda()
            bsz = image.size(0)

            # get seg map
            seg_map = sliding_window_inference(
                inputs=image,
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            # discrete
            seg_map = torch.where(seg_map > 0.5, True, False)

            # post-processing
            seg_map = brats_post_processing(seg_map)

            # calc metric
            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            # output seg map
            if save_pred:
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            # logging
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            # monitor training progress
            if (i == 0) or (i + 1) % args.print_freq == 0:
                mean_metrics = case_metrics_meter.mean()
                logger.info("\t".join([
                    f'{mode.capitalize()}: [{epoch}][{i + 1}/{len(infer_loader)}]', str(batch_time),
                    f"Dice_WT {dice[:, 1].mean():.3f} ({mean_metrics['Dice_WT']:.3f})",
                    f"Dice_TC {dice[:, 0].mean():.3f} ({mean_metrics['Dice_TC']:.3f})",
                    f"Dice_ET {dice[:, 2].mean():.3f} ({mean_metrics['Dice_ET']:.3f})",
                    f"HD95_WT {hd95[:, 1].mean():7.3f} ({mean_metrics['HD95_WT']:7.3f})",
                    f"HD95_TC {hd95[:, 0].mean():7.3f} ({mean_metrics['HD95_TC']:7.3f})",
                    f"HD95_ET {hd95[:, 2].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})",
                ]))

            end = time.time()

        # output case metric csv
        case_metrics_meter.output(save_path)

    # get validation metrics and log to tensorboard
    infer_metrics = case_metrics_meter.mean()
    for key, value in infer_metrics.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)

    return infer_metrics


def save_loss_history(loss_dict, exp_dir):
    """
    Save the loss history to separate text files in the specified directory.

    Parameters:
    - loss_dict: A dictionary where keys are loss names and values are lists of loss values.
    - exp_dir: The directory where the loss history files will be saved.
    """
    # Ensure the experiment directory exists
    os.makedirs(exp_dir, exist_ok=True)

    # Loop through each loss type in the dictionary and save to separate files
    for loss_name, loss_values in loss_dict.items():
        file_path = os.path.join(exp_dir, f"{loss_name}_history.txt")
        with open(file_path, 'w') as f:
            for value in loss_values:
                f.write(f"{value}\n")
def calc_loss_gradient(model, loss_fn, sample, target):
    prediction = model(sample)
    # prediction = prediction[0] if isinstance(prediction, list) else prediction
    # label = target[0] if isinstance(target, list) else target

    bce_loss, dsc_loss = loss_fn(prediction, target)
    loss = bce_loss + dsc_loss

    return torch.autograd.grad(
        loss, list(model.parameters()), allow_unused=True, retain_graph=True
    )

def compute_sample_grads( model, loss_fn, data, targets):
    batch_size = len(data)
    sample_grads = [
        calc_loss_gradient(model, loss_fn, data[i:i+1], targets[i:i+1])
        for i in range(batch_size)
    ]
    return sample_grads

def save_learning_rate_history(learning_rate_history, exp_dir):
    lr_file_path = os.path.join(exp_dir, "learning_rate_history.txt")
    with open(lr_file_path, "a") as f:
        for lr in learning_rate_history:
            f.write(f"{lr}\n")


class TrainModel:
    def determine_meta_weight(self,image,label):
        loss_func = self.loss_2  # same as the loss function used in the outer loop
        net_state_dict = torchopt.extract_state_dict(self.model)

        train_per_sample_gradients = compute_sample_grads(
            self.model, loss_func, image, label
        )

        torchopt.recover_state_dict(self.model, net_state_dict)
        self.optimizer.zero_grad()
        meta_gradients = calc_loss_gradient(self.model, loss_func, self.meta_inputs, self.meta_labels)
        with torch.no_grad():
            # calculate cosine similarity between train and val gradients
            cosine_distance = []
            for grads in train_per_sample_gradients:
                dot_product_term = sum([torch.sum(tg * mg)
                                        for (tg, mg) in zip(grads, meta_gradients)])
                tg_norm_term = sum([torch.sum(tg * tg) for tg in grads])
                vg_norm_term = sum([torch.sum(mg * mg) for mg in meta_gradients])
                cosine_distance_term = dot_product_term / (torch.sqrt(tg_norm_term) * torch.sqrt(vg_norm_term))
                cosine_distance.append(cosine_distance_term)
            cosine_distance = torch.stack(cosine_distance)
            cosine_distance = torch.clamp(cosine_distance, min=0)
            # cosine_distance = (cosine_distance + 1) / 2
            norm_v = torch.sum(cosine_distance)
            if norm_v != 0:
                w_v = cosine_distance / norm_v
            else:
                w_v = cosine_distance
        return w_v.detach()

    def main(self):

        args = parse_seg_args()
        logger, writer = initialization(args)

        # dataloaders

        train_cases, val_cases, test_cases,meta_cases = load_cases_split(args.cases_split)
        train_loader = brats2021.get_train_loader(args, train_cases)
        val_loader = brats2021.get_infer_loader(args, val_cases)
        test_loader = brats2021.get_infer_loader(args, test_cases)
        meta_train = brats2021.get_train_loader(args, meta_cases)

        # model & stuff
        self.model = get_unet(args).cuda()
        if args.data_parallel:
            self.model = nn.DataParallel(self.model).cuda()
        self.optimizer = get_optimizer(args, self.model)
        scheduler = get_scheduler(args, self.optimizer)
        self.loss_fn = clean_SoftDiceBCEWithLogitsLoss().cuda()
        self.loss_2 =SoftDiceBCEWithLogitsLoss().cuda()




        # load model
        def load_pretrained_model(args, model, logger):
            if args.weight_path is not None:
                logger.info("==> Loading pretrain model...")
                assert args.weight_path.endswith(".pth"), "Weight path should end with .pth"

                checkpoint = torch.load(args.weight_path)

                # Extract the state dictionary from the model instance
                if isinstance(checkpoint['model'], torch.nn.Module):
                    model_state = checkpoint['model'].state_dict()
                    logger.info("Extracted state_dict from model instance.")
                else:
                    model_state = checkpoint['model']
                    logger.info("Loaded state_dict directly from checkpoint.")

                model.load_state_dict(model_state)

        load_pretrained_model(args, self.model, logger)
        # train & val
        logger.info("==> Training starts...")
        best_model = {}
        loss_history = []  # Initialize an empty list to store loss values
        loss_history_2 = []
        loss_history_meta = []
        learning_rate_history = []  # Initialize an empty list to store learning rates

        val_leaderboard = LeaderboardBraTS()
        self.meta_dataloader_iter = iter(meta_train)

        for epoch in range(args.epochs):
            self.model.train()
            data_time = AverageMeter('Data', ':6.3f')
            batch_time = AverageMeter('Time', ':6.3f')
            bce_meter = AverageMeter('BCE', ':.4f')
            dsc_meter = AverageMeter('Dice', ':.4f')
            loss_meter = AverageMeter('Loss', ':.4f')
            loss_meter_2= AverageMeter('Loss_2', ':.4f')
            loss_meter_3= AverageMeter('Loss_3', ':.4f')

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, bce_meter, dsc_meter, loss_meter,loss_meter_2,loss_meter_3],
                prefix=f"Train: [{epoch}]")
            end = time.time()


            for i, (image, label, _, _) in enumerate(train_loader):

                if (i + 1) % 1 == 0:
                    pseudo_net = get_unet(args).cuda()
                    pseudo_net.load_state_dict(self.model.state_dict())
                    pseudo_net.train()
                    try:
                        # Attempt to get the next batch; only capture the first two items (meta_inputs, meta_labels)
                        self.meta_inputs, self.meta_labels, *_ = next(self.meta_dataloader_iter)
                    except StopIteration:
                        # Reinitialize the iterator if the end of the dataset is reached
                        meta_dataloader_iter = iter(meta_train)
                        self.meta_inputs, self.meta_labels, *_ = next(meta_dataloader_iter)
                    self.meta_inputs, self.meta_labels = self.meta_inputs.cuda(), self.meta_labels.float().cuda()

                image, label = image.cuda(), label.float().cuda()
                bsz = image.size(0)
                data_time.update(time.time() - end)
                self.optimizer.zero_grad()
                weights = self.determine_meta_weight(image,label)
                output = self.model(image)
                # output = output[0] if isinstance(output, list) else output
                # label = label[0] if isinstance(label, list) else label
                bce_loss, dse_loss = self.loss_fn(output, label)
                total_loss_per_sample = bce_loss + dse_loss # Only if it's a tuple, otherwise, `bce_loss` is already a tensor
                mean_loss_per_sample = total_loss_per_sample.mean(dim=[1, 2, 3, 4])
                loss = mean_loss_per_sample * weights
                total_loss = torch.sum(loss)
                total_loss.backward()
                self.optimizer.step()
                # logging
                torch.cuda.synchronize()
                loss_meter.update(total_loss.item(), bsz)
                batch_time.update(time.time() - end)
                # monitor training progress
                if (i == 0) or (i + 1) % args.print_freq == 0:
                    progress.display(i + 1, logger)
                end = time.time()
            if scheduler is not None:
                scheduler.step()
            train_tb = {
                'bce_loss': bce_meter.avg,
                'dsc_loss': dsc_meter.avg,
                'total_loss': loss_meter.avg,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
            }

            for key, value in train_tb.items():
                writer.add_scalar(f"train/{key}", value, epoch)

        # validation
            if (epoch > 81):
                logger.info(f"==> Validation starts...")
                # inference on validation set
                val_metrics = infer(args, epoch, self.model, val_loader, writer, logger, mode='val')
                # model selection
                val_leaderboard.update(epoch, val_metrics)
                best_model.update({epoch: deepcopy(self.model.state_dict())})
                logger.info(f"==> Validation ends...")
                val_leaderboard.update(epoch, val_metrics)
                if args.save_model:
                    logger.info("==> Saving...")

                    # Save first model
                    state = {'model': self.model, 'epoch': epoch, 'args': args}
                    save_path = os.path.join(args.exp_dir, f"val_epoch_{epoch:02d}", 'model_1_best_ckpt.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(state, save_path)
                    logger.info(f"==> Model 1 saved at {save_path}")


            torch.cuda.empty_cache()

        #ouput final leaderboard and its rank
        val_leaderboard.output(args.exp_dir)
        #
        # test
        logger.info("==> Testing starts...")
        best_epoch = val_leaderboard.get_best_epoch()
        best_model = best_model[best_epoch]
        self.model.load_state_dict(best_model)
        infer(args, best_epoch, self.model, test_loader, writer, logger, mode='test', save_pred=args.save_pred)

        logger.info("==> Testing ends...")


if __name__ == '__main__':
    trainer = TrainModel()
    trainer.main()

