import os
import time
import warnings
from copy import deepcopy
from os.path import join
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from models.unet import *
import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet

from utils.loss import SoftDiceBCEWithLogitsLoss,  clean_SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer,get_clean_optimizer
from utils.scheduler import get_scheduler,get_clean_scheduler

from models.meta import *

import torch
from torch.profiler import profile, record_function, ProfilerActivity


def save_loss_history(loss_history, exp_dir):
    loss_file_path = os.path.join(exp_dir, "loss_history.txt")
    with open(loss_file_path, "a") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")

def save_learning_rate_history(learning_rate_history, exp_dir):
    lr_file_path = os.path.join(exp_dir, "learning_rate_history.txt")
    with open(lr_file_path, "a") as f:
        for lr in learning_rate_history:
            f.write(f"{lr}\n")
def train(self, args, epoch, model,meta_net, train_loader, loss_fn, meta_loss_fn, optimizer, meta_optimizer, scheduler,
          meta_scheduler, scaler,scaler_2, writer, logger):
    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, bce_meter, dsc_meter, loss_meter],
        prefix=f"Train: [{epoch}]")

    end = time.time()

    for i, (image, label, _, _) in enumerate(train_loader):
        # init
        model.train()
        image, label = image.cuda(), label.float().cuda()
        bsz = image.size(0)
        data_time.update(time.time() - end)
        if (i + 1) % 1 == 0:
            pseudo_net = get_unet(args).cuda()
            if args.data_parallel:
                pseudo_net = torch.nn.parallel.DistributedDataParallel.cuda()
            pseudo_net.load_state_dict(model.state_dict())
            pseudo_net.train()
            with autocast((args.amp) and (scaler is not None)):
                pseudo_outputs = pseudo_net(image)
                if isinstance(pseudo_outputs, list):
                    pseudo_outputs = pseudo_outputs[0]
                if isinstance(label, list):
                    label = label[0]
                bce_loss, dsc_loss = meta_loss_fn(pseudo_outputs, label)
                pseudo_loss_vector = bce_loss + dsc_loss
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
            pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

            # Check pseudo_loss for being a scalar
            assert pseudo_loss.dim() == 0, "pseudo_loss must be a scalar"

            # Compute gradients
            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True,
                                               allow_unused=True)

            # Initialize meta-optimizer
            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=args.lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())

            # Perform meta-update step
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            # val_data = next(self.meta_dataloader_iter , None)
            # if val_data is not None:
            #     meta_inputs, meta_labels = val_data
            # else:
            #     cl_data_iterator = iter(self.meta_dataloader_iter)
            #     val_data = next(cl_data_iterator, None)
            #     meta_inputs, meta_labels = val_data
            try:
                # Attempt to get the next batch; only capture the first two items (meta_inputs, meta_labels)
                meta_inputs, meta_labels, *_ = next(self.meta_dataloader_iter)
            except StopIteration:
                # Reinitialize the iterator if the end of the dataset is reached
                self.meta_dataloader_iter = iter(self.meta_dataloader)
                meta_inputs, meta_labels, *_ = next(self.meta_dataloader_iter)

            meta_inputs, meta_labels = meta_inputs.cuda(), meta_labels.float().cuda()
            with autocast((args.amp) and (scaler is not None)):
                meta_outputs = pseudo_net(meta_inputs)
                if isinstance(meta_outputs, list):  # If preds is a list, select the correct element
                    meta_outputs = meta_outputs[0]
                if isinstance(meta_labels, list):  # Similar handling for labels if necessary
                    meta_labels = meta_labels[0]
                bce_loss, dsc_loss = loss_fn(meta_outputs, meta_labels)
                meta_loss = bce_loss + dsc_loss

            meta_optimizer.zero_grad()
            # compute gradient and do optimizer step
            if args.amp and scaler_2 is not None:
                scaler_2.scale(meta_loss).backward()
                if args.clip_grad:
                    scaler_2.unscale_(meta_optimizer)  # enable grad clipping
                    nn.utils.clip_grad_norm_(pseudo_net.parameters(), 10)
                scaler_2.step(meta_optimizer)
                scaler_2.update()

            else:
                meta_loss.backward()
                if args.clip_grad:
                    nn.utils.clip_grad_norm_(pseudo_net.parameters(), 10)
                meta_optimizer.step()

        with autocast((args.amp) and (scaler is not None)):
            outputs = model(image)
            if isinstance(outputs, list):  # If preds is a list, select the correct element
                outputs = outputs[0]
            if isinstance(image, list):  # Similar handling for labels if necessary
                image = image[0]

            bce_loss_2, dsc_loss_2 = meta_loss_fn(outputs, label)
            loss_vector = bce_loss_2 + dsc_loss_2
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
        with torch.no_grad():
            weight = meta_net(loss_vector_reshape)

        loss = torch.mean(weight * loss_vector_reshape)
        optimizer.zero_grad()

        # compute gradient and do optimizer step
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)  # enable grad clipping
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        bce_loss_3 = bce_loss_2.mean()
        dsc_loss_3 = dsc_loss_2.mean()

        # logging
        torch.cuda.synchronize()
        bce_meter.update(bce_loss_3.item(), bsz)
        dsc_meter.update(dsc_loss_3.item(), bsz)
        loss_meter.update(loss.item(), bsz)
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
        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
    }
    avg_loss = loss_meter.avg
    loss_history.append(avg_loss)
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    learning_rate_history.append(current_lr)
    save_learning_rate_history(learning_rate_history, args.exp_dir)

    # Save the loss history to a text file after each epoch
    save_loss_history(loss_history, args.exp_dir)
    for key, value in train_tb.items():
        writer.add_scalar(f"train/{key}", value, epoch)


def infer(args, epoch, model: nn.Module, loss_fn, infer_loader, writer, logger, mode: str, save_pred: bool = False):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()

    # make save epoch folder
    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = os.path.join(args.exp_dir, folder_dir)
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory is created correctly

    # Initialize text file for loss
    loss_txt_path = os.path.join(args.exp_dir, f"{mode}_loss.txt")

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            # get data
            image, label = image.cuda(), label.cuda().float()  # Convert label to float for loss computation
            bsz = image.size(0)

            # get seg map
            # Forward pass to get segmentation map
            seg_map = sliding_window_inference(
                inputs=image,
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )
            seg_map = seg_map > 0.5  # Convert to boolean after inference
            seg_map = brats_post_processing(seg_map)

            # Compute your validation loss (assuming binary cross entropy loss here)
            bce_loss_2, dsc_loss_2 = loss_fn(seg_map.float(), label)
            valloss = bce_loss_2 + dsc_loss_2
            total_loss += valloss.item()
            num_batches += 1

            # Calculate evaluation metrics
            dice = metrics.dice(seg_map, label.bool())
            hd95 = metrics.hd95(seg_map, label.bool())

            # Save predictions if specified
            if save_pred:
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            # Update time meter and metrics meter
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            # Log progress
            if (i == 0) or ((i + 1) % args.print_freq == 0):
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

        # Get validation metrics and log to tensorboard
        infer_metrics = case_metrics_meter.mean()
        for key, value in infer_metrics.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)

        # Compute average loss
        avg_loss = total_loss / num_batches

        # Append average loss to text file
        with open(loss_txt_path, 'a') as f:
            f.write(f"Epoch {epoch if epoch is not None else 'None'} - Average {mode} Loss: {avg_loss:.4f}\n")

        return infer_metrics

learning_rate_history = []
loss_history = []
class TrainModel:


    def main(self):
        args = parse_seg_args()
        logger, writer = initialization(args)


        meta_net = MLP(hidden_size=100, num_layers=1).cuda()
        if args.data_parallel:
            meta_net = nn.DataParallel(meta_net).cuda()
        # dataloaders
        meta_optimizer = get_clean_optimizer(args, meta_net)
        meta_scheduler = get_clean_scheduler(args, meta_optimizer)
        meta_loss = clean_SoftDiceBCEWithLogitsLoss().cuda()

        train_cases, val_cases, test_cases = load_cases_split(args.cases_split)
        train_loader = brats2021.get_train_loader(args, train_cases)
        val_loader   = brats2021.get_infer_loader(args, val_cases)
        test_loader  = brats2021.get_infer_loader(args, test_cases)

        train_clean_cases, val_clean_cases, test_clean_cases = load_cases_split(args.clean_cases_split)
        self.meta_dataloader = brats2021.get_clean_train_loader(args, train_clean_cases)


        self.meta_dataloader_iter = iter(self.meta_dataloader)        # model & stuff
        model = get_unet(args).cuda()
        if args.data_parallel:
            model = nn.DataParallel(model).cuda()
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        loss = SoftDiceBCEWithLogitsLoss().cuda()

        if args.amp:
            scaler = GradScaler()
            logger.info("==> Using AMP (Auto Mixed Precision)")
        else:
            scaler = None


        if args.amp:
            scaler_2 = GradScaler()
            logger.info("==> Using AMP (Auto Mixed Precision)")
        else:
            scaler_2 = None

        # load model
        if args.weight_path is not None:
            logger.info("==> Loading pretrain model...")
            assert args.weight_path.endswith(".pth")
            model_state = torch.load(args.weight_path)['model']
            model.load_state_dict(model_state)

        # train & val
        logger.info("==> Training starts...")
        best_model = {}
        val_leaderboard = LeaderboardBraTS()

        for epoch in range(args.epochs):

            train(self,args, epoch, model,meta_net, train_loader, loss,meta_loss, optimizer,meta_optimizer, scheduler,meta_scheduler, scaler, scaler_2,writer, logger)

            # if ((epoch + 1) % args.eval_freq == 0):
            logger.info(f"==> Validation starts...")
            # inference on validation set
            val_metrics = infer(args, epoch, model,loss, val_loader, writer, logger, mode='val')

            # model selection
            val_leaderboard.update(epoch, val_metrics)
            best_model.update({epoch: deepcopy(model.state_dict())})
            logger.info(f"==> Validation ends...")

            torch.cuda.empty_cache()
        # ouput final leaderboard and its rank
        val_leaderboard.output(args.exp_dir)

        # test
        logger.info("==> Testing starts...")
        best_epoch = val_leaderboard.get_best_epoch()
        best_model = best_model[best_epoch]
        model.load_state_dict(best_model)
        infer(args, best_epoch, model, test_loader, writer, logger, mode='test', save_pred=args.save_pred)

        # save the best model on validation set
        if args.save_model:
            logger.info("==> Saving...")
            state = {'model': best_model, 'epoch': best_epoch, 'args':args}
            torch.save(state, os.path.join(
                args.exp_dir, f"test_epoch_{best_epoch:02d}", f'best_ckpt.pth'))

        logger.info("==> Testing ends...")


if __name__ == '__main__':
    trainer = TrainModel()
    trainer.main()
