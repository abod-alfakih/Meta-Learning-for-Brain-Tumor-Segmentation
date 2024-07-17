from utils.train_utils import (
    save_loss_history,
    save_learning_rate_history,
    train,
    infer,
)
from configs import parse_seg_args
from utils.misc import initialization
from models.unet import MLP, get_unet
import torch.nn as nn
from utils.optim import get_optimizer, get_clean_optimizer
from utils.scheduler import get_scheduler, get_clean_scheduler
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (LeaderboardBraTS,
                         initialization, load_cases_split)
from dataset import brats2021
import torch
import os
from copy import deepcopy

learning_rate_history = []
loss_history = []

class TrainModel:
    def main(self):
        args = parse_seg_args()
        logger, writer = initialization(args)

        v_net = MLP(hidden_size=100, num_layers=1).cuda()
        meta_optimizer = get_clean_optimizer(args, v_net)

        train_cases, val_cases, test_cases = load_cases_split(args.cases_split)

        train_loader = brats2021.get_train_loader(args, train_cases)
        val_loader = brats2021.get_infer_loader(args, val_cases)
        test_loader = brats2021.get_infer_loader(
            args, test_cases
        )  # what is the test loader?

        train_clean_cases, val_clean_cases, test_clean_cases = load_cases_split(
            args.clean_cases_split
        )
        self.meta_dataloader = brats2021.get_clean_train_loader(args, train_clean_cases)
        self.meta_dataloader_iter = iter(self.meta_dataloader)  # model & stuff

        model = get_unet(args).cuda()
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        loss = SoftDiceBCEWithLogitsLoss().cuda()

        # load model
        if args.weight_path is not None:
            logger.info("==> Loading pretrain model...")
            assert args.weight_path.endswith(".pth")
            model_state = torch.load(args.weight_path)["model"]
            model.load_state_dict(model_state)

        # train & val
        logger.info("==> Training starts...")
        best_model = {}
        val_leaderboard = LeaderboardBraTS()

        for epoch in range(args.epochs):

            train(
                self,
                args,
                epoch,
                model,
                v_net,
                train_loader,
                loss,
                optimizer,
                scheduler,
                writer,
                logger,
            )

            if epoch >= 0:
                logger.info(f"==> Validation starts...")
                # inference on validation set
                val_metrics = infer(
                    args, epoch, model, loss, val_loader, writer, logger, mode="val"
                )

                # model selection
                val_leaderboard.update(epoch, val_metrics)
                best_model.update({epoch: deepcopy(model.state_dict())})
                logger.info(f"==> Validation ends...")

            torch.cuda.empty_cache()
        # ouput final leaderboard and its rank
        val_leaderboard.output(args.exp_dir)

        # test
        best_epoch = val_leaderboard.get_best_epoch()
        best_model = best_model[best_epoch]
        model.load_state_dict(best_model)

        # save the best model on validation set
        if args.save_model:
            logger.info("==> Saving...")
            state = {"model": best_model, "epoch": best_epoch, "args": args}
            torch.save(
                state,
                os.path.join(
                    args.exp_dir, f"test_epoch_{best_epoch:02d}", f"best_ckpt.pth"
                ),
            )

        logger.info("==> Testing starts...")
        infer(
            args,
            best_epoch,
            model,
            loss,
            test_loader,
            writer,
            logger,
            mode="test",
            save_pred=args.save_pred,
        )
        logger.info("==> Testing ends...")


if __name__ == "__main__":
    trainer = TrainModel()
    trainer.main()
