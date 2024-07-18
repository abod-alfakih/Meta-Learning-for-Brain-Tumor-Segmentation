from src.utils.registry import DATASET_REGISTRY, MODEL_REGISTRY
import src.losses.brats21 as losses
from .base_model import BaseModel
from tqdm import tqdm
import torch

@MODEL_REGISTRY.register()
class SupervisedTrainer(BaseModel):
    def __init__(
        self,
        arch,
        device,
        optim_type="AdamW",
        lr=1e-3,
        loss_type="SoftDiceBCEWithLogits",
        aux_arch=None,
        mode="train",
    ):
        super().__init__(arch, device, aux_arch, mode)

        if loss_type == "SoftDiceBCEWithLogits":
            self.criterion = losses.SoftDiceBCEWithLogitsLoss()
            self.criterion = self.criterion.to(self.device)
        else:
            ValueError(f"loss type: {loss_type} is not available")

        self.optimizer = self.get_optimizer(
            optim_type=optim_type, params=self.arch.parameters(), lr=lr
        )

    def feed_data(self, batch):
        img, label = batch["image"], batch["label"]
        img = img.to(self.device)
        label = label.float().to(self.device)
        return img, label

    def train_step(self, batch):
        img, label = self.feed_data(batch)
        pred_seg = self.arch(img)

        loss = self.criterion(pred_seg, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_one_epoch(self, dataloader, current_epoch, total_epoch=0):
        self.arch.train()
        running_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Training Epoch {current_epoch}/{total_epoch}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(batch)
                running_loss += loss
                pbar.set_postfix(loss=loss)
                pbar.update(1)
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{current_epoch}/{total_epoch}], Loss: {epoch_loss:.4f}")

