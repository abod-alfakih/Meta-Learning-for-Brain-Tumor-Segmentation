import autorootcwd
import src.data
import src.archs
import src.models
import torch, time, os
from src.utils.registry import DATASET_REGISTRY
from src.utils.registry import MODEL_REGISTRY
from src.utils.registry import ARCH_REGISTRY

from src.utils.vis import save_overlay_nifti
# Retrieve the BRATS21 dataset from

args = {
    "data_root": "data/noise",
    "dataset": "brats2021",
    "cases_split": "data/split/brats2021_split_fold5.csv",
    "batch_size": 8,
    "num_workers": 4,
    "patch_size": 96,
    "pos_ratio": 1.0,
    "neg_ratio": 1.0,
}

brats_dataset = DATASET_REGISTRY.get("BRaTS21TrainDataset")(
    args["data_root"],
    args["dataset"],
    args["cases_split"],
    args["batch_size"],
    args["num_workers"],
    patch_size=args["patch_size"],
)

train_loader = brats_dataset.train_loader
val_loader = brats_dataset.val_loader

start_time = time.time()  # Start time

arch = ARCH_REGISTRY.get("UNETR")(
    img_size=(96, 96, 96),  # Example input size
    in_channels=4,          # Example input channels
    out_channels=3,         # Example output channels
    feature_size=24         # Example feature size
)

model = MODEL_REGISTRY.get('SupervisedTrainer')(arch, device='cuda:0')
model.make_dataloaders(train_loader=train_loader, val_loader=val_loader)

nepoch = 50

for epoch in range(nepoch):
    model.train_val_one_epoch(epoch, nepoch)