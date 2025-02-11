import os
from os.path import join
import torch
import random
import torch
import monai

import numpy as np
import monai.transforms as transforms
import nibabel as nib
from monai.transforms import EnsureChannelFirstd
from torch.utils.data import Dataset, DataLoader
from .dataset_utils import nib_load, RobustZScoreNormalization
# from .dataset_utils import RobustZScoreNormalization
# def nib_load(file_name):
#     if not os.path.exists(file_name):
#         raise FileNotFoundError(f"File not found: {file_name}")
#
#     print(f"Loading data from: {file_name}")
#     proxy = nib.load(file_name)
#     data = proxy.get_fdata()
#     proxy.uncache()
#
#     print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
#     return data
####################################################################################################
# transforms

# def get_brats2021_base_transform():
#     base_transform = [
#         # [B, H, W, D] --> [B, C, H, W, D]
#         transforms.EnsureChannelFirstD(keys=['flair', 't1', 't1ce', 't2', 'label']),
#         transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),
#         RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
#         transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
#         transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
#         transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
#     ]
#     return base_transform
seed=42
def get_brats2021_base_transform():
    base_transform = [
        # Removed EnsureChannelFirstD
        transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),
        RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    ]
    return base_transform


def get_brats2021_train_transform(args, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=seed)

    base_transform = get_brats2021_base_transform()
    data_aug = [
        # crop
        transforms.RandCropByPosNegLabeld(
            keys=["image", 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=1),

        # spatial aug
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        transforms.EnsureTyped(keys=["image", 'label']),
    ]

    return transforms.Compose(base_transform + data_aug)


def get_brats2021_infer_transform(args,seed=42):
    base_transform = get_brats2021_base_transform()
    infer_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
    return transforms.Compose(base_transform + infer_transform)


####################################################################################################
# dataset

class BraTS2021Dataset(Dataset):
    def __init__(self, data_root: str, mode: str, case_names: list = [], transforms=None):
        super(BraTS2021Dataset, self).__init__()

        assert mode in ['train', 'infer'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.case_names = case_names
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple:
        name = self.case_names[index]  # BraTS2021_00000
        base_dir = join(self.data_root, name)  # seg/data/brats21/BraTS2021_00000/BraTS2021_00000



        # flair = np.array(nib_load(os.path.join(base_dir, '_flair.nii.gz')), dtype='float32')
        # t1 = np.array(nib_load(os.path.join(base_dir, 't1.nii.gz')), dtype='float32')
        # t1ce = np.array(nib_load(os.path.join(base_dir, 't1ce.nii.gz')), dtype='float32')
        # t2 = np.array(nib_load(os.path.join(base_dir, 't2.nii.gz')), dtype='float32')
        # mask = np.array(nib_load(os.path.join(base_dir, '_seg.nii.gz')), dtype='float32')

        # Adjust the paths to concatenate the suffix directly to 'name' before adding the extension
        flair_path = os.path.join(base_dir, name + '_flair.nii.gz')
        t1_path = os.path.join(base_dir, name + '_t1.nii.gz')
        t1ce_path = os.path.join(base_dir, name + '_t1ce.nii.gz')
        t2_path = os.path.join(base_dir, name + '_t2.nii.gz')
        mask_path = os.path.join(base_dir, name + '_seg.nii.gz')
        # Load the imaging data using the nib_load function and convert to float32
        flair = np.array(nib_load(flair_path), dtype='float32')
        t1 = np.array(nib_load(t1_path), dtype='float32')
        t1ce = np.array(nib_load(t1ce_path), dtype='float32')
        t2 = np.array(nib_load(t2_path), dtype='float32')
        mask = np.array(nib_load(mask_path), dtype='float32')
        # mask = torch.tensor(nib_load(mask_path), dtype=torch.float32)
        # print("FLAIR sample data:", flair[0, 0, :200])  # Print the first ten elements of the first voxel line
        flair = flair[np.newaxis, ...]
        t1 = t1[np.newaxis, ...]
        t1ce = t1ce[np.newaxis, ...]
        t2 = t2[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        transforms = EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'label'], channel_dim=0)

        item = {'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'label': mask}
        item = self.transforms(item)


        if self.mode == 'train':  # train
            item = item[0]  # [0] for RandCropByPosNegLabeld

        return item['image'], item['label'], index, name

    def __len__(self):
        return len(self.case_names)


####################################################################################################
# dataloaders
# def worker_init_fn(worker_id):
#     torch.manual_seed(42)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)
def get_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args,seed)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)




def get_infer_loader(args, case_names: list):
    infer_transform = get_brats2021_infer_transform(args,seed)
    infer_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='infer',
        case_names=case_names,
        transforms=infer_transform)

    return DataLoader(infer_dataset, batch_size=args.infer_batch_size, shuffle=False,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)



# def get_clean_train_loader(args, case_names: list):
#
#     train_transforms = get_brats2021_train_transform(args)
#     train_dataset = BraTS2021Dataset(
#         data_root=os.path.join(args.clean_data_root, args.dataset),
#         mode='train',
#         case_names=case_names,
#         transforms=train_transforms)
#
#     return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                       drop_last=False, num_workers=args.num_workers, pin_memory=True)

def get_clean_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args,seed)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.clean_data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms)

    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g,persistent_workers=True)