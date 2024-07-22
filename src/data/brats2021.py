import autorootcwd

import os
import time
from os.path import join

from monai.data import (
    LMDBDataset,
    ThreadDataLoader,
)

import monai
import monai.transforms as transforms
from monai.transforms.transform import MapTransform

import numpy as np
import nibabel as nib
import pandas as pd

from src.utils.registry import DATASET_REGISTRY


def load_cases_split(split_path: str):
    df = pd.read_csv(split_path)
    cases_name, cases_split = np.array(df["name"]), np.array(df["split"])
    train_cases = sorted(sorted(list(cases_name[cases_split == "train"])))
    val_cases = sorted(list(cases_name[cases_split == "val"]))
    test_cases = sorted(list(cases_name[cases_split == "test"]))
    meta_cases = sorted(list(cases_name[cases_split == "meta_train"]))


    return train_cases, val_cases, test_cases ,meta_cases


class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = lower
            d[key][mask & (d[key] > upper)] = upper

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()

        return d


def process_f32(img_dir):
    """Set all Voxels that are outside of the brain mask to 0"""
    modalities = ["t2", "t1ce", "flair", "t1"]
    name = os.path.basename(img_dir)
    images = np.stack(
        [
            np.array(
                nib_load(join(img_dir, name) + "_" + i + ".nii.gz"),
                dtype="float32",
                order="C",
            )
            for i in modalities
        ],
        -1,
    )  # [240, 240, 155, 4]
    mask = images.sum(-1) > 0  # [240, 240, 155]

    for k in range(len(modalities)):
        x = images[..., k]
        y = x[mask]  # get brain mask

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]
        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    return images


def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError
        print(file_name)
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def get_brats2021_base_transform():
    base_transform = [
        transforms.LoadImaged(
            keys=["flair", "t1", "t1ce", "t2", "label"],
            ensure_channel_first=True,
            image_only=False,
        ),
        transforms.Orientationd(
            keys=["flair", "t1", "t1ce", "t2", "label"], axcodes="RAS"
        ),
        RobustZScoreNormalization(keys=["flair", "t1", "t1ce", "t2"]),
        transforms.ConcatItemsd(
            keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0
        ),
        transforms.DeleteItemsd(keys=["flair", "t1", "t1ce", "t2"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    ]
    return base_transform


def get_brats2021_train_transform(patch_size, pos_ratio, neg_ratio, mode="train"):
    base_transform = get_brats2021_base_transform()
    data_aug = [
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[patch_size] * 3,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=1,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys="image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        transforms.RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.3)),
        transforms.EnsureTyped(keys=["image", "label"]),
    ]
    if mode == "train":
        return transforms.Compose(base_transform + data_aug)
    elif mode == "val":
        return base_transform
    else:
        raise ValueError("Mode must be 'train' or 'val'")


def get_brats2021_infer_transform():
    base_transform = get_brats2021_base_transform()
    infer_transform = [transforms.EnsureTyped(keys=["image", "label"])]
    return transforms.Compose(base_transform + infer_transform)


@DATASET_REGISTRY.register()
class BRaTS21TrainDataset:
    def __init__(
        self,
        data_root,
        dataset,
        cases_split,
        batch_size,
        num_workers,
        patch_size=96,
        pos_ratio=1,
        neg_ratio=1,
        seed=42,
    ):
        # Set global seed for MONAI
        monai.utils.set_determinism(seed=seed)

        self.data_root = data_root
        self.dataset = dataset
        self.cases_split = cases_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio

        self.train_cases_dict, self.val_cases_dict, self.test_cases_dict = (
            self._prepare_cases(self.cases_split)
        )
        # prepare dictionary for list of files

        self.train_transforms = get_brats2021_train_transform(
            self.patch_size, self.pos_ratio, self.neg_ratio
        )

        self.val_transforms = get_brats2021_train_transform(
            self.patch_size, self.pos_ratio, self.neg_ratio
        )

        self.infer_transforms = get_brats2021_infer_transform()

        # Now create dset
        train_lmdb_dset = self._create_lmdb_dataset(
            self.train_cases_dict, transforms=self.train_transforms
        )
        val_lmdb_dset = self._create_lmdb_dataset(
            self.val_cases_dict, transforms=self.val_transforms
        )

        self.train_loader = self.get_dataloader(
            train_lmdb_dset, shuffle=True, num_workers=self.num_workers
        )
        self.val_loader = self.get_dataloader(
            val_lmdb_dset, shuffle=True, num_workers=self.num_workers
        )

    def _prepare_cases(self, cases_split):
        train_cases, val_cases, test_cases, meta_cases= load_cases_split(cases_split)

        def create_case_dict(case):
            return {
                "flair": os.path.join(
                    self.data_root, self.dataset, case, f"{case}_flair.nii.gz"
                ),
                "t1": os.path.join(
                    self.data_root, self.dataset, case, f"{case}_t1.nii.gz"
                ),
                "t1ce": os.path.join(
                    self.data_root, self.dataset, case, f"{case}_t1ce.nii.gz"
                ),
                "t2": os.path.join(
                    self.data_root, self.dataset, case, f"{case}_t2.nii.gz"
                ),
                "label": os.path.join(
                    self.data_root, self.dataset, case, f"{case}_seg.nii.gz"
                ),
            }

        def validate_case_dict(case_dict):
            for key, path in case_dict.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

        train_cases_dict = []
        val_cases_dict = []
        test_cases_dict = []
        meta_cases_dict = []

        for case in train_cases:
            case_dict = create_case_dict(case)
            validate_case_dict(case_dict)
            train_cases_dict.append(case_dict)

        for case in val_cases:
            case_dict = create_case_dict(case)
            validate_case_dict(case_dict)
            val_cases_dict.append(case_dict)

        for case in test_cases:
            case_dict = create_case_dict(case)
            validate_case_dict(case_dict)
            test_cases_dict.append(case_dict)

        for case in meta_cases:
            case_dict = create_case_dict(case)
            validate_case_dict(case_dict)
            meta_cases_dict.append(case_dict)

        return train_cases_dict, val_cases_dict, test_cases_dict , meta_cases_dict

    def _create_lmdb_dataset(self, cases_dict, transforms):
        lmdb_init_start = time.time()
        LMDB_cache = os.path.join(self.data_root, "lmdb_cache")
        
        if not os.path.exists(LMDB_cache):
            print(f'.... making new cache folder at {LMDB_cache}')
            
            os.makedirs(LMDB_cache)
            
        cache_dset = LMDBDataset(
            cases_dict,
            transform=transforms,
            cache_dir=LMDB_cache,
            lmdb_kwargs={"map_async": True},
        )
        lmdb_init_time = time.time() - lmdb_init_start
        print(f"LMDB init time taken: {lmdb_init_time:.2f} seconds")
        return cache_dset

    def get_dataloader(self, dset, shuffle, num_workers):
        loader = ThreadDataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        return loader


# Usage example
if __name__ == "__main__":
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

    brats_dataset = BRaTS21TrainDataset(
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

    # Iterate over the data loader
    print('---------- For train loader ---------------')

    for batch in train_loader:
        images, labels, file_path = (
            batch["image"],
            batch["label"],
            batch["label_meta_dict"]["filename_or_obj"],
        )
        print(f"Loaded batch with {len(images)} images")
        print(f"Images size: {images.size()}, dtype: {images.dtype}")
        print(f"Labels size: {labels.size()}, dtype: {labels.dtype}")

        # Extract the middle part of the file path
        print("files in mini-batch")
        for path in file_path:
            middle_part = os.path.basename(os.path.dirname(path))
            print(f"File path middle part: {middle_part}")

    # Iterate over the data loader
    print('---------- now for validation loader ---------------')
    for batch in val_loader:
        images, labels, file_path = (
            batch["image"],
            batch["label"],
            batch["label_meta_dict"]["filename_or_obj"],
        )
        print(f"Loaded batch with {len(images)} images")
        print(f"Images size: {images.size()}, dtype: {images.dtype}")
        print(f"Labels size: {labels.size()}, dtype: {labels.dtype}")

        # Extract the middle part of the file path
        print("files in mini-batch")
        for path in file_path:
            middle_part = os.path.basename(os.path.dirname(path))
            print(f"File path middle part: {middle_part}")
    end_time = time.time()  # End time
    total_time = end_time - start_time

    print(f"Total time taken: {total_time:.2f} seconds")
