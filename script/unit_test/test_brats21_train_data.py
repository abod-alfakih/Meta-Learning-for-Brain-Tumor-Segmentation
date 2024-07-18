import autorootcwd
import src.data
import torch, time, os
from src.utils.registry import DATASET_REGISTRY
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

# What is happening
print("==============================================")
print("Unit Testing BRATS 21 Dataset")
print("Author: Kanghyun Ryu (khryu@kist.re.kr)")
print("==============================================")

# Iterate over the data loader
print('---------- For train loader ---------------')
ii = 0
for batch in train_loader:
    images, labels, file_path = (
        batch["image"],
        batch["label"],
        batch["label_meta_dict"]["filename_or_obj"],
    )
    print(f"Loaded batch with {len(images)} images")
    print(f"Images size: {images.size()}, dtype: {images.dtype}")
    print(f"Labels size: {labels.size()}, dtype: {labels.dtype}")

    if ii == 0: 
        save_overlay_nifti(images, labels, file_path='script/unit_test/demo_train_data')


    # Extract the middle part of the file path
    print("files in mini-batch")
    for path in file_path:
        middle_part = os.path.basename(os.path.dirname(path))
        print(f"File path middle part: {middle_part}")

# Iterate over the data loader
print('---------- now for validation loader ---------------')
ii = 0
for batch in val_loader:
    images, labels, file_path = (
        batch["image"],
        batch["label"],
        batch["label_meta_dict"]["filename_or_obj"],
    )
    print(f"Loaded batch with {len(images)} images")
    print(f"Images size: {images.size()}, dtype: {images.dtype}")
    print(f"Labels size: {labels.size()}, dtype: {labels.dtype}")
    
    if ii == 0: 
        save_overlay_nifti(images, labels, file_path='script/unit_test/img/demo_val_data')

    # Extract the middle part of the file path
    print("files in mini-batch")
    for path in file_path:
        middle_part = os.path.basename(os.path.dirname(path))
        print(f"File path middle part: {middle_part}")
        
end_time = time.time()  # End time
total_time = end_time - start_time

print(f"Total time taken: {total_time:.2f} seconds")