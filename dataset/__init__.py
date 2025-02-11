# # from .dataset_utils import nib_load, RobustZScoreNormalization
# import nibabel as nib
#
# # Example path, replace this with one of the printed paths from your debugging
# test_path = 'C:\\Users\\USER\\Master-Degree\\DATA\\BraTS2021\\archive\\BraTS2021_Training_Data\\BraTS2021_00000\\BraTS2021_00000_flair.nii.gz'
#             C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\BraTS2021_01658\BraTS2021_01658_flair.nii.gz
# img = nib.load(test_path)
# print(img.shape)  # This should print the dimensions of the image if successful

# import torch
# import torch.nn.functional as F
#
# # Define example tensors
# tensor1 = torch.tensor([1.0, 1.0, 3.0])
# tensor2 = torch.tensor([-1.0, 0.1, 0.0])
#
# # Reshape tensors to (3, 1) to make them two-dimensional
# # tensor1_reshaped = tensor1.view(-1, 1)
# # print(tensor1_reshaped)
# # tensor2_reshaped = tensor2.view(-1, 1)
# # print(tensor2_reshaped)
# # Compute cosine similarity along the columns (dim=1)
# similarity = F.cosine_similarity(tensor1, tensor2, dim=0)
#
# print(similarity)
#

# import pandas as pd
# import numpy as np
#
# # Load the CSV file
# df = pd.read_csv(r"C:\Users\USER\Master-Degree\3DUNet-BraTS-PyTorch-master\pediatric\data\split\brats2021_split2.csv")
#
# # Shuffle the data
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Calculate the number of rows for each split
# total_len = len(df)
# train_size = int(0.8 * total_len)
# valid_size = int(0.04 * total_len)
# test_size = total_len - train_size - valid_size  # Ensures remaining goes to test
#
# # Create 'split' column as type 'str' to hold string values
# df['split'] = np.nan  # Create 'split' column with NaN values initially
# df['split'] = df['split'].astype(str)  # Cast 'split' column to string type
#
# # Assign 'train' to the first 80%, 'val' to the next 8%, and 'test' to the rest
# df.loc[:train_size, 'split'] = 'train'
# df.loc[train_size:train_size + valid_size, 'split'] = 'val'
# df.loc[train_size + valid_size:, 'split'] = 'test'
#
# # Save to a new CSV file
# df.to_csv(r"C:\Users\USER\Master-Degree\3DUNet-BraTS-PyTorch-master\new\data\split\brats2021_split.csv", index=False)
#
# # Check the resulting DataFrame
# print(df.head())
