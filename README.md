# 3D U-NET BraTS PyTorch

This repository contains a PyTorch implementation of 3D U-Net for Multimodal MRI Brain Tumor Segmentation (BraTS 2021).

## Table of Contents
- [Features](#features)
- [Organize Data Set with DVC](#organize-data-set-with-dvc)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Pipenv](#2-install-pipenv)
  - [3. Activate the Pipenv Shell](#3-activate-the-pipenv-shell)
  - [4. Install the Required Dependencies](#4-install-the-required-dependencies)
  - [5. Install PyTorch (GPU version for CUDA 11.1)](#5-install-pytorch-gpu-version-for-cuda-111)
  - [6. Install DVC](#6-install-dvc)
- [Setup DVC](#setup-dvc)
- [Running the Scripts](#running-the-scripts)

## Features
- 3D U-Net architecture for MRI brain tumor segmentation.
- Supports multimodal MRI images.
- Clean and noisy data processing.

## Organize Data Set with DVC
DVC (Data Version Control) is used to manage and version control your datasets efficiently. 

### 1. Initialize DVC
Initialize DVC in your project directory:
```sh
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```


### 2. Add Data to DVC
Add your dataset to DVC:
```sh
dvc add data/clean/brats2021
dvc add data/noise/brats2021
```

This will create .dvc files for your data, which should be tracked by Git:
```sh
git add data/clean/brats2021.dvc data/noise/brats2021.dvc
git commit -m "Add datasets to DVC"
```
### 3. Configure Google Drive Remote Storage
Configure Google Drive as your remote storage. You need the folder IDs from Google Drive where your data is stored:

Add remote storage for the clean data:
```sh
dvc remote add -d clean_remote gdrive://<clean-folder-id>

```
Add remote storage for the noise data:
```sh
dvc remote add noise_remote gdrive://<noise-folder-id>
```

### 4. Configure Google Drive Remote Storage
To retrieve the datasets, use:
```sh
dvc pull -r clean_remote
dvc pull -r noise_remote
```
## Installation

Before running the 3D U-Net model, you need to set up the environment and install the required dependencies. Follow these steps:

### 1. Clone the Repository

First, download a copy of the project repository to your local machine:

```sh
git clone https://github.com/yourusername/my_3d_unet_project.git
cd my_3d_unet_project
```
### 2. Install DVC
nstall DVC to manage and version control your datasets:
```sh
pip install dvc[gdrive]
```


### 3. Install Pipenv
Pipenv is a tool that manages Python virtual environments and dependencies. If you don't have Pipenv installed, use:
```sh
pip install pipenv
```
### 4. Activate the Pipenv shell
Next, you need to create a virtual environment for the project and activate it. This ensures that all dependencies are installed in an isolated environment, avoiding conflicts with other projects.
```pipenv shell```


### 5. Install the required dependencies
Now, install all the dependencies listed in the Pipfile of the project. This step ensures that all the necessary libraries and packages are available for the project to run correctly.

```sh
pipenv install requests
```

### 6. Install PyTorch (GPU version for CUDA 11.1)
Install PyTorch and related libraries suitable for your CUDA version (replace cu118 with your CUDA version if different):
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### run the scripts
Once the environment is set up and the dependencies are installed, you can run the project's scripts. For example, to start training the model, you would use:
```sh
pipenv run python train_brats2021.py
```

By following the previous steps, you ensure that:

* No Need for Manual Paths: You do not need to manually add any paths for the model. The environment and paths are configured automatically.
* No Additional Packages Required: All necessary packages and dependencies are installed as part of the setup process. You do not need to install any additional packages manually.
The command pipenv run python train_brats2021.py ensures that the script is executed using the correct environment and dependencies managed by Pipenv, providing a seamless setup and execution process.





