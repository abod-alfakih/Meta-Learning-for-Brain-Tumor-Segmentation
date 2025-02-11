3D U-NET BraTS PyTorch
This repository contains a PyTorch implementation of 3D U-Net for Multimodal MRI Brain Tumor Segmentation (BraTS 2021).

Table of Contents
Features
Organize Data Set with DVC
Installation
1. Clone the Repository
2. Install Pipenv
3. Activate the Pipenv Shell
4. Install the Required Dependencies
5. Install PyTorch (GPU version for CUDA 11.1)
6. Install DVC
Setup DVC
Running the Scripts
Features
3D U-Net architecture for MRI brain tumor segmentation.
Supports multimodal MRI images.
Clean and noisy data processing.

# 3D U-NET BraTS PyTorch

This repo is a PyTorch implementation of 3D U-Net  for Multimodal MRI Brain Tumor Segmentation (BraTS 2021).


# Organize Data set
```
3DUNet-BraTS-PyTorch-master/
│
├── data/
│   ├── clean
│   │   ├── brats2021 # but the clean data here
│   ├── noise
│   │   ├── brats2021 # but the noise data here
│   ├── split
└── └──── files_csv
```

## Installation
Before running the 3D UNet model, you need to set up the environment and install the required dependencies. Follow these steps:
### 1. Clone the repository
First, you need to download a copy of the project repository to your local machine. This is done using the git clone command
```
git clone https://github.com/yourusername/my_3d_unet_project.git
cd my_3d_unet_project
```
* git clone https://github.com/yourusername/my_3d_unet_project.git: This command clones (downloads) the repository from GitHub to your local machine.
* cd my_3d_unet_project: This command changes the current directory to the newly cloned repository directory, allowing you to work within the project files.
### 2. Install Pipenv
Pipenv is a tool that manages Python virtual environments and dependencies. If you don't have Pipenv installed, you can install it using the following command:
```pip install pipenv```

### 3. Activate the Pipenv shell
Next, you need to create a virtual environment for the project and activate it. This ensures that all dependencies are installed in an isolated environment, avoiding conflicts with other projects.
```pipenv shell```


### 4. Install the required dependencies
Now, install all the dependencies listed in the Pipfile of the project. This step ensures that all the necessary libraries and packages are available for the project to run correctly.

```pipenv install requests```

### 5. Install PyTorch (GPU version for CUDA 11.1)
Install PyTorch and related libraries suitable for your CUDA version (replace cu118 with your CUDA version if different):
```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

### run the scripts
Once the environment is set up and the dependencies are installed, you can run the project's scripts. For example, to start training the model, you would use:
```pipenv run python train_brats2021.py```

By following the previous steps, you ensure that:

* No Need for Manual Paths: You do not need to manually add any paths for the model. The environment and paths are configured automatically.
* No Additional Packages Required: All necessary packages and dependencies are installed as part of the setup process. You do not need to install any additional packages manually.
The command pipenv run python train_brats2021.py ensures that the script is executed using the correct environment and dependencies managed by Pipenv, providing a seamless setup and execution process.