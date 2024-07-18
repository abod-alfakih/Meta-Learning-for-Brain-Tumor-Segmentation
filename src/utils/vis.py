import numpy as np
import nibabel as nib
import torch
import os

def save_overlay_nifti(image, label, file_path):
    """
    Save an overlay of the image and label to a NIfTI file.

    Parameters:
    image (numpy.ndarray or torch.Tensor): The input image with shape (C, H, W, D) or (B, C, H, W, D).
    label (numpy.ndarray or torch.Tensor): The label with shape (3, H, W, D) or (B, 3, H, W, D).
    file_path (str): The file path where the NIfTI file will be saved. If the file path does not end with '.nii.gz', it will be appended.

    Example:
    >>> import numpy as np
    >>> image = np.random.rand(1, 256, 256, 128)
    >>> label = np.random.randint(0, 2, (3, 256, 256, 128))
    >>> save_overlay_nifti(image, label, 'output/overlay.nii.gz')
    Directory output does not exist. Creating it.
    """
    # Check if image and label are torch tensors, if so, convert to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    # Check if file_path ends with .nii.gz, if not, append it
    if not file_path.endswith('.nii.gz'):
        file_path += '.nii.gz'

    # Handle batch dimension if present
    if image.ndim == 5 and label.ndim == 5:  # (B, C, H, W, D) and (B, 3, H, W, D)
        if image.shape[0] != label.shape[0]:
            raise ValueError("The batch size of image and label must be the same.")
        batch_size = image.shape[0]
        for i in range(batch_size):
            batch_file_path = file_path.replace('.nii.gz', f'_{i+1}.nii.gz')
            save_single_overlay_nifti(image[i], label[i], batch_file_path)
    elif image.ndim == 4 and label.ndim == 4:  # (C, H, W, D) and (3, H, W, D)
        save_single_overlay_nifti(image, label, file_path)
    else:
        raise ValueError("Image and label must have compatible shapes: (C, H, W, D) and (3, H, W, D) or (B, C, H, W, D) and (B, 3, H, W, D).")

def overlay_label_on_image(image, label):
    """Overlay three labels with different intensities on the first channel of the image."""
    # Assuming image has shape (C, H, W, D) and label has shape (3, H, W, D)
    image_gray = image[0].copy()  # Use the first channel as the base for the grayscale image
    intensities = [1, 1, 1]  # Different intensities for three labels

    max_intensity = image_gray.max()  # Get the maximum intensity in image_gray

    for i in range(3):
        mask = label[i] > 0
        image_gray[mask] = intensities[i] * max_intensity  # Apply normalized intensity to the grayscale image

    return image_gray

def save_single_overlay_nifti(image, label, file_path):
    """
    Save a single overlay of the image and label to a NIfTI file.

    Parameters:
    image (numpy.ndarray): The input image with shape (C, H, W, D).
    label (numpy.ndarray): The label with shape (3, H, W, D).
    file_path (str): The file path where the NIfTI file will be saved.
    """
    overlay = overlay_label_on_image(image, label)
    # Convert overlay to numpy array with dtype float32
    overlay_np = overlay.astype(np.float32)
    
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Creating it.")
        os.makedirs(directory)
    
    img_nifti = nib.Nifti1Image(overlay_np, np.eye(4))
    nib.save(img_nifti, file_path)