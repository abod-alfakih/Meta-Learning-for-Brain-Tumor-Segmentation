# import os
# import cv2
# import numpy as np
# import nibabel as nib
# import shutil  # Import shutil for copying files
#
# def dilate_image(image_path, output_path):
#     nifti_img = nib.load(image_path)
#     image_data = nifti_img.get_fdata()
#     edema_mask = (image_data == 2)
#     kernel = np.ones((3, 3))
#     eroded_image_data = cv2.erode(image_data, kernel)
#     eroded_nifti_img = nib.Nifti1Image(eroded_image_data, nifti_img.affine, nifti_img.header)
#
#
#
# def process_cases(input_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#     case_dirs = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
#     for case in case_dirs:
#         case_path = os.path.join(input_directory, case)
#         output_case_path = os.path.join(output_directory, case)
#         if not os.path.exists(output_case_path):
#             os.makedirs(output_case_path)
#         files = os.listdir(case_path)
#         for file in files:
#             if '_seg' in file and file.endswith('.nii.gz'):  # Assuming label files contain 'seg' and end with '.nii'
#                 label_path = os.path.join(case_path, file)
#                 output_label_path = os.path.join(output_case_path, file)
#                 dilate_image(label_path, output_label_path)
#             elif any(x in file for x in ['_t1', '_t2', '_flair', '_t1c']) and file.endswith('.nii.gz'):  # Assuming modality files end with '.nii'
#                 original_file_path = os.path.join(case_path, file)
#                 output_file_path = os.path.join(output_case_path, file)
#                 shutil.copy(original_file_path, output_file_path)  # Copy the file to the output directory
#
#
# # Specify the input and output directories
# input_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\brats2021"
# output_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\PRE"

# # Process all cases
# process_cases(input_directory, output_directory)
#
# #
# import os
# import cv2
# import numpy as np
# import nibabel as nib
# import shutil  # Import shutil for copying files
#
# import os
# import shutil
# import numpy as np
# import cv2
# import nibabel as nib
#
#
# def process_segmentation(image_data):
#     # Define labels (assuming these are the correct labels for edema, enhancing tumor, and tumor core)
#     tumor_core_label = 1
#     edema_label = 2
#     enhancing_tumor_label = 4
#
#     # Create binary masks for each region
#     edema_mask = (image_data == edema_label).astype(np.uint8)
#     enhancing_tumor_mask = (image_data == enhancing_tumor_label).astype(np.uint8)
#     tumor_core_mask = (image_data == tumor_core_label).astype(np.uint8)
#
#     # Combine edema and enhancing tumor masks
#     edema_enhancing_tumor_mask = np.logical_or(edema_mask, enhancing_tumor_mask).astype(np.uint8)
#
#     # Define the structuring element
#     kernel = np.ones((3, 3), np.uint8)
#
#     # Apply erosion to edema and enhancing tumor regions combined
#     eroded_edema_enhancing_tumor = cv2.erode(edema_enhancing_tumor_mask, kernel)
#
#     # Apply dilation to tumor core region
#     dilated_tumor_core = cv2.dilate(tumor_core_mask, kernel)
#
#     # Separate edema and enhancing tumor after erosion
#     eroded_edema = np.logical_and(eroded_edema_enhancing_tumor, edema_mask).astype(np.uint8)
#     eroded_enhancing_tumor = np.logical_and(eroded_edema_enhancing_tumor, enhancing_tumor_mask).astype(np.uint8)
#
#     # Combine the processed regions back into the original label map
#     processed_image_data = np.zeros_like(image_data, dtype=np.uint8)
#     processed_image_data[eroded_edema == 1] = edema_label
#     processed_image_data[eroded_enhancing_tumor == 1] = enhancing_tumor_label
#     processed_image_data[dilated_tumor_core == 1] = tumor_core_label
#
#     # Copy over any other labels (if any other labels exist, handle them appropriately)
#     other_labels = np.isin(image_data, [edema_label, enhancing_tumor_label, tumor_core_label], invert=True)
#     processed_image_data[other_labels] = image_data[other_labels]
#
#     return processed_image_data
#
#
# def process_image(image_path, output_path):
#     nifti_img = nib.load(image_path)
#     image_data = nifti_img.get_fdata().astype(np.uint8)  # Ensure data type is consistent
#
#     processed_image_data = process_segmentation(image_data)
#
#     processed_nifti_img = nib.Nifti1Image(processed_image_data, nifti_img.affine)
#     nib.save(processed_nifti_img, output_path)
#
#
# def process_cases(input_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     case_dirs = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
#
#     for case in case_dirs:
#         case_path = os.path.join(input_directory, case)
#         output_case_path = os.path.join(output_directory, case)
#
#         if not os.path.exists(output_case_path):
#             os.makedirs(output_case_path)
#
#         files = os.listdir(case_path)
#
#         for file in files:
#             if '_seg' in file and file.endswith('.nii.gz'):
#                 # Process label file
#                 label_path = os.path.join(case_path, file)
#                 output_label_path = os.path.join(output_case_path, file)
#                 process_image(label_path, output_label_path)
#
#             elif any(x in file for x in ['_t1', '_t2', '_flair', '_t1c']) and file.endswith('.nii.gz'):
#                 # Copy modality file
#                 original_file_path = os.path.join(case_path, file)
#                 output_file_path = os.path.join(output_case_path, file)
#                 shutil.copy(original_file_path, output_file_path)
#
#
# # Specify the input and output directories
# input_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\brats2021"
# output_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\processed"
#
# # Process all cases
# process_cases(input_directory, output_directory)
#

# import os
# import nibabel as nib
# noisy_folder = r'C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\100%\errored'
#
# clean_folder = r'C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\100%\dilation'
# output_folder = r'C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\100%\New folder'

# import os
# import numpy as np
# import nibabel as nib
#
# # Paths to your clean and noisy Brats data directories
# clean_brats_data_dir = r'C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\full\errored'
# noisy_brats_data_dir = r'C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\full\dilation'
#
#
#
#
# for case_folder in os.listdir(clean_brats_data_dir):
#     case_folder_path = os.path.join(clean_brats_data_dir, case_folder)
#     clean_seg_path = os.path.join(case_folder_path, f'{case_folder}_seg.nii.gz')
#
#     # Check if clean_seg_path exists
#     if not os.path.exists(clean_seg_path):
#         print(f"Clean segmentation file not found: {clean_seg_path}")
#         continue
#
#     try:
#         clean_seg = nib.load(clean_seg_path)
#         clean_seg_data = clean_seg.get_fdata()
#
#         noisy_seg_path = os.path.join(noisy_brats_data_dir, case_folder, f'{case_folder}_seg.nii.gz')
#
#         # Check if noisy_seg_path exists
#         if not os.path.exists(noisy_seg_path):
#             print(f"Noisy segmentation file not found: {noisy_seg_path}")
#             continue
#
#         noisy_seg = nib.load(noisy_seg_path)
#         noisy_seg_data = noisy_seg.get_fdata()
#
#         # Ensure that the dimensions of the segmentation data match
#         if clean_seg_data.shape != noisy_seg_data.shape:
#             print(f"Dimension mismatch for {case_folder}: clean({clean_seg_data.shape}) vs noisy({noisy_seg_data.shape})")
#             continue
#
#         # Create masks for the enhancing tumor (label 4) and edema (label 1 in this case) in the noisy segmentation data
#         edema_mask = (noisy_seg_data == 1)
#
#         # Create a copy of the clean segmentation data
#         updated_seg_data = clean_seg_data.copy()
#
#         # Replace the enhancing tumor (label 4) and edema (label 1) in the clean segmentation data with the noisy segmentation data
#         updated_seg_data[edema_mask] = noisy_seg_data[edema_mask]
#
#         # Save the updated segmentation
#         clean_seg_updated = nib.Nifti1Image(updated_seg_data, clean_seg.affine, clean_seg.header)
#         nib.save(clean_seg_updated, clean_seg_path)
#
#         print(f"Processed {case_folder}")
#     except Exception as e:
#         print(f"Error processing {case_folder}: {e}")
#
# print("Segmentation replacement completed.")
import os
import cv2
import numpy as np
import nibabel as nib
import shutil


def dilate_image(image_path, output_path, label_path_2):
    # Load original image data
    nifti_img = nib.load(image_path)
    image_data = nifti_img.get_fdata().astype(np.uint8)  # Ensure data type for cv2 compatibility

    # Load label image data
    nifti_img_2 = nib.load(label_path_2)
    image_data_2 = nifti_img_2.get_fdata().astype(np.uint8)  # Ensure data type for cv2 compatibility

    # Assuming tumor core label is 1 (adjust this based on your label definitions)
    tumor_core_label = 1
    enhancing_tumor_label = 4

    # Create a mask for the tumor core in original image data
    tumor_core_mask = (image_data == tumor_core_label)

    # Apply erosion to the original image data
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_image_data = cv2.erode(image_data, kernel)

    # Remove tumor core regions from the eroded image
    eroded_image_data[tumor_core_mask] = 0

    # Add original tumor core regions from image_data_2 to the eroded image
    eroded_image_data[tumor_core_mask] = image_data_2[tumor_core_mask]

    # Create a new Nifti image with the eroded data
    eroded_nifti_img = nib.Nifti1Image(eroded_image_data, nifti_img.affine, nifti_img.header)

    # Save the eroded image to the specified output path
    nib.save(eroded_nifti_img, output_path)


def process_cases(input_directory, output_directory, input_directory_2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    case_dirs = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    case_dirs_2 = [d for d in os.listdir(input_directory_2) if os.path.isdir(os.path.join(input_directory_2, d))]

    for case in case_dirs:
        case_path = os.path.join(input_directory, case)
        case_path_2 = os.path.join(input_directory_2, case)
        output_case_path = os.path.join(output_directory, case)

        if not os.path.exists(output_case_path):
            os.makedirs(output_case_path)

        files = [f for f in os.listdir(case_path) if '_seg' in f and f.endswith('.nii.gz')]

        for file in files:
            label_path = os.path.join(case_path, file)
            label_path_2 = os.path.join(case_path_2, file)  # Matching corresponding file from case_path_2

            output_label_path = os.path.join(output_case_path, file)
            dilate_image(label_path, output_label_path, label_path_2)

        for file in os.listdir(case_path):
            if any(x in file for x in ['_t1', '_t2', '_flair', '_t1c']) and file.endswith('.nii.gz'):
                original_file_path = os.path.join(case_path, file)
                output_file_path = os.path.join(output_case_path, file)
                shutil.copy(original_file_path, output_file_path)

# Specify the input and output directories
input_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\brats2021"
input_directory_2 = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\NOISE\20%\dilation"
output_directory = r"C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\PREduse"

# Process the cases
process_cases(input_directory, output_directory, input_directory_2)
