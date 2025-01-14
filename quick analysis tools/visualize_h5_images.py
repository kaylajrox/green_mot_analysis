'''
This code is to quickly visualize an h5 image and a planned cropping zone. By terming top, bottom, left, right positions
then I can put this into other code with the numbers I find I want my image size to be

A rectangle appears on the original image for the cropping zone, and to the right is the cropped image
'''



import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg as the backend

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Independent Parameters for Configuration
folder_path = "S:\\Experiments\\Yb171_MOT_Tweezer_trap\\green_mot_search_camera\\2025\\01\\10\\2s_after_ramp_green_mot"  # Folder path for .h5 files
dataset_path = 'images/cam1/after ramp/frame'  # Path to dataset inside the .h5 file


# starts MUST be a lower number than the end
start_row = 550  # vertical height start (top edge)
end_row = 850  # vertical height end (bottom edge)
start_col = 910  # horizontontal width start point (left edge)
end_col = 1310  # horizontontal width end point (right edge)

# Ensure interactive mode is off
plt.ioff()  # Disable interactive mode

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):  # Check if the file is an .h5 file
        file_path = os.path.join(folder_path, filename)

        # Open the .h5 file
        with h5py.File(file_path, 'r') as f:
            try:
                # Load the image data from the specified dataset path
                image_data = f[dataset_path][:]

                # Check if the dataset is empty
                if image_data.size == 0:
                    print(f"Warning: Dataset in {filename} is empty.")
                    continue  # Skip this file if empty

                # Ensure the cropping region is valid (start < end for both rows and columns)
                if start_row >= end_row or start_col >= end_col:
                    print(f"Warning: Invalid cropping region in {filename}. Skipping...")
                    continue  # Skip this file if the cropping region is invalid

                # Crop the image
                cropped_image = image_data[start_row:end_row, start_col:end_col]

                # Create a figure with two subplots (one for the original image and one for the cropped one)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

                # Show the original image on the left
                ax[0].imshow(image_data, cmap='gray')
                ax[0].set_title(f"Original Image: {filename}")
                ax[0].axis('off')  # Hide axis labels

                # Create a rectangle around the cropped region
                rect = patches.Rectangle((start_col, start_row), end_col - start_col, end_row - start_row,
                                         linewidth=2, edgecolor='r', facecolor='none')  # Red rectangle
                ax[0].add_patch(rect)  # Add the rectangle to the original image

                # Show the cropped image on the right
                ax[1].imshow(cropped_image, cmap='gray')

                # Ensure that the cropped image has valid axis limits
                ax[1].set_title(f"Cropped Image: {filename}")
                ax[1].axis('off')  # Hide axis labels
                ax[1].set_xlim(0, cropped_image.shape[1])  # Set x limits based on the cropped image width
                ax[1].set_ylim(cropped_image.shape[0], 0)  # Set y limits based on the cropped image height (invert y-axis)

                # Display both images
                plt.show()

            except KeyError:
                print(f"Warning: Dataset path '{dataset_path}' not found in {filename}.")
                continue  # Skip this file if the dataset path is incorrect


