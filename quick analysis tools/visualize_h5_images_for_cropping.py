'''
This code is to quickly visualize an h5 image and a planned cropping zone. By terming top, bottom, left, right positions
then I can put this into other code with the numbers I find I want my image size to be.

It currently loops through the ALL h5 file in a specified directory

A rectangle appears on the original image for the cropping zone, and to the right is the cropped image.

Its purpose is to be able to quickly find out the crop numbers you need by manually adjusting and using those numbers for other scripts
'''

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg as the backend

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# THESE ARE THE ONLY PARAMETERS THE USER SHOULD BE CHANGING

# Define the cropping region
top = 350  # vertical height start (top edge)
bottom = 1050  # vertical height end (bottom edge)
left = 950  # horizontal width start point (left edge)
right = 1650  # horizontal width end point (right edge)

data_day_name = '20250123TOF_withBlueMOTBeams'
experiment_title = 'NoRamp_4V'

#===============================================================================================

# Locate the base directory: 'green_mot_analysis'
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'green_mot_analysis'))

# Path to the specific data folder: 'recaptured MOT'
recaptured_mot_folder = os.path.join(base_directory, 'data',data_day_name, experiment_title)

# Check if the folder exists
if not os.path.exists(recaptured_mot_folder):
    raise FileNotFoundError(f"The folder does not exist: {recaptured_mot_folder}")

# List all .h5 files in the 'recaptured MOT' folder
files = [os.path.join(recaptured_mot_folder, f) for f in os.listdir(recaptured_mot_folder) if f.endswith('.h5')]

# Print the files for verification
print("HDF5 files in 'recaptured MOT':")
for file in files:
    print(file)

# Dataset path inside the .h5 file
dataset_path = 'images/cam1/after ramp/frame'

# Ensure interactive mode is off
plt.ioff()  # Disable interactive mode

# Iterate over all .h5 files in the folder
for filename in os.listdir(recaptured_mot_folder):
    if filename.endswith('.h5'):  # Check if the file is an .h5 file
        file_path = os.path.join(recaptured_mot_folder, filename)

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
                if top >= bottom or left >= right:
                    print(f"Warning: Invalid cropping region in {filename}. Skipping...")
                    continue  # Skip this file if the cropping region is invalid

                # Crop the image
                cropped_image = image_data[top:bottom, left:right]

                # Create a figure with two subplots (one for the original image and one for the cropped one)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

                # Show the original image on the left
                ax[0].imshow(image_data, cmap='gray')
                ax[0].set_title(f"Original Image: {filename}")
                ax[0].axis('off')  # Hide axis labels

                # Create a rectangle around the cropped region
                rect = patches.Rectangle((left, top), right - left, bottom - top,
                                         linewidth=2, edgecolor='r', facecolor='none')  # Red rectangle
                ax[0].add_patch(rect)  # Add the rectangle to the original image

                # Show the cropped image on the right
                ax[1].imshow(cropped_image, cmap='gray')
                ax[1].set_title(f"Cropped Image: {filename}")
                ax[1].axis('off')  # Hide axis labels

                # Display both images
                plt.show()

            except KeyError:
                print(f"Warning: Dataset path '{dataset_path}' not found in {filename}.")
                continue  # Skip this file if the dataset path is incorrect


