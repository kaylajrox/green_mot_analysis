import h5py
import numpy as np
import os
import cv2
import matplotlib
import re

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Specify the main directory where subfolders are located
main_folder_path = "data/20250110_first_data"  # Replace with your directory path
image_dataset_path = 'images/cam1/after ramp/frame'

# List to store the file paths, folder names, and image data
file_info = []

# Change cropping region of photos
top = 550
bottom = 850
left = 910
right = 1310

# Adjustable parameters for text position
text_x = 30  # Horizontal position of the text (from the left)
text_y = 45  # Vertical position of the text (from the bottom)


def parse_folder_name(folder_name):
    # Check for the "background" folder
    if "background" in folder_name.lower():
        return "Background"  # Label for background images

    # Check for the "zeeman" folder and label it as "MOT"
    if "zeeman" in folder_name.lower():
        return "MOT"  # Label for Zeeman slower folders

    # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
    match = re.match(r"(\d+)_?(\d*)s?", folder_name)

    if match:
        # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
        if match.group(2):  # If there's a second part like "1_2"
            fraction = f"{match.group(1)}/{match.group(2)}"
            return f"t={fraction}"  # e.g., "t=1/2"
        else:
            return f"t={match.group(1)}s"  # e.g., "t=2s"

    # If no match and folder doesn't start with valid format, return "Unknown"
    return "Unknown"


# Iterate through the subfolders in the main directory
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Check if it is a folder
    if os.path.isdir(subfolder_path):
        # Parse the folder name for experiment information
        parsed_title = parse_folder_name(subfolder_name)

        # Skip folders with invalid or unknown titles (like "zeeman")
        if parsed_title == "Unknown":
            continue

        # Look for .h5 files in the subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".h5"):  # Check for .h5 files
                file_path = os.path.join(subfolder_path, file_name)

                # Open the .h5 file
                with h5py.File(file_path, "r") as h5_file:
                    # Extract frame data
                    frame_data = None
                    if image_dataset_path in h5_file:
                        frame_data = h5_file[image_dataset_path][:]

                        # Apply fixed bounds for cropping
                        cropped_frame = frame_data[top:bottom, left:right]

                        # Store the cropped image and the folder name
                        file_info.append({
                            "folder_name": subfolder_name,
                            "parsed_title": parsed_title,  # Ensure parsed title is included
                            "file_path": file_path,
                            "cropped_image": cropped_frame  # Add cropped image here
                        })

# Number of cropped images to display
num_images = len(file_info)

# Determine the grid size for subplots
cols = 4  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate the number of rows required

# Create a figure for the subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Flatten axes for easy indexing
axes = axes.flatten()

# Display each cropped image in a subplot
for i, info in enumerate(file_info):
    axes[i].imshow(info['cropped_image'], cmap='gray')

    # Add text to the cropped image at adjustable position
    axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
                 ha='left', va='bottom')
    axes[i].axis('off')  # Hide axes for a cleaner look

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Display the plot
plt.tight_layout()
plt.show()
