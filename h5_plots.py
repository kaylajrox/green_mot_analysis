import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the folder path containing the HDF5 files
folder_path = "S:\\Experiments\\Yb171_MOT_Tweezer_trap\\green_mot_search_camera\\2025\\01\\13\\807_65"

# Collect data for plotting
file_data = []  # To store frame data for each file
file_titles = []  # To store titles with B_FINAL and B_INITIAL values

# Iterate through all HDF5 files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".h5"):  # Check for .h5 extension
        file_path = os.path.join(folder_path, file_name)

        with h5py.File(file_path, "r") as h5_file:
            # Extract frame data
            dataset_path = 'images/cam1/after ramp/frame'
            frame_data = None
            if dataset_path in h5_file:
                frame_data = h5_file[dataset_path][:]
                file_data.append(frame_data)

            # Extract B_FINAL and B_INITIAL values from globals
            b_final = None
            b_initial = None
            if 'globals' in h5_file:
                globals_group = h5_file['globals']
                b_final = globals_group.attrs.get('B_FINAL', 'N/A')
                b_initial = globals_group.attrs.get('B_INITIAL', 'N/A')

            # Create title for the file
            title = f"{file_name}\nB_FINAL: {b_final}, B_INITIAL: {b_initial}"
            file_titles.append(title)

# Plot all images in a grid of subplots
num_files = len(file_data)
cols = 3  # Number of columns in the grid
rows = (num_files + cols - 1) // cols  # Calculate rows to fit all files

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Flatten axes for easy indexing
axes = axes.flatten()

for i, (data, title) in enumerate(zip(file_data, file_titles)):
    axes[i].imshow(data, cmap='gray')
    axes[i].set_title(title, fontsize=10)
    axes[i].axis('off')  # Hide axes for a cleaner look

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
