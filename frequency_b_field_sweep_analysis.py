'''
Processing images in h5 files which are all within the same folder

'''


import h5py
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg as the backend
import matplotlib.pyplot as plt
import numpy as np

# Define the folder path containing the HDF5 files
folder_path = "data/20250113_initial_b_freq_parameter_sweep/807_35"

# Change cropping region of photos
top = 550
bottom = 850
left = 910
right = 1310

fixed_bounds = (top, left, bottom, right)  # Example: (top, left, bottom, right)
cropped_data = []  # To store cropped frame data
# Collect data for plotting
file_data = []  # To store frame data for each file
file_titles = []  # To store titles with B_FINAL and B_INITIAL values

# Iterate through all HDF5 files in the folder
for file_name in os.listdir(folder_path):
    print(f" working with filename: {file_name}")
    if file_name.endswith(".h5"):  # Check for .h5 extension
        file_path = os.path.join(folder_path, file_name)

        '''
        import and crop the image data
        '''
        with h5py.File(file_path, "r") as h5_file:
            print((f"filepath: {file_path}"))
            # Extract frame data
            dataset_path = 'images/cam1/after ramp/frame'
            frame_data = None
            if dataset_path in h5_file:
                frame_data = h5_file[dataset_path][:]

                # Apply fixed bounds for cropping
                top, left, bottom, right = fixed_bounds
                cropped_frame = frame_data[top:bottom, left:right]
                cropped_data.append(cropped_frame)
                file_titles.append(file_name)  # Append the file name to titles

            # Extract B_FINAL and B_INITIAL values from globals
            b_final = None
            b_initial = None
            if 'globals' in h5_file:
                globals_group = h5_file['globals']
                b_final = globals_group.attrs.get('B_FINAL', 'N/A')
                b_initial = globals_group.attrs.get('B_INITIAL', 'N/A')
                green_laser_setpoint = globals_group.attrs.get('GREEN_LASER_SET_POINT', 'N/A')

                # Format to 2 decimal places if numeric
                if isinstance(b_final, (int, float)):
                    b_final = f"{b_final:.2f}"
                if isinstance(b_initial, (int, float)):
                    b_initial = f"{b_initial:.2f}"

            # Create title for the file
            title = f"{file_name}\nB_FINAL: {b_final}, B_INITIAL: {b_initial}"
            file_titles.append(title)

# Plot all images in a grid of subplots
cols = 3  # Number of columns in the grid
num_files = len(cropped_data)  # Ensure num_files is calculated from the actual data
if num_files == 0:
    print("No data available for plotting. Could be h5 file formatting not set up correctly, this python script requires all h5 files to be in the same folder, not separate ones so check the folder structure of your shot files")
    exit()  # Exit or handle as needed

rows = (num_files + cols - 1) // cols  # Calculate rows to fit all files

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Flatten axes for easy indexing
axes = axes.flatten()

# Set the laser setpoint for the general title
if 'green_laser_setpoint' in locals():
    green_laser_setpoint = f"GREEN_LASER_SET_POINT: {green_laser_setpoint}"

for i, (data, title) in enumerate(zip(cropped_data, file_titles)):
    axes[i].imshow(data, cmap='gray')
    axes[i].set_title(title, fontsize=8)
    axes[i].axis('off')  # Hide axes for a cleaner look

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Add a single block title for the entire grid
if 'green_laser_setpoint' in locals():
    fig.suptitle(green_laser_setpoint, fontsize=16)

plt.tight_layout()
plt.show()