'''
Visualize the MOT photos at different wait times and plot the intensity count to figure out the
lifetime of the MOT


'''


import h5py
import os
import matplotlib
import cv2
matplotlib.use('TkAgg')  # Use TkAgg as the backend
import matplotlib.pyplot as plt
import numpy as np

# Define folder paths
primary_data_folder = "data/20250114_release_and_recapture_greenMOT/recaptured MOT"
background_block_main_beams_folder = "data/20250114_release_and_recapture_greenMOT/backgrounds1"
background_block_diagonal_beams_folder = "data/20250114_release_and_recapture_greenMOT/backgrounds2"

# Define the cropping region
top, bottom, left, right = 400, 850, 910, 1350
file_titles = []

# Get sorted lists of files
primary_files = sorted([os.path.join(primary_data_folder, f) for f in os.listdir(primary_data_folder) if f.endswith('.h5')])
bg1_files = sorted([os.path.join(background_block_main_beams_folder, f) for f in os.listdir(background_block_main_beams_folder) if f.endswith('.h5')])
bg2_files = sorted([os.path.join(background_block_diagonal_beams_folder, f) for f in os.listdir(background_block_diagonal_beams_folder) if f.endswith('.h5')])

cropped_images = []

# Iterate through files
for primary_file, bg1_file, bg2_file in zip(primary_files, bg1_files, bg2_files):
    with h5py.File(primary_file, "r") as h5_primary, \
         h5py.File(bg1_file, "r") as h5_bg1, \
         h5py.File(bg2_file, "r") as h5_bg2:

        # Extract frame data
        primary_data = h5_primary['images/cam1/after ramp/frame'][:]
        bg1_data = h5_bg1['images/cam1/after ramp/frame'][:]
        bg2_data = h5_bg2['images/cam1/after ramp/frame'][:]

        # Combine and subtract backgrounds
        combined_bg = cv2.add(bg1_data, bg2_data)
        modified_data = cv2.subtract(primary_data, combined_bg)

        # Crop the image
        cropped_image = modified_data[top:bottom, left:right]
        cropped_images.append(cropped_image)

        # Extract metadata for title
        t_wait = h5_primary['globals'].attrs.get('T_WAIT', 'N/A')
        file_titles.append(f"Wait time {t_wait} s")

# Extract the T_WAIT values and calculate the sum of pixel intensities for each image
t_wait_values = []
pixel_sums = []

for cropped_image, title in zip(cropped_images, file_titles):
    # Extract the T_WAIT value from the title
    t_wait = float(title.split(" ")[2])  # Assuming "Wait time X s" format
    t_wait_values.append(t_wait)

    # Calculate the sum of pixel intensities for the cropped image
    pixel_sum = np.sum(cropped_image)
    pixel_sums.append(pixel_sum)

# Sort data by T_WAIT values for a clean plot
sorted_indices = np.argsort(t_wait_values)
t_wait_values = np.array(t_wait_values)[sorted_indices]
pixel_sums = np.array(pixel_sums)[sorted_indices]

# Plot cropped images
cols = 4
num_files = len(cropped_images)
rows = (num_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, (img, title) in enumerate(zip(cropped_images, file_titles)):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(title, fontsize=8)
    axes[i].axis('off')

# Hide unused subplots
for ax in axes[len(cropped_images):]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# Create a second plot: Sum of pixel values vs. time after background subtraction
plt.figure(figsize=(8, 6))
plt.scatter(t_wait_values, pixel_sums, color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
plt.grid(True)
plt.show()


