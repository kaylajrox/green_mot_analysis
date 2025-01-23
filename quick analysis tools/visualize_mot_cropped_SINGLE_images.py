
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg as the backend

import h5py
import numpy as np
import os
import cv2  # OpenCV for video creation
import matplotlib.pyplot as plt

# USER DEFINE PARAMETERS HERE ONLY
# Define the cropping region
top = 350  # vertical height start (top edge)
bottom = 1050  # vertical height end (bottom edge)
left = 950  # horizontal width start point (left edge)
right = 1650  # horizontal width end point (right edge)

#=========================================================================

file_titles = []
frame_list = []  # Store frames for the video

# Locate the base directory
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'green_mot_analysis'))

# Path to the specific data folder
recaptured_mot_folder = os.path.join(base_directory, 'data', '20250123TOF_withBlueMOTBeams', 'NoRamp_4V')

# Check if the folder exists
if not os.path.exists(recaptured_mot_folder):
    raise FileNotFoundError(f"The folder does not exist: {recaptured_mot_folder}")

# List all .h5 files in the folder
files = [os.path.join(recaptured_mot_folder, f) for f in os.listdir(recaptured_mot_folder) if f.endswith('.h5')]

print("HDF5 files in 'recaptured MOT':")
for file in files:
    print(file)

# Dataset path inside the .h5 file
dataset_path = 'images/cam1/after ramp/frame'

# Ensure interactive mode is off
plt.ioff()

# Iterate over all .h5 files
for filename in sorted(os.listdir(recaptured_mot_folder)):  # Sort to maintain order
    if filename.endswith('.h5'):
        file_path = os.path.join(recaptured_mot_folder, filename)

        # Open the .h5 file
        with h5py.File(file_path, 'r') as f:
            try:
                # Load image data
                image_data = f[dataset_path][:]
                if image_data.size == 0:
                    print(f"Warning: Dataset in {filename} is empty.")
                    continue

                    # Validate cropping region
                if top >= bottom or left >= right:
                    print(f"Warning: Invalid cropping region in {filename}. Skipping...")
                    continue

                    # Crop image
                cropped_image = image_data[top:bottom, left:right]

                # Extract T_WAIT and handle missing values
                t_wait = f['globals'].attrs.get('T_WAIT', None)
                if t_wait is not None:
                    t_wait_ms = t_wait * 1e3  # Convert to ms
                    title = f"T_WAIT: {t_wait_ms:.2f} ms"
                else:
                    title = "T_WAIT: N/A"

                file_titles.append(title)

                # Convert image to uint8 format for video
                normalized_image = ((cropped_image - np.min(cropped_image)) /
                                    (np.max(cropped_image) - np.min(cropped_image)) * 255).astype(np.uint8)

                # Convert grayscale to BGR format (needed for OpenCV)
                frame_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

                # Overlay the title text on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                background_color = (0, 0, 0)  # Black background for contrast

                text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
                text_x = 20
                text_y = 50  # Position near the top-left corner

                # Draw black rectangle for better text visibility
                cv2.rectangle(frame_bgr, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10),
                              background_color, -1)

                # Put the title text on the frame
                cv2.putText(frame_bgr, title, (text_x, text_y), font, font_scale, text_color, font_thickness)

                frame_list.append(frame_bgr)  # Store frames for video

                # Display image with title
                plt.figure(figsize=(6, 6))
                plt.imshow(cropped_image, cmap='gray')
                plt.title(title)
                plt.axis('off')

                plt.show()

            except KeyError:
                print(f"Warning: Dataset path '{dataset_path}' not found in {filename}.")
                continue


