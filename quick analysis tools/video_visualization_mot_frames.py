'''
Video creation from a dataset
'''

import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg as the backend

import h5py
import numpy as np
import os
import cv2  # OpenCV for video creation
import matplotlib.pyplot as plt
import re  # For natural sorting


# USER DEFINE PARAMETERS HERE ONLY

# Define the cropping region
top = 350  # vertical height start (top edge)
bottom = 1050  # vertical height end (bottom edge)
left = 950  # horizontal width start point (left edge)
right = 1650  # horizontal width end point (right edge)

data_day_name = '20250123TOF_withBlueMOTBeams'
experiment_title = 'WithRamp_9V_6V'

#=====================================================================
file_titles = []
frame_list = []  # Store frames for the video
t_wait_list = []  # Track extracted wait times



# Locate the base directory
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'green_mot_analysis'))

# Path to the specific data folder
recaptured_mot_folder = os.path.join(base_directory, 'data', data_day_name, experiment_title)

# Check if the folder exists
if not os.path.exists(recaptured_mot_folder):
    raise FileNotFoundError(f"The folder does not exist: {recaptured_mot_folder}")

# Extract experiment information from folder name
folder_name = os.path.basename(recaptured_mot_folder)  # Extract last folder name

# Match "NoRamp_XV" and format it as "No Ramp XV VCA"
match_long_no_ramp = re.match(r"LongImaging_NoRampOnGreen", folder_name)
match_long_ramp = re.match(r"LongImaging_WithRampOnGreen", folder_name)
match_no_ramp = re.match(r"NoRamp_(\d+)V", folder_name)


# Match and format different folder naming conventions
if re.match(r"LongImaging_NoRampOnGreen", folder_name):
    experiment_label = "No-Ramp Long Imaging"
elif re.match(r"LongImaging_WithRampOnGreen", folder_name):
    experiment_label = "Ramp Long Imaging"
elif match := re.match(r"NoRamp_(\d+)V", folder_name):
    experiment_label = f"No Ramp {match.group(1)}V VCA"
elif match := re.match(r"WithRamp_(\d+)V_(\d+)V", folder_name):
    experiment_label = f"With Ramp {match.group(1)}V-{match.group(2)}V"
else:
    experiment_label = "Unknown Experiment"


# List all .h5 files and sort numerically
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


files = sorted(
    [f for f in os.listdir(recaptured_mot_folder) if f.endswith('.h5')],
    key=natural_sort_key
)

print("HDF5 files in 'recaptured MOT':", files)

# Dataset path inside the .h5 file
dataset_path = 'images/cam1/after ramp/frame'

# Ensure interactive mode is off
plt.ioff()

# Process each .h5 file
for filename in files:
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
                title = f"{experiment_label} Wait Time: {t_wait_ms:.2f} ms"
                t_wait_list.append(t_wait_ms)  # Store extracted wait time
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

            print(f"Processed {filename} with {title}")  # Debugging output

        except KeyError:
            print(f"Warning: Dataset path '{dataset_path}' not found in {filename}.")
            continue

        # Generate Video
if frame_list:
    output_video_path = os.path.join(recaptured_mot_folder, f"{folder_name}.mp4")

    # Fixed frame rate to ensure each frame lasts 1 second
    frame_rate = 30  # FPS
    frames_per_image = frame_rate  # Show each image for 1 second

    # Define video parameters
    frame_height, frame_width, _ = frame_list[0].shape

    # Use a safe codec and avoid MPEG4 standard timebase issue
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' instead of 'mpeg4'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Write frames (repeat each image to last for 1 second)
    for frame in frame_list:
        for _ in range(frames_per_image):  # Duplicate each frame to show it for 1 second
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved: {output_video_path}, Frame Rate: {frame_rate} FPS, Each frame lasts 1s.")
else:
    print("No frames were processed. Video was not created.")
