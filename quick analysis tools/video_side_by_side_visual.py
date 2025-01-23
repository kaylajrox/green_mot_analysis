'''
Side by side video comparison
'''



import matplotlib
matplotlib.use('TkAgg')

import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import re

# USER DEFINE PARAMETERS HERE ONLY

# Define the cropping region
top = 350  # vertical height start (top edge)
bottom = 1050  # vertical height end (bottom edge)
left = 950  # horizontal width start point (left edge)
right = 1650  # horizontal width end point (right edge)

origin_data_folder1 = '20250123TOF_withBlueMOTBeams'
file_indata_folder1 = 'WithRamp_9V_2.7V_1ms_step_807.75MHz'
origin_data_folder2 = '20250123TOF_withBlueMOTBeams'
file_indata_folder2 = 'WithRamp_9V_4V_1ms_step_807.75MHz'

#=========================================================================

# Define paths for two different data folders
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'green_mot_analysis'))
folder1 = os.path.join(base_directory, 'data', origin_data_folder1, file_indata_folder1)
folder2 = os.path.join(base_directory, 'data',origin_data_folder2, file_indata_folder2)


def parse_experiment_label(folder_name):
    print(f"Debug: Parsing folder name '{folder_name}'")  # Debugging output

    pattern = r"WithRamp_(\d+)V_([\d\.]+)V_1ms_step_([\d\.]+)MHz"
    match = re.match(pattern, folder_name)

    if match:
        start_voltage = f"{match.group(1)}V_{match.group(2)}V"
        end_frequency = f"{match.group(3)}MHz"
        print(f"Debug: Extracted '{start_voltage}' and '{end_frequency}'")  # Debugging output
        return start_voltage, end_frequency
    else:
        print(f"Warning: Failed to parse folder name '{folder_name}'")  # Debugging output
        return "Unknown Experiment", "Unknown Frequency"
# Function to extract experiment details from folder name
# def parse_experiment_label(folder_name):
#     match = re.match(r"WithRamp_(\d+)V_(\d+\.\d+)V_1ms_step_(\d+\.\d+)MHz", folder_name)
#     if match:
#         start_voltage = f"{match.group(1)}V_{match.group(2)}V"
#         end_frequency = f"{match.group(3)}MHz"
#         return start_voltage, end_frequency
#     else:
#         return "Unknown Experiment", "Unknown Frequency"

# Extract experiment labels
start_voltage1, end_frequency1 = parse_experiment_label(os.path.basename(folder1))
start_voltage2, end_frequency2 = parse_experiment_label(os.path.basename(folder2))

# Log the source folders
print(f"Processing data from:\n- {folder1} ({start_voltage1})\n- {folder2} ({start_voltage2})")

# Ensure both folders have the same ending frequency
if end_frequency1 != end_frequency2:
    raise ValueError(f"Mismatch in ending numbers: {end_frequency1} vs {end_frequency2}")

video_title = f"Comparison_{end_frequency1}.mp4"  # Use the shared ending number

# Function to load and process images from a folder
def load_images_from_folder(folder_path):
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.h5')],
        key=lambda s: [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
    )

    frames = []
    t_waits = []

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        with h5py.File(file_path, 'r') as f:
            try:
                image_data = f['images/cam1/after ramp/frame'][:]
                if image_data.size == 0:
                    continue

                cropped_image = image_data[top:bottom, left:right]
                t_wait = f['globals'].attrs.get('T_WAIT', None)
                print(t_wait)
                t_wait_ms = t_wait * 1e3 if t_wait is not None else None
                t_waits.append(t_wait_ms)

                # Normalize and convert image to uint8
                normalized_image = ((cropped_image - np.min(cropped_image)) /
                                    (np.max(cropped_image) - np.min(cropped_image)) * 255).astype(np.uint8)

                # Convert grayscale to BGR
                frame_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

                frames.append((frame_bgr, t_wait_ms))

            except KeyError:
                continue

    return frames, t_waits

# Load images from both folders
frames1, t_waits1 = load_images_from_folder(folder1)
frames2, t_waits2 = load_images_from_folder(folder2)

# Ensure both lists have the same length
min_length = min(len(frames1), len(frames2))
frames1, frames2 = frames1[:min_length], frames2[:min_length]

# Create side-by-side frames

combined_frames = []
for (frame1, t1), (frame2, t2) in zip(frames1, frames2):
    combined_frame = np.hstack((frame1, frame2))  # Concatenate images side by side

    # Use only one t_wait value (whichever is available)
    t_wait_text = f"Wait time: {t1:.2f} ms" if t1 is not None else "T_WAIT: N/A"

    # Generate the title text
    title_left = f"{start_voltage1}"  # e.g., "9V_2.7V"
    title_right = f"{start_voltage2}"  # e.g., "9V_4V"
    title = f"{t_wait_text} | {title_left}                     {title_right} "

    # Overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]

    text_x = 20
    text_y = 50

    # Draw text background
    cv2.rectangle(combined_frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), background_color, -1)
    # Add text
    cv2.putText(combined_frame, title, (text_x, text_y), font, font_scale, text_color, font_thickness)

# combined_frames = []
# for (frame1, t1), (frame2, t2) in zip(frames1, frames2):
#     combined_frame = np.hstack((frame1, frame2))  # Concatenate images side by side
#
#     # Generate title text
#     t_wait_text = f"T_WAIT: {t1:.2f} ms | {t2:.2f} ms" if t1 is not None and t2 is not None else "T_WAIT: N/A"
#     title = f"{start_voltage1} vs {start_voltage2} | {t_wait_text}"
#
#     # Overlay text
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (255, 255, 255)
#     background_color = (0, 0, 0)
#     text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
#
#     text_x = 20
#     text_y = 50
#
#     # Draw text background
#     cv2.rectangle(combined_frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), background_color, -1)
#     # Add text
#     cv2.putText(combined_frame, title, (text_x, text_y), font, font_scale, text_color, font_thickness)

    combined_frames.append(combined_frame)

# Create the side-by-side comparison video
if combined_frames:
    output_video_path = os.path.join(base_directory, video_title)

    frame_rate = 30
    frames_per_image = frame_rate  # Ensure each frame lasts 1 second
    frame_height, frame_width, _ = combined_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    for frame in combined_frames:
        for _ in range(frames_per_image):
            video_writer.write(frame)

    video_writer.release()
    print(f"Comparison video saved: {output_video_path}")
else:
    print("No frames were processed. Video was not created.")



# import matplotlib
# matplotlib.use('TkAgg')
#
# import h5py
# import numpy as np
# import os
# import cv2
# import matplotlib.pyplot as plt
# import re
#
# # Define the cropping region
# top, bottom = 350, 1050
# left, right = 950, 1650
#
# # Define paths for two different data folders
# base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'green_mot_analysis'))
# folder1 = os.path.join(base_directory, 'data', '20250123TOF_withBlueMOTBeams', 'WithRamp_9V_2.7V_1ms_step_807.65MHz')
# folder2 = os.path.join(base_directory, 'data', '20250123TOF_withBlueMOTBeams', 'WithRamp_9V_4V_1ms_step_807.65MHz')
#
# # Function to extract experiment details from folder name
# def parse_experiment_label(folder_name):
#     match = re.match(r"WithRamp_(\d+)V_(\d+\.\d+)V_1ms_step_(\d+\.\d+)MHz", folder_name)
#     if match:
#         start_voltage = f"{match.group(1)}V_{match.group(2)}V"
#         end_frequency = f"{match.group(3)}MHz"
#         return start_voltage, end_frequency
#     else:
#         return "Unknown Experiment", "Unknown Frequency"
#
# # Extract experiment labels
# start_voltage1, end_frequency1 = parse_experiment_label(os.path.basename(folder1))
# start_voltage2, end_frequency2 = parse_experiment_label(os.path.basename(folder2))
#
# if end_frequency1 != end_frequency2:
#     raise ValueError(f"Mismatch in ending numbers: {end_frequency1} vs {end_frequency2}")
#
# # Function to load and process images from a folder
# def load_images_from_folder(folder_path):
#     files = sorted(
#         [f for f in os.listdir(folder_path) if f.endswith('.h5')],
#         key=lambda s: [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
#     )
#
#     frames = []
#     t_waits = []
#
#     for filename in files:
#         file_path = os.path.join(folder_path, filename)
#
#         with h5py.File(file_path, 'r') as f:
#             try:
#                 image_data = f['images/cam1/after ramp/frame'][:]
#                 if image_data.size == 0:
#                     continue
#
#                 cropped_image = image_data[top:bottom, left:right]
#                 t_wait = f['globals'].attrs.get('T_WAIT', None)
#                 t_wait_ms = t_wait * 1e3 if t_wait is not None else None
#                 t_waits.append(t_wait_ms)
#
#                 # Normalize and convert image to uint8
#                 normalized_image = ((cropped_image - np.min(cropped_image)) /
#                                     (np.max(cropped_image) - np.min(cropped_image)) * 255).astype(np.uint8)
#
#                 # Convert grayscale to BGR
#                 frame_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
#
#                 frames.append((frame_bgr, t_wait_ms))
#
#             except KeyError:
#                 continue
#
#     return frames, t_waits
#
# # Load images from both folders
# frames1, t_waits1 = load_images_from_folder(folder1)
# frames2, t_waits2 = load_images_from_folder(folder2)
#
# # Ensure both lists have the same length
# min_length = min(len(frames1), len(frames2))
# frames1, frames2 = frames1[:min_length], frames2[:min_length]
#
# # Create side-by-side frames
# combined_frames = []
# for (frame1, t1), (frame2, t2) in zip(frames1, frames2):
#     combined_frame = np.hstack((frame1, frame2))  # Concatenate images side by side
#
#     # Generate title text
#     t_wait_text = f"T_WAIT: {t1:.2f} ms | {t2:.2f} ms" if t1 is not None and t2 is not None else "T_WAIT: N/A"
#     title = f"{start_voltage1} vs {experiment_label2} | {t_wait_text}"
#
#     # Overlay text
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (255, 255, 255)
#     background_color = (0, 0, 0)
#     text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
#
#     text_x = 20
#     text_y = 50
#
#     # Draw text background
#     cv2.rectangle(combined_frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), background_color, -1)
#     # Add text
#     cv2.putText(combined_frame, title, (text_x, text_y), font, font_scale, text_color, font_thickness)
#
#     combined_frames.append(combined_frame)
#
# # Create the side-by-side comparison video
# if combined_frames:
#     output_video_path = os.path.join(base_directory, "comparison_video.mp4")
#
#     frame_rate = 30
#     frames_per_image = frame_rate  # Ensure each frame lasts 1 second
#     frame_height, frame_width, _ = combined_frames[0].shape
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
#
#     for frame in combined_frames:
#         for _ in range(frames_per_image):
#             video_writer.write(frame)
#
#     video_writer.release()
#     print(f"Comparison video saved: {output_video_path}")
# else:
#     print("No frames were processed. Video was not created.")
