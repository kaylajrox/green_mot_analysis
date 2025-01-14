
import h5py
import numpy as np
import os
import cv2

# Specify the directory where your .h5 files are located
folder_path = "S:\\Experiments\\Yb171_MOT_Tweezer_trap\\green_mot_search_camera\\2025\\01\\10\\2s_after_ramp_green_mot"
dataset_path = 'images/cam1/after ramp/frame'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):  # Check if the file is an .h5 file
        file_path = os.path.join(folder_path, filename)

        # Open the .h5 file
        with h5py.File(file_path, 'r') as f:
            # Replace 'dataset_name' with the actual name of your dataset in the h5 file
            image_data = f[dataset_path][:]

            # Define the cropping coordinates (if needed)
            start_row, end_row = 50, 200  # Crop from row 50 to row 200
            start_col, end_col = 30, 150  # Crop from column 30 to column 150

            # Crop the image
            cropped_image = image_data[start_row:end_row, start_col:end_col]

            # Define a threshold value for brightness
            brightness_threshold = 200  # Adjust this value as needed

            # Apply thresholding to find bright pixels
            _, bright_pixels = cv2.threshold(cropped_image, brightness_threshold, 255, cv2.THRESH_BINARY)

            # Count the number of bright pixels
            bright_pixel_count = np.count_nonzero(bright_pixels)

            # Print the result for each .h5 file
            print(f"File: {filename} - Number of bright pixels: {bright_pixel_count}")




# import h5py
# import numpy as np
# import cv2
#
# folder_path = "S:\\Experiments\\Yb171_MOT_Tweezer_trap\\green_mot_search_camera\\2025\\01\\10\\2s_after_ramp_green_mot"
#
# # image dataset path
# dataset_path = 'images/cam1/after ramp/frame'
#
# # Open the h5 file
# with h5py.File(dataset_path, 'r') as f:
#     # Replace 'dataset_name' with the actual name of your dataset
#     image_data = f[dataset_path][:]
#
# # Define the cropping coordinates
# # (start_row, end_row, start_col, end_col)
# start_row, end_row = 50, 200  # Crop from row 50 to row 200
# start_col, end_col = 30, 150  # Crop from column 30 to column 150
#
# # Crop the image (2D or 3D)
# cropped_image = image_data[start_row:end_row, start_col:end_col]
#
# # If it's a color image (3D), you can crop it as well
# # Example: Cropping each channel (height, width, channels)
# if len(cropped_image.shape) == 3:
#     cropped_image = cropped_image[:, :, :]
#
# # Show or process the cropped image
# print("Cropped image shape:", cropped_image.shape)



# # Open the h5 file
# with h5py.File('image_data.h5', 'r') as f:
#     # Replace 'dataset_name' with the actual name of your dataset
#     image_data = f['dataset_name'][:]
#
# # If the image is 3D (e.g., (height, width, channels)), you might need to convert it to grayscale first
# if len(image_data.shape) == 3:  # Color image with 3 channels
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#
# # Define a threshold value for brightness
# brightness_threshold = 200  # Adjust this value as needed
#
# # Apply thresholding to find bright pixels
# _, bright_pixels = cv2.threshold(image_data, brightness_threshold, 255, cv2.THRESH_BINARY)
#
# # Count the number of bright pixels
# bright_pixel_count = np.count_nonzero(bright_pixels)
#
# # Print the result
# print(f"Number of bright pixels: {bright_pixel_count}")
