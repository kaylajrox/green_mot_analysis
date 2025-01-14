import h5py
import numpy as np
import os
import cv2
import matplotlib
import re

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Specify the main directory where subfolders are located
main_folder_path = "data/20250117_first_data"  # Replace with your directory path
image_dataset_path = 'images/cam1/after ramp/frame'

# List to store the file paths, folder names, and image data
file_info = []
background_images = []  # List to store background images

# Change cropping region of photos
top = 550
bottom = 850
left = 910
right = 1310

text_x = 10  # Horizontal position of the text (from the left)
text_y = 20  # Vertical position of the text (from the bottom)


def parse_folder_name(folder_name):
    # Check for the "background" folder
    if "background" in folder_name.lower():
        return "Background"  # Label for background images

    # Skip Zeeman slower folders
    if "zeeman" in folder_name.lower():
        return None  # Skip Zeeman slower images by returning None

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
    return None  # No valid label


# Iterate through the subfolders in the main directory to identify background images
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Check if it is a folder
    if os.path.isdir(subfolder_path):
        # Parse the folder name for experiment information
        parsed_title = parse_folder_name(subfolder_name)

        # If it's a background folder, append the background images to the list
        if parsed_title == "Background":
            print(f"Processing background folder: {subfolder_name}")
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".h5"):  # Check for .h5 files
                    file_path = os.path.join(subfolder_path, file_name)

                    # Open the .h5 file
                    with h5py.File(file_path, "r") as h5_file:
                        # Extract frame data for background
                        frame_data = None
                        if image_dataset_path in h5_file:
                            frame_data = h5_file[image_dataset_path][:]

                            # Append background images to list (ensure it's two background images)
                            background_images.append(frame_data)
                            print(f"  Background image added from {file_name}")

# Debugging output to ensure two background images are appended
print(f"Number of background images found: {len(background_images)}")

# Check if there are background images (background1 and background2)
if len(background_images) != 2:
    print("Error: Less than two background images found!")
    exit()

# Now, process all other folders (not background folders)
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Check if it is a folder
    if os.path.isdir(subfolder_path):
        # Parse the folder name for experiment information
        parsed_title = parse_folder_name(subfolder_name)

        # Skip background folders
        if parsed_title == "Background" or parsed_title is None:
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

                        # Apply background subtraction before cropping
                        print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")

                        # Subtract the first background image (background1)
                        background1_subtracted = cv2.subtract(frame_data, background_images[0])
                        print(f"  Background1 subtraction done for {file_name}")

                        # Subtract the second background image (background2)
                        background2_subtracted = cv2.subtract(frame_data, background_images[1])
                        print(f"  Background2 subtraction done for {file_name}")

                        # Combine the results of both subtractions (add the subtracted images)
                        final_subtracted = cv2.add(background1_subtracted, background2_subtracted)
                        print(f"  Combined background subtraction done for {file_name}")

                        # Crop the final background-subtracted image (after both background subtractions)
                        cropped_frame = final_subtracted[top:bottom, left:right]

                        # Extract numeric value from parsed title
                        if parsed_title:
                            numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
                            if numeric_value:
                                if numeric_value.group(2):  # Fraction format like "1/2"
                                    numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
                                else:
                                    numeric_value = float(numeric_value.group(1))
                            else:
                                numeric_value = float('inf')  # If there's no numeric value, assign a large number
                        else:
                            numeric_value = float('inf')  # If parsed_title is None, assign a large number

                        # Store the cropped image information
                        file_info.append({
                            "folder_name": subfolder_name,
                            "parsed_title": parsed_title,  # Ensure parsed title is included
                            "file_path": file_path,
                            "cropped_image": cropped_frame,  # Add cropped image here
                            "numeric_value": numeric_value  # Add numeric value for sorting
                        })

# Calculate the sum of pixel values for each image and subtract both background images before calculation
intensity_info = []
pixel_sums = []
times = []

for info in file_info:
    # Convert cropped image to uint8 if necessary
    cropped_image = info['cropped_image'].astype(np.uint8)

    # Sum the pixel values for the subtracted image
    pixel_sum = np.sum(cropped_image)

    # Store the sum and numeric value for plotting later
    pixel_sums.append(pixel_sum)
    times.append(info['numeric_value'])

    # Store the intensity sum information along with the numeric value for sorting
    intensity_info.append({
        "folder_name": info['folder_name'],
        "parsed_title": info['parsed_title'],
        "file_path": info['file_path'],
        "pixel_sum": pixel_sum,
        "numeric_value": info['numeric_value']
    })

# Sort the data based on the numeric value (from lowest to highest)
file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])

# Number of cropped images to display
num_images = len(file_info_sorted)

# Determine the grid size for subplots
cols = 4  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate the number of rows required

# Create a figure for the subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Flatten axes for easy indexing
axes = axes.flatten()

# Display each cropped image in a subplot
for i, info in enumerate(file_info_sorted):
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

# Create a second plot: Sum of pixel values vs. time after background subtraction
plt.figure(figsize=(8, 6))
plt.scatter(times, pixel_sums, color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
plt.grid(True)
plt.show()

# import h5py
# import numpy as np
# import os
# import cv2
# import matplotlib
# import re
#
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# # Specify the main directory where subfolders are located
# main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# image_dataset_path = 'images/cam1/after ramp/frame'
#
# # List to store the file paths, folder names, and image data
# file_info = []
# background_images = []  # List to store background images
#
# # Change cropping region of photos
# top = 550
# bottom = 850
# left = 910
# right = 1310
#
# text_x = 10  # Horizontal position of the text (from the left)
# text_y = 20  # Vertical position of the text (from the bottom)
#
#
# def parse_folder_name(folder_name):
#     # Check for the "background" folder
#     if "background" in folder_name.lower():
#         return "Background"  # Label for background images
#
#     # Skip Zeeman slower folders
#     if "zeeman" in folder_name.lower():
#         return None  # Skip Zeeman slower images by returning None
#
#     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
#     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
#
#     if match:
#         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
#         if match.group(2):  # If there's a second part like "1_2"
#             fraction = f"{match.group(1)}/{match.group(2)}"
#             return f"t={fraction}"  # e.g., "t=1/2"
#         else:
#             return f"t={match.group(1)}s"  # e.g., "t=2s"
#
#     # If no match and folder doesn't start with valid format, return "Unknown"
#     return None  # No valid label
#
#
# # Iterate through the subfolders in the main directory
# for subfolder_name in os.listdir(main_folder_path):
#     subfolder_path = os.path.join(main_folder_path, subfolder_name)
#
#     # Check if it is a folder
#     if os.path.isdir(subfolder_path):
#         # Parse the folder name for experiment information
#         parsed_title = parse_folder_name(subfolder_name)
#
#         # Skip folders with invalid or unknown titles (like "zeeman")
#         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
#             continue
#
#         # If it's a background folder, append the background images to the list
#         if parsed_title == "Background":
#             for file_name in os.listdir(subfolder_path):
#                 if file_name.endswith(".h5"):  # Check for .h5 files
#                     file_path = os.path.join(subfolder_path, file_name)
#
#                     # Open the .h5 file
#                     with h5py.File(file_path, "r") as h5_file:
#                         # Extract frame data for background
#                         frame_data = None
#                         if image_dataset_path in h5_file:
#                             frame_data = h5_file[image_dataset_path][:]
#
#                             # Append background images to list (ensure it's two background images)
#                             background_images.append(frame_data)
#                             print(f"  Background image added from {file_name}")
#
#         # Look for .h5 files in the other subfolders (not backgrounds)
#         for file_name in os.listdir(subfolder_path):
#             if file_name.endswith(".h5"):  # Check for .h5 files
#                 file_path = os.path.join(subfolder_path, file_name)
#
#                 # Open the .h5 file
#                 with h5py.File(file_path, "r") as h5_file:
#                     # Extract frame data
#                     frame_data = None
#                     if image_dataset_path in h5_file:
#                         frame_data = h5_file[image_dataset_path][:]
#
#                         # Apply background subtraction before cropping
#                         if len(background_images) == 2:  # Ensure we have two backgrounds
#                             # Print the status of background images
#                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
#
#                             # Subtract the first background image (background1)
#                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
#                             print(f"  Background1 subtraction done for {file_name}")
#
#                             # Subtract the second background image (background2)
#                             background2_subtracted = cv2.subtract(frame_data, background_images[1])
#                             print(f"  Background2 subtraction done for {file_name}")
#
#                             # Combine the results of both subtractions (add the subtracted images)
#                             final_subtracted = cv2.add(background1_subtracted, background2_subtracted)
#                             print(f"  Combined background subtraction done for {file_name}")
#
#                             # Crop the final background-subtracted image (after both background subtractions)
#                             cropped_frame = final_subtracted[top:bottom, left:right]
#
#                             # Extract numeric value from parsed title
#                             if parsed_title:
#                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
#                                 if numeric_value:
#                                     if numeric_value.group(2):  # Fraction format like "1/2"
#                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
#                                     else:
#                                         numeric_value = float(numeric_value.group(1))
#                                 else:
#                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
#                             else:
#                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
#
#                             # Store the cropped image information
#                             file_info.append({
#                                 "folder_name": subfolder_name,
#                                 "parsed_title": parsed_title,  # Ensure parsed title is included
#                                 "file_path": file_path,
#                                 "cropped_image": cropped_frame,  # Add cropped image here
#                                 "numeric_value": numeric_value  # Add numeric value for sorting
#                             })
#
# # Check if there are background images (background1 and background2)
# if len(background_images) != 2:
#     print("Error: Less than two background images found!")
#     exit()
#
# # Calculate the sum of pixel values for each image and subtract both background images before calculation
# intensity_info = []
# pixel_sums = []
# times = []
#
# for info in file_info:
#     # Convert cropped image to uint8 if necessary
#     cropped_image = info['cropped_image'].astype(np.uint8)
#
#     # Sum the pixel values for the subtracted image
#     pixel_sum = np.sum(cropped_image)
#
#     # Store the sum and numeric value for plotting later
#     pixel_sums.append(pixel_sum)
#     times.append(info['numeric_value'])
#
#     # Store the intensity sum information along with the numeric value for sorting
#     intensity_info.append({
#         "folder_name": info['folder_name'],
#         "parsed_title": info['parsed_title'],
#         "file_path": info['file_path'],
#         "pixel_sum": pixel_sum,
#         "numeric_value": info['numeric_value']
#     })
#
# # Sort the data based on the numeric value (from lowest to highest)
# file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
#
# # Number of cropped images to display
# num_images = len(file_info_sorted)
#
# # Determine the grid size for subplots
# cols = 4  # Number of columns in the grid
# rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
#
# # Create a figure for the subplots
# fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
#
# # Flatten axes for easy indexing
# axes = axes.flatten()
#
# # Display each cropped image in a subplot
# for i, info in enumerate(file_info_sorted):
#     axes[i].imshow(info['cropped_image'], cmap='gray')
#
#     # Add text to the cropped image at adjustable position
#     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
#                  ha='left', va='bottom')
#     axes[i].axis('off')  # Hide axes for a cleaner look
#
# # Hide unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')
#
# # Display the plot
# plt.tight_layout()
# plt.show()
#
# # Create a second plot: Sum of pixel values vs. time after background subtraction
# plt.figure(figsize=(8, 6))
# plt.scatter(times, pixel_sums, color='blue')
# plt.xlabel('Time (t)')
# plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# plt.grid(True)
# plt.show()
#
# # import h5py
# # import numpy as np
# # import os
# # import cv2
# # import matplotlib
# # import re
# #
# # matplotlib.use('TkAgg')
# # import matplotlib.pyplot as plt
# #
# # # Specify the main directory where subfolders are located
# # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # image_dataset_path = 'images/cam1/after ramp/frame'
# #
# # # List to store the file paths, folder names, and image data
# # file_info = []
# # background_images = []  # List to store background images
# #
# # # Change cropping region of photos
# # top = 550
# # bottom = 850
# # left = 910
# # right = 1310
# #
# # text_x = 10  # Horizontal position of the text (from the left)
# # text_y = 20  # Vertical position of the text (from the bottom)
# #
# #
# # def parse_folder_name(folder_name):
# #     # Check for the "background" folder
# #     if "background" in folder_name.lower():
# #         return "Background"  # Label for background images
# #
# #     # Skip Zeeman slower folders
# #     if "zeeman" in folder_name.lower():
# #         return None  # Skip Zeeman slower images by returning None
# #
# #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# #
# #     if match:
# #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# #         if match.group(2):  # If there's a second part like "1_2"
# #             fraction = f"{match.group(1)}/{match.group(2)}"
# #             return f"t={fraction}"  # e.g., "t=1/2"
# #         else:
# #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# #
# #     # If no match and folder doesn't start with valid format, return "Unknown"
# #     return None  # No valid label
# #
# #
# # # Iterate through the subfolders in the main directory
# # for subfolder_name in os.listdir(main_folder_path):
# #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# #
# #     # Check if it is a folder
# #     if os.path.isdir(subfolder_path):
# #         # Parse the folder name for experiment information
# #         parsed_title = parse_folder_name(subfolder_name)
# #
# #         # Skip folders with invalid or unknown titles (like "zeeman")
# #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# #             continue
# #
# #         # If it's a background folder, append the background images to the list
# #         if parsed_title == "Background":
# #             for file_name in os.listdir(subfolder_path):
# #                 if file_name.endswith(".h5"):  # Check for .h5 files
# #                     file_path = os.path.join(subfolder_path, file_name)
# #
# #                     # Open the .h5 file
# #                     with h5py.File(file_path, "r") as h5_file:
# #                         # Extract frame data for background
# #                         frame_data = None
# #                         if image_dataset_path in h5_file:
# #                             frame_data = h5_file[image_dataset_path][:]
# #
# #                             # Append background images to list (ensure it's two background images)
# #                             background_images.append(frame_data)
# #                             print(f"  Background image added from {file_name}")
# #
# #         # Look for .h5 files in the other subfolders (not backgrounds)
# #         for file_name in os.listdir(subfolder_path):
# #             if file_name.endswith(".h5"):  # Check for .h5 files
# #                 file_path = os.path.join(subfolder_path, file_name)
# #
# #                 # Open the .h5 file
# #                 with h5py.File(file_path, "r") as h5_file:
# #                     # Extract frame data
# #                     frame_data = None
# #                     if image_dataset_path in h5_file:
# #                         frame_data = h5_file[image_dataset_path][:]
# #
# #                         # Apply background subtraction before cropping
# #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# #                             # Print the status of background images
# #                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
# #
# #                             # Subtract the first background image (background1)
# #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# #                             print(f"  Background1 subtraction done for {file_name}")
# #
# #                             # Subtract the second background image (background2)
# #                             background2_subtracted = cv2.subtract(frame_data, background_images[1])
# #                             print(f"  Background2 subtraction done for {file_name}")
# #
# #                             # Combine the results of both subtractions (add the subtracted images)
# #                             final_subtracted = cv2.add(background1_subtracted, background2_subtracted)
# #                             print(f"  Combined background subtraction done for {file_name}")
# #
# #                             # Crop the final background-subtracted image (after both background subtractions)
# #                             cropped_frame = final_subtracted[top:bottom, left:right]
# #
# #                             # Extract numeric value from parsed title
# #                             if parsed_title:
# #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# #                                 if numeric_value:
# #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# #                                     else:
# #                                         numeric_value = float(numeric_value.group(1))
# #                                 else:
# #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# #                             else:
# #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# #
# #                             # Store the cropped image information
# #                             file_info.append({
# #                                 "folder_name": subfolder_name,
# #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# #                                 "file_path": file_path,
# #                                 "cropped_image": cropped_frame,  # Add cropped image here
# #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# #                             })
# #
# # # Check if there are background images (background1 and background2)
# # if len(background_images) != 2:
# #     print("Error: Less than two background images found!")
# #     exit()
# #
# # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # intensity_info = []
# # pixel_sums = []
# # times = []
# #
# # for info in file_info:
# #     # Convert cropped image to uint8 if necessary
# #     cropped_image = info['cropped_image'].astype(np.uint8)
# #
# #     # Sum the pixel values for the subtracted image
# #     pixel_sum = np.sum(cropped_image)
# #
# #     # Store the sum and numeric value for plotting later
# #     pixel_sums.append(pixel_sum)
# #     times.append(info['numeric_value'])
# #
# #     # Store the intensity sum information along with the numeric value for sorting
# #     intensity_info.append({
# #         "folder_name": info['folder_name'],
# #         "parsed_title": info['parsed_title'],
# #         "file_path": info['file_path'],
# #         "pixel_sum": pixel_sum,
# #         "numeric_value": info['numeric_value']
# #     })
# #
# # # Sort the data based on the numeric value (from lowest to highest)
# # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# #
# # # Number of cropped images to display
# # num_images = len(file_info_sorted)
# #
# # # Determine the grid size for subplots
# # cols = 4  # Number of columns in the grid
# # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# #
# # # Create a figure for the subplots
# # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# #
# # # Flatten axes for easy indexing
# # axes = axes.flatten()
# #
# # # Display each cropped image in a subplot
# # for i, info in enumerate(file_info_sorted):
# #     axes[i].imshow(info['cropped_image'], cmap='gray')
# #
# #     # Add text to the cropped image at adjustable position
# #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# #                  ha='left', va='bottom')
# #     axes[i].axis('off')  # Hide axes for a cleaner look
# #
# # # Hide unused subplots
# # for j in range(i + 1, len(axes)):
# #     axes[j].axis('off')
# #
# # # Display the plot
# # plt.tight_layout()
# # plt.show()
# #
# # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # plt.figure(figsize=(8, 6))
# # plt.scatter(times, pixel_sums, color='blue')
# # plt.xlabel('Time (t)')
# # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # plt.grid(True)
# # plt.show()
# #
# # # import h5py
# # # import numpy as np
# # # import os
# # # import cv2
# # # import matplotlib
# # # import re
# # # matplotlib.use('TkAgg')
# # # import matplotlib.pyplot as plt
# # #
# # # # Specify the main directory where subfolders are located
# # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # image_dataset_path = 'images/cam1/after ramp/frame'
# # #
# # # # List to store the file paths, folder names, and image data
# # # file_info = []
# # # background_images = []  # List to store background images
# # #
# # # # Change cropping region of photos
# # # top = 550
# # # bottom = 850
# # # left = 910
# # # right = 1310
# # #
# # # text_x = 10  # Horizontal position of the text (from the left)
# # # text_y = 20  # Vertical position of the text (from the bottom)
# # #
# # # def parse_folder_name(folder_name):
# # #     # Check for the "background" folder
# # #     if "background" in folder_name.lower():
# # #         return "Background"  # Label for background images
# # #
# # #     # Skip Zeeman slower folders
# # #     if "zeeman" in folder_name.lower():
# # #         return None  # Skip Zeeman slower images by returning None
# # #
# # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # #
# # #     if match:
# # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # #         if match.group(2):  # If there's a second part like "1_2"
# # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # #             return f"t={fraction}"  # e.g., "t=1/2"
# # #         else:
# # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # #
# # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # #     return None  # No valid label
# # #
# # #
# # # # Iterate through the subfolders in the main directory
# # # for subfolder_name in os.listdir(main_folder_path):
# # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # #
# # #     # Check if it is a folder
# # #     if os.path.isdir(subfolder_path):
# # #         # Parse the folder name for experiment information
# # #         parsed_title = parse_folder_name(subfolder_name)
# # #
# # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # #             continue
# # #
# # #         # Look for .h5 files in the subfolder
# # #         for file_name in os.listdir(subfolder_path):
# # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # #                 file_path = os.path.join(subfolder_path, file_name)
# # #
# # #                 # Open the .h5 file
# # #                 with h5py.File(file_path, "r") as h5_file:
# # #                     # Extract frame data
# # #                     frame_data = None
# # #                     if image_dataset_path in h5_file:
# # #                         frame_data = h5_file[image_dataset_path][:]
# # #
# # #                         # Apply background subtraction before cropping
# # #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# # #                             # Print the status of background images
# # #                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
# # #
# # #                             # Subtract the first background image (background1)
# # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # #                             print(f"  Background1 subtraction done for {file_name}")
# # #
# # #                             # Subtract the second background image (background2)
# # #                             background2_subtracted = cv2.subtract(frame_data, background_images[1])
# # #                             print(f"  Background2 subtraction done for {file_name}")
# # #
# # #                             # Combine the results of both subtractions (add the subtracted images)
# # #                             final_subtracted = cv2.add(background1_subtracted, background2_subtracted)
# # #                             print(f"  Combined background subtraction done for {file_name}")
# # #
# # #                             # Crop the final background-subtracted image (after both background subtractions)
# # #                             cropped_frame = final_subtracted[top:bottom, left:right]
# # #
# # #                             # Extract numeric value from parsed title
# # #                             if parsed_title:
# # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # #                                 if numeric_value:
# # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # #                                     else:
# # #                                         numeric_value = float(numeric_value.group(1))
# # #                                 else:
# # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # #                             else:
# # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # #
# # #                             # Store the cropped image information
# # #                             file_info.append({
# # #                                 "folder_name": subfolder_name,
# # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # #                                 "file_path": file_path,
# # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # #                             })
# # #
# # #
# # # # Check if there are background images (background1 and background2)
# # # if len(background_images) != 2:
# # #     print("Error: Less than two background images found!")
# # #     exit()
# # #
# # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # intensity_info = []
# # # pixel_sums = []
# # # times = []
# # #
# # # for info in file_info:
# # #     # Convert cropped image to uint8 if necessary
# # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # #
# # #     # Sum the pixel values for the subtracted image
# # #     pixel_sum = np.sum(cropped_image)
# # #
# # #     # Store the sum and numeric value for plotting later
# # #     pixel_sums.append(pixel_sum)
# # #     times.append(info['numeric_value'])
# # #
# # #     # Store the intensity sum information along with the numeric value for sorting
# # #     intensity_info.append({
# # #         "folder_name": info['folder_name'],
# # #         "parsed_title": info['parsed_title'],
# # #         "file_path": info['file_path'],
# # #         "pixel_sum": pixel_sum,
# # #         "numeric_value": info['numeric_value']
# # #     })
# # #
# # # # Sort the data based on the numeric value (from lowest to highest)
# # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # #
# # # # Number of cropped images to display
# # # num_images = len(file_info_sorted)
# # #
# # # # Determine the grid size for subplots
# # # cols = 4  # Number of columns in the grid
# # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # #
# # # # Create a figure for the subplots
# # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # #
# # # # Flatten axes for easy indexing
# # # axes = axes.flatten()
# # #
# # # # Display each cropped image in a subplot
# # # for i, info in enumerate(file_info_sorted):
# # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # #
# # #     # Add text to the cropped image at adjustable position
# # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # #                  ha='left', va='bottom')
# # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # #
# # # # Hide unused subplots
# # # for j in range(i + 1, len(axes)):
# # #     axes[j].axis('off')
# # #
# # # # Display the plot
# # # plt.tight_layout()
# # # plt.show()
# # #
# # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # plt.figure(figsize=(8, 6))
# # # plt.scatter(times, pixel_sums, color='blue')
# # # plt.xlabel('Time (t)')
# # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # plt.grid(True)
# # # plt.show()
# # #
# # #
# # #
# # #
# # # # import h5py
# # # # import numpy as np
# # # # import os
# # # # import cv2
# # # # import matplotlib
# # # # import re
# # # # matplotlib.use('TkAgg')
# # # # import matplotlib.pyplot as plt
# # # #
# # # # # Specify the main directory where subfolders are located
# # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # #
# # # # # List to store the file paths, folder names, and image data
# # # # file_info = []
# # # # background_images = []  # List to store background images
# # # #
# # # # # Change cropping region of photos
# # # # top = 550
# # # # bottom = 850
# # # # left = 910
# # # # right = 1310
# # # #
# # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # #
# # # # def parse_folder_name(folder_name):
# # # #     # Check for the "background" folder
# # # #     if "background" in folder_name.lower():
# # # #         return "Background"  # Label for background images
# # # #
# # # #     # Skip Zeeman slower folders
# # # #     if "zeeman" in folder_name.lower():
# # # #         return None  # Skip Zeeman slower images by returning None
# # # #
# # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # #
# # # #     if match:
# # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # #         if match.group(2):  # If there's a second part like "1_2"
# # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # #         else:
# # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # #
# # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # #     return None  # No valid label
# # # #
# # # #
# # # # # Iterate through the subfolders in the main directory
# # # # for subfolder_name in os.listdir(main_folder_path):
# # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # #
# # # #     # Check if it is a folder
# # # #     if os.path.isdir(subfolder_path):
# # # #         # Parse the folder name for experiment information
# # # #         parsed_title = parse_folder_name(subfolder_name)
# # # #
# # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # #             continue
# # # #
# # # #         # Look for .h5 files in the subfolder
# # # #         for file_name in os.listdir(subfolder_path):
# # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # #
# # # #                 # Open the .h5 file
# # # #                 with h5py.File(file_path, "r") as h5_file:
# # # #                     # Extract frame data
# # # #                     frame_data = None
# # # #                     if image_dataset_path in h5_file:
# # # #                         frame_data = h5_file[image_dataset_path][:]
# # # #
# # # #                         # Apply background subtraction before cropping
# # # #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# # # #                             # Print the status of background images
# # # #                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
# # # #
# # # #                             # Subtract the first background image (background1)
# # # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # # #                             print(f"  Background1 subtraction done for {file_name}")
# # # #
# # # #                             # Subtract the second background image (background2)
# # # #                             final_subtracted = cv2.subtract(background1_subtracted, background_images[1])
# # # #                             print(f"  Background2 subtraction done for {file_name}")
# # # #
# # # #                             # Crop the final background-subtracted image (after both background subtractions)
# # # #                             cropped_frame = final_subtracted[top:bottom, left:right]
# # # #
# # # #                             # Extract numeric value from parsed title
# # # #                             if parsed_title:
# # # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # #                                 if numeric_value:
# # # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # #                                     else:
# # # #                                         numeric_value = float(numeric_value.group(1))
# # # #                                 else:
# # # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # #                             else:
# # # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # #
# # # #                             # Store the cropped image information
# # # #                             file_info.append({
# # # #                                 "folder_name": subfolder_name,
# # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # #                                 "file_path": file_path,
# # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # #                             })
# # # #
# # # #
# # # # # Check if there are background images (background1 and background2)
# # # # if len(background_images) != 2:
# # # #     print("Error: Less than two background images found!")
# # # #     exit()
# # # #
# # # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # # intensity_info = []
# # # # pixel_sums = []
# # # # times = []
# # # #
# # # # for info in file_info:
# # # #     # Convert cropped image to uint8 if necessary
# # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # #
# # # #     # Sum the pixel values for the subtracted image
# # # #     pixel_sum = np.sum(cropped_image)
# # # #
# # # #     # Store the sum and numeric value for plotting later
# # # #     pixel_sums.append(pixel_sum)
# # # #     times.append(info['numeric_value'])
# # # #
# # # #     # Store the intensity sum information along with the numeric value for sorting
# # # #     intensity_info.append({
# # # #         "folder_name": info['folder_name'],
# # # #         "parsed_title": info['parsed_title'],
# # # #         "file_path": info['file_path'],
# # # #         "pixel_sum": pixel_sum,
# # # #         "numeric_value": info['numeric_value']
# # # #     })
# # # #
# # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # #
# # # # # Number of cropped images to display
# # # # num_images = len(file_info_sorted)
# # # #
# # # # # Determine the grid size for subplots
# # # # cols = 4  # Number of columns in the grid
# # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # #
# # # # # Create a figure for the subplots
# # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # #
# # # # # Flatten axes for easy indexing
# # # # axes = axes.flatten()
# # # #
# # # # # Display each cropped image in a subplot
# # # # for i, info in enumerate(file_info_sorted):
# # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # #
# # # #     # Add text to the cropped image at adjustable position
# # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # #                  ha='left', va='bottom')
# # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # #
# # # # # Hide unused subplots
# # # # for j in range(i + 1, len(axes)):
# # # #     axes[j].axis('off')
# # # #
# # # # # Display the plot
# # # # plt.tight_layout()
# # # # plt.show()
# # # #
# # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # plt.figure(figsize=(8, 6))
# # # # plt.scatter(times, pixel_sums, color='blue')
# # # # plt.xlabel('Time (t)')
# # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # # plt.grid(True)
# # # # plt.show()
# # # #
# # # #
# # # #
# # # #
# # # #
# # # # # import h5py
# # # # # import numpy as np
# # # # # import os
# # # # # import cv2
# # # # # import matplotlib
# # # # # import re
# # # # # matplotlib.use('TkAgg')
# # # # # import matplotlib.pyplot as plt
# # # # #
# # # # # # Specify the main directory where subfolders are located
# # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # #
# # # # # # List to store the file paths, folder names, and image data
# # # # # file_info = []
# # # # # background_images = []  # List to store background images
# # # # #
# # # # # # Change cropping region of photos
# # # # # top = 550
# # # # # bottom = 850
# # # # # left = 910
# # # # # right = 1310
# # # # #
# # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # #
# # # # # def parse_folder_name(folder_name):
# # # # #     # Check for the "background" folder
# # # # #     if "background" in folder_name.lower():
# # # # #         return "Background"  # Label for background images
# # # # #
# # # # #     # Skip Zeeman slower folders
# # # # #     if "zeeman" in folder_name.lower():
# # # # #         return None  # Skip Zeeman slower images by returning None
# # # # #
# # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # #
# # # # #     if match:
# # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # #         else:
# # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # #
# # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # #     return None  # No valid label
# # # # #
# # # # #
# # # # # # Iterate through the subfolders in the main directory
# # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # #
# # # # #     # Check if it is a folder
# # # # #     if os.path.isdir(subfolder_path):
# # # # #         # Parse the folder name for experiment information
# # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # #
# # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # #             continue
# # # # #
# # # # #         # Look for .h5 files in the subfolder
# # # # #         for file_name in os.listdir(subfolder_path):
# # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # #
# # # # #                 # Open the .h5 file
# # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # #                     # Extract frame data
# # # # #                     frame_data = None
# # # # #                     if image_dataset_path in h5_file:
# # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # #
# # # # #                         # Apply background subtraction before cropping
# # # # #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# # # # #                             # Print the status of background images
# # # # #                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
# # # # #
# # # # #                             # Subtract the first background image (background1)
# # # # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # # # #                             print(f"  Background1 subtraction done for {file_name}")
# # # # #
# # # # #                             # Subtract the second background image (background2)
# # # # #                             background2_subtracted = cv2.subtract(background1_subtracted, background_images[1])
# # # # #                             print(f"  Background2 subtraction done for {file_name}")
# # # # #
# # # # #                             # Crop the background-subtracted image (after both background subtractions)
# # # # #                             cropped_frame = background2_subtracted[top:bottom, left:right]
# # # # #
# # # # #                             # Extract numeric value from parsed title
# # # # #                             if parsed_title:
# # # # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # #                                 if numeric_value:
# # # # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # # # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # #                                     else:
# # # # #                                         numeric_value = float(numeric_value.group(1))
# # # # #                                 else:
# # # # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # #                             else:
# # # # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # #
# # # # #                             # Store the cropped image information
# # # # #                             file_info.append({
# # # # #                                 "folder_name": subfolder_name,
# # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # #                                 "file_path": file_path,
# # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # #                             })
# # # # #
# # # # #
# # # # # # Check if there are background images (background1 and background2)
# # # # # if len(background_images) != 2:
# # # # #     print("Error: Less than two background images found!")
# # # # #     exit()
# # # # #
# # # # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # # # intensity_info = []
# # # # # pixel_sums = []
# # # # # times = []
# # # # #
# # # # # for info in file_info:
# # # # #     # Convert cropped image to uint8 if necessary
# # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # #
# # # # #     # Sum the pixel values for the subtracted image
# # # # #     pixel_sum = np.sum(cropped_image)
# # # # #
# # # # #     # Store the sum and numeric value for plotting later
# # # # #     pixel_sums.append(pixel_sum)
# # # # #     times.append(info['numeric_value'])
# # # # #
# # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # #     intensity_info.append({
# # # # #         "folder_name": info['folder_name'],
# # # # #         "parsed_title": info['parsed_title'],
# # # # #         "file_path": info['file_path'],
# # # # #         "pixel_sum": pixel_sum,
# # # # #         "numeric_value": info['numeric_value']
# # # # #     })
# # # # #
# # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # #
# # # # # # Number of cropped images to display
# # # # # num_images = len(file_info_sorted)
# # # #
# # # #
# # # # #
# # # # # # Determine the grid size for subplots
# # # # # cols = 4  # Number of columns in the grid
# # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # #
# # # # # # Create a figure for the subplots
# # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # #
# # # # # # Flatten axes for easy indexing
# # # # # axes = axes.flatten()
# # # # #
# # # # # # Display each cropped image in a subplot
# # # # # for i, info in enumerate(file_info_sorted):
# # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # #
# # # # #     # Add text to the cropped image at adjustable position
# # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # #                  ha='left', va='bottom')
# # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # #
# # # # # # Hide unused subplots
# # # # # for j in range(i + 1, len(axes)):
# # # # #     axes[j].axis('off')
# # # # #
# # # # # # Display the plot
# # # # # plt.tight_layout()
# # # # # plt.show()
# # # # #
# # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # plt.figure(figsize=(8, 6))
# # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # plt.xlabel('Time (t)')
# # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # # # plt.grid(True)
# # # # # plt.show()
# # # # #
# # # # #
# # # # #
# # # # #
# # # # # #
# # # # # # import h5py
# # # # # # import numpy as np
# # # # # # import os
# # # # # # import cv2
# # # # # # import matplotlib
# # # # # # import re
# # # # # # matplotlib.use('TkAgg')
# # # # # # import matplotlib.pyplot as plt
# # # # # #
# # # # # # # Specify the main directory where subfolders are located
# # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # #
# # # # # # # List to store the file paths, folder names, and image data
# # # # # # file_info = []
# # # # # # background_images = []  # List to store background images
# # # # # #
# # # # # # # Change cropping region of photos
# # # # # # top = 550
# # # # # # bottom = 850
# # # # # # left = 910
# # # # # # right = 1310
# # # # # #
# # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # #
# # # # # # def parse_folder_name(folder_name):
# # # # # #     # Check for the "background" folder
# # # # # #     if "background" in folder_name.lower():
# # # # # #         return "Background"  # Label for background images
# # # # # #
# # # # # #     # Skip Zeeman slower folders
# # # # # #     if "zeeman" in folder_name.lower():
# # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # #
# # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # #
# # # # # #     if match:
# # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # #         else:
# # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # #
# # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # #     return None  # No valid label
# # # # # #
# # # # # #
# # # # # # # Iterate through the subfolders in the main directory
# # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # #
# # # # # #     # Check if it is a folder
# # # # # #     if os.path.isdir(subfolder_path):
# # # # # #         # Parse the folder name for experiment information
# # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # #
# # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # #             continue
# # # # # #
# # # # # #         # Look for .h5 files in the subfolder
# # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # #
# # # # # #                 # Open the .h5 file
# # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # #                     # Extract frame data
# # # # # #                     frame_data = None
# # # # # #                     if image_dataset_path in h5_file:
# # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # #
# # # # # #                         # Apply background subtraction before cropping
# # # # # #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# # # # # #                             # Subtract the first background image (background1)
# # # # # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # # # # #
# # # # # #                             # Subtract the second background image (background2)
# # # # # #                             background2_subtracted = cv2.subtract(background1_subtracted, background_images[1])
# # # # # #
# # # # # #                             # Crop the background-subtracted image
# # # # # #                             cropped_frame = background2_subtracted[top:bottom, left:right]
# # # # # #
# # # # # #                             # Extract numeric value from parsed title
# # # # # #                             if parsed_title:
# # # # # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # #                                 if numeric_value:
# # # # # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # #                                     else:
# # # # # #                                         numeric_value = float(numeric_value.group(1))
# # # # # #                                 else:
# # # # # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # #                             else:
# # # # # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # #
# # # # # #                             # Store the cropped image information
# # # # # #                             file_info.append({
# # # # # #                                 "folder_name": subfolder_name,
# # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # #                                 "file_path": file_path,
# # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # #                             })
# # # # # #
# # # # # # # Check if there are background images (background1 and background2)
# # # # # # if len(background_images) != 2:
# # # # # #     print("Error: Less than two background images found!")
# # # # # #     exit()
# # # # # #
# # # # # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # # # # intensity_info = []
# # # # # # pixel_sums = []
# # # # # # times = []
# # # # # #
# # # # # # for info in file_info:
# # # # # #     # Convert cropped image to uint8 if necessary
# # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # #
# # # # # #     # Sum the pixel values for the subtracted image
# # # # # #     pixel_sum = np.sum(cropped_image)
# # # # # #
# # # # # #     # Store the sum and numeric value for plotting later
# # # # # #     pixel_sums.append(pixel_sum)
# # # # # #     times.append(info['numeric_value'])
# # # # # #
# # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # #     intensity_info.append({
# # # # # #         "folder_name": info['folder_name'],
# # # # # #         "parsed_title": info['parsed_title'],
# # # # # #         "file_path": info['file_path'],
# # # # # #         "pixel_sum": pixel_sum,
# # # # # #         "numeric_value": info['numeric_value']
# # # # # #     })
# # # # # #
# # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # #
# # # # # # # Number of cropped images to display
# # # # # # num_images = len(file_info_sorted)
# # # # # #
# # # # # # # Determine the grid size for subplots
# # # # # # cols = 4  # Number of columns in the grid
# # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # #
# # # # # # # Create a figure for the subplots
# # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # #
# # # # # # # Flatten axes for easy indexing
# # # # # # axes = axes.flatten()
# # # # # #
# # # # # # # Display each cropped image in a subplot
# # # # # # for i, info in enumerate(file_info_sorted):
# # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # #
# # # # # #     # Add text to the cropped image at adjustable position
# # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # #                  ha='left', va='bottom')
# # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # #
# # # # # # # Hide unused subplots
# # # # # # for j in range(i + 1, len(axes)):
# # # # # #     axes[j].axis('off')
# # # # # #
# # # # # # # Display the plot
# # # # # # plt.tight_layout()
# # # # # # plt.show()
# # # # # #
# # # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # # plt.figure(figsize=(8, 6))
# # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # plt.xlabel('Time (t)')
# # # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # # # # plt.grid(True)
# # # # # # plt.show()
# # # # # #
# # # # # #
# # # # # #
# # # # # #
# # # # # #
# # # # # #
# # # # # # # import h5py
# # # # # # # import numpy as np
# # # # # # # import os
# # # # # # # import cv2
# # # # # # # import matplotlib
# # # # # # # import re
# # # # # # # matplotlib.use('TkAgg')
# # # # # # # import matplotlib.pyplot as plt
# # # # # # #
# # # # # # # # Specify the main directory where subfolders are located
# # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # #
# # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # file_info = []
# # # # # # # background_images = []  # List to store background images
# # # # # # #
# # # # # # # # Change cropping region of photos
# # # # # # # top = 550
# # # # # # # bottom = 850
# # # # # # # left = 910
# # # # # # # right = 1310
# # # # # # #
# # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # #
# # # # # # # def parse_folder_name(folder_name):
# # # # # # #     # Check for the "background" folder
# # # # # # #     if "background" in folder_name.lower():
# # # # # # #         return "Background"  # Label for background images
# # # # # # #
# # # # # # #     # Skip Zeeman slower folders
# # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # #
# # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # #
# # # # # # #     if match:
# # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # #         else:
# # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # #
# # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # #     return None  # No valid label
# # # # # # #
# # # # # # #
# # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # #
# # # # # # #     # Check if it is a folder
# # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # #         # Parse the folder name for experiment information
# # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # #
# # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # #             continue
# # # # # # #
# # # # # # #         # Look for .h5 files in the subfolder
# # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # #
# # # # # # #                 # Open the .h5 file
# # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # #                     # Extract frame data
# # # # # # #                     frame_data = None
# # # # # # #                     if image_dataset_path in h5_file:
# # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # #
# # # # # # #                         # Apply background subtraction before cropping
# # # # # # #                         if len(background_images) == 2:  # Ensure we have two backgrounds
# # # # # # #                             # Print the status of background images
# # # # # # #                             print(f"Subtracting backgrounds from {file_name} in folder {subfolder_name}:")
# # # # # # #
# # # # # # #                             # Subtract the first background image (background1)
# # # # # # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # # # # # #                             print(f"  Background1 subtraction done for {file_name}")
# # # # # # #
# # # # # # #                             # Subtract the second background image (background2)
# # # # # # #                             background2_subtracted = cv2.subtract(background1_subtracted, background_images[1])
# # # # # # #                             print(f"  Background2 subtraction done for {file_name}")
# # # # # # #
# # # # # # #                             # Crop the background-subtracted image
# # # # # # #                             cropped_frame = background2_subtracted[top:bottom, left:right]
# # # # # # #
# # # # # # #                             # Extract numeric value from parsed title
# # # # # # #                             if parsed_title:
# # # # # # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # #                                 if numeric_value:
# # # # # # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # #                                     else:
# # # # # # #                                         numeric_value = float(numeric_value.group(1))
# # # # # # #                                 else:
# # # # # # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # #                             else:
# # # # # # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # #
# # # # # # #                             # Store the cropped image information
# # # # # # #                             file_info.append({
# # # # # # #                                 "folder_name": subfolder_name,
# # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # #                                 "file_path": file_path,
# # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # #                             })
# # # # # # #
# # # # # # #
# # # # # # # # Check if there are background images (background1 and background2)
# # # # # # # if len(background_images) != 2:
# # # # # # #     print("Error: Less than two background images found!")
# # # # # # #     exit()
# # # # # # #
# # # # # # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # # # # # intensity_info = []
# # # # # # # pixel_sums = []
# # # # # # # times = []
# # # # # # #
# # # # # # # for info in file_info:
# # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # #
# # # # # # #     # Sum the pixel values for the subtracted image
# # # # # # #     pixel_sum = np.sum(cropped_image)
# # # # # # #
# # # # # # #     # Store the sum and numeric value for plotting later
# # # # # # #     pixel_sums.append(pixel_sum)
# # # # # # #     times.append(info['numeric_value'])
# # # # # # #
# # # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # # #     intensity_info.append({
# # # # # # #         "folder_name": info['folder_name'],
# # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # #         "file_path": info['file_path'],
# # # # # # #         "pixel_sum": pixel_sum,
# # # # # # #         "numeric_value": info['numeric_value']
# # # # # # #     })
# # # # # # #
# # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # #
# # # # # # # # Number of cropped images to display
# # # # # # # num_images = len(file_info_sorted)
# # # # # # #
# # # # # # # # Determine the grid size for subplots
# # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # #
# # # # # # # # Create a figure for the subplots
# # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # #
# # # # # # # # Flatten axes for easy indexing
# # # # # # # axes = axes.flatten()
# # # # # # #
# # # # # # # # Display each cropped image in a subplot
# # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # #
# # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # #                  ha='left', va='bottom')
# # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # #
# # # # # # # # Hide unused subplots
# # # # # # # for j in range(i + 1, len(axes)):
# # # # # # #     axes[j].axis('off')
# # # # # # #
# # # # # # # # Display the plot
# # # # # # # plt.tight_layout()
# # # # # # # plt.show()
# # # # # # #
# # # # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # # plt.xlabel('Time (t)')
# # # # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # # # # # plt.grid(True)
# # # # # # # plt.show()
# # # # # # #
# # # # # # #
# # # # # # #
# # # # # # # #
# # # # # # # # import h5py
# # # # # # # # import numpy as np
# # # # # # # # import os
# # # # # # # # import cv2
# # # # # # # # import matplotlib
# # # # # # # # import re
# # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # #
# # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # #
# # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # file_info = []
# # # # # # # # background_images = []  # List to store background images
# # # # # # # #
# # # # # # # # # Change cropping region of photos
# # # # # # # # top = 550
# # # # # # # # bottom = 850
# # # # # # # # left = 910
# # # # # # # # right = 1310
# # # # # # # #
# # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # #
# # # # # # # # def parse_folder_name(folder_name):
# # # # # # # #     # Check for the "background" folder
# # # # # # # #     if "background" in folder_name.lower():
# # # # # # # #         return "Background"  # Label for background images
# # # # # # # #
# # # # # # # #     # Skip Zeeman slower folders
# # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # #
# # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # #
# # # # # # # #     if match:
# # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # #         else:
# # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # #
# # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # #     return None  # No valid label
# # # # # # # #
# # # # # # # #
# # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # #
# # # # # # # #     # Check if it is a folder
# # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # #         # Parse the folder name for experiment information
# # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # #
# # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # #             continue
# # # # # # # #
# # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # #
# # # # # # # #                 # Open the .h5 file
# # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # #                     # Extract frame data
# # # # # # # #                     frame_data = None
# # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # #
# # # # # # # #                         # Apply background subtraction before cropping
# # # # # # # #                         if len(background_images) >= 2:  # Ensure we have two backgrounds
# # # # # # # #                             # Subtract the first background image (background1)
# # # # # # # #                             background1_subtracted = cv2.subtract(frame_data, background_images[0])
# # # # # # # #
# # # # # # # #                             # Subtract the second background image (background2)
# # # # # # # #                             background2_subtracted = cv2.subtract(background1_subtracted, background_images[1])
# # # # # # # #
# # # # # # # #                             # Crop the background-subtracted image
# # # # # # # #                             cropped_frame = background2_subtracted[top:bottom, left:right]
# # # # # # # #
# # # # # # # #                             # Extract numeric value from parsed title
# # # # # # # #                             if parsed_title:
# # # # # # # #                                 numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # # #                                 if numeric_value:
# # # # # # # #                                     if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # #                                         numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # #                                     else:
# # # # # # # #                                         numeric_value = float(numeric_value.group(1))
# # # # # # # #                                 else:
# # # # # # # #                                     numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # # #                             else:
# # # # # # # #                                 numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # #
# # # # # # # #                             # Store the cropped image information
# # # # # # # #                             file_info.append({
# # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # #                                 "file_path": file_path,
# # # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # # #                             })
# # # # # # # #
# # # # # # # #
# # # # # # # # # Check if there are background images (background1 and background2)
# # # # # # # # if len(background_images) != 2:
# # # # # # # #     print("Error: Less than two background images found!")
# # # # # # # #     exit()
# # # # # # # #
# # # # # # # # # Calculate the sum of pixel values for each image and subtract both background images before calculation
# # # # # # # # intensity_info = []
# # # # # # # # pixel_sums = []
# # # # # # # # times = []
# # # # # # # #
# # # # # # # # for info in file_info:
# # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # #
# # # # # # # #     # Sum the pixel values for the subtracted image
# # # # # # # #     pixel_sum = np.sum(cropped_image)
# # # # # # # #
# # # # # # # #     # Store the sum and numeric value for plotting later
# # # # # # # #     pixel_sums.append(pixel_sum)
# # # # # # # #     times.append(info['numeric_value'])
# # # # # # # #
# # # # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # # # #     intensity_info.append({
# # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # #         "file_path": info['file_path'],
# # # # # # # #         "pixel_sum": pixel_sum,
# # # # # # # #         "numeric_value": info['numeric_value']
# # # # # # # #     })
# # # # # # # #
# # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # #
# # # # # # # # # Number of cropped images to display
# # # # # # # # num_images = len(file_info_sorted)
# # # # # # # #
# # # # # # # # # Determine the grid size for subplots
# # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # #
# # # # # # # # # Create a figure for the subplots
# # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # #
# # # # # # # # # Flatten axes for easy indexing
# # # # # # # # axes = axes.flatten()
# # # # # # # #
# # # # # # # # # Display each cropped image in a subplot
# # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # #
# # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # #                  ha='left', va='bottom')
# # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # #
# # # # # # # # # Hide unused subplots
# # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # #     axes[j].axis('off')
# # # # # # # #
# # # # # # # # # Display the plot
# # # # # # # # plt.tight_layout()
# # # # # # # # plt.show()
# # # # # # # #
# # # # # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # # # plt.xlabel('Time (t)')
# # # # # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # # # # plt.title('Sum of Pixel Values vs Time (Background1 and Background2 Subtracted)')
# # # # # # # # plt.grid(True)
# # # # # # # # plt.show()
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # # #
# # # # # # # # # import h5py
# # # # # # # # # import numpy as np
# # # # # # # # # import os
# # # # # # # # # import cv2
# # # # # # # # # import matplotlib
# # # # # # # # # import re
# # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # #
# # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # #
# # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # file_info = []
# # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # background_file_paths = []  # List to store file paths of background images
# # # # # # # # #
# # # # # # # # # # Change cropping region of photos
# # # # # # # # # top = 550
# # # # # # # # # bottom = 850
# # # # # # # # # left = 910
# # # # # # # # # right = 1310
# # # # # # # # #
# # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # #
# # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # #     # Check for the "background" folder
# # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # #
# # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # #
# # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # #
# # # # # # # # #     if match:
# # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # #         else:
# # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # #
# # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # #     return None  # No valid label
# # # # # # # # #
# # # # # # # # #
# # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # #
# # # # # # # # #     # Check if it is a folder
# # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # #
# # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # #             continue
# # # # # # # # #
# # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # #
# # # # # # # # #                 # Open the .h5 file
# # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # #                     # Extract frame data
# # # # # # # # #                     frame_data = None
# # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # #
# # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # #
# # # # # # # # #                         # Extract numeric value from parsed title (e.g., "t=2s" -> 2, "t=1/2" -> 1/2)
# # # # # # # # #                         if parsed_title:
# # # # # # # # #                             numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # # # #                             if numeric_value:
# # # # # # # # #                                 if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # #                                     numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # #                                 else:
# # # # # # # # #                                     numeric_value = float(numeric_value.group(1))
# # # # # # # # #                             else:
# # # # # # # # #                                 numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # # # #                         else:
# # # # # # # # #                             numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # #
# # # # # # # # #                         # Store background images separately
# # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # #                             background_file_paths.append(file_path)  # Store the background file path
# # # # # # # # #                         else:
# # # # # # # # #                             # Store the other images for analysis
# # # # # # # # #                             file_info.append({
# # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # #                                 "file_path": file_path,
# # # # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # # # #                             })
# # # # # # # # #
# # # # # # # # # # Count highest intensity pixels for each image and subtract background before calculation
# # # # # # # # # intensity_info = []
# # # # # # # # # pixel_sums = []
# # # # # # # # # times = []
# # # # # # # # #
# # # # # # # # # for info in file_info:
# # # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # # #
# # # # # # # # #     # Match each background to the corresponding image
# # # # # # # # #     # Assuming that the background images are in the same order as the dataset images
# # # # # # # # #     background_image = background_images[file_info.index(info)]  # Use the same index to match
# # # # # # # # #     background_image = background_image.astype(np.uint8)  # Convert background to uint8 if needed
# # # # # # # # #
# # # # # # # # #     # Subtract the background image from the current image
# # # # # # # # #     subtracted_image = cv2.subtract(cropped_image, background_image)
# # # # # # # # #
# # # # # # # # #     # Sum the pixel values for the subtracted image
# # # # # # # # #     pixel_sum = np.sum(subtracted_image)
# # # # # # # # #
# # # # # # # # #     # Store the sum and numeric value for plotting later
# # # # # # # # #     pixel_sums.append(pixel_sum)
# # # # # # # # #     times.append(info['numeric_value'])
# # # # # # # # #
# # # # # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # # # # #     intensity_info.append({
# # # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # # #         "file_path": info['file_path'],
# # # # # # # # #         "pixel_sum": pixel_sum,
# # # # # # # # #         "numeric_value": info['numeric_value']
# # # # # # # # #     })
# # # # # # # # #
# # # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # # #
# # # # # # # # # # Number of cropped images to display
# # # # # # # # # num_images = len(file_info_sorted)
# # # # # # # # #
# # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # #
# # # # # # # # # # Create a figure for the subplots
# # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # #
# # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # axes = axes.flatten()
# # # # # # # # #
# # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # #
# # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # #
# # # # # # # # # # Hide unused subplots
# # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # #     axes[j].axis('off')
# # # # # # # # #
# # # # # # # # # # Display the plot
# # # # # # # # # plt.tight_layout()
# # # # # # # # # plt.show()
# # # # # # # # #
# # # # # # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # # # # plt.xlabel('Time (t)')
# # # # # # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # # # # # plt.title('Sum of Pixel Values vs Time (Background Subtracted)')
# # # # # # # # # plt.grid(True)
# # # # # # # # # plt.show()
# # # # # # # # #
# # # # # # # # #
# # # # # # # # #
# # # # # # # # #
# # # # # # # # # # import h5py
# # # # # # # # # # import numpy as np
# # # # # # # # # # import os
# # # # # # # # # # import cv2
# # # # # # # # # # import matplotlib
# # # # # # # # # # import re
# # # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # #
# # # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # # #
# # # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # # file_info = []
# # # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # #
# # # # # # # # # # # Change cropping region of photos
# # # # # # # # # # top = 550
# # # # # # # # # # bottom = 850
# # # # # # # # # # left = 910
# # # # # # # # # # right = 1310
# # # # # # # # # #
# # # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # # #
# # # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # # #     # Check for the "background" folder
# # # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # # #
# # # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # # #
# # # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # # #
# # # # # # # # # #     if match:
# # # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # # #         else:
# # # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # # #
# # # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # # #     return None  # No valid label
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # # #
# # # # # # # # # #     # Check if it is a folder
# # # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # # #
# # # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # # #             continue
# # # # # # # # # #
# # # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # # #
# # # # # # # # # #                 # Open the .h5 file
# # # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # # #                     # Extract frame data
# # # # # # # # # #                     frame_data = None
# # # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # # #
# # # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # # #
# # # # # # # # # #                         # Extract numeric value from parsed title (e.g., "t=2s" -> 2, "t=1/2" -> 1/2)
# # # # # # # # # #                         if parsed_title:
# # # # # # # # # #                             numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # # # # #                             if numeric_value:
# # # # # # # # # #                                 if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # # #                                     numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # # #                                 else:
# # # # # # # # # #                                     numeric_value = float(numeric_value.group(1))
# # # # # # # # # #                             else:
# # # # # # # # # #                                 numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # # # # #                         else:
# # # # # # # # # #                             numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # # #
# # # # # # # # # #                         # Store background images separately
# # # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # # #                         else:
# # # # # # # # # #                             # Store the other images for analysis
# # # # # # # # # #                             file_info.append({
# # # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # # #                                 "file_path": file_path,
# # # # # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # # # # #                             })
# # # # # # # # # #
# # # # # # # # # # # Compute the average background image (or choose one background frame)
# # # # # # # # # # if background_images:
# # # # # # # # # #     background_avg = np.mean(background_images, axis=0)  # Average over all background images
# # # # # # # # # #     background_avg = background_avg.astype(np.uint8)  # Ensure it's in uint8 format
# # # # # # # # # #
# # # # # # # # # # # Calculate the sum of pixel values for each image and subtract background before calculation
# # # # # # # # # # intensity_info = []
# # # # # # # # # # pixel_sums = []
# # # # # # # # # # times = []
# # # # # # # # # #
# # # # # # # # # # for info in file_info:
# # # # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # # # #
# # # # # # # # # #     # Subtract the background image (using the average background)
# # # # # # # # # #     subtracted_image = cv2.subtract(cropped_image, background_avg)
# # # # # # # # # #
# # # # # # # # # #     # Sum the pixel values for the subtracted image
# # # # # # # # # #     pixel_sum = np.sum(subtracted_image)
# # # # # # # # # #
# # # # # # # # # #     # Store the sum and numeric value for plotting later
# # # # # # # # # #     pixel_sums.append(pixel_sum)
# # # # # # # # # #     times.append(info['numeric_value'])
# # # # # # # # # #
# # # # # # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # # # # # #     intensity_info.append({
# # # # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # # # #         "file_path": info['file_path'],
# # # # # # # # # #         "pixel_sum": pixel_sum,
# # # # # # # # # #         "numeric_value": info['numeric_value']
# # # # # # # # # #     })
# # # # # # # # # #
# # # # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # # # #
# # # # # # # # # # # Number of cropped images to display
# # # # # # # # # # num_images = len(file_info_sorted)
# # # # # # # # # #
# # # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # # #
# # # # # # # # # # # Create a figure for the subplots
# # # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # # #
# # # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # # axes = axes.flatten()
# # # # # # # # # #
# # # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # # #
# # # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # # #
# # # # # # # # # # # Hide unused subplots
# # # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # # #     axes[j].axis('off')
# # # # # # # # # #
# # # # # # # # # # # Display the plot
# # # # # # # # # # plt.tight_layout()
# # # # # # # # # # plt.show()
# # # # # # # # # #
# # # # # # # # # # # Create a second plot: Sum of pixel values vs. time after background subtraction
# # # # # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # # # # # plt.xlabel('Time (t)')
# # # # # # # # # # plt.ylabel('Sum of Pixel Values (After Background Subtraction)')
# # # # # # # # # # plt.title('Sum of Pixel Values vs Time (Background Subtracted)')
# # # # # # # # # # plt.grid(True)
# # # # # # # # # # plt.show()
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # # # import h5py
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import os
# # # # # # # # # # # import cv2
# # # # # # # # # # # import matplotlib
# # # # # # # # # # # import re
# # # # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # #
# # # # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # # # #
# # # # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # # # file_info = []
# # # # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # # #
# # # # # # # # # # # # Change cropping region of photos
# # # # # # # # # # # top = 550
# # # # # # # # # # # bottom = 850
# # # # # # # # # # # left = 910
# # # # # # # # # # # right = 1310
# # # # # # # # # # #
# # # # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # # # #
# # # # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # # # #     # Check for the "background" folder
# # # # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # # # #
# # # # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # # # #
# # # # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # # # #
# # # # # # # # # # #     if match:
# # # # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # # # #         else:
# # # # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # # # #
# # # # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # # # #     return None  # No valid label
# # # # # # # # # # #
# # # # # # # # # # #
# # # # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #     # Check if it is a folder
# # # # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # # # #             continue
# # # # # # # # # # #
# # # # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # # # #
# # # # # # # # # # #                 # Open the .h5 file
# # # # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # # # #                     # Extract frame data
# # # # # # # # # # #                     frame_data = None
# # # # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # # # #
# # # # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # # # #
# # # # # # # # # # #                         # Extract numeric value from parsed title (e.g., "t=2s" -> 2, "t=1/2" -> 1/2)
# # # # # # # # # # #                         if parsed_title:
# # # # # # # # # # #                             numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # # # # # #                             if numeric_value:
# # # # # # # # # # #                                 if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # # # #                                     numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # # # #                                 else:
# # # # # # # # # # #                                     numeric_value = float(numeric_value.group(1))
# # # # # # # # # # #                             else:
# # # # # # # # # # #                                 numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # # # #
# # # # # # # # # # #                         # Store background images separately
# # # # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             # Store the other images for analysis
# # # # # # # # # # #                             file_info.append({
# # # # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # # # #                                 "file_path": file_path,
# # # # # # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # # # # # #                             })
# # # # # # # # # # #
# # # # # # # # # # # # Compute the average background image (or choose one background frame)
# # # # # # # # # # # if background_images:
# # # # # # # # # # #     background_avg = np.mean(background_images, axis=0)  # Average over all background images
# # # # # # # # # # #     background_avg = background_avg.astype(np.uint8)  # Ensure it's in uint8 format
# # # # # # # # # # #
# # # # # # # # # # # # Calculate the sum of pixel values for each image and store number of pixels vs. time for plotting
# # # # # # # # # # # intensity_info = []
# # # # # # # # # # # pixel_sums = []
# # # # # # # # # # # times = []
# # # # # # # # # # #
# # # # # # # # # # # for info in file_info:
# # # # # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # # # # #
# # # # # # # # # # #     # Sum the pixel values for the entire image
# # # # # # # # # # #     pixel_sum = np.sum(cropped_image)
# # # # # # # # # # #
# # # # # # # # # # #     # Store the sum and numeric value for plotting later
# # # # # # # # # # #     pixel_sums.append(pixel_sum)
# # # # # # # # # # #     times.append(info['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # #     # Store the intensity sum information along with the numeric value for sorting
# # # # # # # # # # #     intensity_info.append({
# # # # # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # # # # #         "file_path": info['file_path'],
# # # # # # # # # # #         "pixel_sum": pixel_sum,
# # # # # # # # # # #         "numeric_value": info['numeric_value']
# # # # # # # # # # #     })
# # # # # # # # # # #
# # # # # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # # # Number of cropped images to display
# # # # # # # # # # # num_images = len(file_info_sorted)
# # # # # # # # # # #
# # # # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # # # #
# # # # # # # # # # # # Create a figure for the subplots
# # # # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # # # #
# # # # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # # # axes = axes.flatten()
# # # # # # # # # # #
# # # # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # # # #
# # # # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # # # #
# # # # # # # # # # # # Hide unused subplots
# # # # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # # # #     axes[j].axis('off')
# # # # # # # # # # #
# # # # # # # # # # # # Display the plot
# # # # # # # # # # # plt.tight_layout()
# # # # # # # # # # # plt.show()
# # # # # # # # # # #
# # # # # # # # # # # # Create a second plot: Sum of pixel values vs. time
# # # # # # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # # # # # plt.scatter(times, pixel_sums, color='blue')
# # # # # # # # # # # plt.xlabel('Time (t)')
# # # # # # # # # # # plt.ylabel('Sum of Pixel Values')
# # # # # # # # # # # plt.title('Sum of Pixel Values vs Time')
# # # # # # # # # # # plt.grid(True)
# # # # # # # # # # # plt.show()
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # # # import h5py
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import os
# # # # # # # # # # # import cv2
# # # # # # # # # # # import matplotlib
# # # # # # # # # # # import re
# # # # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # #
# # # # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # # # #
# # # # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # # # file_info = []
# # # # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # # #
# # # # # # # # # # # # Change cropping region of photos
# # # # # # # # # # # top = 550
# # # # # # # # # # # bottom = 850
# # # # # # # # # # # left = 910
# # # # # # # # # # # right = 1310
# # # # # # # # # # #
# # # # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # # # #
# # # # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # # # #     # Check for the "background" folder
# # # # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # # # #
# # # # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # # # #
# # # # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # # # #
# # # # # # # # # # #     if match:
# # # # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # # # #         else:
# # # # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # # # #
# # # # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # # # #     return None  # No valid label
# # # # # # # # # # #
# # # # # # # # # # #
# # # # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #     # Check if it is a folder
# # # # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # # # #             continue
# # # # # # # # # # #
# # # # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # # # #
# # # # # # # # # # #                 # Open the .h5 file
# # # # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # # # #                     # Extract frame data
# # # # # # # # # # #                     frame_data = None
# # # # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # # # #
# # # # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # # # #
# # # # # # # # # # #                         # Extract numeric value from parsed title (e.g., "t=2s" -> 2, "t=1/2" -> 1/2)
# # # # # # # # # # #                         if parsed_title:
# # # # # # # # # # #                             numeric_value = re.search(r"(\d+)(?:/(\d+))?", parsed_title)
# # # # # # # # # # #                             if numeric_value:
# # # # # # # # # # #                                 if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # # # #                                     numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # # # #                                 else:
# # # # # # # # # # #                                     numeric_value = float(numeric_value.group(1))
# # # # # # # # # # #                             else:
# # # # # # # # # # #                                 numeric_value = float('inf')  # If there's no numeric value, assign a large number
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # # # #
# # # # # # # # # # #                         # Store background images separately
# # # # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             # Store the other images for analysis
# # # # # # # # # # #                             file_info.append({
# # # # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # # # #                                 "file_path": file_path,
# # # # # # # # # # #                                 "cropped_image": cropped_frame,  # Add cropped image here
# # # # # # # # # # #                                 "numeric_value": numeric_value  # Add numeric value for sorting
# # # # # # # # # # #                             })
# # # # # # # # # # #
# # # # # # # # # # # # Compute the average background image (or choose one background frame)
# # # # # # # # # # # if background_images:
# # # # # # # # # # #     background_avg = np.mean(background_images, axis=0)  # Average over all background images
# # # # # # # # # # #     background_avg = background_avg.astype(np.uint8)  # Ensure it's in uint8 format
# # # # # # # # # # #
# # # # # # # # # # # # Count highest intensity pixels for each image and store number of pixels vs. time for plotting
# # # # # # # # # # # intensity_info = []
# # # # # # # # # # # pixel_counts = []
# # # # # # # # # # # times = []
# # # # # # # # # # #
# # # # # # # # # # # for info in file_info:
# # # # # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # # # # #
# # # # # # # # # # #     # Find the maximum intensity in the image
# # # # # # # # # # #     max_intensity = np.max(cropped_image)
# # # # # # # # # # #
# # # # # # # # # # #     # Count the number of pixels that have the maximum intensity
# # # # # # # # # # #     max_intensity_count = np.count_nonzero(cropped_image == max_intensity)
# # # # # # # # # # #
# # # # # # # # # # #     # Store pixel count and numeric value for plotting later
# # # # # # # # # # #     pixel_counts.append(max_intensity_count)
# # # # # # # # # # #     times.append(info['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # #     # Store the intensity count information along with the numeric value for sorting
# # # # # # # # # # #     intensity_info.append({
# # # # # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # # # # #         "file_path": info['file_path'],
# # # # # # # # # # #         "max_intensity": max_intensity,
# # # # # # # # # # #         "max_intensity_count": max_intensity_count,
# # # # # # # # # # #         "numeric_value": info['numeric_value']
# # # # # # # # # # #     })
# # # # # # # # # # #
# # # # # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # # # Number of cropped images to display
# # # # # # # # # # # num_images = len(file_info_sorted)
# # # # # # # # # # #
# # # # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # # # #
# # # # # # # # # # # # Create a figure for the subplots
# # # # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # # # #
# # # # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # # # axes = axes.flatten()
# # # # # # # # # # #
# # # # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # # # #
# # # # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # # # #
# # # # # # # # # # # # Hide unused subplots
# # # # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # # # #     axes[j].axis('off')
# # # # # # # # # # #
# # # # # # # # # # # # Display the plot
# # # # # # # # # # # plt.tight_layout()
# # # # # # # # # # # plt.show()
# # # # # # # # # # #
# # # # # # # # # # # # Create a second plot: Number of pixels with max intensity vs. time
# # # # # # # # # # # plt.figure(figsize=(8, 6))
# # # # # # # # # # # plt.scatter(times, pixel_counts, color='blue')
# # # # # # # # # # # plt.xlabel('Time (t)')
# # # # # # # # # # # plt.ylabel('Number of Pixels with Max Intensity')
# # # # # # # # # # # plt.title('Number of Pixels vs Time')
# # # # # # # # # # # plt.grid(True)
# # # # # # # # # # # plt.show()
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # # # import h5py
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import os
# # # # # # # # # # # import cv2
# # # # # # # # # # # import matplotlib
# # # # # # # # # # # import re
# # # # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # #
# # # # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # # # #
# # # # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # # # file_info = []
# # # # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # # #
# # # # # # # # # # # # Change cropping region of photos
# # # # # # # # # # # top = 550
# # # # # # # # # # # bottom = 850
# # # # # # # # # # # left = 910
# # # # # # # # # # # right = 1310
# # # # # # # # # # #
# # # # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # # # #
# # # # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # # # #     # Check for the "background" folder
# # # # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # # # #
# # # # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # # # #
# # # # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # # # #
# # # # # # # # # # #     if match:
# # # # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # # # #         else:
# # # # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # # # #
# # # # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # # # #     return None  # No valid label
# # # # # # # # # # #
# # # # # # # # # # #
# # # # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #     # Check if it is a folder
# # # # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # # # #             continue
# # # # # # # # # # #
# # # # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # # # #
# # # # # # # # # # #                 # Open the .h5 file
# # # # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # # # #                     # Extract frame data
# # # # # # # # # # #                     frame_data = None
# # # # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # # # #
# # # # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # # # #
# # # # # # # # # # #                         # Store background images separately
# # # # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             # Store the other images for analysis
# # # # # # # # # # #                             file_info.append({
# # # # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # # # #                                 "file_path": file_path,
# # # # # # # # # # #                                 "cropped_image": cropped_frame  # Add cropped image here
# # # # # # # # # # #                             })
# # # # # # # # # # #
# # # # # # # # # # # # Compute the average background image (or choose one background frame)
# # # # # # # # # # # if background_images:
# # # # # # # # # # #     background_avg = np.mean(background_images, axis=0)  # Average over all background images
# # # # # # # # # # #     background_avg = background_avg.astype(np.uint8)  # Ensure it's in uint8 format
# # # # # # # # # # #
# # # # # # # # # # # # Sort the file_info based on the numeric value from parsed_title
# # # # # # # # # # # for info in file_info:
# # # # # # # # # # #     if info['parsed_title']:
# # # # # # # # # # #         numeric_value = re.search(r"(\d+)(?:/(\d+))?", info['parsed_title'])
# # # # # # # # # # #         if numeric_value:
# # # # # # # # # # #             if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # # # #                 info['numeric_value'] = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # # # #             else:
# # # # # # # # # # #                 info['numeric_value'] = float(numeric_value.group(1))
# # # # # # # # # # #         else:
# # # # # # # # # # #             info['numeric_value'] = float('inf')  # If there's no numeric value, assign a large number (to handle "Unknown")
# # # # # # # # # # #     else:
# # # # # # # # # # #         info['numeric_value'] = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # # # #
# # # # # # # # # # # # Sort by numeric value (lowest to highest)
# # # # # # # # # # # file_info_sorted = sorted(file_info, key=lambda x: x['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # # # Number of cropped images to display
# # # # # # # # # # # num_images = len(file_info_sorted)
# # # # # # # # # # #
# # # # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # # # #
# # # # # # # # # # # # Create a figure for the subplots
# # # # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # # # #
# # # # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # # # axes = axes.flatten()
# # # # # # # # # # #
# # # # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # # # for i, info in enumerate(file_info_sorted):
# # # # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # # # #
# # # # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # # # #
# # # # # # # # # # # # Hide unused subplots
# # # # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # # # #     axes[j].axis('off')
# # # # # # # # # # #
# # # # # # # # # # # # Display the plot
# # # # # # # # # # # plt.tight_layout()
# # # # # # # # # # # plt.show()
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # #
# # # # # # # # # # # import h5py
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import os
# # # # # # # # # # # import cv2
# # # # # # # # # # # import matplotlib
# # # # # # # # # # # import re
# # # # # # # # # # # matplotlib.use('TkAgg')
# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # #
# # # # # # # # # # # # Specify the main directory where subfolders are located
# # # # # # # # # # # main_folder_path = "data/20250117_first_data"  # Replace with your directory path
# # # # # # # # # # # image_dataset_path = 'images/cam1/after ramp/frame'
# # # # # # # # # # #
# # # # # # # # # # # # List to store the file paths, folder names, and image data
# # # # # # # # # # # file_info = []
# # # # # # # # # # # background_images = []  # List to store background images
# # # # # # # # # # #
# # # # # # # # # # # # Change cropping region of photos
# # # # # # # # # # # top = 550
# # # # # # # # # # # bottom = 850
# # # # # # # # # # # left = 910
# # # # # # # # # # # right = 1310
# # # # # # # # # # #
# # # # # # # # # # # text_x = 10  # Horizontal position of the text (from the left)
# # # # # # # # # # # text_y = 20  # Vertical position of the text (from the bottom)
# # # # # # # # # # # def parse_folder_name(folder_name):
# # # # # # # # # # #     # Check for the "background" folder
# # # # # # # # # # #     if "background" in folder_name.lower():
# # # # # # # # # # #         return "Background"  # Label for background images
# # # # # # # # # # #
# # # # # # # # # # #     # Skip Zeeman slower folders
# # # # # # # # # # #     if "zeeman" in folder_name.lower():
# # # # # # # # # # #         return None  # Skip Zeeman slower images by returning None
# # # # # # # # # # #
# # # # # # # # # # #     # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
# # # # # # # # # # #     match = re.match(r"(\d+)_?(\d*)s?", folder_name)
# # # # # # # # # # #
# # # # # # # # # # #     if match:
# # # # # # # # # # #         # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
# # # # # # # # # # #         if match.group(2):  # If there's a second part like "1_2"
# # # # # # # # # # #             fraction = f"{match.group(1)}/{match.group(2)}"
# # # # # # # # # # #             return f"t={fraction}"  # e.g., "t=1/2"
# # # # # # # # # # #         else:
# # # # # # # # # # #             return f"t={match.group(1)}s"  # e.g., "t=2s"
# # # # # # # # # # #
# # # # # # # # # # #     # If no match and folder doesn't start with valid format, return "Unknown"
# # # # # # # # # # #     return None  # No valid label
# # # # # # # # # # #
# # # # # # # # # # #
# # # # # # # # # # # # Iterate through the subfolders in the main directory
# # # # # # # # # # # for subfolder_name in os.listdir(main_folder_path):
# # # # # # # # # # #     subfolder_path = os.path.join(main_folder_path, subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #     # Check if it is a folder
# # # # # # # # # # #     if os.path.isdir(subfolder_path):
# # # # # # # # # # #         # Parse the folder name for experiment information
# # # # # # # # # # #         parsed_title = parse_folder_name(subfolder_name)
# # # # # # # # # # #
# # # # # # # # # # #         # Skip folders with invalid or unknown titles (like "zeeman")
# # # # # # # # # # #         if parsed_title == "Unknown" or parsed_title is None:  # Skip invalid titles
# # # # # # # # # # #             continue
# # # # # # # # # # #
# # # # # # # # # # #         # Look for .h5 files in the subfolder
# # # # # # # # # # #         for file_name in os.listdir(subfolder_path):
# # # # # # # # # # #             if file_name.endswith(".h5"):  # Check for .h5 files
# # # # # # # # # # #                 file_path = os.path.join(subfolder_path, file_name)
# # # # # # # # # # #
# # # # # # # # # # #                 # Open the .h5 file
# # # # # # # # # # #                 with h5py.File(file_path, "r") as h5_file:
# # # # # # # # # # #                     # Extract frame data
# # # # # # # # # # #                     frame_data = None
# # # # # # # # # # #                     if image_dataset_path in h5_file:
# # # # # # # # # # #                         frame_data = h5_file[image_dataset_path][:]
# # # # # # # # # # #
# # # # # # # # # # #                         # Apply fixed bounds for cropping
# # # # # # # # # # #                         cropped_frame = frame_data[top:bottom, left:right]
# # # # # # # # # # #
# # # # # # # # # # #                         # Store background images separately
# # # # # # # # # # #                         if parsed_title == "Background":
# # # # # # # # # # #                             background_images.append(cropped_frame)
# # # # # # # # # # #                         else:
# # # # # # # # # # #                             # Store the other images for analysis
# # # # # # # # # # #                             file_info.append({
# # # # # # # # # # #                                 "folder_name": subfolder_name,
# # # # # # # # # # #                                 "parsed_title": parsed_title,  # Ensure parsed title is included
# # # # # # # # # # #                                 "file_path": file_path,
# # # # # # # # # # #                                 "cropped_image": cropped_frame  # Add cropped image here
# # # # # # # # # # #                             })
# # # # # # # # # # #
# # # # # # # # # # # # Compute the average background image (or choose one background frame)
# # # # # # # # # # # if background_images:
# # # # # # # # # # #     background_avg = np.mean(background_images, axis=0)  # Average over all background images
# # # # # # # # # # #     background_avg = background_avg.astype(np.uint8)  # Ensure it's in uint8 format
# # # # # # # # # # #
# # # # # # # # # # # # Count highest intensity pixels for each image
# # # # # # # # # # # intensity_info = []
# # # # # # # # # # #
# # # # # # # # # # # for info in file_info:
# # # # # # # # # # #     # Convert cropped image to uint8 if necessary
# # # # # # # # # # #     cropped_image = info['cropped_image'].astype(np.uint8)
# # # # # # # # # # #
# # # # # # # # # # #     # Find the maximum intensity in the image
# # # # # # # # # # #     max_intensity = np.max(cropped_image)
# # # # # # # # # # #
# # # # # # # # # # #     # Count the number of pixels that have the maximum intensity
# # # # # # # # # # #     max_intensity_count = np.count_nonzero(cropped_image == max_intensity)
# # # # # # # # # # #
# # # # # # # # # # #     # Extract numeric value from parsed title (e.g., "t=2s" -> 2, "t=1/2" -> 1/2)
# # # # # # # # # # #     if info['parsed_title']:
# # # # # # # # # # #         numeric_value = re.search(r"(\d+)(?:/(\d+))?", info['parsed_title'])
# # # # # # # # # # #         if numeric_value:
# # # # # # # # # # #             if numeric_value.group(2):  # Fraction format like "1/2"
# # # # # # # # # # #                 numeric_value = float(numeric_value.group(1)) / float(numeric_value.group(2))
# # # # # # # # # # #             else:
# # # # # # # # # # #                 numeric_value = float(numeric_value.group(1))
# # # # # # # # # # #         else:
# # # # # # # # # # #             numeric_value = float('inf')  # If there's no numeric value, assign a large number (to handle "Unknown")
# # # # # # # # # # #     else:
# # # # # # # # # # #         numeric_value = float('inf')  # If parsed_title is None, assign a large number
# # # # # # # # # # #
# # # # # # # # # # #     # Store the intensity count information along with the numeric value for sorting
# # # # # # # # # # #     intensity_info.append({
# # # # # # # # # # #         "folder_name": info['folder_name'],
# # # # # # # # # # #         "parsed_title": info['parsed_title'],
# # # # # # # # # # #         "file_path": info['file_path'],
# # # # # # # # # # #         "max_intensity": max_intensity,
# # # # # # # # # # #         "max_intensity_count": max_intensity_count,
# # # # # # # # # # #         "numeric_value": numeric_value
# # # # # # # # # # #     })
# # # # # # # # # # #
# # # # # # # # # # # # Sort the data based on the numeric value (from lowest to highest)
# # # # # # # # # # # sorted_intensity_info = sorted(intensity_info, key=lambda x: x['numeric_value'])
# # # # # # # # # # #
# # # # # # # # # # # # Display the sorted results
# # # # # # # # # # # for info in sorted_intensity_info:
# # # # # # # # # # #     print(f"File: {info['file_path']} | Max Intensity: {info['max_intensity']} | "
# # # # # # # # # # #           f"Max Intensity Count: {info['max_intensity_count']} | Title: {info['parsed_title']}")
# # # # # # # # # # #
# # # # # # # # # # # # Number of cropped images to display
# # # # # # # # # # # num_images = len(file_info)
# # # # # # # # # # #
# # # # # # # # # # # # Determine the grid size for subplots
# # # # # # # # # # # cols = 4  # Number of columns in the grid
# # # # # # # # # # # rows = (num_images + cols - 1) // cols  # Calculate the number of rows required
# # # # # # # # # # #
# # # # # # # # # # # # Create a figure for the subplots
# # # # # # # # # # # fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# # # # # # # # # # #
# # # # # # # # # # # # Flatten axes for easy indexing
# # # # # # # # # # # axes = axes.flatten()
# # # # # # # # # # #
# # # # # # # # # # # # Display each cropped image in a subplot
# # # # # # # # # # # for i, info in enumerate(file_info):
# # # # # # # # # # #     axes[i].imshow(info['cropped_image'], cmap='gray')
# # # # # # # # # # #
# # # # # # # # # # #     # Add text to the cropped image at adjustable position
# # # # # # # # # # #     axes[i].text(text_x, info['cropped_image'].shape[0] - text_y, info['parsed_title'], color='white', fontsize=12,
# # # # # # # # # # #                  ha='left', va='bottom')
# # # # # # # # # # #     axes[i].axis('off')  # Hide axes for a cleaner look
# # # # # # # # # # #
# # # # # # # # # # # # Hide unused subplots
# # # # # # # # # # # for j in range(i + 1, len(axes)):
# # # # # # # # # # #     axes[j].axis('off')
# # # # # # # # # # #
# # # # # # # # # # # # Display the plot
# # # # # # # # # # # plt.tight_layout()
# # # # # # # # # # # plt.show()
