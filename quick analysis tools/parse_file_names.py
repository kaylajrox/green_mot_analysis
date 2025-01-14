import os
import re

# Specify the main directory where subfolders are located
main_folder_path = "../data/20250117_first_data"  # Replace with your directory path


# Function to parse folder name into readable format
def parse_folder_name(folder_name):
    # Check for the "background" folder
    if "background" in folder_name.lower():
        return "background"

    # Match folders like "2s_after_ramp_green_mot" or "1_2s_after_ramp_green_mot"
    match = re.match(r"(\d+)_?(\d*)s?", folder_name)

    if match:
        # Handle the case where there is just a number followed by "s" (e.g., "2s", "1s", etc.)
        if match.group(2):  # If there's a second part like "1_2"
            fraction = f"{match.group(1)}/{match.group(2)}"
            return f"t={fraction}"  # e.g., "t=1/2"
        else:
            return f"t={match.group(1)}s"  # e.g., "t=2s"

    # If no match, check if it starts with "zeeman" or other formats that should be excluded
    if "zeeman" in folder_name.lower():
        return None  # Skip this folder or label as None

    # If no match and folder doesn't start with valid format, return "Unknown"
    return "Unknown"


# List to store the file paths, folder names, and image data
file_info = []

# Iterate through the subfolders in the main directory
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Check if it is a folder
    if os.path.isdir(subfolder_path):
        # Parse the folder name for experiment information
        parsed_title = parse_folder_name(subfolder_name)

        # Skip folders with invalid or unknown titles (like "zeeman")
        if parsed_title is None or parsed_title == "Unknown":
            continue

        # Add the folder information to file_info
        file_info.append({
            "folder_name": subfolder_name,
            "parsed_title": parsed_title,
            "folder_path": subfolder_path
        })

# Print or process the extracted information
for info in file_info:
    print(f"Folder: {info['folder_name']} -> Parsed Title: {info['parsed_title']} -> Path: {info['folder_path']}")
