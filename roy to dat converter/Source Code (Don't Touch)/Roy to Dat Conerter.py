import os
import shutil
import tkinter as tk
from tkinter import filedialog
import time

def convert_roy_to_dat_copy(roy_file_path, destination_folder):
    # Check if the .roy file exists
    if not os.path.isfile(roy_file_path):
        print(f"The file {roy_file_path} does not exist.")
        return
    
    # Extract the original filename without extension and create the .dat path in the destination folder
    original_filename = os.path.splitext(os.path.basename(roy_file_path))[0]
    dat_file_path = os.path.join(destination_folder, original_filename + '.dat')
    
    # Copy the file to the new destination with the .dat extension
    print("--------------------------------------------------------")
    try:
        shutil.copy(roy_file_path, dat_file_path)
        print(f"File successfully converted to .dat: {dat_file_path}")
    except Exception as e:
        print(f"Error copying file: {e}")

# Function to select the .roy file and destination folder
def select_file_and_destination():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Prompt the user to select a .roy file
    print("Please select a .roy file to convert: ")
    time.sleep(1)
    roy_file_path = filedialog.askopenfilename(
        title="Select a .roy file",
        filetypes=[("ROY files", "*.roy"), ("All files", "*.*")]
    )

    if not roy_file_path:
        print("No .roy file was selected.")
        return
    print("--------------------------------------------------------")
    print(f"Selected file: {roy_file_path}")
    print("--------------------------------------------------------")
    # Prompt the user to select a destination folder
    print("Please select the destination folder for the .dat file.")
    time.sleep(1)
    print("--------------------------------------------------------")
    destination_folder = filedialog.askdirectory(title="Select destination folder")

    if not destination_folder:
        print("No destination folder was selected.")
        return
    print(f"Selected destination folder: {destination_folder}")

    # Convert and save the .dat copy in the chosen destination
    convert_roy_to_dat_copy(roy_file_path, destination_folder)

# Run the file and destination selection function
select_file_and_destination()
