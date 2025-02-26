import pandas as pd
import os
import math
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import time
import zipfile

def print_ascii_art():
    art = r"""
  ______ _ _         _____       _ _ _            
 |  ____(_) |       / ____|     | (_) |           
 | |__   _| | ___  | (___  _ __ | |_| |_ ___ _ __ 
 |  __| | | |/ _ \  \___ \| '_ \| | | __/ _ \ '__|
 | |    | | |  __/  ____) | |_) | | | ||  __/ |   
 |_|    |_|_|\___| |_____/| .__/|_|_|\__\___|_|   
                          | |                     
                          |_|                     
    """
    print(art)

print_ascii_art()

# Function to get the file path using a file dialog
def get_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select the file to split",
        filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv"), ("Excel files", "*.xls;*.xlsx")]
    )
    return file_path

# Function to get the directory path for saving split files
def get_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(
        title="Select the destination folder"
    )
    return directory

print("-------------------------------------------------------------")

# Function to open the folder after splitting
def open_folder(directory):
    os.startfile(directory)

# Main function
def split_file():
    # Step 1: User Inputs
    print("Select the file to split...")
    time.sleep(2)
    splitter_file = get_file()

    if not splitter_file:
        print("No file selected. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    
    # Check for valid file format
    if not (splitter_file.endswith('.tsv') or splitter_file.endswith('.csv') or splitter_file.endswith('.xls') or splitter_file.endswith('.xlsx')):
        print("Invalid file format. Please select a .tsv, .csv, .xls, or .xlsx file. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return

    print(f"Selected file: {splitter_file}")  # Show selected file
    
    print("-------------------------------------------------------------")
    
    # Step 2: Load the file
    print("Loading file...")
    try:
        if splitter_file.endswith('.tsv'):
            df = pd.read_csv(splitter_file, sep='\t', encoding='latin1', low_memory=False)
        elif splitter_file.endswith('.csv'):
            df = pd.read_csv(splitter_file, encoding='latin1', low_memory=False)
        else:
            df = pd.read_excel(splitter_file, encoding='latin1', engine='openpyxl')
    except pd.errors.EmptyDataError:
        print("Error: The file is empty. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    except pd.errors.ParserError:
        print("Error: The file is corrupted or improperly formatted. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    except zipfile.BadZipFile:
        print("Error: The Excel file is corrupted or not a valid .xlsx file. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    except Exception as e:
        print(f"Error loading the file: {e}. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return

    # Check if the DataFrame is empty
    if df.empty:
        print("The file is empty. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    
    # Show the number of rows and columns
    row_size, column_size = df.shape
    print(f"This file contain {row_size} rows x {column_size} columns")  # Show file dimensions

    # Step 3: Filter columns with float or int values
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns
    
    if len(numeric_columns) == 0:
        print("No columns with suitable values found. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return

    # Ask for the number of splits with error handling
    while True:
        try:
            print("-------------------------------------------------------------")
            num_splits = int(input("Enter the number of splits: "))
            if num_splits <= 0:
                raise ValueError("Number of splits must be a positive integer.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive integer.")

    print("-------------------------------------------------------------")
    
    # Display numeric columns to the user and include an option for None
    print("File Columns:")
    print("0. None")  # Option for no summation
    for idx, col in enumerate(numeric_columns):
        print(f"{idx + 1}. {col}")

    # Step 4: Compute the total sum of the selected column before splitting (if not None)
    total_sum = None
    column_to_sum = None
    while True:
        column_choice = input("(Optional) Select a column to sum (Enter the column number or name, or 0 for none): ").strip()
        if column_choice.isdigit():
            column_index = int(column_choice) - 1
            if column_index < 0:
                column_to_sum = None  # User chose "None"
                break
            elif column_index < len(numeric_columns):
                column_to_sum = numeric_columns[column_index]
                break
            else:
                print("Invalid column number. Please try again.")
        else:
            if column_choice.lower() == 'none':
                column_to_sum = None  # User chose "None"
                break
            elif column_choice not in numeric_columns:
                print(f"Column '{column_choice}' not found. Please try again.")
            else:
                column_to_sum = column_choice
                break
            
            
    # Step 5: Compute the total sum of the selected column before splitting (if not None)
    total_sum = df[column_to_sum].sum() if column_to_sum else None

    # Step 6: Split the data into smaller chunks
    split_size = math.ceil(len(df) / num_splits)
    
    print("-------------------------------------------------------------")
    
    # Step 7: Ask user to select the destination directory
    print("Select the destination folder to save the split files...")
    time.sleep(2)
    output_dir = get_directory()
    print(f"Selected destination: {output_dir}")
    
    if not output_dir:
        print("No destination folder selected. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    
    if not os.path.exists(output_dir):
        print("Invalid directory. Please select a valid destination folder. Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting
        return
    
    
    # Create the "Split Files" directory inside the selected destination
    split_files_dir = os.path.join(output_dir, "Split Files")
    os.makedirs(split_files_dir, exist_ok=True)
    
    print("-------------------------------------------------------------")

    # Step 8: Save each split to a separate file with a progress bar
    print("Splitting the file...")
    
    for i in tqdm(range(num_splits), desc="Splitting Progress"):
        start_index = i * split_size
        end_index = min(start_index + split_size, len(df))  # Ensure we do not go out of bounds
        split_df = df.iloc[start_index:end_index]
        split_file_name = os.path.join(split_files_dir, f"split_{i + 1}.csv")
        split_df.to_csv(split_file_name, index=False)

    print("-------------------------------------------------------------")
    
    # Step 9: Output the total sum and the number of splits
    if column_to_sum:
        print(f"Total sum of '{column_to_sum}': {total_sum}")
    else:
        print("No summation performed.")
        
    print(f"Total files split: {num_splits}")
    
    print("-------------------------------------------------------------")
    
    # Step 10: Prompt user if they want to open the folder
    user_response = input("Do you want to open the folder with the split files? (yes/no): ").strip().lower()
    
    if user_response == 'yes':
        open_folder(output_dir)
    else:
        print("Exiting...")
        time.sleep(5)  # Wait for 5 seconds before exiting

if __name__ == "__main__":
    split_file()
