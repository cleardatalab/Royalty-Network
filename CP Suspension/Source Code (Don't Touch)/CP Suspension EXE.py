import os
import pandas as pd  # type: ignore
import tkinter as tk
from tkinter import filedialog
import time
import warnings
import sys  # Import sys to exit the program

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore")

def exit_program():
    """Wait for 5 seconds before closing the program."""
    print("\nClosing the program in 5 seconds...")
    time.sleep(5)
    sys.exit()

def check_and_rename_columns(df, file_type="input"):
    """Rename columns for input files and check missing columns."""
    input_columns = {
        "Master Group": "INPUTINDEX",
        "Song Title": "Input_Title",
        "Song Composers": "Input_Composer",
        "Publisher": "Input_Publisher",
        "Artist": "Input_Artist"
    }

    missing_columns = []
    for original, new in input_columns.items():
        if original not in df.columns:
            missing_columns.append(original)
        else:
            df.rename(columns={original: new}, inplace=True)

    return df

def check_and_rename_cp_columns(df):
    """Rename columns for CP files and check missing columns."""
    cp_columns = {
        "Code": "Cp_Code",
        "Title": "CP_Title",
        "Composer": "Cp_Composers",
        "Main Artist": "Cp_Artists",
        "Publishers": "Cp_Publisher"
    }

    missing_columns = []
    for original, new in cp_columns.items():
        if original not in df.columns:
            missing_columns.append(original)
        else:
            df.rename(columns={original: new}, inplace=True)

    return df

def merge_cp_files(cp_file1, cp_file2):
    """Merge two CP files and rename necessary columns."""
    print("---------------------------\nProcessing files...")

    try:
        cp_df1 = pd.read_csv(cp_file1, encoding='latin1') if cp_file1.endswith('.csv') else pd.read_excel(cp_file1)
        cp_df2 = pd.read_csv(cp_file2, encoding='latin1') if cp_file2.endswith('.csv') else pd.read_excel(cp_file2)
    except Exception as e:
        print(f"ERROR: Unable to read CP files. Details: {e}")
        input("Please take a screenshot and report to Monil!")
        exit_program()

    cp_df1.rename(columns={"Code": "Cp_Code", "Title": "CP_Title", "Composer": "Cp_Composers"}, inplace=True, errors="ignore")
    cp_df2.rename(columns={"Code": "Cp_Code", "Title": "CP_Title", "Composer": "Cp_Composers"}, inplace=True, errors="ignore")

    merged_cp_df = pd.merge(cp_df1, cp_df2, on=["Cp_Code", "CP_Title", "Cp_Composers"], how="outer", suffixes=("", "_drop"))
    merged_cp_df = merged_cp_df.loc[:, ~merged_cp_df.columns.str.endswith('_drop')]
    merged_cp_df = check_and_rename_cp_columns(merged_cp_df)

    return merged_cp_df

def create_folders_and_save_files(destination_folder, input_df, merged_cp_df):
    """Create folders and save generated CSV files."""
    input_combinations = [
        (['INPUTINDEX', 'Input_Title', 'Input_Artist'], 'Input_Title_Artist.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Composer'], 'Input_Title_Composer.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Composer', 'Input_Artist'], 'Input_Title_Composer_Artist.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Composer', 'Input_Publisher'], 'Input_Title_Composer_Publisher.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Composer', 'Input_Publisher', 'Input_Artist'], 'Input_Title_Composer_Publisher_Artist.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Publisher'], 'Input_Title_Publisher.csv'),
        (['INPUTINDEX', 'Input_Title', 'Input_Publisher', 'Input_Artist'], 'Input_Title_Publisher_Artist.csv')
    ]

    for cols, filename in input_combinations:
        input_combined_df = input_df[cols]
        folder_name = filename.replace('Input_', '').replace('.csv', '')
        folder_path = os.path.join(destination_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        input_combined_df.to_csv(os.path.join(folder_path, filename), index=False)

    cp_combinations = [
        (['Cp_Code', 'CP_Title', 'Cp_Artists'], 'CP_Title_Artist.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Composers'], 'CP_Title_Composer.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Composers', 'Cp_Artists'], 'CP_Title_Composer_Artist.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Composers', 'Cp_Publisher'], 'CP_Title_Composer_Publisher.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Composers', 'Cp_Publisher', 'Cp_Artists'], 'CP_Title_Composer_Publisher_Artist.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Publisher'], 'CP_Title_Publisher.csv'),
        (['Cp_Code', 'CP_Title', 'Cp_Publisher', 'Cp_Artists'], 'CP_Title_Publisher_Artist.csv')
    ]

    for cols, filename in cp_combinations:
        try:
            if all(col in merged_cp_df.columns for col in cols):
                cp_combined_df = merged_cp_df[cols]
                folder_name = filename.replace('CP_', '').replace('.csv', '')
                folder_path = os.path.join(destination_folder, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                cp_combined_df.to_csv(os.path.join(folder_path, filename), index=False)
        except KeyError:
            continue

def select_file(file_type):
    """Open file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title=f"Select the {file_type} file",
        filetypes=[("Excel Files", "*.xlsx;*.xls"), ("CSV Files", "*.csv"), ("TSV Files", "*.tsv")]
    )

    if not file_path:
        print(f"ERROR: No {file_type} file selected. Please try again.")
        exit_program()

    return file_path

def select_folder():
    """Open folder dialog to select a destination folder."""
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title="Select Destination Folder")

    if not folder_path:
        print("ERROR: No destination folder selected. Please try again.")
        exit_program()

    return folder_path

def main():
    """Main function to handle file selection and processing."""
    try:
        print("Please select the input file:")
        time.sleep(3)
        input_file = select_file("input")

        print("Please select the first CP file:")
        time.sleep(3)
        cp_file1 = select_file("cp")

        print("Please select the second CP file:")
        time.sleep(3)
        cp_file2 = select_file("cp")

        print("Please select a destination folder to save the files.")
        time.sleep(3)
        destination_folder = select_folder()

        print("---------------------------\nAll files selected successfully!")
        time.sleep(2)

        merged_cp_df = merge_cp_files(cp_file1, cp_file2)
        if merged_cp_df is None:
            exit_program()

        input_df = pd.read_csv(input_file, encoding='latin1') if input_file.endswith('.csv') else pd.read_excel(input_file)
        input_df = check_and_rename_columns(input_df, file_type="input")

        create_folders_and_save_files(destination_folder, input_df, merged_cp_df)

        print("---------------------------\nProcess completed successfully!")

    except Exception as e:
        print(f"ERROR: Unexpected issue occurred. Details: {e}")
        input("---------------------------\nPlease take a screenshot and report to Monil!")
        exit_program()

if __name__ == "__main__":
    main()
