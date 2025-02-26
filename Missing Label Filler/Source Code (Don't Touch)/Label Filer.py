import os
import sys
import glob
import threading  # Import threading module
import pandas as pd  # type: ignore
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar


# Function to get the base path, depending on whether it's a bundled executable or script
def get_base_path():
    if getattr(sys, 'frozen', False):
        # If running as a bundled .exe, get the base path using sys._MEIPASS
        return os.path.dirname(sys.executable)
    else:
        # If running as a script, use the current working directory
        return os.getcwd()


class LabelFillerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Label Fill")  # Set the title of the application window

        # Dynamically construct the CP files path by replacing the device name
        self.cp_files_path = os.path.join(os.path.expanduser("~"), r"The Royalty Network\Share - Documents\MASTER ADMINISTRATOR FILE\DATA APPS\Monil Data Base (Don't Touch)")

        # List of required CP files
        self.required_cp_files = ["CP_1.csv", "CP_2.csv"]

        self.label_file = None  # Variable to store the chosen label file
        self.save_location = None  # Variable to store the chosen save location

        self.setup_ui()  # Set up the UI components
        self.check_files()  # Check if necessary files exist

    def setup_ui(self):
        # UI components for displaying label file information
        tk.Label(self.root, text="Label File:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.label_file_label = tk.Label(self.root, text="No file selected")
        self.label_file_label.grid(row=1, column=0, padx=10, pady=5, columnspan=2, sticky='we')

        # Instruction about column names
        tk.Label(self.root, text="The file must have 'Label Name' and 'ISRC' columns with exact spelling.", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=10, pady=5, columnspan=2, sticky='we')

        # Button for choosing the label file
        tk.Button(self.root, text="Choose Label File", command=self.choose_label_file).grid(row=4, column=0, padx=10, pady=5)

        # Button for choosing the save location
        tk.Button(self.root, text="Choose Save Location", command=self.choose_save_location).grid(row=4, column=1, padx=10, pady=5)
        self.save_location_label = tk.Label(self.root, text="No location selected")
        self.save_location_label.grid(row=5, column=0, padx=10, pady=5, columnspan=2, sticky='we')

        # Button to run the file processing
        self.run_button = tk.Button(self.root, text="Run", command=self.start_processing, state=tk.DISABLED)
        self.run_button.grid(row=6, column=0, padx=10, pady=10, columnspan=2)

        # Progress bar for indicating processing status
        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=7, column=0, padx=10, pady=5, columnspan=2)
        self.progress.grid_remove()  # Initially hide the progress bar

        self.status_label = tk.Label(self.root, text="")  # Label to show processing status
        self.status_label.grid(row=8, column=0, padx=10, pady=5, columnspan=2)

    def check_files(self):
        # Check if all required CP files are available
        missing_files = [file for file in self.required_cp_files if not os.path.exists(os.path.join(self.cp_files_path, file))]

        if missing_files:
            # If any CP files are missing, show a message to the user
            messagebox.showwarning("Missing Files", f"The following CP files are missing:\n" + "\n".join(missing_files))
            self.run_button.config(state=tk.DISABLED)  # Disable the Run button
        else:
            # Enable the Run button only if all required files are present
            self.run_button.config(state=tk.NORMAL)

        # Enable or disable the Run button based on the presence of the label file and CP files
        if not self.label_file or missing_files:
            self.run_button.config(state=tk.DISABLED)  # Disable the Run button if files are missing

    def choose_label_file(self):
        # Function to choose the label file (CSV or Excel)
        label_file = filedialog.askopenfilename(filetypes=[("All Supported Files", "*.csv;*.xls;*.xlsx"),
                                                           ("CSV files", "*.csv"),
                                                           ("Excel files", "*.xls;*.xlsx")])
        if label_file:
            self.label_file = label_file  # Store the selected label file
            self.label_file_label.config(text=os.path.basename(label_file))  # Update the display
            self.check_files()  # Check if the required files exist after choosing label file

    def choose_save_location(self):
        # Function to choose a directory for saving output files
        folder = filedialog.askdirectory()
        if folder:
            self.save_location = folder  # Store the selected directory
            self.save_location_label.config(text=folder)  # Update the display
            self.check_files()  # Check files again after choosing save location

    def start_processing(self):
        # Start the file processing in a separate thread
        threading.Thread(target=self.process_files, daemon=True).start()

    def process_files(self):
        if not self.save_location:
            messagebox.showerror("Error", "Please select a save location.")  # Error if no save location
            return

        self.progress.grid()  # Show the progress bar
        self.progress['value'] = 0  # Initialize progress value
        self.status_label.config(text="Processing files...")  # Update status

        try:
            # Initialize an empty DataFrame for merging
            merged_data = pd.DataFrame()

            # Read the CP files and concatenate them
            for cp_file in self.required_cp_files:
                cp_file_path = os.path.join(self.cp_files_path, cp_file)
                if os.path.exists(cp_file_path):
                    cp_data = pd.read_csv(cp_file_path, on_bad_lines='skip')  # Read CSV file
                    merged_data = pd.concat([merged_data, cp_data], ignore_index=True)  # Merge data
                else:
                    raise FileNotFoundError(f"CP file '{cp_file}' is missing.")  # If file is missing, raise error

            self.progress['value'] = 50  # Update progress
            self.status_label.config(text="Merging data...")  # Update status

            # Read the label file based on its format
            if self.label_file.endswith('.csv'):
                label_data = pd.read_csv(self.label_file)  # Read CSV file
            elif self.label_file.endswith(('.xls', '.xlsx')):
                label_data = pd.read_excel(self.label_file)  # Read Excel file

            # Strip whitespace from column names
            label_data.columns = label_data.columns.str.strip()

            # Check for the exact column names
            required_columns = ['Label Name', 'ISRC']
            if not all(col in label_data.columns for col in required_columns):
                raise ValueError("The file must include the columns 'Label Name' and 'ISRC' exactly as shown.")  # Custom error

            # Fill missing Label Names based on the ISRC mapping
            label_data['Label Name'] = label_data['Label Name'].fillna(
                label_data['ISRC'].map(merged_data.drop_duplicates(subset=['ISRC']).set_index('ISRC')['Main Album Label'])
            )
            label_data['Label Name'].fillna('No Match Found', inplace=True)  # Fill any remaining NaNs

            output_file = os.path.join(self.save_location, 'final_data_file.xlsx')  # Define output file path
            label_data.to_excel(output_file, index=False)  # Save the final label data to an Excel file

            self.progress['value'] = 100  # Complete progress
            self.status_label.config(text="Processing complete!")  # Update status
            if messagebox.askokcancel("Success", f"Files processed and saved at {output_file}.\nWould you like to open the folder?"):
                os.startfile(self.save_location)  # Open folder if user agrees

        except FileNotFoundError as e:
            messagebox.showerror("File Not Found", str(e))  # Show error message for missing files
        except ValueError as e:
            messagebox.showerror("Error", str(e))  # Show error message for invalid data
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")  # Show generic error message
        finally:
            self.progress.grid_remove()  # Hide the progress bar after completion
            self.progress['value'] = 0  # Reset progress value


# Main application runner
if __name__ == "__main__":
    root = tk.Tk()
    app = LabelFillerApp(root)
    root.mainloop()
