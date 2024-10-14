import os
import pandas as pd  # type: ignore
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

class LabelFillerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Label Fill")  # Set the title of the application window

        self.cp_files_path = r"C:\Users\mshah\The Royalty Network\Share - Documents\MASTER ADMINISTRATOR FILE\DATA APPS\Monil Data Base (Don't Touch)"
        self.cp_files = [
            os.path.join(self.cp_files_path, "CP_1.csv"),  # Updated to handle Excel files
            os.path.join(self.cp_files_path, "CP_2.csv")
        ]

        self.label_file = None  # Variable to store the chosen label file
        self.save_location = None  # Variable to store the chosen save location

        self.setup_ui()  # Call the method to set up the UI components
        self.check_files()  # Check for CP files

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
        self.run_button = tk.Button(self.root, text="Run", command=self.process_files, state=tk.DISABLED)
        self.run_button.grid(row=6, column=0, padx=10, pady=10, columnspan=2)

        # Progress bar for indicating processing status
        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=7, column=0, padx=10, pady=5, columnspan=2)
        self.progress.grid_remove()  # Initially hide the progress bar

        self.status_label = tk.Label(self.root, text="")  # Label to show processing status
        self.status_label.grid(row=8, column=0, padx=10, pady=5, columnspan=2)

    def check_files(self):
        # Function to check for the existence of CP files and the label file
        cp_files_found = []
        all_files_exist = True  # Flag to track if all files exist

        for cp_file in self.cp_files:
            if os.path.exists(cp_file):
                cp_files_found.append(os.path.basename(cp_file))  # Add found files to the list
            else:
                cp_files_found.append(f"{os.path.basename(cp_file)} (Not Found)")  # Indicate missing files
                all_files_exist = False  # Set flag to False if any file is missing
                
            # Update the UI to show CP files found or not
            if not all_files_exist:
                messagebox.showwarning("Warning", "CP files not found: \n" + "\n".join(cp_files_found))
                self.root.quit()

        # Enable or disable the Run button based on file existence
        if all_files_exist and self.label_file:
            self.run_button.config(state=tk.NORMAL)  # Enable the Run button
        else:
            self.run_button.config(state=tk.DISABLED)  # Disable the Run button

    def choose_label_file(self):
        # Function to choose the label file
        label_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])  # Updated for Excel files
        if label_file:
            self.label_file = label_file  # Store the selected label file
            self.label_file_label.config(text=os.path.basename(label_file))  # Update the display
            self.check_files()  # Check files again after choosing label file

    def choose_save_location(self):
        # Function to choose a directory for saving output files
        folder = filedialog.askdirectory()
        if folder:
            self.save_location = folder  # Store the selected directory
            self.save_location_label.config(text=folder)  # Update the display
            self.check_files()  # Check files again after choosing save location

    def choose_label_file(self):
        # Allow selection of CSV, XLS, and XLSX files
        label_file = filedialog.askopenfilename(filetypes=[("All Supported Files", "*.csv;*.xls;*.xlsx"), 
                                                        ("CSV files", "*.csv"), 
                                                        ("Excel files", "*.xls;*.xlsx")])
        if label_file:
            self.label_file = label_file  # Store the selected label file
            self.label_file_label.config(text=os.path.basename(label_file))  # Update the display
            self.check_files()  # Check files again after choosing label file

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

            # Read the CP files in chunks and concatenate them
            for cp_file in self.cp_files:
                if os.path.exists(cp_file):
                    self.status_label.config(text="Reading data from Counter Point...")  # Show "Counter Point"
                    self.update()  # Update the UI
                    merged_data = pd.read_csv(cp_file, on_bad_lines='skip')  # Read CSV file

            self.progress['value'] = 50  # Update progress
            self.status_label.config(text="Merging data...")  # Update status

            # Strip whitespace from column names in merged_data
            merged_data.columns = merged_data.columns.str.strip()

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
                raise ValueError("Error: The file must include the columns 'Label Name' and 'ISRC' exactly as shown. Please check that the column names are spelled correctly and use the correct uppercase and lowercase letters.")  # Custom error

            # Fill missing Label Names based on the ISRC mapping
            label_data['Label Name'] = label_data['Label Name'].fillna(
                label_data['ISRC'].map(merged_data.drop_duplicates(subset=['ISRC']).set_index('ISRC')['Main Album Label'])
            )
            label_data['Label Name'].fillna('No Match Found', inplace=True)  # Fill any remaining NaNs

            output_file = os.path.join(self.save_location, 'final_data_file.xlsx')  # Define output file path
            label_data.to_excel(output_file, index=False)  # Save the final label data to an Excel file

            self.progress['value'] = 100  # Complete progress
            self.status_label.config(text="Processing complete!")  # Update status
            if messagebox.askokcancel("Success", f"Files processed and saved at {self.save_location}. Would you like to open the folder?"):
                os.startfile(self.save_location)  # Open the save location in file explorer

        except ValueError as ve:
            messagebox.showerror("Error", str(ve))  # Show custom ValueError message
        except Exception as e:
            messagebox.showerror("Error", "An unexpected error occurred. Contact Monil: " + str(e))  # Handle other exceptions
            self.reset_app_state()  # Reset the application state
        finally:
            self.progress.grid_remove()  # Hide the progress bar after processing


    def update(self):
        # Helper function to update the UI and process idle tasks
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelFillerApp(root)
    root.mainloop()
