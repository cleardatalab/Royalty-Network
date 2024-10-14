# Label Filler Application - User Guide

## Overview

The **Label Filler Application** is designed to process CSV files with missing labels and automatically fill them based on the ISRC. It also handles two internal files (`CP_1` and `CP_2`) located in a specific folder on your computer.

----------------------------------------------------------------------------------------------------------------------------------------

## Step-by-Step Instructions

1. Open the Application by double-clicking the icon.
2. Select a Label File by clicking "Choose Label File," navigating to your CSV, and clicking "Open."
3. Choose a Save Location by clicking "Choose Save Location" and selecting the desired folder.
4. Click the "Run" button to start processing the file.
5. View the results once completed; the output will be saved as final_data_file.csv.

----------------------------------------------------------------------------------------------------------------------------------------

## Important Requirements

Before running the Label Filler Application, ensure that your files meet the following requirements:

1. **Label File (CSV Format)**: 
   - The CSV file you upload must have two specific columns: **“ISRC”** and **“Label Name.”**
   - These columns are **case-sensitive**, which means they must be spelled exactly like this:  
     - **ISRC**  
     - **Label Name**
   - Any file that does not include these two columns will result in an error.

2. **CP Files**:
   - The application relies on two files named **CP_1** and **CP_2**, which are automatically loaded from the directory:  
     `C:\Users\mshah\The Royalty Network\Share - Documents\MASTER ADMINISTRATOR FILE\DATA APPS\Monil Data Base (Don't Touch)`
   - These files should be present in this location for the app to work properly. You do not need to select them manually.

----------------------------------------------------------------------------------------------------------------------------------------

## Troubleshooting

- If you see an error message, check that your label file has both **“ISRC”** and **“Label Name”** columns spelled correctly and ensure that the CP files exist in the required folder.
- Make sure you’ve selected the right file and save location before running the application.

----------------------------------------------------------------------------------------------------------------------------------------

## Contact
 - If you experience any issues with the ISRC column or need additional support, please contact Monil for assistance.