import os
import pandas as pd  # type: ignore

# Columns to keep in the final result
SELECTED_COLUMNS = [
    "Recording 1 ISRC", "Source File", "Recording 1 Display Artist", 
    "Recording 1 Album Title", "Recording 1 Release Date (CWR)", 
    "Recording 1 Label Name", "Composer 1 Surname", "Composer 2 Surname", 
    "Composer 3 Surname", "Composer 4 Surname", "Composer 5 Surname", 
    "Composer 6 Surname", "Composer 7 Surname", "Composer 8 Surname", 
    "Composer 9 Surname", "Publisher 1 Name", "Publisher 2 Name", 
    "Publisher 3 Name", "Publisher 4 Name", "Publisher 5 Name", 
    "Publisher 6 Name", "Publisher 7 Name", "Publisher 8 Name", 
    "Publisher 9 Name", "Composer 1 Share (Total = 100)", 
    "Composer 2 Share (Total = 100)", "Composer 3 Share (Total = 100)", 
    "Composer 4 Share (Total = 100)", "Composer 5 Share (Total = 100)", 
    "Composer 6 Share (Total = 100)", "Composer 7 Share (Total = 100)", 
    "Composer 8 Share (Total = 100)", "Composer 9 Share (Total = 100)"
]

def find_duplicates_and_store_results():
    # Get all .xlsx files in the current folder starting with "CWR"
    xlsx_files = [f for f in os.listdir('.') if f.startswith("CWR") and f.endswith(".xlsx")]

    if not xlsx_files:
        print("No files found starting with 'CWR'.")
        return None  # Exit if no files are found

    all_data = []

    for file in xlsx_files:
        df = pd.read_excel(file)
        df['Source File'] = os.path.basename(file)  # Add source file column
        all_data.append(df[SELECTED_COLUMNS])

    combined_df = pd.concat(all_data, ignore_index=True)

    # Drop rows where ISRC is NaN
    combined_df = combined_df.dropna(subset=["Recording 1 ISRC"])

    # Clean ISRC values (remove spaces and convert to uppercase)
    combined_df["Recording 1 ISRC"] = combined_df["Recording 1 ISRC"].str.strip().str.upper()

    # Group by ISRC to find duplicates
    grouped = combined_df.groupby("Recording 1 ISRC")
    
    # Filter groups that have more than one occurrence (duplicates)
    duplicate_groups = [group for name, group in grouped if len(group) > 1]

    if not duplicate_groups:
        print("No duplicates found based on Recording 1 ISRC.")
        return None

    # Process each duplicate group to combine source files
    processed_groups = []
    for group in duplicate_groups:
        # Get unique source files as comma-separated string
        source_files = ", ".join(sorted(group["Source File"].unique()))
        
        # Take the first row of the group and update its Source File
        first_row = group.iloc[0].copy()
        first_row["Source File"] = source_files
        processed_groups.append(first_row)

    # Create DataFrame from processed groups
    duplicate_data = pd.DataFrame(processed_groups)

    # Sort by ISRC for better readability
    duplicate_data = duplicate_data.sort_values("Recording 1 ISRC")

    # Save results to a file
    output_file = "RESULT.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        duplicate_data.to_excel(writer, sheet_name="Duplicates", index=False)

    print(f"Duplicate handling completed. Results saved to '{output_file}'.")
    print(f"Found {len(duplicate_groups)} groups of duplicates.")

def process_and_remove_duplicates():
    find_duplicates_and_store_results()

if __name__ == "__main__":
    process_and_remove_duplicates()
