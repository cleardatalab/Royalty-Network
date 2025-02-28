import pandas as pd

# Load the Excel file
file_path = 'Brian Thompson - Schedule A.xlsx'

# Read the data, specifying that the header is in the third row (index 2)
df = pd.read_excel(file_path, header=2)

# Check for the presence of the specified columns
required_columns = ['Song', 'Work Title', 'Interested Parties', 'Role']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f'The following columns are missing: {', '.join(missing_columns)}')
else:
    print('All required columns are present.')

    # Define which columns to process
    column_processing = {'Song': 1, 'Title': 1, 'Composer': 1, 'Publisher': 1}

    # Initialize a new DataFrame for the processed data
    processed_data = pd.DataFrame(columns=[col for col, process in column_processing.items() if process == 1])

    # Process the data
    for title, group in df.groupby('Work Title'):
        # Get unique title
        unique_title = title

        # Combine interested parties for Composer, excluding those with Role 'E'
        if column_processing.get('Composer', 0) == 1:
            composers = group[group['Role'] != 'P']['Interested Parties'].tolist()
            composer_str = ', '.join(set(composers))  # Unique names
        else:
            composer_str = ''

        # Combine interested parties for Publisher based on Role 'E'
        if column_processing.get('Publisher', 0) == 1:
            publishers = group[group['Role'] == 'P']['Interested Parties'].tolist()
            publisher_str = ', '.join(set(publishers))  # Unique names
        else:
            publisher_str = ''

        # Create a dictionary for the new row, including only the columns that are set to process
        new_row_data = {col: val for col, val in [('Song', group['Song'].values[0]), ('Title', unique_title), ('Composer', composer_str), ('Publisher', publisher_str)] if col in column_processing and column_processing[col] == 1}

        # Append the processed row to the new DataFrame
        new_row = pd.DataFrame([new_row_data])
        processed_data = pd.concat([processed_data, new_row], ignore_index=True)

    # Reorder columns to put 'Song' before 'Title'
    processed_data = processed_data[['Song', 'Title', 'Composer', 'Publisher']]

    # Save the processed data to a new Excel file
    output_file_path = 'Brian Thompson - Processed_Data.xlsx'
    try:
        processed_data.to_excel(output_file_path, index=False)
        print('Processed data has been saved to', output_file_path)
    except Exception as e:
        print(f'Error saving processed data: {e}')