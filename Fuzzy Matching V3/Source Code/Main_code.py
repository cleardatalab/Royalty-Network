import pandas as pd  # type: ignore
from rapidfuzz import fuzz  # type: ignore
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np  # type: ignore

# Function to calculate similarity between two strings using multiple measures
def calculate_similarity(input_str, target_str):
    # Ensure inputs are strings and convert to lowercase
    input_str = str(input_str).lower().strip()
    target_str = str(target_str).lower().strip()

    # Calculate various similarity scores
    scores = {
        'token_sort_ratio': fuzz.token_sort_ratio(input_str, target_str),
        'partial_ratio': fuzz.partial_ratio(input_str, target_str),
        'token_set_ratio': fuzz.token_set_ratio(input_str, target_str)
    }
    
    # Return the maximum score obtained from different measures
    return max(scores.values())

# Function to process a batch of input rows in parallel
def process_batch(input_rows, cp_titles, cp_composers, cp_publishers, cp_artists, cp_codes, threshold):
    results = []
    
    for idx, input_row in enumerate(input_rows):
        input_title = input_row['Input_Title']
        input_composer = input_row['Input_Composer']
        input_publisher = input_row['Input_Publisher']
        input_artist = input_row['Input_Artist']

        # Initialize match scores and CP fields
        match_title_score = match_composer_score = match_publisher_score = match_artist_score = 0
        cp_code = cp_title = cp_composer = cp_publisher = cp_artist = "['No Match']"

        # Flags for input non-empty and for full no-match status
        is_input_non_empty = False

        # Check Title
        if pd.isna(input_title) or input_title in ['', "['No Match']", "['Empty']"]:
            match_title_score = 0
        else:
            title_similarities = [
                calculate_similarity(input_title, title) for title in cp_titles
            ]
            match_title_score = max(title_similarities) if title_similarities else 0
            if match_title_score >= threshold:
                best_match_index = title_similarities.index(match_title_score)
                cp_code = cp_codes[best_match_index]
                cp_title = cp_titles[best_match_index]
                cp_composer = cp_composers[best_match_index]
                cp_publisher = cp_publishers[best_match_index]
                cp_artist = cp_artists[best_match_index]
                is_input_non_empty = True  # Found a match

        # Check Composer
        if pd.isna(input_composer) or input_composer in ['', "['No Match']", "['Empty']"]:
            match_composer_score = 0
        else:
            composer_similarities = [
                calculate_similarity(input_composer, composer) for composer in cp_composers
            ]
            match_composer_score = max(composer_similarities) if composer_similarities else 0
            if match_composer_score >= threshold:
                is_input_non_empty = True  # Found a match

        # Check Publisher
        if pd.isna(input_publisher) or input_publisher in ['', "['No Match']", "['Empty']"]:
            match_publisher_score = 0
        else:
            publisher_similarities = [
                calculate_similarity(input_publisher, publisher) for publisher in cp_publishers
            ]
            match_publisher_score = max(publisher_similarities) if publisher_similarities else 0
            if match_publisher_score >= threshold:
                is_input_non_empty = True  # Found a match

        # Check Artist
        if pd.isna(input_artist) or input_artist in ['', "['No Match']", "['Empty']"]:
            match_artist_score = 0
        else:
            artist_similarities = [
                calculate_similarity(input_artist, artist) for artist in cp_artists
            ]
            match_artist_score = max(artist_similarities) if artist_similarities else 0
            if match_artist_score >= threshold:
                is_input_non_empty = True  # Found a match

        # Calculate total accuracy only after all checks
        total_accuracy = (match_title_score + match_composer_score + match_publisher_score + match_artist_score) / 4 if is_input_non_empty else 0.0
        
        # Prepare result row
        result_row = {
            "InputINDEX": input_row['INPUTINDEX'],
            "Input_Title": input_title,
            "Input_Composer": input_row['Input_Composer'],
            "Input_Publisher": input_row['Input_Publisher'],
            "Input_Artist": input_row['Input_Artist'],
            "CP_Code": cp_code,
            "CP_Title": cp_title,
            "Cp_Composers": cp_composer,
            "Cp_Publisher": cp_publisher,
            "Cp_Artists": cp_artist,
            "Match_Title_Score": "0%" if cp_title == "['No Match']" else f"{round(match_title_score)}%",
            "Match_Composer_Score": "0%" if cp_composer == "['No Match']" else f"{round(match_composer_score)}%",
            "Match_Publisher_Score": "0%" if cp_publisher == "['No Match']" else f"{round(match_publisher_score)}%",
            "Match_Artist_Score": "0%" if cp_artist == "['No Match']" else f"{round(match_artist_score)}%",
            "Total_Accuracy": "0.00%" if cp_code == "['No Match']" else f"{total_accuracy:.1f}%"
        }

        # Append the result
        results.append(result_row)

    return results

# Main function to split input data into batches and process them in parallel
def generate_matching_table_parallel(input_df, cp_df, threshold=50, num_workers=4):
    cp_titles = cp_df['CP_Title'].tolist()
    cp_composers = cp_df['Cp_Composers'].tolist()
    cp_publishers = cp_df['Cp_Publisher'].tolist()
    cp_artists = cp_df['Cp_Artists'].tolist()
    cp_codes = cp_df['Cp_Code'].tolist()

    # Split input dataframe into batches for parallel processing
    input_batches = np.array_split(input_df.to_dict(orient='records'), num_workers)

    # Use ProcessPoolExecutor to parallelize the batch processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_batch, batch, cp_titles, cp_composers, cp_publishers, cp_artists, cp_codes, threshold)
            for batch in input_batches
        ]
    
    results = []
    for future in futures:
        results.extend(future.result())
    
    return pd.DataFrame(results)

# Read the input and CP files
def read_files(input_file, cp_file):
    def read_file(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    input_df = read_file(input_file)
    cp_df = read_file(cp_file)
    return input_df, cp_df

# Save results to Excel
def save_results_to_excel(results_df, output_file):
    results_df.to_excel(output_file, index=False)

# Main execution block
if __name__ == "__main__":
    input_file = 'Input Songs.xlsx'
    cp_file = 'CP Data.csv'
    output_file = 'RESULT.xlsx'

    print("\nReading Files...")
    # Read input and CP files
    input_df, cp_df = read_files(input_file, cp_file)
    print("\nReading Files Completed...")

    print("Organizing...")

    # Strip any leading/trailing spaces from the column names
    input_df.columns = input_df.columns.str.strip()
    cp_df.columns = cp_df.columns.str.strip()

    # Strip spaces from object columns
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].str.strip()

    for col in cp_df.columns:
        if cp_df[col].dtype == 'object':
            cp_df[col] = cp_df[col].str.strip()
            
    print("Organizing Completed...")

    print("Processing...")

    # Generate the matching table in parallel
    results_df = generate_matching_table_parallel(input_df, cp_df, threshold=60, num_workers=8)  # Adjusted threshold

    # Replace empty or NaN values with "[Empty]"
    results_df.replace('', '[Empty]', inplace=True)
    results_df.fillna('[Empty]', inplace=True)

    # Save the results to an Excel file
    save_results_to_excel(results_df, output_file)

    print(f"Results saved to {output_file}")
    print("\nDone.")
