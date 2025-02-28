import pandas as pd # type: ignore
from fuzzywuzzy import fuzz # type: ignore
from fuzzywuzzy import process # type: ignore
import time
import re
pd.options.mode.chained_assignment = None  # default='warn'
import concurrent.futures
import math
import numpy as np # type: ignore
import multiprocessing
import os
from tqdm import tqdm # type: ignore

print(r"""
  ______                     __  __       _       _     _             
 |  ____|                   |  \/  |     | |     | |   (_)            
 | |__ _   _ _________   _  | \  / | __ _| |_ ___| |__  _ _ __   __ _ 
 |  __| | | |_  /_  / | | | | |\/| |/ _` | __/ __| '_ \| | '_ \ / _` |
 | |  | |_| |/ / / /| |_| | | |  | | (_| | || (__| | | | | | | | (_| |
 |_|   \__,_/___/___|\__, | |_|  |_|\__,_|\__\___|_| |_|_|_| |_|\__, |
                      __/ |                                      __/ |
                     |___/                                      |___/
""")

def titlefunc(data):  # longsearch
    # Extract all titles at once
    titles = CPData['CP_Title']
    
    # Perform fuzzy matching on the entire list of titles in one go
    results = process.extract(data, titles, scorer=fuzz.partial_ratio, limit=30)
    
    return results

def filttitlefunc(data):  # customizable short search
    try:
        # Precompute the length of the input data for reuse
        data_len = len(data)
        
        # Filter CPData['CP_Title'] once based on conditions
        filt = CPData['CP_Title'].str.contains(data[0:2], na=False, flags=re.IGNORECASE)
        filt = filt & (CPData['CP_Title'].str.len() / data_len <= 1.80) & (CPData['CP_Title'].str.len() / data_len >= 0.20)
        filt = filt | (CPData['CP_Title'].str.contains(data[0:2], na=False, flags=re.IGNORECASE) & (data_len / CPData['CP_Title'].str.len() <= 1.80) & (data_len / CPData['CP_Title'].str.len() >= 0.20))

        # Apply the fuzzy matching to only the filtered subset of titles
        filtered_titles = CPData['CP_Title'][filt]
        
        return process.extract(data, filtered_titles, scorer=fuzz.token_sort_ratio, limit=30)

    except Exception as e:
        # Fall back to a full search if an error occurs
        # print(f"Error: {e}")
        # input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return process.extract(data, CPData['CP_Title'], scorer=fuzz.token_sort_ratio, limit=30)

def compfuncaxis1(data):
    # Extract the composer data from CPData
    composers = CPData['Cp_Composers'][data[1]]

    # Perform fuzzy matching on the entire list of composer in one go
    results = process.extract(data[0], composers, scorer=fuzz.partial_ratio, limit=30)
    
    return results

def pubfuncaxis1(data):
    # Extract the publisher data from CPData
    publishers = CPData['Cp_Publisher'][data[1]]

    # Perform fuzzy matching on the entire list of publishers in one go
    results = process.extract(data[0], publishers, scorer=fuzz.token_sort_ratio, limit=30)

    return results

def artfuncaxis1(data):
    # Extract the artists data from CPData
    artists = CPData['Cp_Artists'][data[1]]

    # Perform fuzzy matching on all artists in the list in one go
    results = process.extract(data[0], artists, scorer=fuzz.token_sort_ratio, limit=30)

    return results

def scorer(data):
    # Compute title scores
    titlescores = [i[1] for i in sorted(data[0], key=lambda x: x[2])]

    # Use pre-configured logic for scores
    compscores = score_logic["composer"](data)
    pubscores = score_logic["publisher"](data)
    artistscores = score_logic["artist"](data)

    # Calculate result list
    res_list = [sum(i) for i in zip(titlescores, compscores, pubscores, artistscores)]
    if not res_list:
        return 0
    index_max = max(range(len(res_list)), key=res_list.__getitem__)

    # Retrieve the highest score and related data
    highcodelookup = sorted(data[0], key=lambda x: x[2])[index_max]

    highscoreresult = (
        CPData['Cp_Code'][highcodelookup[2]],
        CPData['CP_Title'][highcodelookup[2]],
        CPData['Cp_Composers'][highcodelookup[2]],
        CPData['Cp_Publisher'][highcodelookup[2]],
        CPData['Cp_Artists'][highcodelookup[2]],
        max(res_list),
        titlescores[index_max],
        compscores[index_max],
        pubscores[index_max],
        artistscores[index_max]
    )

    # Basic matching criteria
    match = []
    if highscoreresult[6] == 100 and highscoreresult[7] == 100 and highscoreresult[8] == 100 and highscoreresult[9] == 0:
        match.append("Perfect Match!")
    if highscoreresult[6] > 90 and highscoreresult[7] > 55 and highscoreresult[8] > 55:
        match.append('Possible Match - title writer publisher 90/55/55+!')
    if highscoreresult[6] > 60 and highscoreresult[7] > 60 and highscoreresult[8] > 60 and highscoreresult[9] > 60:
        match.append("Suggested Match - all 60 or above!")
    if highscoreresult[6] > 86 and highscoreresult[7] > 60:
        match.append('Suggested Match - title and writer 86/60+!')
    if highscoreresult[6] > 90 and highscoreresult[8] > 89:
        match.append('Possible Match - title publisher 90/90+!')
    if highscoreresult[6] > 95:
        match.append('Suggested Match - title 95+!')

    # Precompile regex for replacing
    replace_pattern = re.compile(r'[()/,]')

    # Use sets for matching to ensure faster lookup
    def process_input_and_cp(input_text, cp_text):
        input_set = set(replace_pattern.sub(' ', input_text).split())
        cp_set = set(replace_pattern.sub(' ', cp_text).split())
        match_count = len(input_set & cp_set)
        not_found_count = len(input_set) - match_count
        return match_count, not_found_count, len(input_set), len(cp_set)

    # Title matching
    tmatch, tnotfound, len_inputtitle, len_cptitle = process_input_and_cp(data[4], highscoreresult[1])
    TitlePercentageofCPdatainInput = tmatch / len_cptitle if len_cptitle else 0
    TitlePercentageofInputinCpdata = tmatch / len_inputtitle if len_inputtitle else 0

    # Composer matching
    cmatch, cnotfound, len_inputcomposer, len_cpcomposer = process_input_and_cp(data[5], highscoreresult[2])
    WriterPercentageofCPdatainInput = cmatch / len_cpcomposer if len_cpcomposer else 0
    WriterPercentageofInputinCpdata = cmatch / len_inputcomposer if len_inputcomposer else 0

    # Publisher matching
    pmatch, pnotfound, len_inputpublisher, len_cppublisher = process_input_and_cp(data[6], highscoreresult[3])
    PublisherPercentageofCPdatainInput = pmatch / len_cppublisher if len_cppublisher else 0
    PublisherPercentageofInputinCpdata = pmatch / len_inputpublisher if len_inputpublisher else 0

    # Artist matching
    amatch, anotfound, len_inputartist, len_cpartist = process_input_and_cp(data[7], highscoreresult[4])
    ArtistPercentageofCPdatainInput = amatch / len_cpartist if len_cpartist else 0
    ArtistPercentageofInputinCpdata = amatch / len_inputartist if len_inputartist else 0

    return (
        highscoreresult[0], highscoreresult[1], highscoreresult[2], highscoreresult[3], highscoreresult[4],
        highscoreresult[5], highscoreresult[6], highscoreresult[7], highscoreresult[8], highscoreresult[9],
        match, tmatch, tnotfound, TitlePercentageofCPdatainInput, TitlePercentageofInputinCpdata,
        cmatch, cnotfound, WriterPercentageofCPdatainInput, WriterPercentageofInputinCpdata,
        pmatch, pnotfound, PublisherPercentageofCPdatainInput, PublisherPercentageofInputinCpdata,
        amatch, anotfound, ArtistPercentageofCPdatainInput, ArtistPercentageofInputinCpdata
    )

def normalscript(position):
    # Unpack start, end positions
    start, end = position[0], position[1]
    
    # Slice the relevant portion of the dataframe
    inputsheet_slice = inputsheet.iloc[start:end]

    # Calculate TitleScore and TitleCodeList
    title_scores = inputsheet_slice['Input_Title'].apply(filttitlefunc)
    title_code_list = title_scores.apply(lambda x: [i[2] for i in x])

    # Add TitleScore and TitleCodeList to the sliced dataframe
    inputsheet_slice = inputsheet_slice.assign(
        TitleScore=title_scores,
        TitleCodeList=title_code_list
    )

    # Pre-calculate CompScore, PubScore, and ArtistScore in one go
    comp_scores = inputsheet_slice.apply(lambda x: compfuncaxis1((x['Input_Composer'], x['TitleCodeList'])), axis=1)
    pub_scores = inputsheet_slice.apply(lambda x: pubfuncaxis1((x['Input_Publisher'], x['TitleCodeList'])), axis=1)
    artist_scores = inputsheet_slice.apply(lambda x: artfuncaxis1((x['Input_Artist'], x['TitleCodeList'])), axis=1)

    # Add these scores to the sliced dataframe
    inputsheet_slice = inputsheet_slice.assign(
        CompScore=comp_scores,
        PubScore=pub_scores,
        ArtistScore=artist_scores
    )

    # Apply the scorer function
    scorer_results = inputsheet_slice.apply(
        lambda x: scorer((
            x['TitleScore'], x['CompScore'], x['PubScore'], x['ArtistScore'],
            x['Input_Title'], x['Input_Composer'], x['Input_Publisher'], x['Input_Artist']
        )),
        axis=1
    )

    # Convert scorer results into individual columns
    scorer_columns = [
        'MatchCode', 'MatchTitle', 'MatchComposers', 'MatchPublishers', 'MatchArtists',
        'MatchTotalScore', 'BestTitleScore', 'BestComposerScore', 'BestPublisherScore', 'BestArtistScore',
        'MatchProfile', 'TMatch', 'TNotFound', 'TPercentageofCPdatainInput', 'TPercentageofInputinCpdata',
        'CMatch', 'CNotFound', 'CPercentageofCPdatainInput', 'CPercentageofInputinCpdata',
        'PMatch', 'PNotFound', 'PPercentageofCPdatainInput', 'PPercentageofInputinCpdata',
        'AMatch', 'ANotFound', 'APercentageofCPdatainInput', 'APercentageofInputinCpdata'
    ]
    scorer_df = pd.DataFrame(scorer_results.tolist(), columns=scorer_columns, index=inputsheet_slice.index)

    # Add all columns, including 'Scorer', to the sliced dataframe
    inputsheet_slice = pd.concat([inputsheet_slice, scorer_df], axis=1)

    # Return the modified slice
    return inputsheet_slice

def normalscriptshort():
    # Calculate all scores first, without applying them to the dataframe yet
    title_scores = inputsheet['Input_Title'].apply(filttitlefunc)
    title_code_list = title_scores.apply(lambda x: [i[2] for i in x])

    # Calculate all individual scores (composer, publisher, artist) and add them to the dataframe
    inputsheet['TitleScore'] = title_scores
    inputsheet['TitleCodeList'] = title_code_list

    # Apply functions in one go for CompScore, PubScore, and ArtistScore
    inputsheet[['CompScore', 'PubScore', 'ArtistScore']] = inputsheet[['Input_Composer', 'Input_Publisher', 'Input_Artist', 'TitleCodeList']].apply(
        lambda x: pd.Series({
            'CompScore': compfuncaxis1((x['Input_Composer'], x['TitleCodeList'])),
            'PubScore': pubfuncaxis1((x['Input_Publisher'], x['TitleCodeList'])),
            'ArtistScore': artfuncaxis1((x['Input_Artist'], x['TitleCodeList']))
        }), axis=1)
    
    # Apply functions to add Scorer column
    inputsheet['Scorer'] = inputsheet[['TitleScore', 'CompScore', 'PubScore', 'ArtistScore',
                                       'Input_Title', 'Input_Composer', 'Input_Publisher', 'Input_Artist']].apply(scorer, axis=1)

    # Apply scorer function in one go, and extract necessary fields into new columns
    inputsheet[['MatchCode', 'MatchTitle', 'MatchComposers', 'MatchPublishers', 'MatchArtists',
                'MatchTotalScore', 'BestTitleScore', 'BestComposerScore', 'BestPublisherScore', 'BestArtistScore',
                'MatchProfile', 'TMatch', 'TNotFound', 'TPercentageofCPdatainInput', 'TPercentageofInputinCpdata',
                'CMatch', 'CNotFound', 'CPercentageofCPdatainInput', 'CPercentageofInputinCpdata',
                'PMatch', 'PNotFound', 'PPercentageofCPdatainInput', 'PPercentageofInputinCpdata',
                'AMatch', 'ANotFound', 'APercentageofCPdatainInput', 'APercentageofInputinCpdata']] = inputsheet[['TitleScore', 'CompScore', 'PubScore', 'ArtistScore',
                                                                                                                         'Input_Title', 'Input_Composer', 'Input_Publisher', 'Input_Artist']].apply(
        lambda x: pd.Series(scorer((x['TitleScore'], x['CompScore'], x['PubScore'], x['ArtistScore'], x['Input_Title'],
                                    x['Input_Composer'], x['Input_Publisher'], x['Input_Artist']))), axis=1)

    return inputsheet

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_data_files():
    # Open a file dialog for the user to choose the input file
    Tk().withdraw()  # Hide the root window
    
    # Prompt user to select the input songs file
    print("Select the input songs file (CSV/XLSX/XLS).")
    time.sleep(2)
    input_file_path = askopenfilename(
        title="Select the input songs file",
        filetypes=[("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv")]
    )
    
    print('Loading...')
    time.sleep(1)
    
    if not input_file_path:
        print('--------------------------------------------------')
        print("No input file selected.")
        input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return None, None

    # Load the input songs file
    try:
        if input_file_path.endswith('.csv'):
            inputsheet = pd.read_csv(input_file_path, dtype=str)
        elif input_file_path.endswith(('.xlsx', '.xls')):
            inputsheet = pd.read_excel(input_file_path, dtype=str)
        else:
            raise ValueError("Unsupported input file format. Please select a CSV or Excel file.")
        print(f"Read input file: {input_file_path}")
        inputsheet.fillna('', inplace=True)
    except Exception as e:
        print(f"Error reading the input file: {e}")
        return None, None
    
    print('--------------------------------------------------')
    
    # Check if required columns are present in the input file
    required_input_columns = [
        "INPUTINDEX", "Input_Title", "Input_Composer", "Input_Publisher", "Input_Artist"
    ]
    missing_input_columns = [col for col in required_input_columns if col not in inputsheet.columns]
    if missing_input_columns:
        print("Error: The input file is missing the following required columns:")
        print(", ".join(missing_input_columns))
        print("Please provide a valid input file.")
        input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return None, None
    
    # Prompt user to select the CP data file
    print("Select the CP data file (CSV/XLSX/XLS).")
    time.sleep(2)
    cp_file_path = askopenfilename(
        title="Select the CP data file",
        filetypes=[("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv")]
    )
    
    print('Loading...')
    time.sleep(1)
    
    if not cp_file_path:
        print('--------------------------------------------------')
        print("No CP data file selected.")
        input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return inputsheet, None

    # Load the CP data file
    try:
        if cp_file_path.endswith('.csv'):
            CPData = pd.read_csv(cp_file_path, dtype=str)
        elif cp_file_path.endswith(('.xlsx', '.xls')):
            CPData = pd.read_excel(cp_file_path, dtype=str)
        else:
            raise ValueError("Unsupported CP file format. Please select a CSV or Excel file.")
        print(f"Read CP data file: {cp_file_path}")
        CPData.fillna('', inplace=True)
    except Exception as e:
        print(f"Error reading the CP file: {e}")
        return inputsheet, None
    
    # Check if required columns are present in the CP data file
    required_cp_columns = [
        "Cp_Code", "CP_Title", "Cp_Composers", "Cp_Publisher", "Cp_Artists"
    ]
    missing_cp_columns = [col for col in required_cp_columns if col not in CPData.columns]
    if missing_cp_columns:
        print("Error: The CP data file is missing the following required columns:")
        print(", ".join(missing_cp_columns))
        print("Please provide a valid CP data file.")
        input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return inputsheet, None
    
    print('--------------------------------------------------')
    return inputsheet, CPData

# Usage example
try:
    inputsheet, CPData = load_data_files()
    if inputsheet is not None and CPData is not None:
        print('--------------------------------------------------')
        print("Files loaded successfully.")
        # Proceed with further processing
except Exception as e:
    print('--------------------------------------------------')
    print(f"Error: {e}")
    input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")

from tkinter.filedialog import askdirectory

def select_destination_folder():
    """Prompt the user to select a destination folder."""
    Tk().withdraw()  # Hide the root Tkinter window
    print("Select a destination folder to save the results.")
    time.sleep(2)
    folder_path = askdirectory(title="Select Destination Folder")
    if not folder_path:
        print("No destination folder selected. Please try again.")
        input("-----------------------\nPlease take a screenshot of the error message and report it to Monil.\n-----------------------\n")
        return select_destination_folder()  # Retry prompt
    print(f"Selected destination folder: {folder_path}")
    print('--------------------------------------------------')
    print('Processing...')
    return folder_path

# Check if the entire column is empty
is_column_empty = {
    "composer": inputsheet['Input_Composer'].str.strip().eq('').all(),
    "publisher": inputsheet['Input_Publisher'].str.strip().eq('').all(),
    "artist": inputsheet['Input_Artist'].str.strip().eq('').all()
}

# Pre-configure logic for scores based on column state
score_logic = {
    "composer": lambda data: [0] * 30 if is_column_empty["composer"] else [i[1] for i in sorted(data[1], key=lambda x: x[2])],
    "publisher": lambda data: [0] * 30 if is_column_empty["publisher"] else [i[1] for i in sorted(data[2], key=lambda x: x[2])],
    "artist": lambda data: [0] * 30 if is_column_empty["artist"] else [i[1] for i in sorted(data[3], key=lambda x: x[2])]
}

# Calculate the estimated processing time
print('--------------------------------------------------')
print("Calculating Time...")

# Step 1: Calculate the number of comparisons
num_comparisons = len(inputsheet) * len(CPData)
print(f"Number of comparisons: {num_comparisons}")

# Step 2: Assume you know the time per comparison (in seconds)
time_per_comparison = 0.0000040

# Step 3: Calculate the total estimated time in seconds
estimated_time_seconds = num_comparisons * time_per_comparison

# Check if the estimated time is too small and handle it
if estimated_time_seconds < 1:
    print(f"Estimated processing time: {estimated_time_seconds:.2f} seconds")
else:
    # Step 4: Convert total time into hours, minutes, and seconds
    hours = estimated_time_seconds // 3600
    minutes = (estimated_time_seconds % 3600) // 60
    seconds = estimated_time_seconds % 60

    # Step 5: Format the estimated time in hh:mm:ss format
    formatted_time = f"{int(hours):02}Hr:{int(minutes):02}Min:{int(seconds):02}Sec"

    print('--------------------------------------------------')
    print("The estimated time is approximate and may vary, completing sooner or later depending on your system's configuration and performance. \n(Note: The estimation is based on the file size and may not reflect real-time processing accurately.)")
    
    # Step 6: Display the estimated processing time
    print(f"Estimated processing time: {formatted_time}")
    
    print('--------------------------------------------------')
    input("Press Enter To Continue...")
    print('--------------------------------------------------')
    print("Loading...")


x = len(inputsheet)+1#adding the plus one seems to remove the problem of cutting off the last title
processors = multiprocessing.cpu_count()
#normalscript([0,5])
t = time.localtime()    


# Define the words to be removed from the columns
remove_words = ["UNKNOWN", "COMPOSER", "/", "WRITER", " *JRM*"]

# Clean 'Cp_Composers' column in one step
CPData['Cp_Composers'] = CPData['Cp_Composers'].str.replace('|'.join(remove_words), ' ', regex=True)

# Clean 'CP_Title' column in one step
CPData['CP_Title'] = CPData['CP_Title'].str.replace(' *JRM*', ' ', regex=True).str.replace("WRITER", ' ', regex=True)

# Clean 'Input_Title' column in one step
inputsheet['Input_Title'] = inputsheet['Input_Title'].str.replace(' *JRM*', '', regex=True)

# Clean 'Input_Composer' column in one step
inputsheet['Input_Composer'] = inputsheet['Input_Composer'].str.replace('|'.join(remove_words), ' ', regex=True)


# Create a regex pattern once, using the '|' to join words
remove_words = ["music", "publishing", "limited", "ltd", "edition"]
pat = r'(?i)\b(?:' + '|'.join(remove_words) + r')\b'

# Clean the 'Input_Publisher' column in one go
inputsheet['Input_Publisher'] = inputsheet['Input_Publisher'].str.lower().replace(pat, '', regex=True)

# Clean the 'Cp_Publisher' column in one go
CPData['Cp_Publisher'] = CPData['Cp_Publisher'].str.lower().replace(pat, '', regex=True)


# Record start time for processing
start_time = time.time()

import warnings
import logging

if __name__ == '__main__':
    # Suppress general warnings
    warnings.filterwarnings("ignore")

    # Configure logging to suppress messages from fuzzywuzzy and other libraries
    logging.getLogger().setLevel(logging.ERROR)

    # Initialize the tqdm progress bar within pandas
    tqdm.pandas()
    current_time1 = time.strftime("%b-%d-%Y_%H_%M_%S", time.localtime())

    try:
        # Prompt the user to select a destination folder
        destination_folder = select_destination_folder()
        print(destination_folder)
        
        # Split the range into processors
        q = np.array_split(processors)

        # Parallel execution using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(normalscript, q)

        # Concatenate results and save to Excel
        allresults = pd.concat(results)
        result_file_path = f"{destination_folder}/RESULTS.xlsx"
        allresults.to_excel(result_file_path, index=False)
        print('--------------------------------------------------')
        print(f"Results saved to: {result_file_path}")

    except Exception:
        # Log the exception silently without showing details
        try:
            current_time1 = time.strftime("%b-%d-%Y_%H_%M_%S", time.localtime())
            allresults = normalscriptshort()
            result_file_path = f"{destination_folder}/RESULTS.xlsx"
            allresults.to_excel(result_file_path, index=False)
            print('--------------------------------------------------')
            print(f"Results saved to: {result_file_path}")
        except:
            # Suppress fallback errors silently
            pass


# After processing, record the end time
end_time = time.time()

# Calculate the time taken for the process
time_taken_seconds = end_time - start_time

# Convert time taken into hours, minutes, and seconds
if time_taken_seconds < 1:
    print('--------------------------------------------------')
    print(f"Time taken: {time_taken_seconds:.2f} seconds")
else:
    hours_taken = time_taken_seconds // 3600
    minutes_taken = (time_taken_seconds % 3600) // 60
    seconds_taken = time_taken_seconds % 60

    formatted_time_taken = f"{int(hours_taken):02}:{int(minutes_taken):02}:{int(seconds_taken):02}"
    print('--------------------------------------------------')
    print(f"Time taken: {formatted_time_taken}")
    
input("Press enter to exit....")