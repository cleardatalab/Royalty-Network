import pandas as pd
import pyautogui
import pyperclip
import time
import os
import sys

# Set the path to your screenshots
screenshots_dir = "images"

# Read the Excel data directly
input_file = "Aditya Video Claims 7.19.2024.xlsx"
df = pd.read_excel(input_file)

df['URL'] = df['URL'].ffill()  # Fill empty URLs with the previous valid URL

# Function to wait for a specific image on the screen
def wait_for_image(image_name, timeout=20):
    image_path = os.path.join(screenshots_dir, image_name)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if (location := pyautogui.locateCenterOnScreen(image_path, confidence=0.8)):
            return location
        time.sleep(1)
    raise Exception(f"Timeout: Could not find image {image_name} on screen.")

# Function to convert HH:MM:SS to seconds formatted as "0.00"
def format_time(time_str):
    """Convert time from 'HH:MM:SS to HH:MM:SS' format to seconds formatted as '0.00'."""
    start_time, end_time = time_str.split(' to ')
    start_seconds = convert_to_seconds(start_time)
    end_seconds = convert_to_seconds(end_time)
    return f"{start_seconds // 60}.{start_seconds % 60:02}", f"{end_seconds // 60}.{end_seconds % 60:02}"

def convert_to_seconds(time_str):
    """ Convert HH:MM:SS to total seconds. """
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Function to paste text
def paste_text(text):
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')

# Function to automate claiming
def automate_clamming():
    clamming_url = "https://studio.youtube.com/owner/eJr8cU6VrR4TxH6yrsZUSQ/claims/manual?o=eJr8cU6VrR4TxH6yrsZUSQ&filter=%5B%5D&sort=%7B%22columnType%22%3A%22video%22%2C%22sortOrder%22%3A%22DESCENDING%22%7D"
    
    # Open the clamming URL in Chrome
    os.startfile("chrome.exe")
    pyautogui.hotkey('ctrl', 'l')  # Focus on the address bar
    time.sleep(3)
    paste_text(clamming_url)
    pyautogui.press('enter')
    time.sleep(5)  # Wait for the page to load
    
    last_youtube_url = None  # To keep track of the last valid URL
    last_cp_code = None  # To keep track of the last valid CP Code
    processed_urls = set()  # Set to keep track of processed URLs
    skipped_rows = []  # List to store skipped rows

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        youtube_url = row.iloc[0]
        cp_code = row["CP Code"]  # Change to your actual column name
        
        # Check for empty URL
        if pd.isna(youtube_url):
            if last_youtube_url is None or last_cp_code != cp_code:
                print("Empty URL, skipping...")
                skipped_rows.append(row)  # Store the skipped row
                continue  # Skip if there is no valid last URL or CP Code doesn't match
            else:
                youtube_url = last_youtube_url  # Use the last valid URL
        
        # Check if the URL has already been processed
        if youtube_url in processed_urls:
            print(f"Already processed URL: {youtube_url}, skipping...")
            continue  # Skip this URL if already processed

        last_youtube_url = youtube_url  # Update the last valid URL
        last_cp_code = cp_code  # Update the last valid CP Code
        
        song_used_duration = row.iloc[1]  # Format: "00:00:12 to 00:00:21"

        print(f"Processing URL: {youtube_url} with duration: {song_used_duration}")
        
        # Search for the YouTube URL
        search_bar = wait_for_image("search_bar.png")
        pyautogui.click(search_bar)
        paste_text(youtube_url)
        pyautogui.press('enter')
        print(f"Searched for YouTube URL: {youtube_url}")
        time.sleep(2)

        # Click the dropdown and select asset
        dropdown_button = wait_for_image("dropdown_button.png")
        pyautogui.click(dropdown_button)
        print("Clicked the dropdown button to select the asset.")
        time.sleep(2)
        
        select_asset_button = wait_for_image("select_asset_button.png")
        pyautogui.click(select_asset_button)
        print("Selected the asset.")
        time.sleep(2)

        # Search for CP Code
        select_filter_button = wait_for_image("select_filter_button.png")
        pyautogui.click(select_filter_button)
        paste_text(str(cp_code))
        pyautogui.press('enter')
        time.sleep(2)

        # Select the asset
        radio_button = wait_for_image("radio_button.png")
        pyautogui.click(radio_button)
        time.sleep(2)

        select_button = wait_for_image("select_button.png")
        pyautogui.click(select_button)

        # Handle start and end times
        start_time, end_time = format_time(song_used_duration)
        pyautogui.click(wait_for_image("start_time_input.png"))
        time.sleep(2)
        pyautogui.typewrite(start_time + '\n', interval=0.1)
        print(f"Entered start time: {start_time}")
        
        pyautogui.click(wait_for_image("end_time_input.png"))
        time.sleep(2)
        pyautogui.typewrite(end_time + '\n', interval=0.1)
        print(f"Entered end time: {end_time}")

        # Check for additional claims if any with the same CP_Code
        additional_claims = df[(df.iloc[:, 0] == youtube_url) & (df["CP Code"] == cp_code) & (df.index > index)]  # Get rows after the current one for the same URL and same CP_Code
        
        for _, additional_row in additional_claims.iterrows():
            additional_song_used_duration = additional_row.iloc[1]  # Get the "Song Used Duration"
            print(f"Adding additional claim for duration: {additional_song_used_duration}")

            # Format additional times
            additional_start_time, additional_end_time = format_time(additional_song_used_duration)

            time.sleep(2) # Wait for 2 seconds
            # Click to add another segment
            pyautogui.click(wait_for_image("add_additional_segment_button.png"))
            print("Clicked 'Add Additional Segment' button.")

            # Add a brief delay to ensure the new segment input fields load
            time.sleep(2)

            # Ensure we explicitly focus on the additional start time field
            start_time_field = wait_for_image("start_time_input_additional.png")  # Re-find start time input after adding segment
            time.sleep(2)
            pyautogui.click(start_time_field)  # Ensure start time is clicked
            time.sleep(2)  # Short delay before typing
            pyautogui.typewrite(additional_start_time + '\n', interval=0.1)
            print(f"Entered additional start time: {additional_start_time}")
            time.sleep(2)

            # Ensure we explicitly focus on the additional end time field
            end_time_field = wait_for_image("end_time_input.png")  # Re-find end time input after adding segment
            time.sleep(2)
            pyautogui.click(end_time_field)  # Ensure end time is clicked
            time.sleep(2)  # Short delay before typing
            pyautogui.typewrite(additional_end_time + '\n', interval=0.1)
            print(f"Entered additional end time: {additional_end_time}")
            time.sleep(2)

        # After processing all segments for the current URL, click the claim button
        # Uncomment this line to actually click the claim button
        # pyautogui.click(wait_for_image("claim_button.png"))
        print("Claim Button Clicked for URL:", youtube_url)
        
        # Add the URL to processed set
        processed_urls.add(youtube_url)  # Mark this URL as processed
        time.sleep(3)  # Optional delay between claims

    # Write skipped rows to an Excel file
    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        skipped_df.to_excel("Empty_URL_Claim.xlsx", index=False)
        print("Empty URL rows saved to 'Empty_URL_Claim.xlsx'.")

# Run the automation
if __name__ == "__main__":
    try:
        automate_clamming()
    except Exception as e:
        print(f"Automation stopped due to error: {e}")
        sys.exit(1)