from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import csv
import re
import time
import os
from google.cloud import bigquery

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run browser in headless mode
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Define countries to scrape
date_preset = "90d"
countries = {"Worldwide": "GLOBAL", "USA": "US", "Canada": "CA", "India": "IN"}

# Configuration switch: Set to 1 for web scraping, 0 for local file loading
USE_WEB_SCRAPING = 1  # Change this to 0 to load from local files

# Function to check GCP connection
def check_gcp_connection():
    print("Checking GCP connection...")
    try:
        client = bigquery.Client()
        query = "SELECT COUNT(*) as count FROM `roynet2025.roynet.income_archived_115`"
        results = client.query(query)
        for row in results:
            print(f"GCP connection successful. Found {row.count} records in the table.")
        return True
    except Exception as e:
        print(f"Failed to connect to GCP: {e}")
        return False

# Function to fetch all Song Codes from BigQuery
def fetch_song_codes():
    print("Fetching all Song Codes from BigQuery...")
    client = bigquery.Client()
    query = """
        SELECT Song_Code
        FROM `roynet2025.roynet.income_archived_115`
    """
    results = client.query(query)
    song_codes = [row.Song_Code for row in results]
    print(f"Fetched {len(song_codes)} Song Codes.")
    return song_codes

# Login to Spotify Works
def login():
    print("Starting login process...")
    driver.get("https://works.spotify.com/works?datePreset=" + date_preset + "&country=GLOBAL&page=1")
    wait = WebDriverWait(driver, 10)
    
    email_input = wait.until(EC.presence_of_element_located((By.ID, "login-username")))
    password_input = driver.find_element(By.ID, "login-password")
    login_button = driver.find_element(By.ID, "login-button")
    
    time.sleep(1)  # Short delay to avoid bot detection
    email_input.send_keys("ADD EMAIL HERE")
    time.sleep(1)
    password_input.send_keys("ADD PASSWORD HERE")
    time.sleep(1)
    login_button.click()
    print("Login process completed.")

# Scrape data per country
def scrape_data(country_name, country_code):
    print(f"\nStarting data extraction for {country_name} ({country_code})...")
    
    # Check GCP connection before proceeding
    if not check_gcp_connection():
        print("Exiting process due to GCP connection failure.")
        return
    
    # Fetch Song Codes from BigQuery
    song_codes = fetch_song_codes()
    
    if country_code == "GLOBAL":
        base_url = f"https://works.spotify.com/works?datePreset={date_preset}&country=GLOBAL&page={{}}"
    else:
        base_url = f"https://works.spotify.com/works?datePreset={date_preset}&country={country_code}&page={{}}"

    csv_filename = f"spotify_works_{country_name.lower().replace(' ', '_')}.csv"
    header = ["RANK", "Work ID", "TITLE", "WRITERS", "RECORDINGS", "STREAMS"]
    
    # Open file initially and write header
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            print(f"Created file {csv_filename} with header.")

    total_records = 0
    found_song_codes = []
    not_found_song_codes = []

    for page in range(1, 101):  # Stop at page 100
        print(f"Scraping page {page} for {country_name}...")
        driver.get(base_url.format(page))
        
        try:
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.XPATH, "//table")))
        except:
            print(f"No more pages to scrape for {country_name}.")
            break
        
        # Fetch table data
        table = driver.find_element(By.XPATH, "//table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        page_data = []
        for row in rows[1:]:  # Skip header row
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 5:
                continue  # Skip malformed rows
            
            rank = cols[0].text.strip()
            title_element = cols[1].find_element(By.TAG_NAME, "a")
            title = title_element.text.strip()
            
            # Extract Work ID from `data-test-id` or `href`
            work_id = "N/A"
            work_id_attr = title_element.get_attribute("data-test-id")
            if work_id_attr:
                match = re.search(r'work-link-TRN(\d+)', work_id_attr)
                if match:
                    work_id = match.group(1)
            else:
                href = title_element.get_attribute("href")
                match = re.search(r'/work/TRN(\d+)', href)
                if match:
                    work_id = match.group(1)

            writers = cols[2].text.replace("\n", ", ").strip()
            recordings = cols[3].text.replace(",", "").strip()
            streams = cols[4].text.replace(",", "").strip()
            
            page_data.append([rank, work_id, title, writers, recordings, streams])
        
        if not page_data:
            print(f"No more data found for {country_name}. Stopping.")
            break
        
        total_records += len(page_data)
        print(f"Fetched {len(page_data)} records from page {page} for {country_name}, Total: {total_records}")
        
        for data in page_data:
            work_id = data[1]  # Assuming work_id is at index 1
            if work_id in song_codes:
                found_song_codes.append(data)
                print(f"Work ID {work_id} matches a Song Code.")
            else:
                not_found_song_codes.append(data)
                print(f"Work ID {work_id} does not match any Song Code.")
        
        # Append data to file immediately
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(page_data)
            print(f"Appended {len(page_data)} records to {csv_filename}.")
    
    # Create CSV files for found and not found song codes
    with open(f"found_song_codes_{country_name.lower().replace(' ', '_')}.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        writer.writerows(found_song_codes)  # Write found song codes
        print(f"Created file for found song codes with {len(found_song_codes)} records.")

    with open(f"not_found_song_codes_{country_name.lower().replace(' ', '_')}.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        writer.writerows(not_found_song_codes)  # Write not found song codes
        print(f"Created file for not found song codes with {len(not_found_song_codes)} records.")
    
    print(f"Data extraction for {country_name} completed. Total records saved: {total_records}")

# Run script
login()
for country_name, country_code in countries.items():
    scrape_data(country_name, country_code)

# Close the driver
driver.quit()
print("WebDriver closed. Script execution completed.")