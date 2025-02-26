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

# Set up Chrome options
chrome_options = Options()
#chrome_options.add_argument("--headless")  # Run browser in headless mode
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Define countries to scrape
date_preset = "90d"
countries = {"Worldwide": "GLOBAL", "USA": "US", "Canada": "CA", "India": "IN"}

# Login to Spotify Works
def login():
    print("Starting login process...")
    driver.get("https://works.spotify.com/works?datePreset=" + date_preset + "&country=GLOBAL&page=1")
    wait = WebDriverWait(driver, 10)
    
    email_input = wait.until(EC.presence_of_element_located((By.ID, "login-username")))
    password_input = driver.find_element(By.ID, "login-password")
    login_button = driver.find_element(By.ID, "login-button")
    
    time.sleep(1)  # Short delay to avoid bot detection
    email_input.send_keys("monil@roynet.com")
    time.sleep(1)
    password_input.send_keys("Jinal@619619")
    time.sleep(1)
    
    login_button.click()
    print("Login successful.")
    wait.until(EC.presence_of_element_located((By.XPATH, "//table")))

# Scrape data per country
def scrape_data(country_name, country_code):
    print(f"\nStarting data extraction for {country_name} ({country_code})...")
    
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
    
    total_records = 0
    
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
            
            # ✅ Extract Work ID from `data-test-id` or `href`
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
        
        # Append data to file immediately
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(page_data)
    
    print(f"Data for {country_name} saved to {csv_filename}")

# Run script
login()
for country_name, country_code in countries.items():
    scrape_data(country_name, country_code)

driver.quit()
print("Scraping complete for all selected countries.")
