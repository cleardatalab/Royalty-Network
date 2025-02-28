import os
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import datetime

# API Credentials
CLIENT_SECRETS_FILE = r"CLIENT SECRET FILE"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtubepartner",
    "https://www.googleapis.com/auth/youtubepartner-content-owner-readonly",
]

CONTENT_OWNER_ID = "CONTENT OWENER ID"

# Directory Paths
BASE_PATH = r"PATH TO SAVE FILE"
SAVE_PATH = os.path.join(BASE_PATH, "Youtube Claming")

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Error log path
ERROR_LOG_PATH = os.path.join(SAVE_PATH, "error_log.csv")
if not os.path.exists(ERROR_LOG_PATH):
    pd.DataFrame(columns=["Timestamp", "Endpoint", "Error Message", "Response"]).to_csv(ERROR_LOG_PATH, index=False)


def authenticate_with_google():
    """
    Authenticate with Google and return credentials.
    """
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=8080)
    return credentials.token


def fetch_data_from_api(endpoint, params, access_token):
    """
    Fetch paginated data from YouTube Partner API.
    """
    base_url = "https://www.googleapis.com/youtube/partner/v1"
    url = f"{base_url}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    all_data = []
    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()

            if "items" in data and data["items"]:
                all_data.extend(data["items"])
            else:
                print(f"Warning: No items found for {endpoint}")

            # Handle pagination
            next_page_token = data.get("nextPageToken")
            if next_page_token:
                params["pageToken"] = next_page_token
            else:
                break

        except requests.exceptions.RequestException as e:
            error_message = str(e)
            save_error_to_log(endpoint, error_message, response.text if response else "No response")
            return []  # Return an empty list on failure

    return all_data


def save_error_to_log(endpoint, error_message, response_content):
    """
    Save error details to a CSV log file.
    """
    error_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Endpoint": endpoint,
        "Error Message": error_message,
        "Response": response_content,
    }
    df = pd.DataFrame([error_data])
    df.to_csv(ERROR_LOG_PATH, mode="a", header=False, index=False)
    print(f"Error logged to {ERROR_LOG_PATH}")


def save_to_csv(data, filename):
    """
    Save data to CSV in the 'Youtube Claming' folder.
    """
    file_path = os.path.join(SAVE_PATH, filename)
    if data:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print(f"No data found for {filename}. Check error logs.")


def fetch_all_details(content_owner_id, access_token):
    """
    Fetch and save assets and claims data.
    """
    print("Fetching assets...")
    assets_params = {"onBehalfOfContentOwner": content_owner_id}
    assets = fetch_data_from_api("assets", assets_params, access_token)
    save_to_csv(assets, "assets.csv")

    print("Fetching claims...")
    claims_params = {"onBehalfOfContentOwner": content_owner_id}
    claims = fetch_data_from_api("claims", claims_params, access_token)
    save_to_csv(claims, "claims.csv")


def main():
    print("Authenticating with Google...")
    token = authenticate_with_google()
    print("Authentication successful!")

    print("Fetching all details...")
    fetch_all_details(CONTENT_OWNER_ID, token)


if __name__ == "__main__":
    main()
