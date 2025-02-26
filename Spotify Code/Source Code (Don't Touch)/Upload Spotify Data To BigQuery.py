import requests
import json
import csv
import time
import os
from datetime import datetime

# ---------------------------
# Spotify Token & Helper Functions
# ---------------------------
def get_spotify_token():
    """Fetch a valid Spotify API token."""
    try:
        print("Fetching access token from Spotify...")
        token_url = "https://open.spotify.com/get_access_token?reason=transport&productType=web_player"
        response = requests.get(token_url)
        if response.status_code != 200:
            print(f"Failed to fetch token. Status code: {response.status_code}")
            return None
        token_data = response.json()
        if "accessToken" not in token_data:
            print("No access token in response")
            return None
        print("Access token successfully retrieved.")
        return token_data["accessToken"]
    except Exception as e:
        print(f"Error getting token: {str(e)}")
        return None

def clean_artist_names(artist_string):
    """
    Removes duplicate artist names while maintaining their original order.
    """
    seen = set()
    unique_artists = []
    for artist in artist_string.split(', '):
        if artist not in seen:
            seen.add(artist)
            unique_artists.append(artist)
    return ', '.join(unique_artists)

# ---------------------------
# Album and Playcount + Credit Functions
# ---------------------------
def get_artist_albums(artist_id, token):
    """Get up to 5 albums for a given artist using the Spotify API."""
    print(f"Fetching albums for artist {artist_id}...")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {"include_groups": "album", "limit": 5}
    try:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 200:
            data = r.json()
            albums = data.get("items", [])
            if not albums:
                print(f"No albums found for artist {artist_id}.")
                return []
            album_details = []
            for album in albums:
                artist_names = [artist["name"] for artist in album.get("artists", []) if "name" in artist]
                cleaned_artist_name = clean_artist_names(', '.join(artist_names))
                album_details.append((
                    album["id"],
                    album["name"],
                    cleaned_artist_name,
                    album.get("release_date", "Unknown")
                ))
            print(f"Album details (cleaned): {album_details}")
            return album_details
        else:
            print(f"Failed to fetch albums for artist {artist_id}: {r.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching albums for artist {artist_id}: {str(e)}")
        return []

def get_playcount_and_credit(album_id, album_name, artist_name, release_date, token):
    """
    For a given album, fetch each track's playcount data and immediately fetch its credit data.
    Returns a list of merged records.
    """
    merged_records = []
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "operationName": "queryAlbumTracks",
        "variables": json.dumps({
            "uri": f"spotify:album:{album_id}",
            "offset": 0,
            "limit": 300
        }),
        "extensions": json.dumps({
            "persistedQuery": {
                "version": 1,
                "sha256Hash": "3ea563e1d68f486d8df30f69de9dcedae74c77e684b889ba7408c589d30f7f2e"
            }
        })
    }
    try:
        r_playcount = requests.get("https://api-partner.spotify.com/pathfinder/v1/query", headers=headers, params=params).json()
        if "data" in r_playcount and "album" in r_playcount["data"]:
            print(f"Playcount data retrieved for album {album_name}.")
            for track_item in r_playcount["data"]["album"]["tracks"]["items"]:
                uid = track_item.get("uid", "UID not available")
                track_data = track_item.get("track", {})
                uri = track_data.get("uri", "URI not available")
                playcount = track_data.get("playcount", "Playcount not available")
                track_name = track_data.get("name", "Name not available")
                
                # Extract track ID from URI (expected format: "spotify:track:<TRACK_ID>")
                parts = uri.split(":")
                if len(parts) == 3 and parts[1] == "track":
                    track_id = parts[2]
                else:
                    print(f"Unexpected URI format for track: {uri}. Using empty track id.")
                    track_id = ""
                
                # Fetch credits immediately if we have a valid track_id
                if track_id:
                    credits = get_spotify_credits(track_id, token)
                else:
                    credits = generate_empty_credit_result(track_id)
                
                # Merge playcount and credit data into a single record
                merged_record = {
                    "Album ID": album_id,
                    "Album Name": album_name,
                    "Artist Name": artist_name,
                    "Track Name": track_name,
                    "Playcount": playcount,
                    "URI": uri,
                    "UID": uid,
                    "Release Date": release_date,
                    "Timestamp": datetime.now(),  # raw datetime; converted when saving
                    "Track ID": track_id,
                    "Song Name": credits.get("Song Name", "-"),
                    "Performed by": credits.get("Performed by", "-"),
                    "Written by": credits.get("Written by", "-"),
                    "Produced by": credits.get("Produced by", "-"),
                    "Source": credits.get("Source", "-")
                }
                merged_records.append(merged_record)
                # Small delay between credit requests
                time.sleep(0.5)
        else:
            print(f"Failed to get playcount for album {album_id}.")
    except Exception as e:
        print(f"Error fetching playcount for album {album_name}: {e}")
    return merged_records

# ---------------------------
# Track Credits Functions
# ---------------------------
def get_spotify_credits(track_id, token):
    """
    Get credits for a Spotify track using web scraping of the internal API.
    Returns a dictionary with credit details.
    """
    try:
        print(f"\nProcessing credits for Track ID: {track_id}")
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}',
            'App-Platform': 'WebPlayer',
            'Referer': 'https://open.spotify.com/',
            'Origin': 'https://open.spotify.com'
        }
        metadata_url = f"https://api.spotify.com/v1/tracks/{track_id}"
        print("Fetching track metadata...")
        response = session.get(metadata_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch track metadata. Status code: {response.status_code}")
            return generate_empty_credit_result(track_id)
        track_data = response.json()
        song_name = track_data.get('name', 'Unknown')
        album_id = track_data.get('album', {}).get('id')
        if album_id:
            album_url = f"https://api.spotify.com/v1/albums/{album_id}"
            album_response = session.get(album_url, headers=headers)
            if album_response.status_code == 200:
                album_data = album_response.json()
                label = album_data.get('label', '-')
            else:
                label = '-'
        else:
            label = '-'
        credits_url = f"https://spclient.wg.spotify.com/track-credits-view/v0/experimental/{track_id}/credits"
        print("Fetching credits...")
        response = session.get(credits_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch credits. Status code: {response.status_code}")
            return generate_empty_credit_result(track_id)
        credits_data = response.json()
        performed_by = []
        written_by = []
        produced_by = []
        for role in credits_data.get('roleCredits', []):
            role_name = role.get('roleTitle', '').lower()
            artists = [artist.get('name') for artist in role.get('artists', []) if artist.get('name')]
            if 'performer' in role_name or 'artist' in role_name:
                performed_by.extend(artists)
            elif 'writer' in role_name or 'composer' in role_name or 'lyricist' in role_name:
                written_by.extend(artists)
            elif 'producer' in role_name:
                produced_by.extend(artists)
        # Remove duplicates while preserving order
        performed_by = list(dict.fromkeys(performed_by))
        written_by = list(dict.fromkeys(written_by))
        produced_by = list(dict.fromkeys(produced_by))
        result = {
            "Song Name": song_name,
            "Performed by": ", ".join(performed_by) if performed_by else "-",
            "Written by": ", ".join(written_by) if written_by else "-",
            "Produced by": ", ".join(produced_by) if produced_by else "-",
            "Source": label
        }
        print("[SUCCESS] Credits fetched.")
        return result
    except Exception as e:
        print(f"Error getting credits for {track_id}: {str(e)}")
        return generate_empty_credit_result(track_id)

def generate_empty_credit_result(track_id):
    """Return a default credits dictionary if fetching fails."""
    return {
        "Song Name": "Not Found",
        "Performed by": "-",
        "Written by": "-",
        "Produced by": "-",
        "Source": "-"
    }

# ---------------------------
# CSV Saving Function
# ---------------------------
def save_data_to_csv(final_data, file_name="spotify_merged_data.csv"):
    """
    Save the merged data to a CSV file with the following columns:
      Album ID, Album Name, Artist Name, Track Name, Playcount, URI, UID,
      Release Date, Timestamp, Track ID, Song Name, Performed by, Written by,
      Produced by, Source
    """
    fieldnames = [
        "Album ID", "Album Name", "Artist Name", "Track Name", "Playcount", "URI",
        "UID", "Release Date", "Timestamp", "Track ID", "Song Name",
        "Performed by", "Written by", "Produced by", "Source"
    ]
    try:
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in final_data:
                # Convert Timestamp datetime to ISO format string if needed
                if isinstance(record.get("Timestamp"), datetime):
                    record["Timestamp"] = record["Timestamp"].isoformat()
                writer.writerow(record)
        print(f"Data successfully saved to CSV file: {file_name}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")

# ---------------------------
# Main Processing Function
# ---------------------------
def main():
    print("Starting Spotify data collection with immediate credit merging...")
    token = get_spotify_token()
    if not token:
        print("Exiting script due to missing token.")
        return

    # List of artist IDs (Replace with your actual artist IDs)
    artist_ids = [
        "7uIbLdzzSEqnX0Pkrb56cR"
    ]

    final_data = []
    # Process each artist and for each album, immediately merge playcount with credit data.
    for artist_id in artist_ids:
        print(f"\nProcessing artist ID: {artist_id}")
        album_details = get_artist_albums(artist_id, token)
        for album_id, album_name, artist_name, release_date in album_details:
            merged_records = get_playcount_and_credit(album_id, album_name, artist_name, release_date, token)
            final_data.extend(merged_records)
            time.sleep(0.5)  # brief delay between album requests

    # Save the merged records to CSV
    save_data_to_csv(final_data)
    print("Script executed successfully!")

if __name__ == '__main__':
    main()
