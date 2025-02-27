import requests
import json
import csv
import time
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# ---------------------------
# BigQuery Table Validation
# ---------------------------
def validate_and_create_table(client, project_id, dataset_id, table_name):
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    expected_schema = [
        bigquery.SchemaField("Album ID", "STRING"),
        bigquery.SchemaField("Album Name", "STRING"),
        bigquery.SchemaField("Artist Name", "STRING"),
        bigquery.SchemaField("Track Name", "STRING"),
        bigquery.SchemaField("Playcount", "INTEGER"),
        bigquery.SchemaField("URI", "STRING"),
        bigquery.SchemaField("UID", "STRING"),
        bigquery.SchemaField("Release Date", "STRING"),
        bigquery.SchemaField("Timestamp", "TIMESTAMP"),
        bigquery.SchemaField("Track ID", "STRING"),
        bigquery.SchemaField("Song Name", "STRING"),
        bigquery.SchemaField("Performed by", "STRING"),
        bigquery.SchemaField("Written by", "STRING"),
        bigquery.SchemaField("Produced by", "STRING"),
        bigquery.SchemaField("Source", "STRING"),
    ]
    
    try:
        table = client.get_table(table_id)
        # Check if the table schema matches expected schema
        actual_columns = set(field.name for field in table.schema)
        expected_columns = set(field.name for field in expected_schema)
        if actual_columns != expected_columns:
            print("Table schema does not match expected schema. Recreating table...")
            client.delete_table(table_id, not_found_ok=True)
            table = bigquery.Table(table_id, schema=expected_schema)
            table = client.create_table(table)
            print(f"Table {table_id} created with updated schema.")
        else:
            print(f"Table {table_id} already exists and schema is valid.")
    except NotFound:
        print("Table not found. Creating new table...")
        table = bigquery.Table(table_id, schema=expected_schema)
        table = client.create_table(table)
        print(f"Table {table_id} created.")
    return table

def safe_playcount(value):
    try:
        return int(value)
    except:
        return 0

def load_data_to_bigquery(final_data, client, project_id, dataset_id, table_name):
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    rows_to_insert = []
    for record in final_data:
        row = {
            "Album ID": record.get("Album ID", "-"),
            "Album Name": record.get("Album Name", "-"),
            "Artist Name": record.get("Artist Name", "-"),
            "Track Name": record.get("Track Name", "-"),
            "Playcount": safe_playcount(record.get("Playcount")),
            "URI": record.get("URI", "-"),
            "UID": record.get("UID", "-"),
            "Release Date": record.get("Release Date", "-"),
            "Timestamp": record.get("Timestamp").isoformat() if isinstance(record.get("Timestamp"), datetime) else record.get("Timestamp"),
            "Track ID": record.get("Track ID", "-"),
            "Song Name": record.get("Song Name", "-"),
            "Performed by": record.get("Performed by", "-"),
            "Written by": record.get("Written by", "-"),
            "Produced by": record.get("Produced by", "-"),
            "Source": record.get("Source", "-")
        }
        rows_to_insert.append(row)
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print(f"Successfully inserted {len(rows_to_insert)} rows into {table_id}.")

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
# Album, Playcount & Credit Functions
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
                
                merged_record = {
                    "Album ID": album_id,
                    "Album Name": album_name,
                    "Artist Name": artist_name,
                    "Track Name": track_name,
                    "Playcount": playcount,
                    "URI": uri,
                    "UID": uid,
                    "Release Date": release_date,
                    "Timestamp": datetime.now(),
                    "Track ID": track_id,
                    "Song Name": credits.get("Song Name", "-"),
                    "Performed by": credits.get("Performed by", "-"),
                    "Written by": credits.get("Written by", "-"),
                    "Produced by": credits.get("Produced by", "-"),
                    "Source": credits.get("Source", "-")
                }
                merged_records.append(merged_record)
                time.sleep(0.5)
        else:
            print(f"Failed to get playcount for album {album_id}.")
    except Exception as e:
        print(f"Error fetching playcount for album {album_name}: {e}")
    return merged_records

def get_spotify_credits(track_id, token):
    """
    Get credits for a Spotify track using the internal API.
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
# Main Processing Function
# ---------------------------
def main():
    print("Starting Spotify data collection with immediate credit merging...")
    token = get_spotify_token()
    if not token:
        print("Exiting script due to missing token.")
        return

    # BigQuery configuration
    project_id = "roynet2025"
    dataset_id = "api_dataset"
    table_name = "spotifytable"
    bq_client = bigquery.Client(project=project_id)
    
    # Validate and/or create table with expected schema
    validate_and_create_table(bq_client, project_id, dataset_id, table_name)

    # List of artist IDs (Replace with your actual artist IDs)
    artist_ids = [
    "7uIbLdzzSEqnX0Pkrb56cR", "4YRxDV8wJFPHPTeXepOstw", "5T2I75UlGBcWd5nVyfmL13",
    "71hjb5G92mGoKRSAW3Cj00", "4kq8z3vydHjPDggxb16ErB", "4zCH9qm4R2DADamUHMCa6O", "0ZvsEkINadmEV4qzS4ollh",
    "0a1SidMjD8D6EHvJph4n2H", "2dlCuzBPpSIeyY4ZCJBKGS", "7osFcSwjlRPwxZdVALIOuC", "6dhXvR5MsnlwYguRuqoapR",
    "0C4gtx1iHMfuaQ73GKWvtZ", "1Gh0pCAxpjw0Iq3JMoVAwO", "4FGPzWzgjURDNT7JQ8pYgH", "0vTVU0KH0CVzijsoKGsTPl",
    "4rdJkXHNrMgowlwUdQAg8T", "4E0HD2PMY8kQJIjlShrLUS", "1J1pGfTqp5ReVIX8Z1Wzsg", "3DHtfeD4PsmR9YGhCP4VF7",
    "4nqQTosM2Mbg7iRjvJU0N0", "47UhY4DqayBiq2gp43WOcZ", "1iCnM8foFssWlPRLfAbIwo", "67FB4n52MgexGQIG8s0yUH",
    "2IprcYDAYTYzCl4AJH3AuT", "4EnymklUyqZwvmHQGlRssl", "6TWEX2qTj9b0bBsXSVCMKM", "0NlNru2YcUz6RbnpYGQz26",
    "54seKvtsZauR1iauN0ptpo", "6pV5zH2LzjOUHaAvENdMMa", "6TKygPpVT29oGUogu4J9Ec", "4gvjmrtzydbMpyJaXUtwvP",
    "1WNmfSqydnt1FDJKg3l6lw", "21YCHE0ZFflbHVTsyrCpgh", "24b0qNYNgeOfpP5rbljIB3", "0ocxWXtgr9tJW60xV5ZufT",
    "6NvsNA4Ea62yJh7ePTS8gz", "2BPwxhCvvcb8xDl8GWIjbh", "7MxGWmiAbqjNOGmj23wbWf", "0zmxCsd8aIJHfNC95gdT2i",
    "6P5ccCJCe8A4s9tDSTNFzF", "6fb3I3Q54izgnOMtiZbOBA",
    # Newly added artist IDs
    "3J4UgGSEUJOzdHkssLTHBj", "2NIfJ8TeZ3rfPIEWSPCZNy", "1IueXOQyABrMOprrzwQJWN", "0LyfQWJT6nXafLPZqxe9Of",
    "2NoJ7NuNs9nyj8Thoh1kbu", "1nqAxAdynShWqEoMImqzto", "0Y6dVaC9DZtPNH4591M42W", "3plJVWt88EqjvtuB4ZDRV3",
    "2oZcMYiKpjaA2Et5mU3RPP", "1xCzIWADhHc3qmVesDX7Gp", "7Hvc2DEwgI2n5lD1hk6EUx", "7BPZmKlfEYVLsuTCEzav0w",
    "0LcJLqbBmaGUft1e9Mm8HV", "26prf4s7uhVnLIWwFKf5i6", "7DJ333ShuiNV3gwmajAVfr", "0gsJgj7k1eMp55jA0Mpyxb",
    "0s1ec6aPpRZ4DCj15w1EFg", "0bfX8pF8kuHNCs57Ms4jZb", "6f5kUMXGROFtdAtxXZwing", "1lq29gvVmonIIomaiiafub",
    "1DLsogyGi0pwPtwV78D8uZ", "4h11UhaLnKa6WKk8AIWYrk", "4KWTAlx2RvbpseOGMEmROg", "1uoxul8H3vteREJQfvrReR",
    "6DhqmXo63IFrko6ZMrGfpM", "13yg8YI3uzQ9MIRb4py8Ys", "0OzxPXyowUEQ532c9AmHUR", "3b6Xu46myEwqKjHhCb5PFt",
    "6Xb4ezwoAQC4516kI89nWz", "1YZhNFBxkEB5UKTgMDvot4", "1ZwdS5xdxEREPySFridCfh", "7ueOlHsDGBjqZfjpDj4oJO",
    "2kEU1NNJORHNJabD0tdO1E", "7jgnJBnpZTiGnCF2Wvka2Z", "5tm3TvF1iMzxJahJJQu3qN", "778Snztf3N5DXp0kHGFl3g",
    "2wdilDjBFjtfm30BczhsPa", "3LcyxBLuNKj7qPQnH5aTOQ", "4OqCkbr31VLuR8lpHAZvBr", "56uDOSQrD9uvMezSmBfLM4",
    "68JafKRPdyHOelH8FFYhGv", "5oDtp2FC8VqBjTx1aT4P5j", "6U6zWkFtgM3UU5c1hBlGCD", "3Isy6kedDrgPYoTS1dazA9",
    "6VCt6Dh7TaZF330ZFeNHv5", "6KUP1yDyDiJlpHeNts73D4", "4wes2odxtMtswRxLopJJmn", "0146m8i9zvXFasHl91avpE",
    "2mVTkiwfm4ic6DnHpmFq8K", "5VAyiDhBinVfc6RM5RKnLa", "6Nb5kOMBK5OsZWWLwmQ0pK", "71jzN72g8qWMCMkWC5p1Z0",
    "57K5h2Rf5M9i0buAVVBpmH"
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

    # Load merged records into BigQuery
    load_data_to_bigquery(final_data, bq_client, project_id, dataset_id, table_name)
    print("Script executed successfully!")

if __name__ == '__main__':
    main()
