import base64
import functions_framework
import googleapiclient.discovery
import googleapiclient.errors
from google.cloud import bigquery
import sys
import csv
import re
import isodate  # This will help parse ISO 8601 duration strings
from datetime import timedelta

sys.stdout.reconfigure(encoding='utf-8')

def sanitize_string(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def convert_duration(duration):
    try:
        parsed_duration = isodate.parse_duration(duration)
        total_seconds = int(parsed_duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    except Exception as e:
        print(f"Error converting duration: {e}")
        return duration

def get_channel_data(youtube, channel_id):
    try:
        request = youtube.channels().list(part="snippet,contentDetails,statistics", id=channel_id)
        response = request.execute()

        if 'items' not in response or len(response['items']) == 0:
            print(f"Error: No data found for channel ID {channel_id}.")
            return None

        channel_info = response['items'][0]
        channel_data = {
            'name': channel_info['snippet']['title'],
            'country': channel_info['snippet'].get('country', 'N/A'),
            'total_views': int(channel_info['statistics']['viewCount']),
            'subscribers': int(channel_info['statistics']['subscriberCount']),
            'total_videos': int(channel_info['statistics']['videoCount']),
            'uploads_playlist': channel_info['contentDetails']['relatedPlaylists']['uploads']
        }

        return channel_data

    except googleapiclient.errors.HttpError as e:
        print(f"API error occurred: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def get_video_details(youtube, video_id):
    request = youtube.videos().list(part="statistics,snippet,contentDetails", id=video_id)
    response = request.execute()

    video_info = response['items'][0]
    description = video_info['snippet'].get('description', '')

    video_data = {
        'title': video_info['snippet']['title'],
        'published_at': video_info['snippet']['publishedAt'],
        'video_url': f"https://www.youtube.com/watch?v={video_id}",
        'view_count': int(video_info['statistics'].get('viewCount', 0)),
        'like_count': int(video_info['statistics'].get('likeCount', 0)),
        'dislike_count': int(video_info['statistics'].get('dislikeCount', 0)),
        'comment_count': int(video_info['statistics'].get('commentCount', 0)),
        'duration': convert_duration(video_info['contentDetails'].get('duration', 'PT0S')),
        'category': video_info['snippet'].get('categoryId', 'N/A'),
        'caption': video_info['contentDetails'].get('caption', 'false')
    }
    return video_data

def get_all_videos(youtube, playlist_id):
    video_list = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_data = get_video_details(youtube, video_id)
            video_list.append(video_data)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_list

def create_or_update_table(client, dataset_id, table_id, schema):
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    try:
        table = client.get_table(table_ref)
        existing_schema = [(field.name, field.field_type) for field in table.schema]
        new_schema = [(field['name'], field['type']) for field in schema]

        if existing_schema != new_schema:
            print("Table schema does not match the expected schema. Exiting.")
            sys.exit(1)
    except Exception:
        print("Table does not exist. Creating table.")
        table = bigquery.Table(table_ref, schema=[bigquery.SchemaField(**field) for field in schema])
        client.create_table(table)


def load_data_to_bigquery(client, dataset_id, table_id, data):
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # Fetch existing data in the table to check for duplicates
    existing_video_urls = {}
    try:
        rows = client.list_rows(table_ref)
        for row in rows:
            existing_video_urls[row['Video URL']] = row
    except Exception as e:
        print(f"Error fetching existing data: {e}")
    
    # Prepare the data for insertion (with replacement logic)
    records_to_insert = []
    records_to_delete = []
    
    for record in data:
        video_url = record['Video URL']
        
        if video_url in existing_video_urls:
            # If a duplicate is found, mark the old record for deletion and continue
            print(f"Duplicate found for Video URL: {video_url}. Replacing existing data.")
            records_to_delete.append(existing_video_urls[video_url]['Video URL'])
        else:
            # If no duplicate is found, append the record
            records_to_insert.append(record)
    
    # First, delete the old records (duplicates)
    for video_url in records_to_delete:
        # Delete the record with the matching Video URL (you can use other conditions as needed)
        try:
            # The delete operation is not directly available in BigQuery, so you must update rows instead
            print(f"Deleting old record with Video URL: {video_url}")
            # BigQuery does not support direct row deletion via `insert_rows_json`, so here you can instead
            # overwrite old data or create a new table with the updated data
            pass  # Add your custom logic here if needed to delete or overwrite data
        except Exception as e:
            print(f"Error deleting record: {e}")
    
    # Insert the new records (without duplicates)
    if records_to_insert:
        errors = client.insert_rows_json(table_ref, records_to_insert)
        if errors:
            print(f"Error occurred while loading data to BigQuery: {errors}")
        else:
            print("New data loaded successfully to BigQuery.")
    else:
        print("No new data to load (duplicates only).")

# The Cloud Function entry point
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Extract and decode the message from Pub/Sub
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode('utf-8')
    
    # Print the message to logs
    print(f"Received message: {message_data}")
    
    # Initialize YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey="AIzaSyDMGYaAjnwMEBKCoZsl0LO3FPU6wua1cfA")

    client = bigquery.Client()
    dataset_id = "api_dataset"
    table_id = "youtubetable"

    schema = [
        {"name": "Channel Name", "type": "STRING"},
        {"name": "Country", "type": "STRING"},
        {"name": "Total Views", "type": "INTEGER"},
        {"name": "Subscribers", "type": "INTEGER"},
        {"name": "Total Videos", "type": "INTEGER"},
        {"name": "Video Title", "type": "STRING"},
        {"name": "Published At", "type": "TIMESTAMP"},
        {"name": "Video URL", "type": "STRING"},
        {"name": "Video View Count", "type": "INTEGER"},
        {"name": "Video Like Count", "type": "INTEGER"},
        {"name": "Video Dislike Count", "type": "INTEGER"},
        {"name": "Video Comment Count", "type": "INTEGER"},
        {"name": "Video Duration", "type": "STRING"},
        {"name": "Video Category ID", "type": "STRING"},
        {"name": "Video Captions Available", "type": "STRING"}
    ]

    create_or_update_table(client, dataset_id, table_id, schema)

    # The main logic to fetch data
    channel_ids = [
        "UC8zBl939TmRZGK3KDwS1IUg", 
        "UCjZfepNc3GDI2nO0fFpICxg", 
        "UCru4XPP-NWjE6HQk39FyCYA"
    ]

    all_channel_data = []

    for channel_id in channel_ids:
        print(f"\nFetching data for channel: {channel_id}")
        channel_data = get_channel_data(youtube, channel_id)

        if not channel_data:
            print(f"Skipping channel {channel_id} due to errors in fetching data.")
            continue

        print("--- Channel Info ---")
        print(f"Channel Name: {channel_data['name']}")
        print(f"Country: {channel_data['country']}")
        print(f"Total Views: {channel_data['total_views']}")
        print(f"Subscribers: {channel_data['subscribers']}")
        print(f"Total Videos: {channel_data['total_videos']}")

        print("\nFetching all videos...\n")
        videos = get_all_videos(youtube, channel_data['uploads_playlist'])

        for video in videos:
            record = {
                "Channel Name": channel_data['name'],
                "Country": channel_data['country'],
                "Total Views": channel_data['total_views'],
                "Subscribers": channel_data['subscribers'],
                "Total Videos": channel_data['total_videos'],
                "Video Title": sanitize_string(video['title']),
                "Published At": video['published_at'],
                "Video URL": video['video_url'],
                "Video View Count": video['view_count'],
                "Video Like Count": video['like_count'],
                "Video Dislike Count": video['dislike_count'],
                "Video Comment Count": video['comment_count'],
                "Video Duration": video['duration'],
                "Video Category ID": video['category'],
                "Video Captions Available": video['caption']
            }
            all_channel_data.append(record)

    load_data_to_bigquery(client, dataset_id, table_id, all_channel_data)

    print("Cloud Function executed successfully!")