import os
import pandas as pd
from google.cloud import bigquery, storage
import paramiko
from paramiko import SSHClient, RSAKey
from io import StringIO

# SFTP configuration
SFTP_HOST = "ftp-data.themlc.com"
SFTP_PORT = 22  # Default SFTP port
SFTP_USERNAME = "royaltynetwork"
GCS_BUCKET = "mlc_file"
SFTP_PRIVATE_KEY_BLOB = "RoyNetPrivKey_openssh"
PRIVATE_KEY_PASSWORD = "goodQu3en31"
REMOTE_BASE_PATH = "/public-database/"

# BigQuery configuration
DATASET_ID = 'mlc_dataset'
TABLE_NAME = 'unclaimedworkrightshares'
WRITE_MODE = 'WRITE_APPEND'
SCHEMA = [
    "FeedProvidersRightShareId", "FeedProvidersRecordingId",
    "FeedProvidersWorkId", "ISRC", "DspRecordingId", "RecordingTitle",
    "RecordingSubTitle", "AlternativeRecordingTitle", "DisplayArtistName",
    "DisplayArtistISNI", "Duration", "UnclaimedPercentage",
    "PercentileForPrioritisation", "snapshotid"
]
CHUNK_SIZE = 5000000

def delete_table_if_exists(dataset_id, table_name):
    """
    Deletes the BigQuery table if it exists.
    """
    try:
        print(f"[INFO] Checking if BigQuery table {dataset_id}.{table_name} exists...")
        bq_client = bigquery.Client()
        table_id = f"{bq_client.project}.{dataset_id}.{table_name}"
        bq_client.delete_table(table_id, not_found_ok=True)
        print(f"[INFO] Table {table_id} deleted successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to delete BigQuery table: {e}")

def get_private_key_from_gcs(bucket_name, key_blob_name):
    """
    Retrieves the SFTP private key from GCS and loads it into a Paramiko RSAKey.
    """
    try:
        print("[INFO] Retrieving SFTP private key from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key_blob_name)
        key_data_bytes = blob.download_as_bytes()
        key_text = key_data_bytes.decode("utf-8")
        key_io = StringIO(key_text)
        return RSAKey.from_private_key(key_io, password=PRIVATE_KEY_PASSWORD)
    except Exception as e:
        print(f"[ERROR] Failed to retrieve private key from GCS: {e}")
        return None

def get_latest_tsv_file(host, port, username, private_key, base_path):
    """
    Finds the latest timestamped folder and retrieves the TSV file path.
    """
    try:
        print("[INFO] Connecting to SFTP to find latest file...")
        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(host, port=port, username=username, pkey=private_key)
        
        sftp = ssh_client.open_sftp()
        directories = sftp.listdir(base_path)
        timestamped_dirs = sorted([d for d in directories if d.startswith("BWARM_PADPIDA")], reverse=True)
        
        if not timestamped_dirs:
            raise Exception("No timestamped directories found.")
        
        latest_dir = timestamped_dirs[0]
        latest_path = f"{base_path}{latest_dir}/unclaimedworkrightshares.tsv"
        print(f"[INFO] Latest file found: {latest_path}")
        
        sftp.close()
        ssh_client.close()
        
        return latest_path
    except Exception as e:
        print(f"[ERROR] Error retrieving latest TSV file: {e}")
        return None

def download_chunk_from_sftp(host, port, username, private_key, remote_path, chunk_size, start_byte):
    """
    Downloads a chunk of the file from an SFTP server.
    """
    try:
        print(f"[INFO] Connecting to SFTP server {host}...")
        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(host, port=port, username=username, pkey=private_key)
        
        sftp = ssh_client.open_sftp()
        print(f"[INFO] Opening remote file {remote_path}...")
        remote_file = sftp.open(remote_path, 'r')
        remote_file.seek(start_byte)
        
        print(f"[INFO] Downloading chunk (Starting byte: {start_byte}, Size: {chunk_size} bytes)...")
        chunk = remote_file.read(chunk_size).decode('utf-8')
        
        next_start = start_byte + len(chunk) if len(chunk) == chunk_size else None
        
        remote_file.close()
        sftp.close()
        ssh_client.close()
        
        print(f"[INFO] Downloaded {len(chunk)} bytes.")
        return chunk, next_start
    except Exception as e:
        print(f"[ERROR] Error downloading file chunk from SFTP: {e}")
        return None, start_byte

def process_and_upload_chunk(chunk, schema, dataset_id, table_name, write_mode='WRITE_APPEND'):
    """
    Processes and uploads a chunk of data to BigQuery.
    """
    try:
        print("[INFO] Processing chunk for BigQuery...")
        chunk_io = StringIO(chunk)
        chunk_df = pd.read_csv(chunk_io, sep='\t', header=None, dtype=str)
        
        chunk_df.columns = range(chunk_df.shape[1])
        last_valid_index = chunk_df.notna().cumsum(axis=1).idxmax(axis=1)
        max_valid_index = last_valid_index.max()
        chunk_trimmed = chunk_df.iloc[:, :max_valid_index + 1]
        
        if chunk_trimmed.shape[1] < len(schema):
            extra_columns = len(schema) - chunk_trimmed.shape[1]
            chunk_trimmed = pd.concat([chunk_trimmed,
                                       pd.DataFrame([[None] * extra_columns] * chunk_trimmed.shape[0],
                                                    columns=schema[-extra_columns:])], axis=1)
        chunk_trimmed.columns = schema
        
        bq_client = bigquery.Client()
        table_id = f"{bq_client.project}.{dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition=write_mode)
        load_job = bq_client.load_table_from_dataframe(chunk_trimmed, table_id, job_config=job_config)
        load_job.result()
        print(f"[SUCCESS] Chunk uploaded to BigQuery table '{table_id}'.")
    except Exception as e:
        print(f"[ERROR] Error processing or uploading chunk: {e}")

def process_file_from_sftp_in_chunks():
    """
    Deletes BigQuery table if it exists, then downloads the latest TSV file in chunks from SFTP and uploads to BigQuery.
    """
    delete_table_if_exists(DATASET_ID, TABLE_NAME)
    
    private_key = get_private_key_from_gcs(GCS_BUCKET, SFTP_PRIVATE_KEY_BLOB)
    if not private_key:
        return
    
    latest_file = get_latest_tsv_file(SFTP_HOST, SFTP_PORT, SFTP_USERNAME, private_key, REMOTE_BASE_PATH)
    if not latest_file:
        print("[ERROR] No valid TSV file found. Exiting process.")
        return
    
    start_byte = 0
    chunk_counter = 1
    while True:
        print(f"\n[INFO] Starting download for chunk {chunk_counter} (Starting byte: {start_byte})...")
        chunk, next_start_byte = download_chunk_from_sftp(SFTP_HOST, SFTP_PORT, SFTP_USERNAME, private_key, latest_file, CHUNK_SIZE, start_byte)
        
        if not chunk:
            print("\n[INFO] No more chunks to download. Process completed.")
            break
        
        process_and_upload_chunk(chunk, SCHEMA, DATASET_ID, TABLE_NAME, WRITE_MODE)
        
        if next_start_byte is None:
            break
        
        start_byte = next_start_byte
        chunk_counter += 1

if __name__ == "__main__":
    print("\n[INFO] Starting file processing...\n")
    process_file_from_sftp_in_chunks()
    print("\n[SUCCESS] File processing completed!")
