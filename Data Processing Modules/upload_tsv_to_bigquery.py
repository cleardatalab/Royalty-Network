import pandas as pd  # type: ignore
from io import StringIO
import os
from google.cloud import bigquery

# Local path and configuration
LOCAL_PATH = r"C:\Users\mshah\Desktop\MLC_OG\Latest Unclame File"
TSV_FILE = 'unclaimedworkrightshares.tsv'
DATASET_ID = 'mlc_dataset'
TABLE_NAME = 'unclaimedworkrightshares'
WRITE_MODE = 'WRITE_APPEND'  # Ensure data is appended, not replaced
SCHEMA = [
    "FeedProvidersRightShareId", "FeedProvidersRecordingId",
    "FeedProvidersWorkId", "ISRC", "DspRecordingId", "RecordingTitle",
    "RecordingSubTitle", "AlternativeRecordingTitle", "DisplayArtistName",
    "DisplayArtistISNI", "Duration", "UnclaimedPercentage",
    "PercentileForPrioritisation", "snapshotid"
]

CHUNK_SIZE = 500000  # Adjust the chunk size based on your system's memory capacity

def process_file_from_local(path, tsv_file, schema, dataset_id, table_name, write_mode='WRITE_APPEND'):
    """
    Processes a large TSV file from a local path, chunks it, and loads it into BigQuery.

    Parameters:
    - path (str): Local file path where the TSV file is stored.
    - tsv_file (str): The name of the TSV file in the local directory.
    - schema (list): The schema list of column names.
    - dataset_id (str): BigQuery dataset ID.
    - table_name (str): BigQuery table name.
    - write_mode (str): 'WRITE_APPEND' to append new data to the table.
    """
    print(f"\n----------- Starting to process the file: {tsv_file} from local path: {path} -----------")

    # Step 1: Open the file in chunks using pandas
    file_path = os.path.join(path, tsv_file)
    
    # Set up the BigQuery client
    bq_client = bigquery.Client(project='roynet2025')
    table_id = f"{bq_client.project}.{dataset_id}.{table_name}"

    # Step 2: Process the file in chunks to avoid memory overload
    chunk_iter = pd.read_csv(file_path, sep='\t', header=None, dtype=str, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(chunk_iter):
        print(f"  -> Processing chunk {i + 1}...")

        # Step 3: Clean the chunk and adjust schema length
        chunk.columns = range(chunk.shape[1])  # Assign numerical column names for cleanup
        last_valid_index = chunk.notna().cumsum(axis=1).idxmax(axis=1)
        max_valid_index = last_valid_index.max()
        chunk_trimmed = chunk.iloc[:, :max_valid_index + 1]

        if chunk_trimmed.shape[1] < len(schema):
            extra_columns = len(schema) - chunk_trimmed.shape[1]
            chunk_trimmed = pd.concat([
                chunk_trimmed,
                pd.DataFrame([[None] * extra_columns] * chunk_trimmed.shape[0], columns=schema[-extra_columns:])
            ], axis=1)

        chunk_trimmed.columns = schema  # Assign schema to the chunk

        # Step 4: Load the chunk into BigQuery using WRITE_APPEND mode
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition=write_mode  # Ensure we're appending data, not replacing
            )
            load_job = bq_client.load_table_from_dataframe(
                chunk_trimmed, table_id, job_config=job_config
            )
            load_job.result()  # Wait for the job to finish
            print(f"  -> Chunk {i + 1} successfully loaded into BigQuery table '{table_id}'.")

        except Exception as e:
            print(f"  -> Error loading chunk {i + 1} into BigQuery: {e}")
            continue  # Skip this chunk and continue with the next chunk

    print(f"  -> All chunks processed and loaded into BigQuery successfully.")

if __name__ == "__main__":
    # Execute the BigQuery load process for the local file
    process_file_from_local(
        path=LOCAL_PATH,
        tsv_file=TSV_FILE,
        schema=SCHEMA,
        dataset_id=DATASET_ID,
        table_name=TABLE_NAME,
        write_mode=WRITE_MODE
    )
    # Print confirmation
    print("Local execution completed successfully!")
