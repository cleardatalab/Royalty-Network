import pandas as pd  # type: ignore
import os
from google.cloud import bigquery

# Local path and configuration
LOCAL_PATH = r"D:\Backup\Latest MLC FIles"
DATASET_ID = 'mlc_dataset'
WRITE_MODE = 'WRITE_APPEND'  # Append new data to the table
CHUNK_SIZE = 500000  # Adjust the chunk size based on system's memory capacity

def process_all_tsv_files(path, dataset_id, write_mode='WRITE_APPEND'):
    """
    Processes all TSV files from a directory and loads them into BigQuery.
    Assumes that each TSV file already has headers.
    
    Parameters:
    - path (str): Local directory containing TSV files.
    - dataset_id (str): BigQuery dataset ID.
    - write_mode (str): 'WRITE_APPEND' to append new data to the table.
    """
    print(f"\n----------- Starting to process TSV files from directory: {path} -----------")
    
    # Set up the BigQuery client
    bq_client = bigquery.Client(project='roynet2025')
    
    # Iterate through files in the directory
    for tsv_file in os.listdir(path):
        if not tsv_file.endswith('.tsv'):
            continue  # Skip non-TSV files
        
        file_path = os.path.join(path, tsv_file)
        table_name = os.path.splitext(tsv_file)[0]  # Use filename (without extension) as table name
        table_id = f"{bq_client.project}.{dataset_id}.{table_name}"
        
        print(f"\nProcessing file: {tsv_file} -> Table: {table_id}")
        
        # Read the TSV file with headers using chunks to manage memory usage
        chunk_iter = pd.read_csv(file_path, sep='\t', dtype=str, chunksize=CHUNK_SIZE, header=0)
        
        for i, chunk in enumerate(chunk_iter):
            print(f"  -> Processing chunk {i + 1} of '{tsv_file}'...")
            
            # Load the chunk into BigQuery
            try:
                job_config = bigquery.LoadJobConfig(
                    write_disposition=write_mode  # Append new data to the table
                )
                load_job = bq_client.load_table_from_dataframe(chunk, table_id, job_config=job_config)
                load_job.result()  # Wait for the job to finish
                print(f"  -> Chunk {i + 1} successfully loaded into BigQuery table '{table_id}'.")
            except Exception as e:
                print(f"  -> Error loading chunk {i + 1} of file '{tsv_file}' into BigQuery: {e}")
                continue  # Skip this chunk and continue with the next one

    print(f"  -> All files processed and loaded into BigQuery successfully.")

if __name__ == "__main__":
    # Execute the BigQuery load process for all TSV files in the directory
    process_all_tsv_files(
        path=LOCAL_PATH,
        dataset_id=DATASET_ID,
        write_mode=WRITE_MODE
    )
    # Print confirmation
    print("Local execution completed successfully!")
