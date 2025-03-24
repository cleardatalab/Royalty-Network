import pandas as pd
from google.cloud import bigquery
import os
import re
import csv  # Import csv module for correct quoting

# Set your variables
FILE_PATH = r"muma_03_05_2025.txt"
DATASET_ID = "roynet"
TABLE_ID = "income_archived_115"
PROJECT_ID = "roynet2025"
CHUNK_SIZE = 500000

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

def clean_column_name(col_name):
    """Remove invalid characters from column names."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", col_name.strip())

def infer_schema_as_string(file_path, sample_size=10000):
    """Infer schema where all fields are treated as STRING."""
    try:
        df_sample = pd.read_csv(
            file_path, 
            nrows=sample_size, 
            dtype=str, 
            encoding='latin1', 
            on_bad_lines='skip', 
            sep=',',  # Use comma as separator
            quotechar='"',  # Treat double quotes as enclosing characters
            quoting=csv.QUOTE_MINIMAL  # Preserve quoted values with commas
        )

        # Clean column names
        df_sample.columns = [clean_column_name(col) for col in df_sample.columns]

        schema = [bigquery.SchemaField(col, "STRING") for col in df_sample.columns]
        return schema

    except Exception as e:
        print(f"Schema inference failed: {e}")
        return []

def load_data_to_bigquery(file_path, dataset_id, table_id, project_id, chunk_size=500000):
    """Load large CSV file into BigQuery, converting all data to STRING."""
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        schema = infer_schema_as_string(file_path)
        if not schema:
            print("Schema inference failed. Aborting data load.")
            return

        job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
        total_rows = 0
        chunk_count = 0

        for chunk in pd.read_csv(
            file_path, 
            chunksize=chunk_size, 
            dtype=str, 
            encoding='latin1', 
            low_memory=False, 
            on_bad_lines='skip', 
            sep=',',  # Use comma as separator
            quotechar='"',  # Treat double quotes as enclosing characters
            quoting=csv.QUOTE_MINIMAL  # Preserve quoted values with commas
        ):
            # Clean column names
            chunk.columns = [clean_column_name(col) for col in chunk.columns]

            chunk = chunk.astype(str)  # Convert all columns to STRING
            chunk.replace({"NULL": None, "nan": None, "NaN": None}, inplace=True)  # Handle NULL values

            job = client.load_table_from_dataframe(chunk, table_ref, job_config=job_config)
            job.result()  # Wait for job to complete

            chunk_count += 1
            total_rows += len(chunk)
            print(f"Chunk {chunk_count}: Loaded {len(chunk)} rows. Total rows loaded: {total_rows}")

        print("Data loading completed successfully.")

    except Exception as e:
        print(f"Data loading failed: {e}")

# Run the function
load_data_to_bigquery(FILE_PATH, DATASET_ID, TABLE_ID, PROJECT_ID, CHUNK_SIZE)
