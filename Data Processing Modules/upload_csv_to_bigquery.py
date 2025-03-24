import pandas as pd
from google.cloud import bigquery
import pyarrow as pa
import pyarrow.parquet as pq

# File Path
csv_file = r"claim_report_royaltynetwork_C_v1-1.csv\asset_share_lifetime_asset_share_v1_1_20250223.csv"

# BigQuery Details
PROJECT_ID = "roynet2025"
DATASET_ID = "Youtube_Dataset"
TABLE_NAME = "asset_share"

# Read CSV
print("Reading CSV file...")
df = pd.read_csv(csv_file, delimiter=",", quoting=3, encoding="utf-8", on_bad_lines="skip", low_memory=False)
print(f"Successfully read CSV file with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Detect Column Data Types
column_types = {}
mixed_columns = {}

print("Checking column data types...\n")
for col in df.columns:
    unique_types = set(type(x) for x in df[col].dropna())
    
    if len(unique_types) == 1:
        dtype = list(unique_types)[0]
        if dtype == int:
            column_types[col] = "INTEGER"
        elif dtype == float:
            column_types[col] = "FLOAT"
        else:
            column_types[col] = "STRING"
    else:
        mixed_columns[col] = unique_types
        column_types[col] = "STRING"

# Print Column Types
for col, dtype in column_types.items():
    if col in mixed_columns:
        print(f"{col} - mixed value {mixed_columns[col]} (converted to STRING)")
    else:
        print(f"{col} - {dtype.lower()}")

print("\nData type check completed.\n")

# Convert Mixed-Type Columns to String
for col in mixed_columns:
    df[col] = df[col].astype(str)

# Upload to BigQuery
client = bigquery.Client(project=PROJECT_ID)
table_ref = client.dataset(DATASET_ID).table(TABLE_NAME)

def upload_to_bigquery(df, column_types):
    print("Starting data upload to BigQuery...")
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(col, dtype) for col, dtype in column_types.items()],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for job completion
    print("Data upload completed successfully.")

upload_to_bigquery(df, column_types)
