import ftplib
import os
import pyodbc
import logging

# ------------------------------------
# Configure Logging
# ------------------------------------
logging.basicConfig(
    filename='automate_db_backup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------
# FTP Configuration
# ------------------------------------
FTP_SERVER = 'ftp.yourserver.com'  # Change to your FTP server
USERNAME = 'your_username'           # Change to your FTP username
PASSWORD = 'your_password'           # Change to your FTP password
REMOTE_FILE_PATH = '/path/to/backup.bak'  # Change to the path of the SQL backup file on the FTP server
LOCAL_FILE_PATH = 'backup.bak'       # Local path to save the downloaded file

# ------------------------------------
# SQL Server Configuration
# ------------------------------------
SQL_SERVER_CONNECTION_STRING = (
    'Driver={SQL Server};'
    'Server=your_sql_server;'
    'UID=your_sql_username;'          # Change to your SQL Server username
    'PWD=your_sql_password;'          # Change to your SQL Server password
)

NEW_DATABASE_NAME = 'NewDatabase'   # Change to your desired new database name
RESTORED_DATABASE_NAME = 'RestoredDB'  # Change to your desired name for the restored database

# ------------------------------------
# Download FTP File
# ------------------------------------
def download_ftp_file():
    """Download the database backup file from FTP server."""
    try:
        logging.info("Connecting to FTP server.")
        with ftplib.FTP(FTP_SERVER) as ftp:
            ftp.login(user=USERNAME, passwd=PASSWORD)
            logging.info(f"Connected to FTP server: {FTP_SERVER}")
            
            with open(LOCAL_FILE_PATH, 'wb') as local_file:
                ftp.retrbinary(f'RETR {REMOTE_FILE_PATH}', local_file.write)
        logging.info("File downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download file: {e}")

# ------------------------------------
# Restore Database from .bak File
# ------------------------------------
def restore_database():
    """Restore database from .bak file."""
    try:
        logging.info(f"Restoring database from {LOCAL_FILE_PATH}.")
        with pyodbc.connect(SQL_SERVER_CONNECTION_STRING) as conn:
            cursor = conn.cursor()
            cursor.execute(f"ALTER DATABASE {RESTORED_DATABASE_NAME} SET SINGLE_USER WITH ROLLBACK IMMEDIATE;")
            cursor.execute(f"RESTORE DATABASE {RESTORED_DATABASE_NAME} FROM DISK = '{os.path.abspath(LOCAL_FILE_PATH)}' WITH REPLACE;")
            cursor.execute(f"ALTER DATABASE {RESTORED_DATABASE_NAME} SET MULTI_USER;")
            conn.commit()
        logging.info(f"Database {RESTORED_DATABASE_NAME} restored successfully.")
    except Exception as e:
        logging.error(f"Failed to restore database: {e}")

# ------------------------------------
# Update Main Database
# ------------------------------------
def update_main_database():
    """Update main database with only new or changed records."""
    try:
        logging.info("Updating main database with new or changed records.")
        with pyodbc.connect(SQL_SERVER_CONNECTION_STRING) as conn:
            cursor = conn.cursor()

            # Create Staging Table if it doesn't exist
            cursor.execute("""
            IF OBJECT_ID('tempdb..#StagingTable') IS NOT NULL
                DROP TABLE #StagingTable;
            CREATE TABLE #StagingTable (
                Id INT PRIMARY KEY,
                Column1 VARCHAR(255),
                Column2 INT
                -- Add other columns as needed
            );
            """)

            # Insert data into Staging Table
            cursor.execute(f"""
            INSERT INTO #StagingTable (Id, Column1, Column2)
            SELECT Id, Column1, Column2 FROM {RESTORED_DATABASE_NAME}.dbo.MainTable;
            """)

            # Insert updated values into Main Table
            cursor.execute("""
            INSERT INTO MainTable (Id, Column1, Column2)
            SELECT s.Id, s.Column1, s.Column2
            FROM #StagingTable s
            LEFT JOIN MainTable m ON s.Id = m.Id
            WHERE m.Id IS NULL OR s.Column1 <> m.Column1 OR s.Column2 <> m.Column2;
            """)
            conn.commit()
        logging.info("Main database updated successfully.")
    except Exception as e:
        logging.error(f"Failed to update main database: {e}")

# ------------------------------------
# Create New Database
# ------------------------------------
def create_new_database():
    """Create a new database and populate it with selected values."""
    try:
        logging.info(f"Creating new database: {NEW_DATABASE_NAME}.")
        with pyodbc.connect(SQL_SERVER_CONNECTION_STRING) as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {NEW_DATABASE_NAME};")
            cursor.execute(f"USE {NEW_DATABASE_NAME};")

            # Create table in the new database
            cursor.execute("""
            CREATE TABLE SelectedValues (
                Id INT PRIMARY KEY,
                Column1 VARCHAR(255)
                -- Add other columns as needed
            );
            """)

            # Insert selected values from MainTable
            cursor.execute(f"""
            INSERT INTO {NEW_DATABASE_NAME}.dbo.SelectedValues (Id, Column1)
            SELECT Id, Column1
            FROM MainTable
            WHERE Column2 > 100;  -- Specify your selection conditions here
            """)
            conn.commit()
        logging.info(f"{NEW_DATABASE_NAME} created and populated successfully.")
    except Exception as e:
        logging.error(f"Failed to create new database: {e}")

# ------------------------------------
# Main Execution
# ------------------------------------
def main():
    download_ftp_file()
    restore_database()
    update_main_database()
    create_new_database()

if __name__ == "__main__":
    main()
