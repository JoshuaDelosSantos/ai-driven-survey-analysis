"""
Learning Content Data Loader for Sentiment Analysis Project

This module loads learning content data from CSV files into the PostgreSQL database.
It provides functionality to load CSV data in batches with proper error handling
and duplicate detection.

Functions:
- load_learning_content_data(): Loads learning content data from CSV into PostgreSQL
- count_existing_records(): Counts existing records in the learning_content table

Usage:
    python load_learning_content_data.py

Dependencies:
- db_connector: Database operations
- pandas: CSV data processing
- datetime: Timestamp logging
"""

import db_connector
import pandas as pd
from datetime import datetime
import os

def count_existing_records(connection):
    """
    Count the number of existing records in the learning_content table.
    
    Args:
        connection: Database connection object
        
    Returns:
        int: Number of existing records, or 0 if table doesn't exist
    """
    try:
        count_query = "SELECT COUNT(*) FROM learning_content"
        result = db_connector.fetch_data(count_query, connection=connection)
        return result[0][0]
    except Exception as e:
        print(f"[{datetime.now()}] Warning: Could not count existing records: {e}")
        return 0

def load_learning_content_data(csv_file_path):
    """
    Load learning content data from CSV file into the PostgreSQL database.
    Handles duplicate prevention and provides detailed logging.
    """
    try:
        # 1. Read CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"[{datetime.now()}] Read {len(df)} rows from {csv_file_path}")

        # Ensure column names in CSV match what we expect for the DB
        # Expected columns: surrogate_key, name, content_id, content_type, target_level, governing_bodies
        expected_columns = {'surrogate_key', 'name', 'content_id', 'content_type', 'target_level', 'governing_bodies'}
        if not expected_columns.issubset(df.columns):
            print(f"[{datetime.now()}] Error: CSV file is missing one or more required columns: {expected_columns}")
            print(f"[{datetime.now()}] Found columns: {df.columns.tolist()}")
            return

    except FileNotFoundError:
        print(f"[{datetime.now()}] Error: The file {csv_file_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"[{datetime.now()}] Error: The file {csv_file_path} is empty.")
        return
    except Exception as e:
        print(f"[{datetime.now()}] Error reading CSV file {csv_file_path}: {e}")
        return

    connection = None
    try:
        # 2. Establish database connection
        connection = db_connector.get_db_connection()
        if not connection:
            print(f"[{datetime.now()}] Could not connect to the database. Aborting.")
            return

        # 3. Check existing record count
        existing_count = count_existing_records(connection)
        print(f"[{datetime.now()}] Existing records in learning_content table: {existing_count}")

        # 4. Prepare data for batch insertion
        table_name = "learning_content"
        columns = ['surrogate_key', 'name', 'content_id', 'content_type', 'target_level', 'governing_bodies']

        # Show first few rows for verification
        print(f"[{datetime.now()}] Sample data (first 3 rows):")
        for idx, row in df.head(3).iterrows():
            print(f"  Row {idx + 1}: surrogate_key={row['surrogate_key']}, name='{row['name'][:50]}{'...' if len(row['name']) > 50 else ''}'")

        # Convert DataFrame rows to a list of tuples with proper data types
        if 'surrogate_key' in df.columns and df['surrogate_key'].isnull().any():
            print(f"[{datetime.now()}] Warning: 'surrogate_key' column contains null values. These rows might be skipped or cause errors if 'surrogate_key' is a PRIMARY KEY.")
            df.dropna(subset=['surrogate_key'], inplace=True)
        
        # Convert surrogate_key to integer if it's not already, to match INTEGER PRIMARY KEY
        df['surrogate_key'] = df['surrogate_key'].astype(int)
        df['content_id'] = df['content_id'].astype(int)

        data_to_insert = [tuple(x) for x in df[columns].to_numpy()]

        if not data_to_insert:
            print(f"[{datetime.now()}] No valid data to insert after processing the DataFrame.")
            return
        # 5. Check for potential duplicates based on surrogate_key
        if existing_count > 0:
            print(f"[{datetime.now()}] Checking for duplicate surrogate_keys...")
            
            existing_keys_query = "SELECT surrogate_key FROM learning_content"
            existing_keys_result = db_connector.fetch_data(existing_keys_query, connection=connection)
            existing_keys = {row[0] for row in existing_keys_result}
            
            csv_keys = {row[0] for row in data_to_insert}
            duplicate_keys = existing_keys.intersection(csv_keys)
            
            if duplicate_keys:
                print(f"[{datetime.now()}] Warning: Found {len(duplicate_keys)} potential duplicate surrogate_keys:")
                print(f"[{datetime.now()}] Duplicate keys: {sorted(list(duplicate_keys))[:10]}{'...' if len(duplicate_keys) > 10 else ''}")
                print(f"[{datetime.now()}] Skipping insert to prevent conflicts. Please review data manually.")
                return
            else:
                print(f"[{datetime.now()}] No duplicate surrogate_keys found, proceeding with insert")

        # 6. Insert data into the database
        print(f"[{datetime.now()}] Attempting to insert/process {len(data_to_insert)} rows into '{table_name}' table...")
        
        # Create the INSERT query for the learning_content table
        insert_query = f"INSERT INTO {table_name} (surrogate_key, name, content_id, content_type, target_level, governing_bodies) VALUES (%s, %s, %s, %s, %s, %s)"
        
        # Use the correct function signature
        rows_inserted = db_connector.batch_insert_data(insert_query, data_to_insert, connection)
        print(f"[{datetime.now()}] Successfully inserted {rows_inserted} rows into '{table_name}' table")

    except Exception as e:
        print(f"[{datetime.now()}] An error occurred during the data loading process: {e}")
    finally:
        # 7. Close database connection
        if connection:
            db_connector.close_db_connection(connection)

if __name__ == "__main__":
    # Get the correct path to the learning_content.csv file
    # The CSV is in the src/csv/ directory, and this script is in src/db/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'csv', 'learning_content.csv')
    
    print(f"[{datetime.now()}] Learning content data loading script started")
    print("=" * 60)
    print(f"[{datetime.now()}] Looking for CSV file at: {csv_path}")
    
    if os.path.exists(csv_path):
        try:
            load_learning_content_data(csv_path)
            print("=" * 60)
            print(f"[{datetime.now()}] Learning content data loading script completed successfully")
        except Exception as e:
            print("=" * 60)
            print(f"[{datetime.now()}] Learning content data loading script failed: {e}")
    else:
        print(f"[{datetime.now()}] Error: CSV file not found at {csv_path}")
        print(f"[{datetime.now()}] Please ensure the learning_content.csv file is in the src/csv/ directory")
