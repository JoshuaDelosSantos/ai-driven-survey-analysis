"""
User Data Loader for Sentiment Analysis Project

This module loads user data from CSV files into the PostgreSQL database.
It reads user information (user_id, user_level, agency) from CSV format
and inserts it into the 'users' table using batch operations for efficiency.

Functions:
- load_users_from_csv_to_db(): Main function to load CSV data into database

Usage:
    python load_user_data.py

Dependencies:
- pandas: CSV file processing
- os: File path operations
- db_connector: Database operations
"""

import pandas as pd
import os
import db_connector  # Direct import since we're in the same directory

def load_users_from_csv_to_db(csv_file_path):
    """
    Reads user data from a CSV file and inserts it into the 'users' table in PostgreSQL.
    """
    try:
        # 1. Read CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Read {len(df)} rows from {csv_file_path}")

        # Ensure column names in CSV match what we expect for the DB
        # Expected columns from User - Sheet1.csv: user_id, user_level, agency
        # These will map directly to the 'users' table columns
        if not {'user_id', 'user_level', 'agency'}.issubset(df.columns):
            print("Error: CSV file is missing one or more required columns: 'user_id', 'user_level', 'agency'.")
            print(f"Found columns: {df.columns.tolist()}")
            return

    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file_path} is empty.")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    db_conn = None
    try:
        # 2. Establish database connection
        db_conn = db_connector.get_db_connection()
        if not db_conn:
            print("Could not connect to the database. Aborting.")
            return

        # 3. Prepare data for batch insertion
        # The 'users' table columns are: user_id, user_level, agency
        table_name = "users"
        columns = ['user_id', 'user_level', 'agency']

        # Convert DataFrame rows to a list of tuples
        # Ensure data types are compatible; pandas usually handles this well from CSV.
        # If user_id is read as float by pandas due to NaNs (though not expected for this key), cast to int.
        if 'user_id' in df.columns and df['user_id'].isnull().any():
            print("Warning: 'user_id' column contains null values. These rows might be skipped or cause errors if 'user_id' is a PRIMARY KEY.")
            df.dropna(subset=['user_id'], inplace=True) # Remove rows where user_id is NaN
        
        # Convert user_id to integer if it's not already, to match INTEGER PRIMARY KEY
        df['user_id'] = df['user_id'].astype(int)

        data_to_insert = [tuple(x) for x in df[columns].to_numpy()]

        if not data_to_insert:
            print("No valid data to insert after processing the DataFrame.")
            return

        # 4. Insert data into the database
        print(f"Attempting to insert/process {len(data_to_insert)} rows into '{table_name}' table...")
        
        # Create the INSERT query for the users table
        insert_query = f"INSERT INTO {table_name} (user_id, user_level, agency) VALUES (%s, %s, %s)"
        
        # Use the correct function signature
        rows_inserted = db_connector.batch_insert_data(insert_query, data_to_insert, db_conn)
        print(f"Successfully inserted {rows_inserted} rows into '{table_name}' table")

    except Exception as e:
        print(f"An error occurred during the data loading process: {e}")
    finally:
        # 5. Close database connection
        if db_conn:
            db_connector.close_db_connection(db_conn)

if __name__ == '__main__':
    # Get the correct path to the user.csv file
    # The CSV is in the src/csv/ directory, and this script is in src/sentiment-analysis/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'csv', 'user.csv')
    
    print(f"Looking for CSV file at: {csv_path}")
    
    if os.path.exists(csv_path):
        load_users_from_csv_to_db(csv_path)
    else:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the user.csv file is in the src/csv/ directory")