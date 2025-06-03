"""
Evaluation Data Loader for Sentiment Analysis Project

This module loads evaluation data from CSV files into the PostgreSQL database.
It reads evaluation information (response_id, user_id, learning_content_surrogate_key, 
course_end_date, etc.) from CSV format and inserts it into the 'evaluation' table
using batch operations for efficiency.

Functions:
- load_evaluation_from_csv_to_db(): Main function to load CSV data into database

Usage:
    python load_evaluation_data.py

Dependencies:
- pandas: CSV file processing
- os: File path operations
- datetime: Timestamp logging
- db_connector: Database operations
"""

import pandas as pd
import os
from datetime import datetime
import db_connector  # Direct import since we're in the same directory

def load_evaluation_from_csv_to_db(csv_file_path):
    """
    Reads evaluation data from a CSV file and inserts it into the 'evaluation' table in PostgreSQL.
    """
    try:
        # 1. Read CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"[{datetime.now()}] Read {len(df)} rows from {csv_file_path}")

        # Ensure column names in CSV match what we expect for the DB
        expected_columns = {
            'response_id', 'user_id', 'learning_content_surrogate_key', 'course_end_date', 
            'course_delivery_type', 'agency', 'attendance_motivation', 'positive_learning_experience', 
            'effective_use_of_time', 'relevant_to_work', 'did_experience_issue', 
            'did_experience_issue_detail', 'facilitator_skills', 'had_guest_speakers', 
            'guest_contribution', 'knowledge_level_prior', 'course_application', 
            'course_application_other', 'course_application_timeframe', 'general_feedback'
        }
        
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

        # 3. Prepare data for batch insertion
        table_name = "evaluation"
        columns = [
            'response_id', 'user_id', 'learning_content_surrogate_key', 'course_end_date',
            'course_delivery_type', 'agency', 'attendance_motivation', 'positive_learning_experience',
            'effective_use_of_time', 'relevant_to_work', 'did_experience_issue',
            'did_experience_issue_detail', 'facilitator_skills', 'had_guest_speakers',
            'guest_contribution', 'knowledge_level_prior', 'course_application',
            'course_application_other', 'course_application_timeframe', 'general_feedback'
        ]

        # Convert DataFrame rows to a list of tuples
        # Handle NaN values and ensure correct data types
        if 'response_id' in df.columns and df['response_id'].isnull().any():
            print(f"[{datetime.now()}] Warning: 'response_id' column contains null values. These rows will be skipped.")
            df.dropna(subset=['response_id'], inplace=True)
        
        # Convert ID columns to integers
        df['response_id'] = df['response_id'].astype(int)
        df['user_id'] = df['user_id'].astype(int)
        df['learning_content_surrogate_key'] = df['learning_content_surrogate_key'].astype(int)
        
        # Handle nulls in numeric columns
        numeric_columns = ['positive_learning_experience', 'effective_use_of_time', 'relevant_to_work']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Replace empty strings with None for text fields
        text_columns = [
            'attendance_motivation', 'did_experience_issue', 'did_experience_issue_detail',
            'facilitator_skills', 'guest_contribution', 'course_application',
            'course_application_other', 'general_feedback'
        ]
        for col in text_columns:
            df[col] = df[col].replace('', None)
        
        data_to_insert = [tuple(x) for x in df[columns].to_numpy()]

        if not data_to_insert:
            print(f"[{datetime.now()}] No valid data to insert after processing the DataFrame.")
            return

        # 4. Check for potential duplicates based on response_id
        try:
            existing_count_query = "SELECT COUNT(*) FROM evaluation"
            existing_count_result = db_connector.fetch_data(existing_count_query, connection=connection)
            existing_count = existing_count_result[0][0]
            
            if existing_count > 0:
                print(f"[{datetime.now()}] Checking for duplicate response_ids...")
                
                existing_ids_query = "SELECT response_id FROM evaluation"
                existing_ids_result = db_connector.fetch_data(existing_ids_query, connection=connection)
                existing_ids = {row[0] for row in existing_ids_result}
                
                csv_ids = {row[0] for row in data_to_insert}
                duplicate_ids = existing_ids.intersection(csv_ids)
                
                if duplicate_ids:
                    print(f"[{datetime.now()}] Warning: Found {len(duplicate_ids)} potential duplicate response_ids")
                    print(f"[{datetime.now()}] Duplicate IDs: {sorted(list(duplicate_ids))[:10]}{'...' if len(duplicate_ids) > 10 else ''}")
                    print(f"[{datetime.now()}] Skipping insert to prevent conflicts. Please review data manually.")
                    return
                else:
                    print(f"[{datetime.now()}] No duplicate response_ids found, proceeding with insert")
        except Exception as e:
            print(f"[{datetime.now()}] Error checking for duplicates: {e}")
            print(f"[{datetime.now()}] Proceeding with insert anyway")

        # 5. Insert data into the database
        print(f"[{datetime.now()}] Attempting to insert {len(data_to_insert)} rows into '{table_name}' table...")
        
        # Create placeholders for all columns
        placeholders = ', '.join(['%s'] * len(columns))
        
        # Create the INSERT query for the evaluation table
        insert_query = f"""
        INSERT INTO {table_name} 
        ({', '.join(columns)}) 
        VALUES ({placeholders})
        """
        
        # Use the batch insert function
        rows_inserted = db_connector.batch_insert_data(insert_query, data_to_insert, connection)
        print(f"[{datetime.now()}] Successfully inserted {rows_inserted} rows into '{table_name}' table")
        
        # Show sample of inserted data
        if rows_inserted > 0:
            sample_query = """
            SELECT e.response_id, e.user_id, u.user_level, e.learning_content_surrogate_key, 
                   l.name, e.course_delivery_type, e.positive_learning_experience
            FROM evaluation e
            JOIN users u ON e.user_id = u.user_id
            JOIN learning_content l ON e.learning_content_surrogate_key = l.surrogate_key
            ORDER BY e.response_id
            LIMIT 3
            """
            
            try:
                sample_results = db_connector.fetch_data(sample_query, connection=connection)
                print(f"[{datetime.now()}] Sample of inserted data with user and content details:")
                for row in sample_results:
                    print(f"  - Response ID: {row[0]}, User: {row[1]} ({row[2]}), Content: {row[4]}, Type: {row[5]}, Rating: {row[6]}")
            except Exception as e:
                print(f"[{datetime.now()}] Could not fetch sample data: {e}")

    except Exception as e:
        print(f"[{datetime.now()}] An error occurred during the data loading process: {e}")
    finally:
        # 6. Close database connection
        if connection:
            db_connector.close_db_connection(connection)

if __name__ == '__main__':
    # Get the correct path to the evaluation.csv file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'csv', 'evaluation.csv')
    
    print(f"[{datetime.now()}] Evaluation data loading script started")
    print("=" * 60)
    print(f"[{datetime.now()}] Looking for CSV file at: {csv_path}")
    
    if os.path.exists(csv_path):
        try:
            load_evaluation_from_csv_to_db(csv_path)
            print("=" * 60)
            print(f"[{datetime.now()}] Evaluation data loading script completed successfully")
        except Exception as e:
            print("=" * 60)
            print(f"[{datetime.now()}] Evaluation data loading script failed: {e}")
    else:
        print(f"[{datetime.now()}] Error: CSV file not found at {csv_path}")
        print(f"[{datetime.now()}] Please ensure the evaluation.csv file is in the src/csv/ directory")
