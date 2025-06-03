"""
Evaluation Table Creator for Sentiment Analysis Project

This module creates the 'evaluation' table in the PostgreSQL database if it doesn't exist.
It handles table creation with proper schema definition and provides informative
logging about the table's existence status.

Functions:
- create_evaluation_table(): Creates the evaluation table with proper schema

Usage:
    python create_evaluation_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def create_evaluation_table():
    """
    Create the evaluation table with appropriate schema for the CSV data.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting evaluation table creation process")
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if evaluation table already exists
        if db_connector.table_exists('evaluation', connection):
            print(f"[{datetime.now()}] Evaluation table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'evaluation' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing evaluation table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            return
        
        # Create evaluation table if it doesn't exist
        print(f"[{datetime.now()}] Evaluation table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE evaluation (
            response_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            learning_content_surrogate_key INTEGER NOT NULL,
            course_end_date DATE,
            course_delivery_type VARCHAR(255),
            agency VARCHAR(255),
            attendance_motivation TEXT,
            positive_learning_experience INTEGER,
            effective_use_of_time INTEGER,
            relevant_to_work INTEGER,
            did_experience_issue TEXT,
            did_experience_issue_detail TEXT,
            facilitator_skills TEXT,
            had_guest_speakers VARCHAR(50),
            guest_contribution TEXT,
            knowledge_level_prior VARCHAR(255),
            course_application TEXT,
            course_application_other TEXT,
            course_application_timeframe VARCHAR(255),
            general_feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (learning_content_surrogate_key) REFERENCES learning_content(surrogate_key)
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] Evaluation table created successfully")
        
        # Verify table creation and show structure
        if db_connector.table_exists('evaluation', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'evaluation' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] New evaluation table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
        else:
            print(f"[{datetime.now()}] Warning: Table creation may have failed")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error during table creation process: {e}")
        raise
    finally:
        if connection:
            db_connector.close_db_connection(connection)

if __name__ == "__main__":
    print(f"[{datetime.now()}] Evaluation table creation script started")
    print("=" * 60)
    
    try:
        create_evaluation_table()
        print("=" * 60)
        print(f"[{datetime.now()}] Evaluation table creation script completed successfully")
    except Exception as e:
        print("=" * 60)
        print(f"[{datetime.now()}] Evaluation table creation script failed: {e}")
