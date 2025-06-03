"""
Attendance Table Creator for Sentiment Analysis Project

This module creates the 'attendance' table in the PostgreSQL database if it doesn't exist.
It handles table creation with proper schema definition and provides informative
logging about the table's existence status.

Functions:
- create_attendance_table(): Creates the attendance table with proper schema
- table_exists(): Checks if the attendance table already exists

Usage:
    python create_attendance_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def table_exists(table_name, connection):
    """
    Check if a table exists in the current database.
    
    Args:
        table_name (str): Name of the table to check
        connection: Database connection object
        
    Returns:
        bool: True if table exists, False otherwise
    """
    check_query = """
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = %s
    )
    """
    
    try:
        result = db_connector.fetch_data(check_query, (table_name,), connection=connection)
        return result[0][0]  # Returns True/False
    except Exception as e:
        print(f"[{datetime.now()}] Error checking table existence: {e}")
        return False

def create_attendance_table():
    """
    Create the attendance table with appropriate schema for the CSV data.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting attendance table creation process")
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if attendance table already exists
        if table_exists('attendance', connection):
            print(f"[{datetime.now()}] Attendance table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'attendance' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing attendance table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            return
        
        # Create attendance table if it doesn't exist
        print(f"[{datetime.now()}] Attendance table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE attendance (
            attendance_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            learning_content_surrogate_key INTEGER NOT NULL,
            date_effective DATE NOT NULL,
            status VARCHAR(255) NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (learning_content_surrogate_key) REFERENCES learning_content(surrogate_key)
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] Attendance table created successfully")
        
        # Verify table creation and show structure
        if table_exists('attendance', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'attendance' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] New attendance table structure:")
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
    print(f"[{datetime.now()}] Attendance table creation script started")
    print("=" * 60)
    
    try:
        create_attendance_table()
        print("=" * 60)
        print(f"[{datetime.now()}] Attendance table creation script completed successfully")
    except Exception as e:
        print("=" * 60)
        print(f"[{datetime.now()}] Attendance table creation script failed: {e}")
