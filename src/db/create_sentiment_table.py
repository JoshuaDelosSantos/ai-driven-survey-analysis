"""
Sentiment Table Creator for AI-Driven Analysis project

Creates the sentiment scores table if it does not already exist.
Checks for table existence, logs status, and defines schema with:
- response_id (FK to evaluation.response_id)
- column_name (TEXT)
- neg, neu, pos (FLOAT)
- created_at timestamp default NOW()
Primary key: (response_id, column_name)
"""
import db_connector
from datetime import datetime

SENTIMENT_TABLE = db_connector.os.getenv('SENTIMENT_TABLE', 'evaluation_sentiment')


def create_sentiment_table():
    """
    Create the sentiment table with columns for each score and metadata.
    Idempotent: skips creation if table already exists.
    """
    connection = None
    try:
        print(f"[{datetime.now()}] Starting sentiment table creation process")
        connection = db_connector.get_db_connection()

        if db_connector.table_exists(SENTIMENT_TABLE, connection):
            print(f"[{datetime.now()}] '{SENTIMENT_TABLE}' already exists, skipping creation")
            return

        print(f"[{datetime.now()}] '{SENTIMENT_TABLE}' does not exist, creating new table")
        create_query = f"""
        CREATE TABLE {SENTIMENT_TABLE} (
            response_id INTEGER NOT NULL,
            column_name TEXT NOT NULL,
            neg FLOAT,
            neu FLOAT,
            pos FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (response_id, column_name),
            FOREIGN KEY (response_id) REFERENCES evaluation(response_id)
        )
        """
        db_connector.execute_query(create_query, connection=connection)
        print(f"[{datetime.now()}] '{SENTIMENT_TABLE}' created successfully")

    except Exception as e:
        print(f"[{datetime.now()}] Error creating '{SENTIMENT_TABLE}': {e}")
        raise
    finally:
        if connection:
            db_connector.close_db_connection(connection)


if __name__ == '__main__':
    print(f"[{datetime.now()}] Running sentiment table creation script")
    print('=' * 50)
    create_sentiment_table()
    print(f"[{datetime.now()}] Sentiment table creation script completed")
