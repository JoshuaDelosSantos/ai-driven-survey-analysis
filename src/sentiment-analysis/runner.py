"""
Runner script for Sentiment Analysis Module

- Ensures the sentiment table exists by invoking the dedicated DB script
- Instantiates core components (analyzer, DB operations, processor)
- Executes the sentiment analysis pipeline over all evaluations
"""
import os
import sys
import subprocess
from datetime import datetime

# Configure import paths for standalone script execution
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
# Add src and current module dir to sys.path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, SCRIPT_DIR)

# Path to sentiment table creation script
DB_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'db', 'create_sentiment_table.py')

from analyser import SentimentAnalyser
from db_operations import DBOperations
from data_processor import DataProcessor


def main():
    print(f"[{datetime.now()}] Starting sentiment analysis pipeline")

    # Ensure sentiment table exists
    print(f"[{datetime.now()}] Checking/creating sentiment table via DB script")
    subprocess.run(["python", DB_SCRIPT], check=True)

    # Initialize components
    analyser = SentimentAnalyser()
    db_ops = DBOperations()
    processor = DataProcessor(db_ops, analyser)

    # Run processing
    processor.process_all()

    print(f"[{datetime.now()}] Sentiment analysis pipeline completed successfully")


if __name__ == '__main__':
    main()
