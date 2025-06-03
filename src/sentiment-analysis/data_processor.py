"""
Data Processor for Sentiment Analysis Module

Defines DataProcessor:
 - fetches evaluation rows
 - iterates over free-text columns
 - calls SentimentAnalyser.analyse()
 - persists scores via DBOperations.write_sentiment()
"""
from .config import EVALUATION_TABLE, FREE_TEXT_COLUMNS
from .analyser import SentimentAnalyser
from .db_operations import DBOperations
from db.db_connector import fetch_data

class DataProcessor:
    def __init__(self, db_ops: DBOperations, analyser: SentimentAnalyser):
        self.db_ops = db_ops
        self.analyser = analyser

    def process_all(self):
        """
        Process all evaluations:
        - Fetch response_id and free-text columns
        - Analyse each non-empty text field
        - Write sentiment scores to the database
        """
        # Build SELECT query
        cols = ["response_id"] + FREE_TEXT_COLUMNS
        col_list = ", ".join(cols)
        query = f"SELECT {col_list} FROM {EVALUATION_TABLE}"

        rows = fetch_data(query)
        print(f"Processing {len(rows)} evaluation rows...")

        for row in rows:
            response_id = row[0]
            # Iterate each free-text column by index
            for idx, column in enumerate(FREE_TEXT_COLUMNS, start=1):
                text = row[idx]
                if text and text.strip():
                    try:
                        scores = self.analyser.analyse(text)
                        self.db_ops.write_sentiment(response_id, column, scores)
                    except Exception as e:
                        print(f"Error processing response {response_id}, column '{column}': {e}")
