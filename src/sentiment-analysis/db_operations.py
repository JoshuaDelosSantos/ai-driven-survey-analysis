"""
DB operations for Sentiment Analysis Module

Defines DBOperations:
 - write_sentiment(response_id, column, scores): upserts sentiment scores into sentiment table
"""
from db.db_connector import execute_query
from config import SENTIMENT_TABLE, SCORE_COLUMNS

class DBOperations:
    def __init__(self):
        # No persistent connection; execute_query handles its own connection lifecycle
        pass

    def write_sentiment(self, response_id: int, column: str, scores: dict) -> int:
        """
        Upsert sentiment scores for a given response and column into the sentiment table.

        Args:
            response_id (int): evaluation response identifier
            column (str): name of the free-text column analyzed
            scores (dict): mapping of score labels to probabilities, e.g. {'neg':0.1,'neu':0.8,'pos':0.1}

        Returns:
            int: number of rows affected by the operation
        """
        # Prepare query for upsert
        query = f"""
        INSERT INTO {SENTIMENT_TABLE} (response_id, column_name, {', '.join(SCORE_COLUMNS)})
        VALUES (%s, %s, {', '.join(['%s'] * len(SCORE_COLUMNS))})
        ON CONFLICT (response_id, column_name) DO UPDATE
          SET {', '.join([f"{col}=EXCLUDED.{col}" for col in SCORE_COLUMNS])};
        """
        # Build parameters tuple
        params = [response_id, column] + [scores.get(col) for col in SCORE_COLUMNS]

        # Execute upsert
        rows_affected = execute_query(query, tuple(params))
        return rows_affected
