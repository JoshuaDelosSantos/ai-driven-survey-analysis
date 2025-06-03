"""
Configuration for Sentiment Analysis Module

Centralises constants and environment-based settings:
- MODEL_NAME: Hugging Face model identifier
- EVALUATION_TABLE: source table for free-text responses
- SENTIMENT_TABLE: target table to store scores
- FREE_TEXT_COLUMNS: list of evaluation columns to analyze
- SCORE_COLUMNS: sentiment score keys for each model
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face model identifier (override via env variable)
MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME", 
    "cardiffnlp/twitter-roberta-base-sentiment"
)

# Source and target database tables (override via env variables)
EVALUATION_TABLE = os.getenv("EVALUATION_TABLE", "evaluation")
SENTIMENT_TABLE = os.getenv("SENTIMENT_TABLE", "evaluation_sentiment")

# Columns in the evaluation table containing free-text feedback to analyse
FREE_TEXT_COLUMNS = os.getenv(
    "FREE_TEXT_COLUMNS",
    "did_experience_issue_detail,course_application_other,general_feedback"
).split(",")

# Keys used in sentiment score output
SCORE_COLUMNS = os.getenv("SCORE_COLUMNS", "neg,neu,pos").split(",")
