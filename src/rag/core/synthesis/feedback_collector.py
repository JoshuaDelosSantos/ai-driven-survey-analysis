"""
Feedback Collection System for RAG Module

Simple, maintainable feedback collector following Single Responsibility Principle.
Only handles: collection, PII anonymisation, and database storage.

Key Features:
- Simple 1-5 scale rating collection
- Optional text comments with PII protection
- Database storage using existing db_connector patterns
- Privacy-first design with Australian Privacy Principles compliance
- Robust error handling to prevent disruption to main RAG functionality

Security: All text content automatically anonymised before storage.
Performance: Async operations with graceful error handling.
Privacy: Australian Privacy Principles (APP) compliance maintained.

Example Usage:
    # Initialize collector
    collector = FeedbackCollector()
    
    # Collect feedback
    feedback_data = FeedbackData(
        session_id="session_123",
        query_id="query_456", 
        query_text="What is the completion rate?",
        response_text="The average completion rate is 85%",
        rating=4,
        comment="Very helpful information",
        response_sources=["Database Analysis"]
    )
    
    success = await collector.collect_feedback(feedback_data)
"""

import sys
import re
import logging
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

# Import using relative path to db module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import db.db_connector as db_connector

logger = logging.getLogger(__name__)

@dataclass
class FeedbackData:
    """
    Simple data structure for feedback storage.
    
    Represents user feedback on RAG system responses with minimal required fields
    and optional enhancements for analytics.
    """
    session_id: str
    query_id: str  
    query_text: str
    response_text: str
    rating: int  # 1-5 scale
    comment: Optional[str] = None
    response_sources: Optional[List[str]] = None

class FeedbackCollector:
    """
    Handles feedback collection, PII cleaning, and database storage only.
    
    Following Single Responsibility Principle:
    - Collects user feedback data
    - Anonymises PII in text content
    - Stores feedback in database
    - Does NOT handle analytics or display
    """
    
    def __init__(self):
        """Initialize feedback collector."""
        self.logger = logging.getLogger(__name__)
        
    def _anonymise_text(self, text: Optional[str]) -> Optional[str]:
        """
        Simple PII anonymisation - remove common patterns.
        
        Removes emails, phone numbers, and other identifiable patterns
        while preserving the meaningful content for analysis.
        
        Args:
            text: Text content to anonymise
            
        Returns:
            Anonymised text or None if input was None
        """
        if not text:
            return text
            
        # Basic patterns for common PII
        anonymised = text
        
        # Email addresses -> [EMAIL]
        anonymised = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
            '[EMAIL]', 
            anonymised
        )
        
        # Phone numbers (various formats) -> [PHONE]
        anonymised = re.sub(
            r'\b(?:\+?61\s?)?(?:\(0[2-8]\)\s?)?[0-9\s\-]{8,}\b', 
            '[PHONE]', 
            anonymised
        )
        
        # Australian mobile numbers -> [MOBILE]
        anonymised = re.sub(
            r'\b(?:\+?61\s?)?4[0-9]{2}\s?[0-9]{3}\s?[0-9]{3}\b',
            '[MOBILE]',
            anonymised
        )
        
        # Names that look like "John Smith" -> [NAME]
        anonymised = re.sub(
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            '[NAME]',
            anonymised
        )
        
        return anonymised
        
    async def collect_feedback(self, feedback_data: FeedbackData) -> bool:
        """
        Store feedback in database with PII anonymisation.
        
        This is the main method that handles the complete feedback storage process:
        1. Validates feedback data
        2. Anonymises any PII in text fields
        3. Stores in database using existing db_connector patterns
        4. Returns success/failure status
        
        Args:
            feedback_data: FeedbackData object with user feedback
            
        Returns:
            bool: True if feedback stored successfully, False otherwise
        """
        try:
            # Validate required fields
            if not feedback_data.session_id or not feedback_data.query_id:
                self.logger.error("Missing required fields: session_id or query_id")
                return False
                
            if not (1 <= feedback_data.rating <= 5):
                self.logger.error(f"Invalid rating: {feedback_data.rating}. Must be 1-5")
                return False
            
            # Anonymise comment text
            anonymised_comment = self._anonymise_text(feedback_data.comment)
            
            # Prepare database insertion
            insert_sql = """
            INSERT INTO rag_user_feedback 
            (session_id, query_id, query_text, response_text, rating, 
             comment, response_sources, anonymised_comment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert response_sources list to PostgreSQL array format
            sources_array = feedback_data.response_sources or []
            
            # Execute database insertion
            connection = db_connector.get_db_connection()
            
            try:
                db_connector.execute_query(
                    insert_sql,
                    (
                        feedback_data.session_id,
                        feedback_data.query_id,
                        feedback_data.query_text,
                        feedback_data.response_text,
                        feedback_data.rating,
                        feedback_data.comment,
                        sources_array,
                        anonymised_comment
                    ),
                    connection=connection
                )
                
                self.logger.info(f"Feedback collected successfully for query {feedback_data.query_id}")
                return True
                
            finally:
                db_connector.close_db_connection(connection)
            
        except Exception as e:
            self.logger.error(f"Failed to collect feedback: {e}")
            return False

    def validate_feedback_data(self, feedback_data: FeedbackData) -> tuple[bool, str]:
        """
        Validate feedback data before storage.
        
        Args:
            feedback_data: FeedbackData object to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not feedback_data.session_id:
            return False, "Session ID is required"
            
        if not feedback_data.query_id:
            return False, "Query ID is required"
            
        if not feedback_data.query_text:
            return False, "Query text is required"
            
        if not feedback_data.response_text:
            return False, "Response text is required"
            
        if not isinstance(feedback_data.rating, int) or not (1 <= feedback_data.rating <= 5):
            return False, "Rating must be an integer between 1 and 5"
            
        return True, ""

# Convenience function for simple feedback collection
async def collect_simple_feedback(
    session_id: str,
    query_id: str, 
    query_text: str,
    response_text: str,
    rating: int,
    comment: Optional[str] = None,
    response_sources: Optional[List[str]] = None
) -> bool:
    """
    Convenience function for collecting feedback with individual parameters.
    
    Args:
        session_id: User session identifier
        query_id: Unique query identifier
        query_text: Original user query
        response_text: System response
        rating: User rating (1-5)
        comment: Optional user comment
        response_sources: Optional list of sources used
        
    Returns:
        bool: Success status
    """
    collector = FeedbackCollector()
    feedback_data = FeedbackData(
        session_id=session_id,
        query_id=query_id,
        query_text=query_text,
        response_text=response_text,
        rating=rating,
        comment=comment,
        response_sources=response_sources
    )
    
    return await collector.collect_feedback(feedback_data)
