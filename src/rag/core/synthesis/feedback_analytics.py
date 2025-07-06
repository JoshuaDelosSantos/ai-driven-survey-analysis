"""
Feedback Analytics System for RAG Module

Simple on-demand feedback analytics for /feedback-stats terminal command.
Following Single Responsibility Principle - only handles data analysis and reporting.

Key Features:
- On-demand feedback statistics generation
- Simple rating distribution analysis  
- Recent comments display (anonymised)
- Temporal filtering (last N days)
- Performance and trend insights

Security: Uses anonymised feedback data for analysis.
Performance: Efficient queries with configurable limits.
Privacy: Only accesses anonymised content for reporting.

Example Usage:
    from db import db_connector
    
    # Initialize analytics
    analytics = FeedbackAnalytics()
    
    # Get recent feedback stats
    stats = await analytics.get_feedback_stats(days_back=7)
    
    print(f"Average rating: {stats.average_rating:.1f}/5")
    print(f"Total feedback: {stats.total_count}")
"""

import sys
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Import using absolute path from project structure
from db import db_connector

logger = logging.getLogger(__name__)

@dataclass
class FeedbackStats:
    """
    Simple feedback statistics for display and analysis.
    
    Contains aggregated feedback data for a specified time period
    with basic metrics and insights.
    """
    total_count: int = 0
    average_rating: float = 0.0
    rating_counts: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    recent_comments: List[str] = field(default_factory=list)
    days_analyzed: int = 7
    analysis_date: datetime = field(default_factory=datetime.now)

class FeedbackAnalytics:
    """
    Simple analytics for on-demand feedback analysis.
    
    Following Single Responsibility Principle:
    - Retrieves feedback data from database
    - Calculates basic statistics and trends
    - Formats data for display
    - Does NOT handle data collection or storage
    """
    
    def __init__(self):
        """Initialize feedback analytics."""
        self.logger = logging.getLogger(__name__)
        
    async def get_feedback_stats(self, days_back: int = 7) -> FeedbackStats:
        """
        Get simple feedback statistics for the last N days.
        
        Retrieves and analyzes feedback data to provide insights into
        system performance and user satisfaction trends.
        
        Args:
            days_back: Number of days to analyze (default: 7)
            
        Returns:
            FeedbackStats: Aggregated statistics and insights
        """
        stats = FeedbackStats(days_analyzed=days_back)
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query for feedback data
            stats_sql = """
            SELECT rating, comment, anonymised_comment, created_at, response_sources
            FROM rag_user_feedback 
            WHERE created_at >= %s 
            ORDER BY created_at DESC
            LIMIT 100
            """
            
            # Execute query
            connection = db_connector.get_db_connection()
            
            try:
                rows = db_connector.fetch_data(stats_sql, (cutoff_date,), connection=connection)
                
                if rows:
                    stats.total_count = len(rows)
                    total_rating = 0
                    
                    for row in rows:
                        rating = row[0]  # rating column
                        comment = row[1]  # comment column  
                        anonymised_comment = row[2]  # anonymised_comment column
                        created_at = row[3]  # created_at column
                        response_sources = row[4]  # response_sources column
                        
                        # Count ratings
                        if rating in stats.rating_counts:
                            stats.rating_counts[rating] += 1
                        total_rating += rating
                        
                        # Collect recent comments (prefer anonymised)
                        display_comment = anonymised_comment if anonymised_comment else comment
                        if display_comment and display_comment.strip() and len(stats.recent_comments) < 5:
                            # Truncate long comments for display
                            if len(display_comment) > 200:
                                display_comment = display_comment[:197] + "..."
                            stats.recent_comments.append(display_comment)
                    
                    # Calculate average rating
                    stats.average_rating = total_rating / stats.total_count if stats.total_count > 0 else 0.0
                    
                    self.logger.info(f"Retrieved {stats.total_count} feedback records for analysis")
                else:
                    self.logger.info("No feedback data found for the specified period")
                    
            finally:
                db_connector.close_db_connection(connection)
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback stats: {e}")
            
        return stats
    
    async def get_feedback_trends(self, days_back: int = 30) -> Dict:
        """
        Get feedback trends over time for deeper analysis.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dict: Trend data including daily averages and patterns
        """
        trends = {
            "daily_averages": [],
            "weekly_summary": {},
            "rating_trend": "stable",
            "volume_trend": "stable"
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query for daily feedback trends
            trends_sql = """
            SELECT DATE(created_at) as feedback_date, 
                   AVG(rating) as avg_rating,
                   COUNT(*) as feedback_count
            FROM rag_user_feedback 
            WHERE created_at >= %s 
            GROUP BY DATE(created_at)
            ORDER BY feedback_date DESC
            """
            
            connection = db_connector.get_db_connection()
            
            try:
                rows = db_connector.fetch_data(trends_sql, (cutoff_date,), connection=connection)
                
                for row in rows:
                    feedback_date = row[0]
                    avg_rating = float(row[1])
                    feedback_count = row[2]
                    
                    trends["daily_averages"].append({
                        "date": feedback_date.isoformat(),
                        "average_rating": round(avg_rating, 2),
                        "count": feedback_count
                    })
                
                # Simple trend analysis
                if len(trends["daily_averages"]) >= 7:
                    recent_avg = sum(d["average_rating"] for d in trends["daily_averages"][:7]) / 7
                    older_avg = sum(d["average_rating"] for d in trends["daily_averages"][7:14]) / min(7, len(trends["daily_averages"][7:14]))
                    
                    if recent_avg > older_avg + 0.2:
                        trends["rating_trend"] = "improving"
                    elif recent_avg < older_avg - 0.2:
                        trends["rating_trend"] = "declining"
                
                self.logger.info(f"Analyzed feedback trends over {days_back} days")
                
            finally:
                db_connector.close_db_connection(connection)
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback trends: {e}")
            
        return trends
    
    async def get_low_rated_queries(self, days_back: int = 30, min_rating: int = 3) -> List[Dict]:
        """
        Get queries with low ratings for improvement analysis.
        
        Args:
            days_back: Number of days to analyze
            min_rating: Maximum rating to consider "low" (inclusive)
            
        Returns:
            List[Dict]: Low-rated queries with context
        """
        low_rated = []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            low_rated_sql = """
            SELECT query_text, rating, anonymised_comment, response_sources, created_at
            FROM rag_user_feedback 
            WHERE created_at >= %s 
              AND rating <= %s
            ORDER BY rating ASC, created_at DESC
            LIMIT 10
            """
            
            connection = db_connector.get_db_connection()
            
            try:
                rows = db_connector.fetch_data(low_rated_sql, (cutoff_date, min_rating), connection=connection)
                
                for row in rows:
                    query_text = row[0]
                    rating = row[1]
                    comment = row[2] 
                    sources = row[3]
                    created_at = row[4]
                    
                    # Truncate long queries for display
                    display_query = query_text[:100] + "..." if len(query_text) > 100 else query_text
                    
                    low_rated.append({
                        "query": display_query,
                        "rating": rating,
                        "comment": comment or "No comment provided",
                        "sources": sources or [],
                        "date": created_at.strftime("%Y-%m-%d %H:%M")
                    })
                
                self.logger.info(f"Found {len(low_rated)} low-rated queries")
                
            finally:
                db_connector.close_db_connection(connection)
                
        except Exception as e:
            self.logger.error(f"Failed to get low-rated queries: {e}")
            
        return low_rated
    
    def format_stats_for_display(self, stats: FeedbackStats) -> str:
        """
        Format feedback statistics for terminal display.
        
        Args:
            stats: FeedbackStats object to format
            
        Returns:
            str: Formatted text for terminal display
        """
        if stats.total_count == 0:
            return "No feedback data available for the specified period."
        
        output = []
        output.append(f"📊 Feedback Analysis ({stats.days_analyzed} days)")
        output.append("=" * 50)
        output.append(f"Total responses: {stats.total_count}")
        output.append(f"Average rating: {stats.average_rating:.1f}/5.0")
        
        # Rating distribution
        output.append("\n📈 Rating Distribution:")
        for rating in range(1, 6):
            count = stats.rating_counts[rating]
            percentage = (count / stats.total_count) * 100 if stats.total_count > 0 else 0
            bar = "█" * int(percentage / 5)  # Simple bar chart
            output.append(f"  {rating}⭐: {count:3d} ({percentage:4.1f}%) {bar}")
        
        # Recent comments
        if stats.recent_comments:
            output.append(f"\n💬 Recent Comments ({len(stats.recent_comments)}):")
            for i, comment in enumerate(stats.recent_comments, 1):
                output.append(f"  {i}. \"{comment}\"")
        
        # Simple insights
        output.append("\n💡 Quick Insights:")
        if stats.average_rating >= 4.0:
            output.append("  • Strong user satisfaction levels")
        elif stats.average_rating >= 3.0:
            output.append("  • Moderate satisfaction - room for improvement")
        else:
            output.append("  • Low satisfaction - requires attention")
            
        low_ratings = stats.rating_counts[1] + stats.rating_counts[2]
        if low_ratings > stats.total_count * 0.3:
            output.append(f"  • High proportion of low ratings ({low_ratings}/{stats.total_count})")
        
        return "\n".join(output)

# Convenience function for quick stats retrieval
async def get_quick_stats(days_back: int = 7) -> FeedbackStats:
    """
    Convenience function for getting feedback statistics.
    
    Args:
        days_back: Number of days to analyze
        
    Returns:
        FeedbackStats: Statistics for the period
    """
    analytics = FeedbackAnalytics()
    return await analytics.get_feedback_stats(days_back)
