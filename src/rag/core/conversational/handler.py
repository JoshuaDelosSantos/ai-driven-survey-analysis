"""
Conversational Handler for RAG Agent

This module implements conversational intelligence for the RAG system, handling
greetings, system questions, off-topic queries, and other conversational interactions
with a friendly but professional Australian tone.

Key Features:
- Pattern-based conversational query recognition
- Australian-friendly response templates
- Simple pattern learning mechanism
- Context-aware schema guidance
- Seamless integration with data analysis pipeline

Pattern Categories:
- Greetings: "Hello", "Hi", "Good morning", "How are you?"
- System Questions: "What can you do?", "How do you work?", "What data do you have?"
- Politeness: "Thank you", "Please", "Goodbye"
- Off-topic: Non-data-related queries
- Meta: Questions about the system itself

Example Usage:
    # Initialize conversational handler
    handler = ConversationalHandler()
    
    # Handle greeting
    response = handler.handle_conversational_query("Hello! How are you?")
    print(response.content)  # "G'day! I'm working well, thank you for asking..."
    
    # Handle system question
    response = handler.handle_conversational_query("What can you do?")
    print(response.content)  # "I'm here to help you analyse survey data..."
    
    # Handle off-topic query
    response = handler.handle_conversational_query("What's the weather like?")
    print(response.content)  # "I'm focused on helping with data analysis..."
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationalPattern(Enum):
    """Enumeration of conversational pattern categories."""
    GREETING = "greeting"
    SYSTEM_QUESTION = "system_question"
    POLITENESS = "politeness"
    OFF_TOPIC = "off_topic"
    META = "meta"
    UNKNOWN = "unknown"


@dataclass
class ConversationalResponse:
    """Response structure for conversational interactions."""
    content: str
    confidence: float
    pattern_type: ConversationalPattern
    suggested_queries: Optional[List[str]] = None
    learning_feedback: Optional[str] = None


@dataclass
class PatternLearningData:
    """Data structure for pattern learning and improvement."""
    pattern: str
    frequency: int
    success_rate: float
    last_used: datetime
    feedback_scores: List[float]
    
    def update_success_rate(self, was_successful: bool) -> None:
        """Update success rate based on feedback."""
        self.feedback_scores.append(1.0 if was_successful else 0.0)
        self.success_rate = sum(self.feedback_scores) / len(self.feedback_scores)
        self.last_used = datetime.now()


class ConversationalHandler:
    """
    Handles conversational interactions with Australian-friendly responses.
    
    This class manages pattern recognition, response generation, and simple
    pattern learning for conversational queries that don't require data analysis.
    """
    
    def __init__(self):
        """Initialize the conversational handler."""
        self.pattern_learning: Dict[str, PatternLearningData] = {}
        self.response_templates = self._initialize_response_templates()
        self.pattern_matchers = self._initialize_pattern_matchers()
        
    def _initialize_response_templates(self) -> Dict[ConversationalPattern, List[str]]:
        """Initialize response templates with Australian-friendly language."""
        return {
            ConversationalPattern.GREETING: [
                "G'day! I'm working well, thank you for asking. I'm here to help you analyse survey and training data from the Australian Public Service. How can I assist you today?",
                "Hello! I'm doing well, thanks for checking in. I'm ready to help you explore your survey data and training feedback. What would you like to know?",
                "Hi there! I'm operating smoothly and ready to help. I can analyse survey responses, training evaluations, and user feedback. What can I help you discover?",
                "Good to hear from you! I'm functioning well and excited to help you dive into your data. Whether you need statistics, feedback analysis, or insights, I'm here for you."
            ],
            ConversationalPattern.SYSTEM_QUESTION: [
                "I'm here to help you analyse survey data and training feedback from the Australian Public Service. I can:\n• Provide statistical analysis of survey responses\n• Search through user feedback and comments\n• Analyse training evaluations and learning outcomes\n• Generate insights from attendance and engagement data\n\nTry asking me about user satisfaction, training effectiveness, or specific feedback themes!",
                "I'm a data analysis assistant specialising in survey and training data. I can help you:\n• Understand participation patterns and trends\n• Explore user feedback and experiences\n• Analyse training outcomes and effectiveness\n• Generate statistical summaries and insights\n\nWhat aspect of your data would you like to explore?",
                "I'm designed to help you make sense of your survey and training data. My capabilities include:\n• Statistical analysis and reporting\n• Semantic search through feedback text\n• Training evaluation analysis\n• User engagement and satisfaction insights\n\nWhat questions do you have about your data?"
            ],
            ConversationalPattern.POLITENESS: [
                "You're very welcome! I'm happy to help with your data analysis needs.",
                "My pleasure! Feel free to ask me anything about your survey or training data.",
                "No worries at all! I'm here whenever you need help with data analysis.",
                "Glad I could help! Don't hesitate to ask if you have more questions about your data."
            ],
            ConversationalPattern.OFF_TOPIC: [
                "I'm focused on helping with data analysis and survey insights. While I can't help with general questions, I'd be happy to assist you with:\n• Survey response analysis\n• Training feedback exploration\n• User satisfaction insights\n• Statistical summaries\n\nWhat would you like to know about your data?",
                "I specialise in survey and training data analysis rather than general topics. I can help you:\n• Analyse user feedback and responses\n• Explore training effectiveness\n• Generate statistical insights\n• Search through feedback comments\n\nWhat data questions can I help you with?",
                "While I can't assist with general queries, I'm excellent at helping with data analysis! I can:\n• Examine survey trends and patterns\n• Analyse training outcomes\n• Search feedback for specific themes\n• Provide statistical summaries\n\nWhat aspects of your data interest you most?"
            ],
            ConversationalPattern.META: [
                "I'm a RAG (Retrieval-Augmented Generation) system designed specifically for Australian Public Service survey and training data. I combine statistical analysis with semantic search to help you understand your data better. I can work with structured data for statistics and unstructured feedback for insights.",
                "I work by first understanding what type of analysis you need, then either running statistical queries on your structured data or searching through feedback text for relevant insights. I'm designed to be helpful, accurate, and privacy-conscious with your data.",
                "I use a combination of database queries for statistical analysis and vector search for exploring feedback text. This allows me to handle both quantitative questions (like 'how many users completed training') and qualitative questions (like 'what did users think about the experience')."
            ]
        }
    
    def _initialize_pattern_matchers(self) -> Dict[ConversationalPattern, List[str]]:
        """Initialize regex patterns for conversational query recognition."""
        return {
            ConversationalPattern.GREETING: [
                r'\b(hello|hi|hey|g\'?day|good\s+(morning|afternoon|evening|day))\b',
                r'\bhow\s+are\s+you\b',
                r'\bhow\s+are\s+things\b',
                r'\bhow\'s\s+it\s+going\b',
                r'\bnice\s+to\s+meet\s+you\b'
            ],
            ConversationalPattern.SYSTEM_QUESTION: [
                r'\bwhat\s+can\s+you\s+do\b',
                r'\bwhat\s+are\s+you\s+capable\s+of\b',
                r'\bwhat\s+kind\s+of\s+help\b',
                r'\bwhat\s+data\s+do\s+you\s+have\b',
                r'\bwhat\s+information\s+do\s+you\s+have\b',
                r'\bwhat\s+can\s+you\s+help\s+with\b',
                r'\bwhat\s+are\s+your\s+capabilities\b',
                r'\bwhat\s+services\s+do\s+you\s+provide\b'
            ],
            ConversationalPattern.POLITENESS: [
                r'\b(thank\s+you|thanks|ta|cheers)\b',
                r'\b(please|if\s+you\s+could|would\s+you\s+mind)\b',
                r'\b(goodbye|bye|see\s+you|catch\s+you\s+later)\b',
                r'\b(appreciate\s+it|much\s+appreciated)\b'
            ],
            ConversationalPattern.META: [
                r'\bhow\s+do\s+you\s+work\b',
                r'\bwhat\s+are\s+you\b',
                r'\bwho\s+are\s+you\b',
                r'\bwhat\s+kind\s+of\s+system\b',
                r'\bhow\s+were\s+you\s+built\b',
                r'\bwhat\s+technology\s+do\s+you\s+use\b',
                r'\bwhat\s+is\s+your\s+architecture\b'
            ]
        }
    
    def is_conversational_query(self, query: str) -> Tuple[bool, ConversationalPattern, float]:
        """
        Determine if a query is conversational and identify its pattern.
        
        Args:
            query: The user query to analyse
            
        Returns:
            Tuple of (is_conversational, pattern_type, confidence)
        """
        query_lower = query.lower().strip()
        
        # Check for data-related keywords that suggest it's NOT conversational
        data_keywords = [
            'user', 'training', 'course', 'survey', 'feedback', 'data', 'analysis',
            'report', 'statistic', 'satisfaction', 'evaluation', 'attendance',
            'completion', 'agency', 'level', 'score', 'rating', 'response'
        ]
        
        # If query contains data keywords, it's likely not purely conversational
        data_keyword_count = sum(1 for keyword in data_keywords if keyword in query_lower)
        if data_keyword_count > 2:
            return False, ConversationalPattern.UNKNOWN, 0.0
        
        # Check conversational patterns
        best_pattern = ConversationalPattern.UNKNOWN
        best_confidence = 0.0
        
        for pattern, regex_list in self.pattern_matchers.items():
            for regex in regex_list:
                if re.search(regex, query_lower):
                    # Calculate confidence based on pattern strength and query length
                    base_confidence = 0.8
                    
                    # Adjust confidence based on query characteristics
                    if len(query.split()) <= 5:  # Short queries are more likely conversational
                        base_confidence += 0.1
                    
                    if data_keyword_count > 0:  # Reduce confidence if data keywords present
                        base_confidence -= 0.2 * data_keyword_count
                    
                    confidence = max(0.1, min(0.95, base_confidence))
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pattern = pattern
        
        # Check for generic off-topic patterns
        if best_pattern == ConversationalPattern.UNKNOWN and data_keyword_count == 0:
            # Look for non-data-related question patterns
            question_patterns = [
                r'\bwhat\s+is\s+the\s+weather\b',
                r'\bwhat\s+time\s+is\s+it\b',
                r'\btell\s+me\s+about\s+\w+\b',
                r'\bwho\s+won\s+the\s+game\b',
                r'\bwhat\'s\s+new\s+in\s+\w+\b'
            ]
            
            for pattern in question_patterns:
                if re.search(pattern, query_lower):
                    return True, ConversationalPattern.OFF_TOPIC, 0.7
        
        is_conversational = best_confidence > 0.5
        return is_conversational, best_pattern, best_confidence
    
    def handle_conversational_query(self, query: str) -> ConversationalResponse:
        """
        Handle a conversational query and generate an appropriate response.
        
        Args:
            query: The conversational query to handle
            
        Returns:
            ConversationalResponse with generated content and metadata
        """
        is_conv, pattern_type, confidence = self.is_conversational_query(query)
        
        if not is_conv:
            # This shouldn't happen if classification is working correctly
            logger.warning(f"Non-conversational query passed to conversational handler: {query}")
            return ConversationalResponse(
                content="I'm not sure how to respond to that as a conversational query. Let me know if you have questions about your survey or training data!",
                confidence=0.3,
                pattern_type=ConversationalPattern.UNKNOWN,
                suggested_queries=self._get_suggested_queries()
            )
        
        # Get appropriate response template
        templates = self.response_templates.get(pattern_type, [])
        if not templates:
            templates = self.response_templates[ConversationalPattern.OFF_TOPIC]
        
        # Select response (for now, use first template - could be randomized)
        response_content = templates[0]
        
        # Add suggested queries for certain patterns
        suggested_queries = None
        if pattern_type in [ConversationalPattern.SYSTEM_QUESTION, ConversationalPattern.OFF_TOPIC]:
            suggested_queries = self._get_suggested_queries()
        
        # Record pattern usage for learning
        self._record_pattern_usage(query, pattern_type)
        
        return ConversationalResponse(
            content=response_content,
            confidence=confidence,
            pattern_type=pattern_type,
            suggested_queries=suggested_queries
        )
    
    def _get_suggested_queries(self) -> List[str]:
        """Get suggested data analysis queries to help users."""
        return [
            "How satisfied were users with their training experience?",
            "What feedback did users provide about virtual learning?",
            "How many users completed courses by agency?",
            "What are the most common themes in user feedback?",
            "Show me training completion rates by user level"
        ]
    
    def _record_pattern_usage(self, query: str, pattern_type: ConversationalPattern) -> None:
        """Record pattern usage for simple learning mechanism."""
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        
        if pattern_key not in self.pattern_learning:
            self.pattern_learning[pattern_key] = PatternLearningData(
                pattern=pattern_key,
                frequency=1,
                success_rate=0.8,  # Start with reasonable default
                last_used=datetime.now(),
                feedback_scores=[]
            )
        else:
            self.pattern_learning[pattern_key].frequency += 1
            self.pattern_learning[pattern_key].last_used = datetime.now()
    
    def provide_pattern_feedback(self, query: str, pattern_type: ConversationalPattern, 
                               was_helpful: bool) -> None:
        """
        Provide feedback on pattern recognition for learning improvement.
        
        Args:
            query: The original query
            pattern_type: The identified pattern type
            was_helpful: Whether the response was helpful
        """
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        
        if pattern_key in self.pattern_learning:
            self.pattern_learning[pattern_key].update_success_rate(was_helpful)
        
        logger.info(f"Pattern feedback recorded: {pattern_key} = {'helpful' if was_helpful else 'not helpful'}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern usage and learning."""
        stats = {
            "total_patterns": len(self.pattern_learning),
            "pattern_details": {}
        }
        
        for pattern_key, data in self.pattern_learning.items():
            stats["pattern_details"][pattern_key] = {
                "frequency": data.frequency,
                "success_rate": data.success_rate,
                "last_used": data.last_used.isoformat(),
                "feedback_count": len(data.feedback_scores)
            }
        
        return stats
