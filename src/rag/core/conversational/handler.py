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
    # Basic greetings
    GREETING = "greeting"
    GREETING_FORMAL = "greeting_formal"
    GREETING_CASUAL = "greeting_casual"
    GREETING_TIME_AWARE = "greeting_time_aware"
    
    # System questions
    SYSTEM_QUESTION = "system_question"
    SYSTEM_QUESTION_CAPABILITIES = "system_question_capabilities"
    SYSTEM_QUESTION_DATA = "system_question_data"
    SYSTEM_QUESTION_METHODOLOGY = "system_question_methodology"
    
    # Politeness
    POLITENESS = "politeness"
    POLITENESS_THANKS = "politeness_thanks"
    POLITENESS_PLEASE = "politeness_please"
    POLITENESS_GOODBYE = "politeness_goodbye"
    
    # Off-topic
    OFF_TOPIC = "off_topic"
    OFF_TOPIC_WEATHER = "off_topic_weather"
    OFF_TOPIC_NEWS = "off_topic_news"
    OFF_TOPIC_PERSONAL = "off_topic_personal"
    
    # Meta questions
    META = "meta"
    META_ARCHITECTURE = "meta_architecture"
    META_TECHNOLOGY = "meta_technology"
    META_METHODOLOGY = "meta_methodology"
    
    # Help requests
    HELP_REQUEST = "help_request"
    HELP_NAVIGATION = "help_navigation"
    HELP_UNDERSTANDING = "help_understanding"
    
    # Feedback
    FEEDBACK = "feedback"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    FEEDBACK_SUGGESTION = "feedback_suggestion"
    
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
    """Enhanced data structure for pattern learning and improvement with LLM integration."""
    pattern: str
    frequency: int
    success_rate: float
    last_used: datetime
    feedback_scores: List[float]
    template_effectiveness: Dict[str, float]  # Track which templates work best
    context_success: Dict[str, float]  # Success rates in different contexts
    user_satisfaction: float
    
    # LLM effectiveness tracking for Phase 1 integration
    llm_routing_attempts: int = 0
    llm_routing_successes: int = 0
    llm_fallback_rate: float = 0.0
    llm_confidence_threshold: float = 0.7
    vector_confidence_boost: float = 0.0  # How much vector search improves confidence
    edge_case_frequency: int = 0  # Track how often this pattern represents edge cases
    
    # Phase 3 learning integration fields
    llm_usage_count: int = 0  # Total number of times LLM was used for this pattern
    llm_effectiveness: float = 0.8  # Current LLM effectiveness score (0.0-1.0)
    llm_avg_response_time: float = 0.0  # Average response time when using LLM (ms)
    template_vs_llm_preference: str = "hybrid"  # Learned preference: "template", "llm", or "hybrid"
    
    def update_success_rate(self, was_successful: bool, template_used: str = None, context: str = "general") -> None:
        """Update success rate based on feedback with enhanced tracking."""
        score = 1.0 if was_successful else 0.0
        self.feedback_scores.append(score)
        self.success_rate = sum(self.feedback_scores) / len(self.feedback_scores)
        self.last_used = datetime.now()
        
        # Track template effectiveness
        if template_used:
            if template_used not in self.template_effectiveness:
                self.template_effectiveness[template_used] = 0.0
            
            # Update template effectiveness with exponential moving average
            current_effectiveness = self.template_effectiveness[template_used]
            self.template_effectiveness[template_used] = (current_effectiveness * 0.8) + (score * 0.2)
        
        # Track context success
        if context not in self.context_success:
            self.context_success[context] = 0.0
        
        current_context_success = self.context_success[context]
        self.context_success[context] = (current_context_success * 0.8) + (score * 0.2)
        
        # Update overall user satisfaction
        self.user_satisfaction = (self.user_satisfaction * 0.9) + (score * 0.1)
    
    def update_llm_effectiveness(self, llm_was_used: bool, llm_was_successful: bool = None) -> None:
        """Update LLM routing effectiveness for learning-based LLM integration."""
        if llm_was_used:
            self.llm_routing_attempts += 1
            if llm_was_successful:
                self.llm_routing_successes += 1
        
        # Calculate fallback rate
        if self.llm_routing_attempts > 0:
            self.llm_fallback_rate = 1.0 - (self.llm_routing_successes / self.llm_routing_attempts)
    
    def should_use_llm(self, base_confidence: float) -> bool:
        """Determine if LLM routing should be used based on learning data."""
        # Use LLM for edge cases or when template confidence is low
        if base_confidence < self.llm_confidence_threshold:
            return True
        
        # Use LLM if this pattern has high edge case frequency
        if self.edge_case_frequency > 3 and self.llm_fallback_rate < 0.3:
            return True
        
        # Use LLM if it has proven effective for this pattern
        if (self.llm_routing_attempts > 5 and 
            self.llm_routing_successes / self.llm_routing_attempts > 0.8):
            return True
        
        return False
    
    def update_vector_confidence_boost(self, vector_boost: float) -> None:
        """Update how much vector search improves confidence for this pattern."""
        # Use exponential moving average to track vector effectiveness
        self.vector_confidence_boost = (self.vector_confidence_boost * 0.7) + (vector_boost * 0.3)
    
    def get_best_template(self) -> Optional[str]:
        """Get the most effective template based on learning data."""
        if not self.template_effectiveness:
            return None
        
        return max(self.template_effectiveness, key=self.template_effectiveness.get)
    
    def get_context_confidence(self, context: str) -> float:
        """Get confidence for this pattern in a specific context."""
        return self.context_success.get(context, self.success_rate)


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
        
        # Phase 1: Enhanced Component Integration
        self.pattern_classifier = None  # Will be initialized lazily
        self.llm_routing_enabled = True  # Feature flag for LLM routing
        self.vector_boost_enabled = True  # Feature flag for vector confidence boost
        
    def _initialize_response_templates(self) -> Dict[ConversationalPattern, List[str]]:
        """Initialize response templates with Australian-friendly language and enhanced variety."""
        return {
            # Basic greetings
            ConversationalPattern.GREETING: [
                "G'day! I'm working well, thank you for asking. I'm here to help you analyse survey and training data from the Australian Public Service. How can I assist you today?",
                "Hello! I'm doing well, thanks for checking in. I'm ready to help you explore your survey data and training feedback. What would you like to know?",
                "Hi there! I'm operating smoothly and ready to help. I can analyse survey responses, training evaluations, and user feedback. What can I help you discover?",
                "Good to hear from you! I'm functioning well and excited to help you dive into your data. Whether you need statistics, feedback analysis, or insights, I'm here for you."
            ],
            
            # Formal greetings
            ConversationalPattern.GREETING_FORMAL: [
                "Good day. I'm pleased to assist you with your survey and training data analysis requirements. How may I help you today?",
                "Hello. I'm ready to provide professional assistance with your Australian Public Service data analysis needs. What can I help you with?",
                "Good morning/afternoon. I'm here to help you analyse survey responses and training feedback. How can I assist you today?"
            ],
            
            # Casual greetings
            ConversationalPattern.GREETING_CASUAL: [
                "Hey there! I'm ready to help you make sense of your survey data. What's on your mind?",
                "G'day mate! I'm here to help with all your data analysis needs. What would you like to explore?",
                "Hi! I'm your data analysis assistant - how can I help you today?",
                "Hey! Ready to dive into some data? I'm here to help you understand your survey and training information."
            ],
            
            # Time-aware greetings
            ConversationalPattern.GREETING_TIME_AWARE: [
                "Good morning! Hope you're having a great start to your day. I'm ready to help you explore your survey and training data.",
                "Good afternoon! I'm here and ready to help with your data analysis this afternoon. What can I help you discover?",
                "Good evening! I'm available to help you analyse your survey data this evening. How can I assist you?"
            ],
            
            # System capabilities
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: [
                "I'm here to help you analyse survey data and training feedback from the Australian Public Service. I can:\n• Provide statistical analysis of survey responses\n• Search through user feedback and comments\n• Analyse training evaluations and learning outcomes\n• Generate insights from attendance and engagement data\n• Help you understand participation patterns and trends\n\nTry asking me about user satisfaction, training effectiveness, or specific feedback themes!",
                "I'm a data analysis specialist focusing on survey and training data. My capabilities include:\n• Statistical analysis and reporting\n• Semantic search through feedback text\n• Training evaluation analysis\n• User engagement and satisfaction insights\n• Trend analysis across different agencies and user levels\n\nWhat aspect of your data interests you most?",
                "I'm designed to help you make sense of your survey and training data. I can help you:\n• Understand participation patterns and trends\n• Explore user feedback and experiences\n• Analyse training outcomes and effectiveness\n• Generate statistical summaries and insights\n• Find specific themes in user comments\n\nWhat questions do you have about your data?"
            ],
            
            # Data-specific questions
            ConversationalPattern.SYSTEM_QUESTION_DATA: [
                "I work with Australian Public Service survey and training data, including:\n• User survey responses and feedback\n• Training evaluation data\n• Course completion and attendance records\n• User satisfaction ratings\n• Feedback comments and suggestions\n\nThe data covers various agencies, user levels, and training programmes. What specific aspect would you like to explore?",
                "I have access to comprehensive survey and training data from the Australian Public Service:\n• Survey responses from various agencies\n• Training feedback and evaluations\n• User engagement and satisfaction data\n• Course completion statistics\n• Qualitative feedback and comments\n\nWhat type of analysis or insights are you looking for?",
                "My data includes survey responses, training evaluations, and user feedback from Australian Public Service personnel. I can help you explore:\n• Satisfaction levels across different programmes\n• Training effectiveness and completion rates\n• User feedback themes and suggestions\n• Participation patterns by agency and level\n\nWhat would you like to analyse?"
            ],
            
            # Methodology questions
            ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY: [
                "I use a combination of statistical analysis and semantic search to help you understand your data:\n• SQL queries for statistical analysis and aggregations\n• Vector search for exploring feedback text and comments\n• Hybrid approaches combining both methods\n• Schema-aware responses to ensure accurate results\n\nI can handle both quantitative questions (like participation rates) and qualitative questions (like user experiences).",
                "I work by first understanding what type of analysis you need, then either:\n• Running statistical queries on structured data for numbers and trends\n• Searching through feedback text for relevant insights and themes\n• Combining both approaches for comprehensive analysis\n\nI'm designed to be helpful, accurate, and privacy-conscious with your data. What would you like to explore?",
                "I'm a RAG (Retrieval-Augmented Generation) system that combines database queries with semantic search:\n• For statistical questions: I query the database directly\n• For feedback exploration: I search through text using semantic matching\n• For complex analysis: I combine both approaches\n\nThis allows me to handle both 'how many' questions and 'what did people think' questions effectively."
            ],
            
            # Thanks responses
            ConversationalPattern.POLITENESS_THANKS: [
                "You're very welcome! I'm happy to help with your data analysis needs anytime.",
                "My pleasure! Feel free to ask me anything about your survey or training data.",
                "No worries at all! I'm here whenever you need help with data analysis.",
                "Glad I could help! Don't hesitate to ask if you have more questions about your data.",
                "Too right! I'm always happy to help you understand your data better."
            ],
            
            # Please responses
            ConversationalPattern.POLITENESS_PLEASE: [
                "Of course! I'm here to help. What would you like to know about your data?",
                "Absolutely! I'm happy to assist with your data analysis needs.",
                "Certainly! I'm ready to help you explore your survey and training data.",
                "No problem at all! I'm here to help you understand your data better."
            ],
            
            # Goodbye responses
            ConversationalPattern.POLITENESS_GOODBYE: [
                "Thanks for using the system! Feel free to come back anytime you need help analysing your data.",
                "See you later! I'll be here whenever you need assistance with your survey data.",
                "Goodbye! Don't hesitate to return if you have more questions about your data.",
                "Catch you later! I'm always here to help with your data analysis needs.",
                "Cheers! Come back anytime you need help understanding your survey information."
            ],
            
            # Weather off-topic
            ConversationalPattern.OFF_TOPIC_WEATHER: [
                "I can't help with weather information, but I'm excellent at analysing survey data and training feedback! I can help you explore user satisfaction, training effectiveness, or feedback themes. What would you like to know about your data?",
                "While I can't provide weather updates, I can help you understand the climate of opinion in your survey data! I can analyse user feedback, satisfaction trends, or training outcomes. What interests you most?"
            ],
            
            # News off-topic
            ConversationalPattern.OFF_TOPIC_NEWS: [
                "I'm not able to provide news updates, but I can help you analyse the latest trends in your survey data! I can explore user feedback, training effectiveness, or satisfaction patterns. What would you like to discover?",
                "While I can't help with current events, I can help you understand what's happening in your survey data! I can analyse feedback trends, user satisfaction, or training outcomes. What catches your interest?"
            ],
            
            # Personal off-topic
            ConversationalPattern.OFF_TOPIC_PERSONAL: [
                "I focus on professional data analysis rather than personal topics. I'm here to help you understand your survey and training data, including user feedback, satisfaction trends, and training effectiveness. What would you like to explore?",
                "While I can't help with personal matters, I'm excellent at helping you understand your professional data! I can analyse survey responses, training feedback, or user satisfaction. What interests you most?"
            ],
            
            # General off-topic
            ConversationalPattern.OFF_TOPIC: [
                "I'm focused on helping with data analysis and survey insights. While I can't help with general questions, I'd be happy to assist you with:\n• Survey response analysis\n• Training feedback exploration\n• User satisfaction insights\n• Statistical summaries\n• Trend analysis\n\nWhat would you like to know about your data?",
                "I specialise in survey and training data analysis rather than general topics. I can help you:\n• Analyse user feedback and responses\n• Explore training effectiveness\n• Generate statistical insights\n• Search through feedback comments\n• Understand participation patterns\n\nWhat data questions can I help you with?",
                "While I can't assist with general queries, I'm excellent at helping with data analysis! I can:\n• Examine survey trends and patterns\n• Analyse training outcomes\n• Search feedback for specific themes\n• Provide statistical summaries\n• Help you understand user satisfaction\n\nWhat aspects of your data interest you most?"
            ],
            
            # Architecture meta
            ConversationalPattern.META_ARCHITECTURE: [
                "I'm built using a RAG (Retrieval-Augmented Generation) architecture that combines:\n• Database queries for statistical analysis\n• Vector search for semantic text exploration\n• LangGraph for intelligent query routing\n• Australian privacy-compliant processing\n\nThis allows me to handle both structured data queries and unstructured feedback analysis effectively.",
                "My architecture combines several technologies:\n• SQL databases for structured survey data\n• Vector embeddings for semantic search of feedback text\n• Language models for natural language understanding\n• Australian PII protection throughout the pipeline\n\nThis hybrid approach lets me answer both statistical and qualitative questions about your data."
            ],
            
            # Technology meta
            ConversationalPattern.META_TECHNOLOGY: [
                "I use modern AI and database technologies including:\n• Python with async processing for performance\n• Vector databases for semantic search\n• SQL databases for statistical analysis\n• LangGraph for intelligent workflow management\n• Australian privacy-compliant data handling\n\nAll designed to help you understand your survey and training data effectively.",
                "My technology stack includes:\n• Machine learning models for natural language understanding\n• Vector embeddings for semantic similarity search\n• Database systems for efficient data retrieval\n• Privacy-preserving processing techniques\n\nThis combination enables comprehensive analysis of both structured and unstructured data."
            ],
            
            # Methodology meta
            ConversationalPattern.META_METHODOLOGY: [
                "I follow a structured methodology:\n1. Understand your question and classify the type of analysis needed\n2. Choose the appropriate approach (statistical, semantic, or hybrid)\n3. Retrieve relevant data while protecting privacy\n4. Generate comprehensive, accurate responses\n5. Provide source attribution and confidence levels\n\nThis ensures you get reliable, trustworthy insights from your data.",
                "My analytical approach involves:\n• Query classification to understand what you're asking\n• Intelligent routing to the most appropriate analysis method\n• Privacy-compliant data retrieval and processing\n• Confidence scoring to indicate result reliability\n• Clear explanations of findings and limitations\n\nThis methodology ensures accurate, useful insights from your survey data."
            ],
            
            # Navigation help
            ConversationalPattern.HELP_NAVIGATION: [
                "I'm here to help you navigate through your data analysis! You can ask me questions like:\n• 'How satisfied were users with training?'\n• 'What feedback did people give about virtual learning?'\n• 'Show me completion rates by agency'\n• 'What are the main themes in user comments?'\n\nJust ask in natural language - I'll understand what you're looking for!",
                "Need help getting started? I can help you explore your data in several ways:\n• Ask about specific metrics (satisfaction, completion rates, etc.)\n• Explore user feedback and comments\n• Compare data across different groups or time periods\n• Find trends and patterns in your data\n\nWhat would you like to explore first?"
            ],
            
            # Understanding help
            ConversationalPattern.HELP_UNDERSTANDING: [
                "I'm happy to help you understand your data better! I can:\n• Explain what different metrics mean\n• Help you interpret survey results\n• Clarify statistical findings\n• Suggest related questions to explore\n• Provide context for your data\n\nWhat specifically would you like help understanding?",
                "No worries - I'm here to make your data easier to understand! I can:\n• Break down complex statistics into simple terms\n• Explain survey methodology and results\n• Help you identify key insights\n• Suggest actionable next steps\n• Clarify any confusing results\n\nWhat part of your data would you like me to explain?"
            ],
            
            # General help
            ConversationalPattern.HELP_REQUEST: [
                "I'm here to help! I can assist you with:\n• Understanding your survey and training data\n• Exploring user feedback and satisfaction\n• Analysing training effectiveness\n• Finding trends and patterns\n• Answering specific questions about your data\n\nWhat would you like help with today?",
                "Happy to help! I specialise in making survey and training data easy to understand. I can:\n• Answer questions about user satisfaction\n• Explore feedback themes and trends\n• Analyse training outcomes\n• Provide statistical insights\n• Help you discover patterns in your data\n\nWhat can I help you explore?"
            ],
            
            # Positive feedback
            ConversationalPattern.FEEDBACK_POSITIVE: [
                "Thank you for the positive feedback! I'm glad I could help you understand your data better. Is there anything else you'd like to explore?",
                "Thanks! I'm pleased the analysis was helpful. I'm here whenever you need more insights from your survey data.",
                "Great to hear! I'm always working to provide useful insights from your data. What else would you like to discover?",
                "Thanks for letting me know! I'm here to help you get the most value from your survey and training data."
            ],
            
            # Negative feedback
            ConversationalPattern.FEEDBACK_NEGATIVE: [
                "I appreciate your feedback and I'm sorry the response wasn't as helpful as expected. Could you let me know what specific information you were looking for? I'd be happy to try a different approach.",
                "Thank you for the feedback. I'd like to improve - could you tell me more about what you were hoping to find? I can try analysing your data from a different angle.",
                "I appreciate you letting me know. To better help you, could you clarify what specific insights you're looking for? I have several ways to analyse your data.",
                "Thanks for the feedback. I'm always learning how to provide better insights. What specific aspect of your data would you like me to focus on?"
            ],
            
            # Suggestion feedback
            ConversationalPattern.FEEDBACK_SUGGESTION: [
                "Thank you for the suggestion! I'm always looking to improve how I help you analyse your data. Your feedback helps me provide better insights.",
                "I appreciate your suggestion! User feedback helps me understand how to better serve your data analysis needs. Thanks for taking the time to share your thoughts.",
                "Thanks for the suggestion! I'm designed to continuously improve based on user feedback. Your input helps me provide more useful insights.",
                "Great suggestion! I value user feedback as it helps me better understand how to analyse and present your data effectively."
            ]
        }
    
    def _initialize_pattern_matchers(self) -> Dict[ConversationalPattern, List[str]]:
        """Initialize regex patterns for conversational query recognition."""
        return {
            # Basic greetings
            ConversationalPattern.GREETING: [
                r'\b(hello|hi|hey|g\'?day|good\s+(morning|afternoon|evening|day))\b',
                r'\bhow\s+are\s+you\b',
                r'\bhow\s+are\s+things\b',
                r'\bhow\'s\s+it\s+going\b',
                r'\bnice\s+to\s+meet\s+you\b'
            ],
            
            # Formal greetings
            ConversationalPattern.GREETING_FORMAL: [
                r'\bgood\s+(morning|afternoon|evening|day)\b',
                r'\bpleasure\s+to\s+meet\s+you\b',
                r'\bthank\s+you\s+for\s+your\s+time\b',
                r'\bi\s+would\s+like\s+to\s+enquire\b'
            ],
            
            # Casual greetings
            ConversationalPattern.GREETING_CASUAL: [
                r'\b(hey|hi|yo|sup|what\'s\s+up)\b',
                r'\bhowdy\b',
                r'\bwhat\'s\s+happening\b',
                r'\bhow\'s\s+things\b'
            ],
            
            # Time-aware greetings
            ConversationalPattern.GREETING_TIME_AWARE: [
                r'\bgood\s+morning\b',
                r'\bgood\s+afternoon\b',
                r'\bgood\s+evening\b',
                r'\bhave\s+a\s+good\s+(morning|afternoon|evening|day|night)\b'
            ],
            
            # System capabilities questions
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: [
                r'\bwhat\s+can\s+you\s+do\b',
                r'\bwhat\s+are\s+you\s+capable\s+of\b',
                r'\bwhat\s+are\s+your\s+capabilities\b',
                r'\bwhat\s+services\s+do\s+you\s+provide\b',
                r'\bwhat\s+kind\s+of\s+help\b',
                r'\bwhat\s+can\s+you\s+help\s+with\b',
                r'\bwhat\s+functions\s+do\s+you\s+have\b'
            ],
            
            # Data-specific questions
            ConversationalPattern.SYSTEM_QUESTION_DATA: [
                r'\bwhat\s+data\s+do\s+you\s+have\b',
                r'\bwhat\s+information\s+do\s+you\s+have\b',
                r'\bwhat\s+kind\s+of\s+data\b',
                r'\bwhat\s+datasets\b',
                r'\bwhat\s+sources\b',
                r'\bwhat\s+database\b',
                r'\btell\s+me\s+about\s+the\s+data\b'
            ],
            
            # Methodology questions
            ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY: [
                r'\bhow\s+do\s+you\s+work\b',
                r'\bhow\s+does\s+this\s+work\b',
                r'\bwhat\s+is\s+your\s+process\b',
                r'\bhow\s+do\s+you\s+analyse\b',
                r'\bwhat\s+method\s+do\s+you\s+use\b',
                r'\bhow\s+do\s+you\s+find\s+information\b'
            ],
            
            # Thanks patterns
            ConversationalPattern.POLITENESS_THANKS: [
                r'\b(thank\s+you|thanks|ta|cheers)\b',
                r'\bmuch\s+appreciated\b',
                r'\bappreciate\s+it\b',
                r'\bthanks\s+a\s+lot\b',
                r'\bthank\s+you\s+very\s+much\b',
                r'\bthanks\s+mate\b'
            ],
            
            # Please patterns
            ConversationalPattern.POLITENESS_PLEASE: [
                r'\bplease\b',
                r'\bif\s+you\s+could\b',
                r'\bwould\s+you\s+mind\b',
                r'\bcould\s+you\s+please\b',
                r'\bi\s+would\s+appreciate\b'
            ],
            
            # Goodbye patterns
            ConversationalPattern.POLITENESS_GOODBYE: [
                r'\b(goodbye|bye|see\s+you|catch\s+you\s+later)\b',
                r'\btake\s+care\b',
                r'\bhave\s+a\s+good\s+(day|night|weekend)\b',
                r'\bsee\s+you\s+later\b',
                r'\bcheers\b'
            ],
            
            # Weather off-topic
            ConversationalPattern.OFF_TOPIC_WEATHER: [
                r'\bwhat\'s\s+the\s+weather\b',
                r'\bhow\'s\s+the\s+weather\b',
                r'\bis\s+it\s+(raining|sunny|hot|cold)\b',
                r'\bweather\s+forecast\b',
                r'\bwill\s+it\s+rain\b'
            ],
            
            # News off-topic
            ConversationalPattern.OFF_TOPIC_NEWS: [
                r'\bwhat\'s\s+in\s+the\s+news\b',
                r'\blatest\s+news\b',
                r'\bcurrent\s+events\b',
                r'\bwhat\'s\s+happening\s+in\s+the\s+world\b',
                r'\bnews\s+update\b'
            ],
            
            # Personal off-topic
            ConversationalPattern.OFF_TOPIC_PERSONAL: [
                r'\bhow\s+old\s+are\s+you\b',
                r'\bwhere\s+do\s+you\s+live\b',
                r'\bdo\s+you\s+have\s+(family|friends)\b',
                r'\bwhat\s+do\s+you\s+like\s+to\s+do\b',
                r'\btell\s+me\s+about\s+yourself\b'
            ],
            
            # Architecture meta
            ConversationalPattern.META_ARCHITECTURE: [
                r'\bwhat\s+is\s+your\s+architecture\b',
                r'\bhow\s+are\s+you\s+built\b',
                r'\bwhat\s+is\s+your\s+structure\b',
                r'\bsystem\s+architecture\b',
                r'\bhow\s+are\s+you\s+designed\b'
            ],
            
            # Technology meta
            ConversationalPattern.META_TECHNOLOGY: [
                r'\bwhat\s+technology\s+do\s+you\s+use\b',
                r'\bwhat\s+are\s+you\s+built\s+with\b',
                r'\bwhat\s+programming\s+language\b',
                r'\bwhat\s+tools\s+do\s+you\s+use\b',
                r'\bwhat\s+software\b'
            ],
            
            # Methodology meta
            ConversationalPattern.META_METHODOLOGY: [
                r'\bwhat\s+is\s+rag\b',
                r'\bhow\s+does\s+rag\s+work\b',
                r'\bwhat\s+is\s+retrieval\s+augmented\s+generation\b',
                r'\bexplain\s+your\s+methodology\b',
                r'\bhow\s+do\s+you\s+process\s+queries\b'
            ],
            
            # Navigation help
            ConversationalPattern.HELP_NAVIGATION: [
                r'\bhow\s+do\s+i\s+(start|begin|navigate)\b',
                r'\bwhere\s+do\s+i\s+start\b',
                r'\bhow\s+do\s+i\s+ask\s+questions\b',
                r'\bhow\s+do\s+i\s+use\s+this\b',
                r'\bi\s+don\'t\s+know\s+how\s+to\s+start\b'
            ],
            
            # Understanding help
            ConversationalPattern.HELP_UNDERSTANDING: [
                r'\bi\s+don\'t\s+understand\b',
                r'\bthis\s+is\s+confusing\b',
                r'\bcan\s+you\s+explain\b',
                r'\bwhat\s+does\s+this\s+mean\b',
                r'\bi\'m\s+confused\b',
                r'\bhelp\s+me\s+understand\b'
            ],
            
            # General help
            ConversationalPattern.HELP_REQUEST: [
                r'\bhelp\b',
                r'\bi\s+need\s+help\b',
                r'\bcan\s+you\s+help\b',
                r'\bassist\s+me\b',
                r'\bi\'m\s+stuck\b',
                r'\bwhat\s+should\s+i\s+do\b'
            ],
            
            # Positive feedback
            ConversationalPattern.FEEDBACK_POSITIVE: [
                r'\bthis\s+is\s+(good|great|helpful|useful|excellent)\b',
                r'\bi\s+like\s+this\b',
                r'\bwell\s+done\b',
                r'\bthis\s+works\s+well\b',
                r'\bvery\s+helpful\b',
                r'\bperfect\b'
            ],
            
            # Negative feedback
            ConversationalPattern.FEEDBACK_NEGATIVE: [
                r'\bthis\s+(doesn\'t\s+work|is\s+not\s+working|is\s+wrong)\b',
                r'\bi\s+don\'t\s+like\s+this\b',
                r'\bthis\s+is\s+(bad|poor|unhelpful)\b',
                r'\bnot\s+what\s+i\s+wanted\b',
                r'\bthis\s+doesn\'t\s+help\b'
            ],
            
            # Suggestion feedback
            ConversationalPattern.FEEDBACK_SUGGESTION: [
                r'\byou\s+should\b',
                r'\bit\s+would\s+be\s+better\s+if\b',
                r'\bi\s+suggest\b',
                r'\bmy\s+suggestion\b',
                r'\bcan\s+you\s+improve\b',
                r'\bwhat\s+about\s+adding\b'
            ]
        }
    
    def is_conversational_query(self, query: str) -> Tuple[bool, ConversationalPattern, float]:
        """
        Determine if a query is conversational and identify its pattern with enhanced recognition.
        
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
            'completion', 'agency', 'level', 'score', 'rating', 'response', 'trend'
        ]
        
        # If query contains multiple data keywords, it's likely not purely conversational
        data_keyword_count = sum(1 for keyword in data_keywords if keyword in query_lower)
        
        # But allow some conversational queries that mention data in a meta way
        meta_data_patterns = [
            r'\bwhat\s+data\s+do\s+you\s+have\b',
            r'\bwhat\s+information\s+do\s+you\s+have\b',
            r'\btell\s+me\s+about\s+the\s+data\b'
        ]
        
        is_meta_data_query = any(re.search(pattern, query_lower) for pattern in meta_data_patterns)
        
        if data_keyword_count > 2 and not is_meta_data_query:
            return False, ConversationalPattern.UNKNOWN, 0.0
        
        # Check conversational patterns in order of specificity
        best_pattern = ConversationalPattern.UNKNOWN
        best_confidence = 0.0
        
        # Define pattern priority (more specific patterns checked first)
        pattern_priority = [
            # Specific greetings
            ConversationalPattern.GREETING_FORMAL,
            ConversationalPattern.GREETING_CASUAL,
            ConversationalPattern.GREETING_TIME_AWARE,
            
            # Specific system questions
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES,
            ConversationalPattern.SYSTEM_QUESTION_DATA,
            ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY,
            
            # Specific politeness
            ConversationalPattern.POLITENESS_THANKS,
            ConversationalPattern.POLITENESS_PLEASE,
            ConversationalPattern.POLITENESS_GOODBYE,
            
            # Specific off-topic
            ConversationalPattern.OFF_TOPIC_WEATHER,
            ConversationalPattern.OFF_TOPIC_NEWS,
            ConversationalPattern.OFF_TOPIC_PERSONAL,
            
            # Specific meta
            ConversationalPattern.META_ARCHITECTURE,
            ConversationalPattern.META_TECHNOLOGY,
            ConversationalPattern.META_METHODOLOGY,
            
            # Specific help
            ConversationalPattern.HELP_NAVIGATION,
            ConversationalPattern.HELP_UNDERSTANDING,
            
            # Specific feedback
            ConversationalPattern.FEEDBACK_POSITIVE,
            ConversationalPattern.FEEDBACK_NEGATIVE,
            ConversationalPattern.FEEDBACK_SUGGESTION,
            
            # General patterns (checked last)
            ConversationalPattern.GREETING,
            ConversationalPattern.SYSTEM_QUESTION,
            ConversationalPattern.POLITENESS,
            ConversationalPattern.OFF_TOPIC,
            ConversationalPattern.META,
            ConversationalPattern.HELP_REQUEST,
            ConversationalPattern.FEEDBACK
        ]
        
        for pattern in pattern_priority:
            if pattern not in self.pattern_matchers:
                continue
                
            regex_list = self.pattern_matchers[pattern]
            for regex in regex_list:
                if re.search(regex, query_lower):
                    # Calculate confidence based on pattern specificity and query characteristics
                    base_confidence = self._calculate_pattern_confidence(pattern, query, data_keyword_count)
                    
                    if base_confidence > best_confidence:
                        best_confidence = base_confidence
                        best_pattern = pattern
                        break  # Use first match for this pattern type
        
        # Check for generic off-topic patterns if no specific pattern found
        if best_pattern == ConversationalPattern.UNKNOWN and data_keyword_count == 0:
            # Look for non-data-related question patterns
            generic_question_patterns = [
                r'\bwhat\s+is\s+the\s+weather\b',
                r'\bwhat\s+time\s+is\s+it\b',
                r'\btell\s+me\s+about\s+\w+\b',
                r'\bwho\s+won\s+the\s+game\b',
                r'\bwhat\'s\s+new\s+in\s+\w+\b',
                r'\bhow\s+do\s+i\s+\w+\b'
            ]
            
            for pattern in generic_question_patterns:
                if re.search(pattern, query_lower):
                    return True, ConversationalPattern.OFF_TOPIC, 0.7
        
        is_conversational = best_confidence > 0.5
        return is_conversational, best_pattern, best_confidence
    
    def _calculate_pattern_confidence(self, pattern: ConversationalPattern, query: str, data_keyword_count: int) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence varies by pattern specificity
        pattern_base_confidence = {
            # High confidence for specific patterns
            ConversationalPattern.GREETING_FORMAL: 0.9,
            ConversationalPattern.GREETING_CASUAL: 0.9,
            ConversationalPattern.GREETING_TIME_AWARE: 0.95,
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: 0.95,
            ConversationalPattern.SYSTEM_QUESTION_DATA: 0.95,
            ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY: 0.95,
            ConversationalPattern.POLITENESS_THANKS: 0.9,
            ConversationalPattern.POLITENESS_GOODBYE: 0.9,
            ConversationalPattern.OFF_TOPIC_WEATHER: 0.95,
            ConversationalPattern.OFF_TOPIC_NEWS: 0.95,
            ConversationalPattern.META_ARCHITECTURE: 0.9,
            ConversationalPattern.META_TECHNOLOGY: 0.9,
            ConversationalPattern.HELP_NAVIGATION: 0.85,
            ConversationalPattern.HELP_UNDERSTANDING: 0.85,
            ConversationalPattern.FEEDBACK_POSITIVE: 0.8,
            ConversationalPattern.FEEDBACK_NEGATIVE: 0.8,
            
            # Medium confidence for general patterns
            ConversationalPattern.GREETING: 0.8,
            ConversationalPattern.SYSTEM_QUESTION: 0.8,
            ConversationalPattern.POLITENESS: 0.75,
            ConversationalPattern.OFF_TOPIC: 0.7,
            ConversationalPattern.META: 0.75,
            ConversationalPattern.HELP_REQUEST: 0.7,
            ConversationalPattern.FEEDBACK: 0.7
        }
        
        base_confidence = pattern_base_confidence.get(pattern, 0.6)
        
        # Adjust confidence based on query characteristics
        query_words = len(query.split())
        
        # Short queries are more likely to be conversational
        if query_words <= 3:
            base_confidence += 0.1
        elif query_words <= 5:
            base_confidence += 0.05
        
        # Reduce confidence if data keywords are present (except for meta data queries)
        if data_keyword_count > 0 and not pattern.value.startswith('system_question_data'):
            base_confidence -= 0.15 * data_keyword_count
        
        # Boost confidence for time-aware patterns if time words are present
        if pattern == ConversationalPattern.GREETING_TIME_AWARE:
            time_words = ['morning', 'afternoon', 'evening', 'night']
            if any(word in query.lower() for word in time_words):
                base_confidence += 0.05
        
        return max(0.1, min(0.95, base_confidence))
    
    def handle_conversational_query(self, query: str) -> ConversationalResponse:
        """
        Handle a conversational query and generate an appropriate response with enhanced sophistication.
        
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
        
        # Phase 1: Enhanced pattern recognition with vector boost
        enhanced_confidence = confidence
        llm_routing_recommended = False
        
        if self.vector_boost_enabled:
            enhanced_confidence, llm_routing_recommended = self._enhance_pattern_confidence(
                query, pattern_type, confidence
            )
        
        # Determine if LLM routing should be used based on learning data
        should_use_llm = self._should_use_llm_routing(query, pattern_type, enhanced_confidence)
        
        # Get appropriate response template
        response_content = self._select_response_template(pattern_type, query)
        
        # Record pattern usage for learning (including enhanced data)
        self._record_enhanced_pattern_usage(
            query, pattern_type, response_content[:50], enhanced_confidence, 
            llm_routing_recommended, should_use_llm
        )
        
        # Get suggested queries for user guidance
        suggested_queries = self._get_suggested_queries()
        
        return ConversationalResponse(
            content=response_content,
            confidence=enhanced_confidence,
            pattern_type=pattern_type,
            suggested_queries=suggested_queries,
            learning_feedback=f"Vector boost: {self.vector_boost_enabled}, LLM routing: {should_use_llm}"
        )
    
    def _select_response_template(self, pattern_type: ConversationalPattern, query: str) -> str:
        """
        Select the most appropriate response template with enhanced intelligence.
        
        Args:
            pattern_type: The identified pattern type
            query: The original query for context
            
        Returns:
            Selected response content
        """
        # Get templates for this pattern type
        templates = self.response_templates.get(pattern_type, [])
        
        # Fall back to general patterns if specific pattern has no templates
        if not templates:
            fallback_patterns = {
                ConversationalPattern.GREETING_FORMAL: ConversationalPattern.GREETING,
                ConversationalPattern.GREETING_CASUAL: ConversationalPattern.GREETING,
                ConversationalPattern.GREETING_TIME_AWARE: ConversationalPattern.GREETING,
                ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES,
                ConversationalPattern.SYSTEM_QUESTION_DATA: ConversationalPattern.SYSTEM_QUESTION_DATA,
                ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY: ConversationalPattern.SYSTEM_QUESTION_METHODOLOGY,
                ConversationalPattern.POLITENESS_THANKS: ConversationalPattern.POLITENESS_THANKS,
                ConversationalPattern.POLITENESS_PLEASE: ConversationalPattern.POLITENESS_PLEASE,
                ConversationalPattern.POLITENESS_GOODBYE: ConversationalPattern.POLITENESS_GOODBYE,
                ConversationalPattern.OFF_TOPIC_WEATHER: ConversationalPattern.OFF_TOPIC_WEATHER,
                ConversationalPattern.OFF_TOPIC_NEWS: ConversationalPattern.OFF_TOPIC_NEWS,
                ConversationalPattern.OFF_TOPIC_PERSONAL: ConversationalPattern.OFF_TOPIC_PERSONAL,
                ConversationalPattern.META_ARCHITECTURE: ConversationalPattern.META_ARCHITECTURE,
                ConversationalPattern.META_TECHNOLOGY: ConversationalPattern.META_TECHNOLOGY,
                ConversationalPattern.META_METHODOLOGY: ConversationalPattern.META_METHODOLOGY,
                ConversationalPattern.HELP_NAVIGATION: ConversationalPattern.HELP_NAVIGATION,
                ConversationalPattern.HELP_UNDERSTANDING: ConversationalPattern.HELP_UNDERSTANDING,
                ConversationalPattern.FEEDBACK_POSITIVE: ConversationalPattern.FEEDBACK_POSITIVE,
                ConversationalPattern.FEEDBACK_NEGATIVE: ConversationalPattern.FEEDBACK_NEGATIVE,
                ConversationalPattern.FEEDBACK_SUGGESTION: ConversationalPattern.FEEDBACK_SUGGESTION
            }
            
            fallback_pattern = fallback_patterns.get(pattern_type, ConversationalPattern.OFF_TOPIC)
            templates = self.response_templates.get(fallback_pattern, 
                                                   self.response_templates[ConversationalPattern.OFF_TOPIC])
        
        # Enhanced template selection logic
        selected_template = self._intelligent_template_selection(templates, query, pattern_type)
        
        return selected_template
    
    def _intelligent_template_selection(self, templates: List[str], query: str, pattern_type: ConversationalPattern) -> str:
        """
        Intelligently select the best template based on context and query characteristics.
        
        Args:
            templates: Available response templates
            query: Original query for context
            pattern_type: The pattern type
            
        Returns:
            Best selected template
        """
        if not templates:
            return "I'm here to help you with your survey and training data analysis. What would you like to explore?"
        
        # For single template, return it
        if len(templates) == 1:
            return templates[0]
        
        # Time-aware selection for greetings
        if pattern_type in [ConversationalPattern.GREETING_TIME_AWARE, ConversationalPattern.GREETING]:
            from datetime import datetime
            current_hour = datetime.now().hour
            
            # Morning (5-12), Afternoon (12-17), Evening (17-22)
            if 5 <= current_hour < 12:
                # Prefer morning responses
                morning_templates = [t for t in templates if 'morning' in t.lower()]
                if morning_templates:
                    return morning_templates[0]
            elif 12 <= current_hour < 17:
                # Prefer afternoon responses
                afternoon_templates = [t for t in templates if 'afternoon' in t.lower()]
                if afternoon_templates:
                    return afternoon_templates[0]
            elif 17 <= current_hour < 22:
                # Prefer evening responses
                evening_templates = [t for t in templates if 'evening' in t.lower()]
                if evening_templates:
                    return evening_templates[0]
        
        # Formality detection
        formal_indicators = ['please', 'would you', 'could you', 'i would like', 'i require']
        casual_indicators = ['hey', 'hi', 'yo', 'what\'s up', 'sup']
        
        query_lower = query.lower()
        is_formal = any(indicator in query_lower for indicator in formal_indicators)
        is_casual = any(indicator in query_lower for indicator in casual_indicators)
        
        if is_formal and pattern_type == ConversationalPattern.GREETING:
            # Use more formal templates
            formal_templates = [t for t in templates if 'pleased' in t.lower() or 'professional' in t.lower()]
            if formal_templates:
                return formal_templates[0]
        elif is_casual and pattern_type == ConversationalPattern.GREETING:
            # Use more casual templates
            casual_templates = [t for t in templates if 'mate' in t.lower() or 'hey' in t.lower()]
            if casual_templates:
                return casual_templates[0]
        
        # Pattern learning integration - use most successful template
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        if pattern_key in self.pattern_learning:
            pattern_data = self.pattern_learning[pattern_key]
            
            # Use learning data to select best template
            if pattern_data.success_rate > 0.7:  # High success rate
                best_template = pattern_data.get_best_template()
                if best_template and best_template in templates:
                    return best_template
            
            # Check context-specific success
            context = self._determine_query_context(query)
            context_confidence = pattern_data.get_context_confidence(context)
            if context_confidence > 0.8:
                # Use template that works well in this context
                best_template = pattern_data.get_best_template()
                if best_template and best_template in templates:
                    return best_template
        
        # Default: return first template (could be randomized for variety)
        return templates[0]
    
    def _get_suggested_queries(self) -> List[str]:
        """Get suggested data analysis queries to help users."""
        return [
            "How satisfied were users with their training experience?",
            "What feedback did users provide about virtual learning?",
            "How many users completed courses by agency?",
            "What are the most common themes in user feedback?",
            "Show me training completion rates by user level"
        ]
    
    def _record_pattern_usage(self, query: str, pattern_type: ConversationalPattern, template_used: str = None) -> None:
        """Record pattern usage for sophisticated learning mechanism."""
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        
        # Determine context for learning
        context = self._determine_query_context(query)
        
        if pattern_key not in self.pattern_learning:
            self.pattern_learning[pattern_key] = PatternLearningData(
                pattern=pattern_key,
                frequency=1,
                success_rate=0.8,  # Start with reasonable default
                last_used=datetime.now(),
                feedback_scores=[],
                template_effectiveness={},
                context_success={},
                user_satisfaction=0.8
            )
        else:
            self.pattern_learning[pattern_key].frequency += 1
            self.pattern_learning[pattern_key].last_used = datetime.now()
        
        # Track template usage
        if template_used:
            pattern_data = self.pattern_learning[pattern_key]
            if template_used not in pattern_data.template_effectiveness:
                pattern_data.template_effectiveness[template_used] = 0.8  # Default effectiveness
    
    def _determine_query_context(self, query: str) -> str:
        """Determine the context of a query for learning purposes."""
        query_lower = query.lower()
        
        # Time context
        from datetime import datetime
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            time_context = "morning"
        elif 12 <= current_hour < 17:
            time_context = "afternoon"
        elif 17 <= current_hour < 22:
            time_context = "evening"
        else:
            time_context = "night"
        
        # Formality context
        formal_indicators = ['please', 'would you', 'could you', 'i would like', 'i require']
        casual_indicators = ['hey', 'hi', 'yo', 'what\'s up', 'sup']
        
        if any(indicator in query_lower for indicator in formal_indicators):
            formality_context = "formal"
        elif any(indicator in query_lower for indicator in casual_indicators):
            formality_context = "casual"
        else:
            formality_context = "neutral"
        
        # Length context
        word_count = len(query.split())
        if word_count <= 3:
            length_context = "short"
        elif word_count <= 7:
            length_context = "medium"
        else:
            length_context = "long"
        
        return f"{time_context}_{formality_context}_{length_context}"
    
    def provide_pattern_feedback(self, query: str, pattern_type: ConversationalPattern, 
                               was_helpful: bool, template_used: str = None, 
                               llm_was_used: bool = False) -> None:
        """
        Provide enhanced feedback on pattern recognition for learning improvement.
        
        Args:
            query: The original query
            pattern_type: The identified pattern type
            was_helpful: Whether the response was helpful
            template_used: Which template was used for the response
            llm_was_used: Whether LLM routing was used (Phase 1 enhancement)
        """
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        context = self._determine_query_context(query)
        
        if pattern_key in self.pattern_learning:
            pattern_data = self.pattern_learning[pattern_key]
            
            # Update traditional success rate
            pattern_data.update_success_rate(was_helpful, template_used, context)
            
            # Update LLM effectiveness if LLM was used
            if llm_was_used:
                pattern_data.update_llm_effectiveness(
                    llm_was_used=True,
                    llm_was_successful=was_helpful
                )
        
        logger.info(
            f"Enhanced pattern feedback recorded: {pattern_key} = "
            f"{'helpful' if was_helpful else 'not helpful'} "
            f"(template: {template_used}, context: {context}, llm_used: {llm_was_used})"
        )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system for optimization including LLM routing data."""
        insights = {
            "total_patterns": len(self.pattern_learning),
            "most_successful_patterns": [],
            "least_successful_patterns": [],
            "best_templates": {},
            "context_performance": {},
            "improvement_suggestions": [],
            # Phase 1: LLM routing insights
            "llm_routing_stats": {
                "total_attempts": 0,
                "total_successes": 0,
                "average_fallback_rate": 0.0,
                "patterns_using_llm": 0
            },
            "vector_boost_effectiveness": {},
            "edge_case_patterns": []
        }
        
        if not self.pattern_learning:
            return insights
        
        # Traditional analysis
        sorted_patterns = sorted(
            self.pattern_learning.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )
        
        # Most and least successful patterns
        insights["most_successful_patterns"] = [
            {
                "pattern": pattern,
                "success_rate": data.success_rate,
                "frequency": data.frequency,
                "user_satisfaction": data.user_satisfaction
            }
            for pattern, data in sorted_patterns[:5]
        ]
        
        insights["least_successful_patterns"] = [
            {
                "pattern": pattern,
                "success_rate": data.success_rate,
                "frequency": data.frequency,
                "user_satisfaction": data.user_satisfaction
            }
            for pattern, data in sorted_patterns[-3:] if data.frequency > 5
        ]
        
        # Best templates across all patterns
        for pattern, data in self.pattern_learning.items():
            best_template = data.get_best_template()
            if best_template:
                insights["best_templates"][pattern] = {
                    "template": best_template[:100] + "..." if len(best_template) > 100 else best_template,
                    "effectiveness": data.template_effectiveness[best_template]
                }
        
        # Context performance analysis
        context_aggregates = {}
        for pattern, data in self.pattern_learning.items():
            for context, success_rate in data.context_success.items():
                if context not in context_aggregates:
                    context_aggregates[context] = []
                context_aggregates[context].append(success_rate)
        
        for context, rates in context_aggregates.items():
            insights["context_performance"][context] = {
                "average_success_rate": sum(rates) / len(rates),
                "pattern_count": len(rates)
            }
        
        # Phase 1: LLM routing analytics
        total_llm_attempts = 0
        total_llm_successes = 0
        patterns_using_llm = 0
        
        for pattern, data in self.pattern_learning.items():
            if data.llm_routing_attempts > 0:
                patterns_using_llm += 1
                total_llm_attempts += data.llm_routing_attempts
                total_llm_successes += data.llm_routing_successes
                
                # Track vector boost effectiveness
                if data.vector_confidence_boost > 0:
                    insights["vector_boost_effectiveness"][pattern] = {
                        "boost_amount": data.vector_confidence_boost,
                        "success_rate": data.success_rate
                    }
                
                # Track edge case patterns
                if data.edge_case_frequency > 2:
                    insights["edge_case_patterns"].append({
                        "pattern": pattern,
                        "edge_case_frequency": data.edge_case_frequency,
                        "llm_fallback_rate": data.llm_fallback_rate,
                        "success_rate": data.success_rate
                    })
        
        insights["llm_routing_stats"]["total_attempts"] = total_llm_attempts
        insights["llm_routing_stats"]["total_successes"] = total_llm_successes
        insights["llm_routing_stats"]["patterns_using_llm"] = patterns_using_llm
        
        if total_llm_attempts > 0:
            insights["llm_routing_stats"]["average_fallback_rate"] = (
                total_llm_attempts - total_llm_successes
            ) / total_llm_attempts
        
        # Enhanced improvement suggestions
        low_performing_patterns = [
            pattern for pattern, data in self.pattern_learning.items()
            if data.success_rate < 0.6 and data.frequency > 3
        ]
        
        if low_performing_patterns:
            insights["improvement_suggestions"].append(
                f"Consider improving responses for patterns: {', '.join(low_performing_patterns[:3])}"
            )
        
        # LLM-specific suggestions
        high_fallback_patterns = [
            pattern for pattern, data in self.pattern_learning.items()
            if data.llm_fallback_rate > 0.5 and data.llm_routing_attempts > 5
        ]
        
        if high_fallback_patterns:
            insights["improvement_suggestions"].append(
                f"High LLM fallback rates for patterns: {', '.join(high_fallback_patterns[:3])}"
            )
        
        underused_contexts = [
            context for context, performance in insights["context_performance"].items()
            if performance["pattern_count"] < 3
        ]
        
        if underused_contexts:
            insights["improvement_suggestions"].append(
                f"Gather more data for contexts: {', '.join(underused_contexts[:3])}"
            )
        
        return insights
    
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
    
    # Phase 1: Enhanced Component Integration Methods
    
    async def _initialize_pattern_classifier(self):
        """Lazy initialization of pattern classifier to avoid circular imports."""
        if self.pattern_classifier is None:
            try:
                from .pattern_classifier import ConversationalPatternClassifier
                self.pattern_classifier = ConversationalPatternClassifier()
                await self.pattern_classifier.initialize()
                logger.info("Pattern classifier initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize pattern classifier: {e}")
                self.vector_boost_enabled = False
    
    def _enhance_pattern_confidence(self, query: str, pattern_type: ConversationalPattern, base_confidence: float) -> Tuple[float, bool]:
        """
        Enhance pattern confidence using vector similarity and return LLM routing recommendation.
        
        Args:
            query: User query
            pattern_type: Identified pattern type
            base_confidence: Template-based confidence
            
        Returns:
            Tuple of (enhanced_confidence, llm_routing_recommended)
        """
        if not self.vector_boost_enabled or self.pattern_classifier is None:
            return base_confidence, False
        
        try:
            # This would be async in real implementation, simplified for Phase 1
            # For now, use synchronous boost based on pattern learning data
            pattern_key = f"{pattern_type.value}_{len(query.split())}"
            
            if pattern_key in self.pattern_learning:
                pattern_data = self.pattern_learning[pattern_key]
                
                # Apply learned vector confidence boost
                vector_boost = pattern_data.vector_confidence_boost
                enhanced_confidence = min(0.95, base_confidence + vector_boost)
                
                # Recommend LLM routing for edge cases
                llm_routing_recommended = (
                    pattern_data.edge_case_frequency > 2 or
                    enhanced_confidence < 0.6 or
                    pattern_data.llm_fallback_rate > 0.4
                )
                
                return enhanced_confidence, llm_routing_recommended
            
            return base_confidence, False
            
        except Exception as e:
            logger.warning(f"Vector confidence enhancement failed: {e}")
            return base_confidence, False
    
    def _should_use_llm_routing(self, query: str, pattern_type: ConversationalPattern, confidence: float) -> bool:
        """
        Determine if LLM routing should be used based on learning data and pattern analysis.
        
        Args:
            query: User query
            pattern_type: Identified pattern type
            confidence: Enhanced confidence score
            
        Returns:
            Boolean indicating if LLM routing should be used
        """
        if not self.llm_routing_enabled:
            return False
        
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        
        # Use learning data to make routing decision
        if pattern_key in self.pattern_learning:
            pattern_data = self.pattern_learning[pattern_key]
            return pattern_data.should_use_llm(confidence)
        
        # Default routing logic for new patterns
        # Use LLM for low confidence or complex queries
        if confidence < 0.6:
            return True
        
        # Use LLM for longer, potentially complex queries
        if len(query.split()) > 10:
            return True
        
        # Use LLM for unknown patterns
        if pattern_type == ConversationalPattern.UNKNOWN:
            return True
        
        return False
    
    def _record_enhanced_pattern_usage(self, query: str, pattern_type: ConversationalPattern, 
                                     template_used: str, enhanced_confidence: float,
                                     llm_routing_recommended: bool, llm_routing_used: bool) -> None:
        """Record enhanced pattern usage including LLM routing decisions."""
        # Record traditional pattern usage
        self._record_pattern_usage(query, pattern_type, template_used)
        
        # Record LLM routing data
        pattern_key = f"{pattern_type.value}_{len(query.split())}"
        
        if pattern_key in self.pattern_learning:
            pattern_data = self.pattern_learning[pattern_key]
            
            # Update LLM effectiveness tracking
            pattern_data.update_llm_effectiveness(
                llm_was_used=llm_routing_used,
                llm_was_successful=None  # Will be updated with feedback
            )
            
            # Update vector confidence boost based on enhancement
            original_confidence = enhanced_confidence - pattern_data.vector_confidence_boost
            actual_boost = enhanced_confidence - original_confidence
            pattern_data.update_vector_confidence_boost(actual_boost)
            
            # Track edge case frequency
            if llm_routing_recommended:
                pattern_data.edge_case_frequency += 1
