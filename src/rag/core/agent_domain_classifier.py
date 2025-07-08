"""
LLM-based domain relevance classifier for RAG system.

This module provides schema-aware domain classification to determine if queries
are relevant to the Australian Public Service training and survey data.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.settings import get_settings
from ..utils.llm_utils import get_llm


class DomainRelevanceClassifier:
    """
    LLM-based classifier to determine if queries are relevant to the schema.
    
    Uses the data-dictionary.json to understand what data is available and
    determines if user queries can reasonably be answered using that data.
    """
    
    def __init__(self, llm=None):
        """Initialize the domain classifier."""
        self.llm = llm or get_llm()
        self.settings = get_settings()
        self._schema_context = None
        self._load_schema_context()
    
    def _load_schema_context(self) -> None:
        """Load and prepare schema context for domain classification."""
        try:
            # Load the data dictionary
            schema_path = Path(__file__).parent.parent.parent / "csv" / "data-dictionary.json"
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            # Create a concise domain description for the LLM
            self._schema_context = self._create_domain_description(schema_data)
            
        except Exception as e:
            # Fallback to basic description if schema loading fails
            self._schema_context = """
            Australian Public Service training and survey data including:
            - User information (staff levels, agencies)
            - Learning content (courses, training materials)
            - Attendance records (enrollment, completion status)
            - Evaluation feedback (post-course surveys, ratings, comments)
            """
    
    def _create_domain_description(self, schema_data: Dict[str, Any]) -> str:
        """Create a concise domain description from the schema."""
        domain_desc = "Australian Public Service training and survey data system containing:\n\n"
        
        for table_name, table_info in schema_data.items():
            desc = table_info.get('description', '')
            domain_desc += f"• **{table_name.title()}**: {desc}\n"
            
            # Add key column examples to show data scope
            key_columns = []
            for col in table_info.get('columns', [])[:3]:  # First 3 columns as examples
                col_desc = col.get('description', col.get('name', ''))
                key_columns.append(f"{col['name']} ({col_desc})")
            
            if key_columns:
                domain_desc += f"  Fields: {', '.join(key_columns)}\n"
            domain_desc += "\n"
        
        domain_desc += """
        **What this system can analyze:**
        - Training completion rates and participation statistics
        - Course effectiveness and learning outcomes
        - Participant feedback, satisfaction ratings, and comments
        - Agency-level training performance and trends
        - Staff development and skill building progress
        
        **What this system cannot analyze:**
        - General knowledge questions unrelated to training
        - Personal advice or recommendations
        - Topics outside government training and surveys
        - Manufacturing, DIY, or how-to guides
        - Entertainment, shopping, travel, or lifestyle queries
        """
        
        return domain_desc
    
    async def is_query_domain_relevant(self, query: str) -> Dict[str, Any]:
        """
        Determine if a query is relevant to the training/survey domain.
        
        Args:
            query: User query to classify
            
        Returns:
            Dict with:
            - is_relevant: bool
            - confidence: float (0.0-1.0)
            - reason: str explaining the decision
            - suggested_redirect: Optional[str] for off-topic queries
        """
        try:
            classification_prompt = f"""
You are a domain classifier for an Australian Public Service training and survey data analysis system.

**SYSTEM DOMAIN:**
{self._schema_context}

**USER QUERY:** "{query}"

**TASK:** Determine if this query can be reasonably answered using the training and survey data described above.

**CLASSIFICATION RULES:**
1. RELEVANT: Query asks about training, courses, staff development, participation, feedback, or survey data
2. RELEVANT: Query wants statistics or analysis related to learning outcomes, attendance, or evaluation data
3. RELEVANT: Query asks about agency performance, user levels, or organizational training metrics
4. NOT RELEVANT: Query is general knowledge, personal advice, or completely unrelated topics
5. NOT RELEVANT: Query is about manufacturing, DIY, entertainment, travel, shopping, etc.
6. NOT RELEVANT: Query asks about topics that would require external knowledge beyond training data

You must respond with ONLY a JSON object in this exact format:
{{
    "is_relevant": true,
    "confidence": 0.95,
    "reason": "Brief explanation of why this query is/isn't relevant to training data",
    "category": "training_statistics"
}}

Valid categories: training_statistics, feedback_analysis, attendance_data, general_knowledge, unrelated_topic

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT:"""

            # Get LLM classification
            response = await self.llm.ainvoke(classification_prompt)
            
            # Clean and parse response
            response_text = response.content.strip()
            
            # Remove any markdown formatting or extra text
            if "```json" in response_text:
                # Extract JSON from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract JSON from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Try to find JSON object if there's extra text
            if not response_text.startswith("{"):
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                # Validate required fields
                if not all(key in result for key in ['is_relevant', 'confidence', 'reason']):
                    raise ValueError("Missing required fields in LLM response")
                
                # Ensure confidence is a float
                result['confidence'] = float(result['confidence'])
                
                # Add suggested redirect for irrelevant queries
                if not result['is_relevant']:
                    result['suggested_redirect'] = self._generate_redirect_message(query, result.get('reason', ''))
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                # Enhanced fallback parsing if JSON fails
                content_lower = response.content.lower()
                
                # Look for clear indicators
                is_relevant_indicators = ["relevant", "training", "survey", "data", "statistics"]
                not_relevant_indicators = ["not relevant", "unrelated", "general knowledge", "outside scope"]
                
                # Count indicators
                relevant_count = sum(1 for indicator in is_relevant_indicators if indicator in content_lower)
                not_relevant_count = sum(1 for indicator in not_relevant_indicators if indicator in content_lower)
                
                if not_relevant_count > relevant_count:
                    is_relevant = False
                    confidence = 0.6  # Medium confidence for fallback
                else:
                    is_relevant = True
                    confidence = 0.3  # Low confidence for fallback
                
                return {
                    "is_relevant": is_relevant,
                    "confidence": confidence,
                    "reason": f"Fallback classification (JSON parse error: {e}). Raw response: {response.content[:200]}",
                    "category": "unknown",
                    "suggested_redirect": None if is_relevant else self._generate_redirect_message(query, "Unable to parse detailed reasoning")
                }
                
        except Exception as e:
            # Complete fallback - assume relevant to avoid blocking valid queries
            return {
                "is_relevant": True,
                "confidence": 0.3,
                "reason": f"Classification failed ({e}), defaulting to relevant for safety",
                "category": "error_fallback",
                "suggested_redirect": None
            }
    
    def _generate_redirect_message(self, query: str, reason: str) -> str:
        """Generate a helpful redirect message for off-topic queries."""
        return f"""I understand you're asking: "{query}"

However, {reason}

**Here's what I can help you with:**

📊 **Training Statistics:**
• Completion rates by agency or role level
• Participation numbers and trends
• Course effectiveness metrics
• Learning outcome measurements

💬 **Participant Feedback:**
• Comments about course content and delivery
• Satisfaction ratings and experiences
• Suggestions for improvement
• Technical issues or accessibility feedback

📈 **Combined Analysis:**
• Statistical trends with supporting feedback
• Correlation between ratings and comments
• Comprehensive program evaluations

**Example questions you could ask:**
• "How many staff completed cybersecurity training this quarter?"
• "What feedback did participants give about virtual learning?"
• "Analyze satisfaction trends with supporting comments"

Would you like to ask about any of these topics instead?"""


# Async helper function for integration
async def check_domain_relevance(query: str, llm=None) -> Dict[str, Any]:
    """
    Quick async function to check if a query is domain-relevant.
    
    Args:
        query: User query to check
        llm: Optional LLM instance (will create if not provided)
        
    Returns:
        Classification result dict
    """
    classifier = DomainRelevanceClassifier(llm=llm)
    return await classifier.is_query_domain_relevant(query)
