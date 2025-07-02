"""
APS-specific patterns and weights for rule-based query classification.

T            r'\b(?:what (?:do|did|does).*think|what.*opinions)\b',
            r'\b(?:what (?:are|were).*saying|what.*opinions)\b',
            r'\b(?:generally say|typically say|usually say)\b',
            r'\b(?:feedback about|comments about|opinions on|thoughts on)\b',
            r'\b(?:what.*feedback|feedback.*provided|feedback.*given)\b',module contains all the Australian Public Service domain-specific 
patterns and their associated weights for accurate query classification
in the learning analytics context.

Example Usage:
    # Access compiled patterns
    patterns = aps_patterns.compiled_patterns
    sql_patterns = patterns["SQL"]
    vector_patterns = patterns["VECTOR"]
    hybrid_patterns = patterns["HYBRID"]
    
    # Get weighted patterns for confidence scoring
    weighted_patterns = aps_pattern_weights.get_all_weighted_patterns()
    
    # Test a query against SQL patterns
    query = "How many Level 6 users completed training?"
    matches = []
    for pattern in sql_patterns:
        if pattern.search(query.lower()):
            matches.append(pattern.pattern)
    
    # Check pattern weights
    sql_weights = aps_pattern_weights.get_sql_weights()
    high_weight_patterns = [p for p, w in sql_weights.items() if w == "high"]
    
    # Get all pattern statistics
    total_patterns = (
        len(patterns["SQL"]) + 
        len(patterns["VECTOR"]) + 
        len(patterns["HYBRID"])
    )
    print(f"Total APS patterns loaded: {total_patterns}")
"""

import re
from typing import Any, Dict, List, Pattern


class APSPatterns:
    """Container for APS-specific classification patterns."""
    
    def __init__(self):
        """Initialize APS-specific patterns with domain knowledge."""
        
        # Core SQL patterns for statistical analysis
        self.sql_patterns = [
            # Core statistical patterns (preserved from original)
            r'\b(?:count|how many|number of)\b',
            r'\b(?:average|mean|avg)\b',
            r'\b(?:percentage|percent|%)\b',
            r'\b(?:breakdown by|group by|categorized by)\b',
            r'\b(?:statistics|stats|statistical)\b',
            r'\b(?:total|sum|aggregate)\b',
            r'\b(?:compare numbers|numerical comparison)\b',
            r'\b(?:completion rate|enrollment rate)\b',
            r'\b(?:agency breakdown|level breakdown|user level)\b',
            
            # Enhanced APS-specific statistical patterns
            r'\b(?:executive level|level [1-6]|EL[12]|APS [1-6])\b.*(?:completion|attendance|performance)',
            r'\b(?:agency|department|portfolio)\b.*(?:breakdown|comparison|statistics)',
            r'\b(?:learning pathway|professional development|capability framework)\b.*(?:metrics|data)',
            r'\b(?:mandatory training|compliance training)\b.*(?:rates|numbers|tracking)',
            r'\b(?:face-to-face|virtual|blended)\b.*(?:delivery|attendance|completion)',
            r'\b(?:cost per|budget|resource allocation)\b.*(?:training|learning)',
            r'\b(?:quarterly|annual|yearly)\b.*(?:training|learning|development)\b.*(?:report|summary)',
            r'\b(?:participation rate|dropout rate|success rate)\b',
            r'\b(?:training hours|contact hours|learning hours)\b.*(?:total|average|per)',
            r'\b(?:geographical|location|state)\b.*(?:breakdown|distribution)'
        ]
        
        # Core VECTOR patterns for feedback analysis
        self.vector_patterns = [
            # Core feedback patterns (enhanced to catch more variations)
            r'\b(?:what (?:did|do|does).*say|what.*said)\b',
            r'\b(?:what (?:do|did|does).*think|what.*thought)\b',
            r'\b(?:what (?:are|were).*saying|what.*opinions)\b',
            r'\b(?:generally say|typically say|usually say)\b',
            r'\b(?:feedback about|comments about|opinions on|thoughts on)\b',
            r'\b(?:experiences with|experience of)\b',
            r'\b(?:user feedback|participant feedback|student feedback)\b',
            r'\b(?:comments|opinions|thoughts|feelings)\b',
            r'\b(?:issues mentioned|problems reported)\b',
            r'\b(?:satisfaction|dissatisfaction)\b',
            r'\b(?:testimonials|reviews|responses)\b',
            r'\b(?:what people think|user opinions)\b',
            
            # Enhanced APS-specific feedback patterns
            r'\b(?:participant|delegate|attendee)\b.*(?:experience|reflection|view)',
            r'\b(?:training quality|course quality|learning experience)\b.*(?:feedback|assessment)',
            r'\b(?:facilitator|presenter|instructor)\b.*(?:effectiveness|skill|performance)',
            r'\b(?:venue|location|facilities)\b.*(?:issues|problems|concerns)',
            r'\b(?:accessibility|inclusion|diversity)\b.*(?:feedback|experience)',
            r'\b(?:technical issues|platform problems|system difficulties)\b',
            r'\b(?:relevance to role|workplace application|practical use)\b',
            r'\b(?:course content|curriculum|material)\b.*(?:feedback|quality|relevance)',
            r'\b(?:learning outcomes|skill development|capability building)\b.*(?:feedback|experience)',
            r'\b(?:recommendation|would recommend|likelihood to recommend)\b',
            
            # Content feedback patterns (evaluation table - Phase 2 enhancement)
            r'\b(?:course|learning|training).*(?:feedback|evaluation|review|assessment|rating)\b',
            r'\b(?:content|material|curriculum|lesson).*(?:feedback|quality|usefulness|relevance)\b',
            r'\b(?:instructor|facilitator|trainer|presenter).*(?:feedback|rating|performance|effectiveness)\b',
            r'\b(?:session|workshop|module|unit).*(?:feedback|evaluation|review)\b',
            r'\b(?:delivery method|teaching style|presentation).*(?:feedback|opinion|view)\b',
            r'\b(?:learning experience|educational experience).*(?:feedback|evaluation|satisfaction)\b',
            r'\b(?:post-course|post-training|after.*(course|training)).*(?:feedback|survey|evaluation)\b',
            r'\b(?:formal feedback|structured feedback|evaluation forms?)\b'
        ]
        
        # HYBRID patterns for combined analysis
        self.hybrid_patterns = [
            # Core hybrid patterns (preserved from original)
            r'\b(?:analyze satisfaction|analyze feedback)\b',
            r'\b(?:compare feedback across|feedback trends)\b',
            r'\b(?:sentiment by agency|satisfaction by level)\b',
            r'\b(?:trends in opinions|opinion trends)\b',
            r'\b(?:comprehensive analysis|detailed analysis)\b',
            r'\b(?:both.*and|statistics.*feedback|numbers.*comments)\b',
            r'\b(?:quantitative.*qualitative|statistical.*sentiment)\b',
            
            # General feedback aggregation patterns (should be HYBRID)
            r'\b(?:what (?:do|did|does).* generally|what.*overall|what.*in general)\b',
            r'\b(?:generally.*(?:say|think|feel|report|mention))\b',
            r'\b(?:overall.*(?:feedback|opinion|satisfaction|experience))\b',
            r'\b(?:common.*(?:feedback|themes|issues|concerns))\b',
            r'\b(?:typical.*(?:feedback|response|comment))\b',
            r'\b(?:main.*(?:feedback|concerns|issues|themes))\b',
            r'\b(?:summary.*(?:feedback|opinions|comments))\b',
            r'\b(?:aggregate.*(?:feedback|sentiment|opinions))\b',
            
            # Enhanced APS-specific hybrid patterns
            r'\b(?:analyse|analyze)\b.*(?:satisfaction|effectiveness)\b.*(?:across|by|between)',
            r'\b(?:training ROI|return on investment|cost-benefit)\b.*(?:analysis|evaluation)',
            r'\b(?:performance impact|capability improvement|skill development)\b.*(?:measurement|assessment)',
            r'\b(?:stakeholder satisfaction|user experience)\b.*(?:metrics|analysis)',
            r'\b(?:trend analysis|pattern identification|insight generation)\b',
            r'\b(?:comprehensive|holistic|integrated)\b.*(?:evaluation|assessment|review)',
            r'\b(?:correlate|correlation)\b.*(?:satisfaction|feedback)\b.*(?:with|and)\b.*(?:completion|performance)',
            r'\b(?:demographic analysis|cohort analysis)\b.*(?:feedback|satisfaction)'
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = {
            "SQL": [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_patterns],
            "VECTOR": [re.compile(pattern, re.IGNORECASE) for pattern in self.vector_patterns],
            "HYBRID": [re.compile(pattern, re.IGNORECASE) for pattern in self.hybrid_patterns]
        }


class APSPatternWeights:
    """Weighted patterns for enhanced confidence scoring."""
    
    def __init__(self):
        """Initialize weighted pattern system."""
        
        # Pattern weighting system for improved confidence calibration
        self.pattern_weights = {
            "SQL": {
                "high_confidence": [
                    r'\b(?:count|how many|number of)\b',
                    r'\b(?:percentage|percent|%)\b',
                    r'\b(?:total|sum|aggregate)\b',
                    r'\b(?:completion rate|enrollment rate)\b',
                    r'\b(?:executive level|level [1-6]|EL[12]|APS [1-6])\b.*(?:completion|attendance|performance)',
                    r'\b(?:participation rate|dropout rate|success rate)\b'
                ],
                "medium_confidence": [
                    r'\b(?:breakdown by|group by|categorized by)\b',
                    r'\b(?:statistics|stats|statistical)\b',
                    r'\b(?:average|mean|avg)\b',
                    r'\b(?:agency|department|portfolio)\b.*(?:breakdown|comparison|statistics)',
                    r'\b(?:training hours|contact hours|learning hours)\b.*(?:total|average|per)'
                ],
                "low_confidence": [
                    r'\b(?:compare numbers|numerical comparison)\b',
                    r'\b(?:quarterly|annual|yearly)\b.*(?:training|learning|development)\b.*(?:report|summary)',
                    r'\b(?:geographical|location|state)\b.*(?:breakdown|distribution)'
                ]
            },
            "VECTOR": {
                "high_confidence": [
                    r'\b(?:what (?:did|do|does).*say|feedback about)\b',
                    r'\b(?:what (?:do|did|does).*think|what.*opinions)\b',
                    r'\b(?:comments|opinions|thoughts)\b',
                    r'\b(?:participant|delegate|attendee)\b.*(?:experience|reflection|view)',
                    r'\b(?:technical issues|platform problems|system difficulties)\b',
                    r'\b(?:recommendation|would recommend|likelihood to recommend)\b',
                    # Content feedback patterns (Phase 2 enhancement)
                    r'\b(?:course|learning|training).*(?:feedback|evaluation|review|assessment|rating)\b',
                    r'\b(?:content|material|curriculum|lesson).*(?:feedback|quality|usefulness|relevance)\b',
                    r'\b(?:instructor|facilitator|trainer|presenter).*(?:feedback|rating|performance|effectiveness)\b'
                ],
                "medium_confidence": [
                    r'\b(?:experiences with|satisfaction)\b',
                    r'\b(?:training quality|course quality|learning experience)\b.*(?:feedback|assessment)',
                    r'\b(?:facilitator|presenter|instructor)\b.*(?:effectiveness|skill|performance)',
                    r'\b(?:relevance to role|workplace application|practical use)\b',
                    r'\b(?:session|workshop|module|unit).*(?:feedback|evaluation|review)\b',
                    r'\b(?:learning experience|educational experience).*(?:feedback|evaluation|satisfaction)\b'
                ],
                "low_confidence": [
                    r'\b(?:feelings|thoughts)\b',
                    r'\b(?:venue|location|facilities)\b.*(?:issues|problems|concerns)',
                    r'\b(?:accessibility|inclusion|diversity)\b.*(?:feedback|experience)',
                    r'\b(?:post-course|post-training|after.*(course|training)).*(?:feedback|survey|evaluation)\b'
                ]
            },
            "HYBRID": {
                "high_confidence": [
                    r'\b(?:analyze satisfaction|comprehensive analysis)\b',
                    r'\b(?:what (?:do|did|does).* generally|what.*overall|what.*in general)\b',
                    r'\b(?:generally.*(?:say|think|feel|report|mention))\b',
                    r'\b(?:overall.*(?:feedback|opinion|satisfaction|experience))\b',
                    r'\b(?:what.*generally.*(?:say|think|report|mention))\b',  # Extra weight for generally
                    r'\b(?:what.*feedback.*(?:give|gave|provide|provided))\b',  # Direct "give feedback" match
                    r'\b(?:training ROI|return on investment|cost-benefit)\b.*(?:analysis|evaluation)',
                    r'\b(?:correlate|correlation)\b.*(?:satisfaction|feedback)\b.*(?:with|and)\b.*(?:completion|performance)'
                ],
                "medium_confidence": [
                    r'\b(?:trends in|patterns in)\b',
                    r'\b(?:common.*(?:feedback|themes|issues|concerns))\b',
                    r'\b(?:typical.*(?:feedback|response|comment))\b',
                    r'\b(?:main.*(?:feedback|concerns|issues|themes))\b',
                    r'\b(?:summary.*(?:feedback|opinions|comments))\b',
                    r'\b(?:aggregate.*(?:feedback|sentiment|opinions))\b',
                    r'\b(?:performance impact|capability improvement|skill development)\b.*(?:measurement|assessment)',
                    r'\b(?:stakeholder satisfaction|user experience)\b.*(?:metrics|analysis)',
                    r'\b(?:demographic analysis|cohort analysis)\b.*(?:feedback|satisfaction)'
                ],
                "low_confidence": [
                    r'\b(?:both.*and|detailed analysis)\b',
                    r'\b(?:comprehensive|holistic|integrated)\b.*(?:evaluation|assessment|review)',
                    r'\b(?:trend analysis|pattern identification|insight generation)\b'
                ]
            }
        }
        
        # Compile weighted patterns for enhanced confidence calculation
        self.compiled_weighted_patterns = {}
        for category in ["SQL", "VECTOR", "HYBRID"]:
            self.compiled_weighted_patterns[category] = {}
            for confidence_level in ["high_confidence", "medium_confidence", "low_confidence"]:
                self.compiled_weighted_patterns[category][confidence_level] = [
                    re.compile(pattern, re.IGNORECASE) 
                    for pattern in self.pattern_weights[category][confidence_level]
                ]
    
    def get_weighted_patterns_for_category(self, category: str) -> Dict[str, List[Pattern]]:
        """
        Get compiled weighted patterns for a specific category.
        
        Args:
            category: Classification category (SQL, VECTOR, HYBRID)
            
        Returns:
            Dictionary of compiled patterns by confidence level
        """
        return self.compiled_weighted_patterns.get(category, {})
    
    def get_all_weighted_patterns(self) -> Dict[str, Dict[str, List[Pattern]]]:
        """
        Get all compiled weighted patterns.
        
        Returns:
            Full dictionary of compiled weighted patterns
        """
        return self.compiled_weighted_patterns


class APSDomainKnowledge:
    """APS domain-specific knowledge for classification enhancement."""
    
    def __init__(self):
        """Initialize APS domain knowledge."""
        
        # APS domain-specific indicators
        self.aps_domain_indicators = [
            "executive level", "el1", "el2", "aps", "agency", "department", 
            "portfolio", "capability framework", "professional development",
            "mandatory training", "compliance", "participant", "delegate",
            "facilitator", "virtual learning", "face-to-face", "blended"
        ]
        
        # Ambiguity markers that reduce confidence
        self.ambiguity_markers = [
            "maybe", "perhaps", "might be", "could be", "not sure",
            "unclear", "confusing", "ambiguous", "general", "broad",
            "anything", "something", "everything", "help", "assist"
        ]
        
        # Keywords for fallback classification
        self.fallback_keywords = {
            "sql": ["count", "many", "number", "average", "percentage", "total", "breakdown"],
            "vector": ["feedback", "comment", "say", "opinion", "experience", "think"],
            "hybrid": ["analyze", "analysis", "comprehensive", "trends", "correlation", "impact"]
        }
    
    def get_aps_indicators(self) -> List[str]:
        """Get list of APS domain indicators."""
        return self.aps_domain_indicators.copy()
    
    def get_ambiguity_markers(self) -> List[str]:
        """Get list of ambiguity markers."""
        return self.ambiguity_markers.copy()
    
    def get_fallback_keywords(self) -> Dict[str, List[str]]:
        """Get fallback keywords for simple classification."""
        return {
            category: keywords.copy() 
            for category, keywords in self.fallback_keywords.items()
        }


class FeedbackTableClassifier:
    """
    Table-specific feedback classification for Phase 2 enhancement.
    
    Distinguishes between:
    - Content feedback: evaluation table (course/training feedback)
    - System feedback: rag_user_feedback table (RAG system feedback)
    
    This addresses the core issue where LLM was incorrectly joining
    rag_user_feedback with learning_content instead of using evaluation table.
    """
    
    def __init__(self):
        """Initialize table-specific feedback patterns."""
        
        # Content feedback patterns (should use evaluation table)
        self.content_feedback_patterns = [
            # Direct course/training feedback
            r'\b(?:course|training|learning|workshop|session).*(?:feedback|evaluation|review|rating|assessment)\b',
            r'\b(?:content|material|curriculum|lesson|module).*(?:feedback|quality|usefulness|relevance|rating)\b',
            r'\b(?:instructor|facilitator|trainer|presenter|teacher).*(?:feedback|rating|performance|effectiveness)\b',
            r'\b(?:learning experience|educational experience|training experience)\b.*(?:feedback|evaluation|satisfaction)\b',
            r'\b(?:delivery method|teaching style|presentation style).*(?:feedback|opinion|view|rating)\b',
            
            # Post-event feedback
            r'\b(?:post-course|post-training|after.*(course|training|session)).*(?:feedback|survey|evaluation)\b',
            r'\b(?:end-of-course|course completion|training completion).*(?:feedback|survey|evaluation)\b',
            r'\b(?:participant|delegate|attendee|student).*(?:evaluation|feedback|rating|survey)\b',
            
            # Formal feedback structures
            r'\b(?:formal feedback|structured feedback|evaluation forms?|feedback forms?|survey responses?)\b',
            r'\b(?:likert scale|rating scale|satisfaction scale|effectiveness scale)\b',
            r'\b(?:course evaluation|training evaluation|program evaluation|learning evaluation)\b',
            
            # Quality and effectiveness feedback
            r'\b(?:training quality|course quality|content quality|delivery quality).*(?:feedback|assessment|evaluation)\b',
            r'\b(?:learning outcomes|skill development|knowledge gain).*(?:feedback|evaluation|effectiveness)\b',
            r'\b(?:workplace application|practical use|relevance to role).*(?:feedback|evaluation|rating)\b',
            
            # Improvement and recommendation feedback
            r'\b(?:improvement suggestions|recommendations|suggestions for).*(?:course|training|content|delivery)\b',
            r'\b(?:would recommend|likelihood to recommend|recommendation rate)\b.*(?:course|training|program)\b',
            
            # Enhanced patterns to catch the original query
            r'\b(?:feedback|comments|evaluation|review|rating|assessment).*(?:about|regarding|for|on).*(?:course|training|learning|content|session|workshop|program|module)\b',
            r'\b(?:what|how).*(?:feedback|comments|evaluation|review|rating).*(?:users?|participants?|attendees?|students?).*(?:give|gave|provide|provided).*(?:about|regarding|for|on).*(?:course|training|learning|content)\b',
            r'\b(?:users?|participants?|attendees?|students?).*(?:feedback|comments|evaluation|review|rating|assessment).*(?:about|regarding|for|on).*(?:course|training|learning|content|session|workshop|program)\b',
            r'\b(?:what|how).*(?:users?|participants?|attendees?|students?).*(?:say|said|think|thought|feel|felt).*(?:about|regarding).*(?:course|training|learning|content|session|workshop|program)\b'
        ]
        
        # System feedback patterns (should use rag_user_feedback table)
        self.system_feedback_patterns = [
            # RAG system specific feedback
            r'\b(?:rag|retrieval|search|query|question).*(?:feedback|experience|performance|accuracy)\b',
            r'\b(?:system|platform|tool|interface).*(?:feedback|usability|performance|issues)\b',
            r'\b(?:answer quality|response quality|search results|query results).*(?:feedback|evaluation|rating)\b',
            
            # Technical system feedback
            r'\b(?:technical issues|system problems|platform difficulties|interface problems)\b',
            r'\b(?:search functionality|query processing|information retrieval)\b.*(?:feedback|issues|problems)\b',
            r'\b(?:user interface|user experience|system navigation).*(?:feedback|usability|satisfaction)\b',
            
            # AI/LLM specific feedback
            r'\b(?:ai response|ai answer|artificial intelligence|machine learning).*(?:feedback|accuracy|quality)\b',
            r'\b(?:chatbot|virtual assistant|automated response).*(?:feedback|performance|effectiveness)\b',
            r'\b(?:natural language|language processing|text analysis).*(?:feedback|accuracy|quality)\b',
            
            # Enhanced patterns for system feedback
            r'\b(?:feedback|comments|evaluation|review|rating).*(?:about|regarding|for|on).*(?:system|platform|interface|search|query|response|answer|ai|rag)\b',
            r'\b(?:what|how).*(?:feedback|comments|evaluation|review).*(?:users?|participants?).*(?:give|gave|provide|provided).*(?:about|regarding|for|on).*(?:system|platform|interface|search|query|response|answer)\b'
        ]
        
        # Compile patterns for performance
        self.compiled_content_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.content_feedback_patterns
        ]
        self.compiled_system_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.system_feedback_patterns
        ]
    
    def classify_feedback_type(self, query: str) -> Dict[str, Any]:
        """
        Classify whether a feedback query should use evaluation or rag_user_feedback table.
        
        Args:
            query: Query text to classify
            
        Returns:
            Dict with classification result:
            {
                'table_type': 'content' | 'system' | 'unclear',
                'recommended_table': 'evaluation' | 'rag_user_feedback' | None,
                'confidence': float (0-1),
                'matched_patterns': List[str],
                'reasoning': str
            }
        """
        query_lower = query.lower()
        
        # Count pattern matches
        content_matches = []
        system_matches = []
        
        # Check content feedback patterns
        for pattern in self.compiled_content_patterns:
            match = pattern.search(query_lower)
            if match:
                content_matches.append(pattern.pattern)
        
        # Check system feedback patterns
        for pattern in self.compiled_system_patterns:
            match = pattern.search(query_lower)
            if match:
                system_matches.append(pattern.pattern)
        
        # Calculate scores
        content_score = len(content_matches)
        system_score = len(system_matches)
        total_matches = content_score + system_score
        
        # Determine classification
        if content_score > system_score and content_score > 0:
            table_type = 'content'
            recommended_table = 'evaluation'
            confidence = min(0.9, 0.5 + (content_score * 0.1))
            reasoning = f"Query contains {content_score} content feedback indicators, suggesting course/training evaluation feedback"
        elif system_score > content_score and system_score > 0:
            table_type = 'system'
            recommended_table = 'rag_user_feedback'
            confidence = min(0.9, 0.5 + (system_score * 0.1))
            reasoning = f"Query contains {system_score} system feedback indicators, suggesting RAG system feedback"
        elif total_matches == 0:
            table_type = 'unclear'
            recommended_table = None
            confidence = 0.0
            reasoning = "No specific feedback type indicators found"
        else:
            # Equal matches - unclear
            table_type = 'unclear'
            recommended_table = None
            confidence = 0.3
            reasoning = f"Mixed indicators: {content_score} content, {system_score} system - clarification needed"
        
        return {
            'table_type': table_type,
            'recommended_table': recommended_table,
            'confidence': confidence,
            'matched_patterns': content_matches + system_matches,
            'reasoning': reasoning,
            'content_matches': content_matches,
            'system_matches': system_matches
        }
    
    def get_table_usage_guidance(self, feedback_classification: Dict[str, Any]) -> str:
        """
        Generate specific SQL guidance based on feedback classification.
        
        Args:
            feedback_classification: Result from classify_feedback_type()
            
        Returns:
            String with specific table usage guidance for SQL generation
        """
        table_type = feedback_classification.get('table_type')
        recommended_table = feedback_classification.get('recommended_table')
        
        if table_type == 'content' and recommended_table == 'evaluation':
            return (
                "For content feedback queries, use the 'evaluation' table which contains "
                "structured course/training feedback including ratings, satisfaction scores, "
                "and formal evaluation responses. Join with 'learning_content' and 'users' "
                "tables as needed for context."
            )
        elif table_type == 'system' and recommended_table == 'rag_user_feedback':
            return (
                "For system feedback queries, use the 'rag_user_feedback' table which contains "
                "feedback about the RAG system itself, search quality, and technical issues. "
                "This table is separate from course content feedback."
            )
        else:
            return (
                "Feedback type unclear - consider asking for clarification about whether the "
                "user wants course/training feedback (use evaluation table) or system/platform "
                "feedback (use rag_user_feedback table)."
            )


# Create singleton instances for easy import
aps_patterns = APSPatterns()
aps_pattern_weights = APSPatternWeights()
aps_domain_knowledge = APSDomainKnowledge()
feedback_table_classifier = FeedbackTableClassifier()
