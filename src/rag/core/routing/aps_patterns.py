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
from typing import Dict, List, Pattern


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
            r'\b(?:recommendation|would recommend|likelihood to recommend)\b'
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
                    r'\b(?:what.*feedback|feedback.*provided|feedback.*given)\b',
                    r'\b(?:comments|opinions|thoughts)\b',
                    r'\b(?:participant|delegate|attendee)\b.*(?:experience|reflection|view)',
                    r'\b(?:technical issues|platform problems|system difficulties)\b',
                    r'\b(?:recommendation|would recommend|likelihood to recommend)\b'
                ],
                "medium_confidence": [
                    r'\b(?:experiences with|satisfaction)\b',
                    r'\b(?:training quality|course quality|learning experience)\b.*(?:feedback|assessment)',
                    r'\b(?:facilitator|presenter|instructor)\b.*(?:effectiveness|skill|performance)',
                    r'\b(?:relevance to role|workplace application|practical use)\b'
                ],
                "low_confidence": [
                    r'\b(?:feelings|thoughts)\b',
                    r'\b(?:venue|location|facilities)\b.*(?:issues|problems|concerns)',
                    r'\b(?:accessibility|inclusion|diversity)\b.*(?:feedback|experience)'
                ]
            },
            "HYBRID": {
                "high_confidence": [
                    r'\b(?:analyze satisfaction|comprehensive analysis)\b',
                    r'\b(?:what (?:do|did|does).* generally|what.*overall|what.*in general)\b',
                    r'\b(?:generally.*(?:say|think|feel|report|mention))\b',
                    r'\b(?:overall.*(?:feedback|opinion|satisfaction|experience))\b',
                    r'\b(?:what.*generally.*(?:say|think|report|mention))\b',  # Extra weight for generally
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


# Create singleton instances for easy import
aps_patterns = APSPatterns()
aps_pattern_weights = APSPatternWeights()
aps_domain_knowledge = APSDomainKnowledge()
