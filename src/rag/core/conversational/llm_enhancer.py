"""
Conversational LLM Enhancer

This module provides minimal LLM enhancement for conversational responses when template
confidence is low, while preserving all existing functionality and maximizing component reuse.

Key Features:
- Only enhances when template confidence < 0.7 (preserves existing system as primary)
- Reuses existing LLMManager infrastructure
- Integrates with existing PIIDetector for privacy protection
- Maintains Australian-friendly tone and style
- Preserves all existing templates and patterns
- Minimal overhead, maximum component reuse

Classes:
- ConversationalLLMEnhancer: Main enhancement class with conservative approach
- EnhancedResponse: Response with enhancement metadata

Integration:
- Reuses existing LLMManager from utils
- Leverages existing PIIDetector infrastructure
- Preserves ConversationalResponse structure
- Integrates with existing audit logging

Example Usage:
    enhancer = ConversationalLLMEnhancer()
    await enhancer.initialize()
    
    # Only enhance low-confidence responses
    response = await enhancer.enhance_response_if_needed(
        query="G'day, how can you help with our survey analysis?",
        template_response="I can help you analyse survey data...",
        confidence=0.65,
        pattern_type=ConversationalPattern.SYSTEM_QUESTION
    )
    
    if response.enhancement_used:
        print("LLM enhancement applied")
    else:
        print("Template response preserved")
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from .handler import ConversationalPattern, ConversationalResponse
from ..privacy.pii_detector import AustralianPIIDetector
from ...utils.llm_utils import get_llm, LLMManager
from ...utils.logging_utils import get_logger
from ...config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class EnhancedResponse:
    """Extended response with enhancement metadata."""
    content: str
    confidence: float
    pattern_type: ConversationalPattern
    enhancement_used: bool = False
    llm_processing_time: float = 0.0
    privacy_check_passed: bool = True
    original_template: Optional[str] = None
    enhancement_reason: Optional[str] = None


class ConversationalLLMEnhancer:
    """
    Minimal LLM enhancement for conversational responses that preserves
    existing functionality while providing intelligent augmentation when needed.
    
    Conservative Approach:
    - Only enhances when template confidence < 0.7
    - Falls back to templates for PII-containing queries
    - Preserves Australian tone and style
    - Maintains all existing response patterns
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None, 
                 pii_detector: Optional[AustralianPIIDetector] = None):
        """Initialize the LLM enhancer with existing infrastructure."""
        self.llm_manager = llm_manager
        self.pii_detector = pii_detector
        self.settings = get_settings()
        self.is_initialized = False
        
        # Enhancement thresholds
        self.confidence_threshold = getattr(self.settings, 'llm_enhancement_threshold', 0.7)
        self.max_enhancement_time = getattr(self.settings, 'max_llm_enhancement_time', 5.0)
        
        # Enhancement prompts that preserve Australian tone
        self.enhancement_prompts = {
            ConversationalPattern.GREETING: self._get_greeting_enhancement_prompt(),
            ConversationalPattern.SYSTEM_QUESTION: self._get_system_question_enhancement_prompt(),
            ConversationalPattern.HELP_REQUEST: self._get_help_request_enhancement_prompt(),
            ConversationalPattern.POLITENESS: self._get_politeness_enhancement_prompt(),
            ConversationalPattern.OFF_TOPIC: self._get_off_topic_enhancement_prompt(),
        }
        
    async def initialize(self) -> None:
        """Initialize the LLM enhancer with necessary components."""
        if self.is_initialized:
            return
            
        try:
            # Initialize LLM manager if not provided
            if self.llm_manager is None:
                self.llm_manager = LLMManager()
                await self.llm_manager.initialize()
                
            # Initialize PII detector if not provided
            if self.pii_detector is None:
                self.pii_detector = AustralianPIIDetector()
                await self.pii_detector.initialize()
                
            self.is_initialized = True
            logger.info("ConversationalLLMEnhancer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationalLLMEnhancer: {e}")
            raise
            
    async def enhance_response_if_needed(
        self, 
        query: str, 
        template_response: str, 
        confidence: float,
        pattern_type: ConversationalPattern,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedResponse:
        """
        Enhance response only when template confidence is low.
        
        Args:
            query: Original user query
            template_response: Template-generated response
            confidence: Template confidence score
            pattern_type: Detected conversational pattern
            context: Optional context information
            
        Returns:
            EnhancedResponse with potential LLM enhancement
        """
        if not self.is_initialized:
            await self.initialize()
            
        start_time = datetime.now()
        
        try:
            # Conservative approach: only enhance if confidence < threshold
            if confidence >= self.confidence_threshold:
                logger.debug(f"Template confidence {confidence:.2f} >= {self.confidence_threshold}, using template")
                return EnhancedResponse(
                    content=template_response,
                    confidence=confidence,
                    pattern_type=pattern_type,
                    enhancement_used=False,
                    enhancement_reason="Template confidence sufficient"
                )
            
            # Privacy check - fallback to template for PII queries
            pii_result = await self.pii_detector.detect_and_anonymise(query)
            if not pii_result.anonymisation_applied and len(pii_result.entities_detected) == 0:
                privacy_check_passed = True
                safe_query = query
            else:
                logger.warning(f"PII detected in query, falling back to template")
                return EnhancedResponse(
                    content=template_response,
                    confidence=confidence,
                    pattern_type=pattern_type,
                    enhancement_used=False,
                    privacy_check_passed=False,
                    enhancement_reason="PII detected, template fallback"
                )
            
            # Apply LLM enhancement
            enhanced_content = await self._apply_llm_enhancement(
                safe_query, template_response, pattern_type, context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedResponse(
                content=enhanced_content,
                confidence=min(confidence + 0.2, 1.0),  # Boost confidence moderately
                pattern_type=pattern_type,
                enhancement_used=True,
                llm_processing_time=processing_time,
                privacy_check_passed=privacy_check_passed,
                original_template=template_response,
                enhancement_reason=f"Low template confidence: {confidence:.2f}"
            )
            
        except Exception as e:
            logger.error(f"LLM enhancement failed, falling back to template: {e}")
            return EnhancedResponse(
                content=template_response,
                confidence=confidence,
                pattern_type=pattern_type,
                enhancement_used=False,
                enhancement_reason=f"Enhancement failed: {str(e)}"
            )
    
    async def _apply_llm_enhancement(
        self, 
        query: str, 
        template_response: str, 
        pattern_type: ConversationalPattern,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Apply LLM enhancement while preserving Australian tone and style."""
        
        enhancement_prompt = self.enhancement_prompts.get(
            pattern_type, 
            self._get_default_enhancement_prompt()
        )
        
        # Build enhancement prompt
        prompt = enhancement_prompt.format(
            user_query=query,
            template_response=template_response,
            context_info=self._format_context(context) if context else "No additional context"
        )
        
        try:
            # Use existing LLM infrastructure
            response = await self.llm_manager.generate(
                prompt=prompt,
                max_tokens=200,  # Keep responses concise
                temperature=0.3  # Conservative temperature for consistency
            )
            
            enhanced_content = response.content.strip()
            
            # Validate enhancement maintains appropriate tone
            if self._validate_enhanced_response(enhanced_content, template_response):
                return enhanced_content
            else:
                logger.warning("Enhanced response failed validation, using template")
                return template_response
                
        except Exception as e:
            logger.error(f"LLM enhancement generation failed: {e}")
            return template_response
    
    def _validate_enhanced_response(self, enhanced: str, template: str) -> bool:
        """Validate that enhanced response maintains appropriate tone and length."""
        # Length check - shouldn't be dramatically longer
        if len(enhanced) > len(template) * 2:
            return False
            
        # Tone check - should maintain professional but friendly Australian tone
        inappropriate_terms = ['mate', 'bloody', 'crikey']  # Too casual for professional context
        for term in inappropriate_terms:
            if term.lower() in enhanced.lower():
                return False
                
        # Should contain appropriate Australian politeness markers
        appropriate_markers = ['thanks', 'please', 'happy to', 'glad to', 'certainly']
        has_appropriate_tone = any(marker in enhanced.lower() for marker in appropriate_markers)
        
        return has_appropriate_tone or len(enhanced) < 100  # Short responses get a pass
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for LLM prompt."""
        if not context:
            return "No additional context"
            
        formatted_parts = []
        for key, value in context.items():
            if isinstance(value, (str, int, float)):
                formatted_parts.append(f"{key}: {value}")
                
        return ", ".join(formatted_parts) if formatted_parts else "No additional context"
    
    def _get_greeting_enhancement_prompt(self) -> str:
        """Get enhancement prompt for greeting patterns."""
        return """You are an Australian government data analysis assistant. A user has greeted you, and you have a template response. Please enhance the response to be more welcoming and informative while maintaining a professional but friendly Australian tone.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. Warmly welcoming but professional
2. Briefly informative about your capabilities
3. Australian-friendly but appropriate for government context
4. No longer than 150 words

Enhanced Response:"""

    def _get_system_question_enhancement_prompt(self) -> str:
        """Get enhancement prompt for system question patterns."""
        return """You are an Australian government data analysis assistant. A user is asking about your capabilities or the system, and you have a template response. Please enhance it to be more informative and helpful while maintaining professionalism.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. More detailed about specific capabilities
2. Include relevant examples if appropriate
3. Maintain professional but approachable tone
4. Focus on survey and learning data analysis capabilities
5. No longer than 200 words

Enhanced Response:"""

    def _get_help_request_enhancement_prompt(self) -> str:
        """Get enhancement prompt for help request patterns."""
        return """You are an Australian government data analysis assistant. A user is asking for help, and you have a template response. Please enhance it to be more helpful and specific while maintaining a supportive tone.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. More specific about how you can help
2. Include concrete next steps or examples
3. Encouraging and supportive
4. Professional but warm
5. No longer than 180 words

Enhanced Response:"""

    def _get_politeness_enhancement_prompt(self) -> str:
        """Get enhancement prompt for politeness patterns."""
        return """You are an Australian government data analysis assistant. A user is being polite (thanking, saying goodbye, etc.), and you have a template response. Please enhance it to be appropriately reciprocal while maintaining professionalism.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. Appropriately reciprocal in politeness
2. Professional but genuine
3. Brief and to the point
4. Australian-friendly
5. No longer than 100 words

Enhanced Response:"""

    def _get_off_topic_enhancement_prompt(self) -> str:
        """Get enhancement prompt for off-topic patterns."""
        return """You are an Australian government data analysis assistant. A user has asked something off-topic, and you have a template response that redirects them. Please enhance it to be more understanding and helpful while maintaining focus.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. Understanding and polite about the off-topic nature
2. Gently redirect to your capabilities
3. Offer specific examples of what you can help with
4. Professional but not dismissive
5. No longer than 150 words

Enhanced Response:"""

    def _get_default_enhancement_prompt(self) -> str:
        """Get default enhancement prompt for unrecognized patterns."""
        return """You are an Australian government data analysis assistant. A user has made a query, and you have a template response. Please enhance it to be more helpful and informative while maintaining professionalism.

User Query: {user_query}
Template Response: {template_response}
Context: {context_info}

Please enhance the template response to be:
1. More helpful and informative
2. Professional but friendly
3. Appropriate for Australian government context
4. No longer than 175 words

Enhanced Response:"""
