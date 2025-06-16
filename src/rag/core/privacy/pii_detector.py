"""
Australian-Specific PII Detection and Anonymisation

This module implements mandatory PII detection and anonymisation using Microsoft Presidio
with Australian-specific patterns and entities. All text must pass through this module
before being sent to LLMs or used for embedding generation.

Security: Mandatory PII anonymisation before external processing
Privacy: Australian Privacy Principles (APP) compliance
Performance: Async operations with batch processing support
"""

import asyncio
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Use absolute imports to avoid pytest import issues
import sys
from pathlib import Path

# Add the rag module to path if not already there
rag_path = Path(__file__).parent.parent.parent
if str(rag_path) not in sys.path:
    sys.path.insert(0, str(rag_path))

# Import with fallback for testing
try:
    from utils.logging_utils import get_logger
    from config.settings import get_settings
    logger = get_logger(__name__)
except Exception:
    # Fallback for testing - create a simple logger that doesn't require config
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Mock get_settings for testing
    def get_settings():
        from types import SimpleNamespace
        return SimpleNamespace()


@dataclass
class PIIDetectionResult:
    """Result of PII detection and anonymisation process."""
    original_text: str
    anonymised_text: str
    entities_detected: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    processing_time: float
    anonymisation_applied: bool


class AustralianPIIDetector:
    """
    Australian-specific PII detection and anonymisation using Microsoft Presidio.
    
    Implements mandatory PII detection for free-text fields before any LLM processing
    or embedding generation. Includes Australian-specific patterns and entities.
    """
    
    def __init__(self):
        """Initialise the PII detector with Australian-specific patterns."""
        print("DEBUG: Creating AustralianPIIDetector instance")
        print("DEBUG: About to call get_settings()")
        try:
            self.settings = get_settings()
            print("DEBUG: get_settings() completed successfully")
        except Exception as e:
            print(f"DEBUG: get_settings() failed: {e}, using mock settings")
            from types import SimpleNamespace
            self.settings = SimpleNamespace()
        
        self.analyzer = None
        self.anonymizer = None
        self._initialised = False
        
        # Australian-specific PII patterns
        self.australian_patterns = {
            'ABN': r'ABN\s*:?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})',
            'ACN': r'ACN\s*:?\s*(\d{3}\s?\d{3}\s?\d{3})',
            'Medicare': r'Medicare\s*:?\s*(\d{4}\s?\d{5}\s?\d{1})',
            'Tax_File_Number': r'TFN\s*:?\s*(\d{3}\s?\d{3}\s?\d{3})',
            'Australian_Business_Number': r'(?:ABN|Australian Business Number)\s*:?\s*(\d{11}|\d{2}\s\d{3}\s\d{3}\s\d{3})',
            'Australian_Company_Number': r'(?:ACN|Australian Company Number)\s*:?\s*(\d{9}|\d{3}\s\d{3}\s\d{3})',
            'Australian_Phone': r'(?:\+61|0)[2-478](?:[ -]?\d){8}',
            'Australian_Postcode': r'\b(?:0[289][0-9]{2}|[1-9][0-9]{3})\b',
            'Australian_State': r'\b(?:NSW|VIC|QLD|WA|SA|TAS|NT|ACT)\b',
        }
        
        # Confidence thresholds for different entity types
        self.confidence_thresholds = {
            'PERSON': 0.7,
            'EMAIL_ADDRESS': 0.8,
            'PHONE_NUMBER': 0.7,
            'LOCATION': 0.6,
            'ORGANIZATION': 0.6,
            'DATE_TIME': 0.5,
            'IP_ADDRESS': 0.9,
            'CREDIT_CARD': 0.8,
            'IBAN_CODE': 0.8,
            'US_SSN': 0.8,
            'AU_ABN': 0.8,
            'AU_ACN': 0.8,
            'AU_TFN': 0.9,
            'AU_MEDICARE': 0.8,
        }
        
        # Anonymisation replacements
        self.replacement_tokens = {
            'PERSON': '[PERSON]',
            'EMAIL_ADDRESS': '[EMAIL]',
            'PHONE_NUMBER': '[PHONE]',
            'LOCATION': '[LOCATION]',
            'ORGANIZATION': '[ORGANISATION]',
            'DATE_TIME': '[DATE]',
            'IP_ADDRESS': '[IP_ADDRESS]',
            'CREDIT_CARD': '[CREDIT_CARD]',
            'IBAN_CODE': '[BANK_ACCOUNT]',
            'US_SSN': '[SSN]',
            'AU_ABN': '[ABN]',
            'AU_ACN': '[ACN]',
            'AU_TFN': '[TFN]',
            'AU_MEDICARE': '[MEDICARE]',
        }
    
    async def initialise(self) -> None:
        """
        Initialise Presidio analyzer and anonymizer.
        
        Raises:
            RuntimeError: If Presidio is not available or initialisation fails
        """
        if self._initialised:
            return
            
        try:
            # Configure NLP engine for Australian English
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            
            nlp_engine_provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = nlp_engine_provider.create_engine()
            
            # Initialise analyzer with Australian patterns
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            
            # Add Australian-specific recognizers
            await self._add_australian_recognizers()
            
            # Initialise anonymizer
            self.anonymizer = AnonymizerEngine()
            
            self._initialised = True
            logger.info("Australian PII detector initialised successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialise PII detector: {e}")
            raise RuntimeError(f"PII detector initialisation failed: {e}")
    
    async def _add_australian_recognizers(self) -> None:
        """Add Australian-specific PII recognizers to Presidio."""
        from presidio_analyzer import PatternRecognizer, Pattern
        
        # Australian Business Number (ABN) recognizer
        abn_pattern = Pattern(
            name="abn_pattern",
            regex=r'\b(?:ABN\s*:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b',
            score=0.9
        )
        abn_recognizer = PatternRecognizer(
            supported_entity="AU_ABN",
            patterns=[abn_pattern],
            name="australian_abn_recognizer"
        )
        self.analyzer.registry.add_recognizer(abn_recognizer)
        
        # Australian Company Number (ACN) recognizer - more flexible
        acn_pattern = Pattern(
            name="acn_pattern",
            regex=r'\bACN\s*(?:is\s*|:?\s*)(\d{3}\s?\d{3}\s?\d{3})\b',
            score=1.0
        )
        acn_recognizer = PatternRecognizer(
            supported_entity="AU_ACN",
            patterns=[acn_pattern],
            name="australian_acn_recognizer"
        )
        self.analyzer.registry.add_recognizer(acn_recognizer)
        
        # Australian Tax File Number (TFN) recognizer - more flexible
        tfn_pattern = Pattern(
            name="tfn_pattern",
            regex=r'\bTFN\s*(?:is\s*|:?\s*)(\d{3}\s?\d{3}\s?\d{3})\b',
            score=1.0
        )
        tfn_recognizer = PatternRecognizer(
            supported_entity="AU_TFN", 
            patterns=[tfn_pattern],
            name="australian_tfn_recognizer"
        )
        self.analyzer.registry.add_recognizer(tfn_recognizer)
        
        # Australian Medicare Number recognizer - match only the number
        medicare_pattern = Pattern(
            name="medicare_pattern", 
            regex=r'(?<=Medicare\s*(?:No\.?\s*|Number\.?\s*):?\s*)(\d{4}\s?\d{5}\s?\d{1})\b',
            score=1.0
        )
        medicare_recognizer = PatternRecognizer(
            supported_entity="AU_MEDICARE",
            patterns=[medicare_pattern],
            name="australian_medicare_recognizer"
        )
        self.analyzer.registry.add_recognizer(medicare_recognizer)
        
        # Debug: Log the recognizers that were added
        all_recognizers = [r.name for r in self.analyzer.registry.recognizers]
        logger.info(f"All recognizers in registry: {all_recognizers}")
        
        # Test pattern directly
        import re
        test_text = "Company ABN: 53 004 085 616"
        pattern = r'\b(?:ABN\s*:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b'
        matches = re.findall(pattern, test_text, re.IGNORECASE)
        logger.info(f"Direct regex test - Pattern: {pattern}")
        logger.info(f"Direct regex test - Text: {test_text}")  
        logger.info(f"Direct regex test - Matches: {matches}")
        
        logger.info("Australian-specific PII recognizers loaded")
    
    async def detect_and_anonymise(self, text: str) -> PIIDetectionResult:
        """
        Detect and anonymise PII in text using Australian-specific patterns.
        
        Args:
            text: Input text to process
            
        Returns:
            PIIDetectionResult: Complete results of PII detection and anonymisation
            
        Raises:
            RuntimeError: If PII detection fails
        """
        if not self._initialised:
            await self.initialise()
            
        start_time = datetime.now()
        
        try:
            if self.analyzer and self.anonymizer:
                result = await self._presidio_detection(text)
            else:
                result = await self._fallback_detection(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            logger.info(
                f"PII detection completed: {len(result.entities_detected)} entities found, "
                f"anonymisation={'applied' if result.anonymisation_applied else 'not needed'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            raise RuntimeError(f"PII detection failed: {e}")
    
    async def _presidio_detection(self, text: str) -> PIIDetectionResult:
        """Perform PII detection using Microsoft Presidio."""
        # Analyze text for PII (including Australian-specific entities)
        analyzer_results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=[
                'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'LOCATION',
                'DATE_TIME', 'IP_ADDRESS', 'CREDIT_CARD',
                'AU_ABN', 'AU_ACN', 'AU_TFN', 'AU_MEDICARE'  # Australian entities
            ]
        )
        
        # Filter results by confidence threshold
        filtered_results = [
            result for result in analyzer_results
            if result.score >= self.confidence_thresholds.get(result.entity_type, 0.5)
        ]
        
        # Anonymise if PII detected
        if filtered_results:
            from presidio_anonymizer.entities import OperatorConfig
            
            operators = {
                entity_type: OperatorConfig(operator_name="replace", params={"new_value": replacement})
                for entity_type, replacement in self.replacement_tokens.items()
            }
            
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=filtered_results,
                operators=operators
            )
            anonymised_text = anonymized_result.text
            anonymisation_applied = True
        else:
            anonymised_text = text
            anonymisation_applied = False
        
        # Extract entity information
        entities_detected = [
            {
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end]
            }
            for result in filtered_results
        ]
        
        # Calculate confidence scores by entity type
        confidence_scores = {}
        for result in filtered_results:
            entity_type = result.entity_type
            if entity_type not in confidence_scores:
                confidence_scores[entity_type] = []
            confidence_scores[entity_type].append(result.score)
        
        # Average confidence scores
        confidence_scores = {
            entity_type: sum(scores) / len(scores)
            for entity_type, scores in confidence_scores.items()
        }
        
        return PIIDetectionResult(
            original_text=text,
            anonymised_text=anonymised_text,
            entities_detected=entities_detected,
            confidence_scores=confidence_scores,
            processing_time=0.0,  # Will be set by caller
            anonymisation_applied=anonymisation_applied
        )
    
    async def _fallback_detection(self, text: str) -> PIIDetectionResult:
        """Fallback PII detection using regex patterns when Presidio is unavailable."""
        logger.warning("Using fallback PII detection - install Presidio for full functionality")
        
        entities_detected = []
        anonymised_text = text
        confidence_scores = {}
        
        # Apply Australian-specific regex patterns
        for entity_type, pattern in self.australian_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities_detected.append({
                    'entity_type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.8,  # Default confidence for regex matches
                    'text': match.group()
                })
                
                # Replace with anonymisation token
                replacement = self.replacement_tokens.get(entity_type, f'[{entity_type}]')
                anonymised_text = anonymised_text.replace(match.group(), replacement)
        
        # Basic email detection
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.finditer(email_pattern, text)
        for match in email_matches:
            entities_detected.append({
                'entity_type': 'EMAIL_ADDRESS',
                'start': match.start(),
                'end': match.end(),
                'score': 0.9,
                'text': match.group()
            })
            anonymised_text = anonymised_text.replace(match.group(), '[EMAIL]')
        
        # Basic phone number detection (Australian format)
        phone_pattern = r'(?:\+61|0)[2-478](?:[ -]?\d){8}'
        phone_matches = re.finditer(phone_pattern, text)
        for match in phone_matches:
            entities_detected.append({
                'entity_type': 'AU_PHONE',
                'start': match.start(),
                'end': match.end(),
                'score': 0.8,
                'text': match.group()
            })
            anonymised_text = anonymised_text.replace(match.group(), '[PHONE]')
        
        anonymisation_applied = len(entities_detected) > 0
        
        return PIIDetectionResult(
            original_text=text,
            anonymised_text=anonymised_text,
            entities_detected=entities_detected,
            confidence_scores=confidence_scores,
            processing_time=0.0,  # Will be set by caller
            anonymisation_applied=anonymisation_applied
        )
    
    async def batch_process(self, texts: List[str]) -> List[PIIDetectionResult]:
        """
        Process multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List[PIIDetectionResult]: Results for each input text
        """
        if not self._initialised:
            await self.initialise()
        
        tasks = [self.detect_and_anonymise(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the batch
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for text {i}: {result}")
                # Return safe fallback result
                processed_results.append(PIIDetectionResult(
                    original_text=texts[i],
                    anonymised_text="[TEXT_PROCESSING_ERROR]",
                    entities_detected=[],
                    confidence_scores={},
                    processing_time=0.0,
                    anonymisation_applied=True
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.analyzer:
            # Presidio doesn't require explicit cleanup
            pass
        logger.info("PII detector closed")


# Global instance for singleton pattern
_pii_detector_instance: Optional[AustralianPIIDetector] = None


async def get_pii_detector() -> AustralianPIIDetector:
    """
    Get the global PII detector instance.
    
    Returns:
        AustralianPIIDetector: Initialised PII detector
    """
    global _pii_detector_instance
    
    if _pii_detector_instance is None:
        _pii_detector_instance = AustralianPIIDetector()
        await _pii_detector_instance.initialise()
    
    return _pii_detector_instance


# Convenience function for direct usage
async def detect_and_anonymise_text(text: str) -> PIIDetectionResult:
    """
    Convenience function to detect and anonymise PII in text.
    
    Args:
        text: Input text to process
        
    Returns:
        PIIDetectionResult: Complete results of PII detection and anonymisation
    """
    detector = await get_pii_detector()
    return await detector.detect_and_anonymise(text)
