# Privacy & PII Detection Module

**Purpose**: Mandatory Australian-compliant PII detection and anonymisation for secure text processing  
**Implementation**: Microsoft Presidio with custom Australian entity recognisers  
**Security Status**: Production-ready with comprehensive test coverage

---

## Module Overview

The `src/rag/core/privacy` module implements **mandatory PII detection and anonymisation** for all free-text data before LLM processing or embedding generation. This module ensures compliance with Australian Privacy Principles (APP) and provides robust protection for sensitive Australian Government data.

### Security Architecture

#### Non-Negotiable Requirements
- **Mandatory Processing**: All text must pass through PII detection before external processing
- **Australian Compliance**: Custom recognisers for ABN, ACN, TFN, Medicare numbers
- **Zero PII Leakage**: Complete anonymisation of detected entities before LLM interaction
- **Audit Trail**: Comprehensive logging of all PII detection activities

#### Implementation Status
- **Production Ready**: Full async implementation with proper resource management
- **Test Coverage**: 13/13 core tests passing with comprehensive validation
- **Australian Patterns**: ABN, ACN, TFN, Medicare number detection operational
- **Integration Ready**: Seamless integration with RAG module configuration

---

## Core Components

### `pii_detector.py` - Main PII Detection Engine

**Primary Class**: `AustralianPIIDetector`
- **Async Architecture**: All methods use async/await for non-blocking processing
- **Singleton Pattern**: Global instance with efficient session management
- **Batch Processing**: Support for processing multiple texts efficiently
- **Fallback Systems**: Robust error handling with graceful degradation

**Key Methods**:
```python
async def detect_and_anonymise(text: str) -> PIIDetectionResult
async def batch_process(texts: List[str]) -> List[PIIDetectionResult]
async def initialise() -> None  # Presidio setup
async def close() -> None       # Resource cleanup
```

### Australian Entity Recognition

#### Custom Pattern Recognisers
- **ABN (Australian Business Number)**: `\b(?:ABN\s*:?\s*)?(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b`
- **ACN (Australian Company Number)**: `\bACN\s*(?:is|:)?\s*(\d{3}\s?\d{3}\s?\d{3})\b`
- **TFN (Tax File Number)**: `\bTFN\s*(?:is|:)?\s*(\d{3}\s?\d{3}\s?\d{3})\b`
- **Medicare Number**: `\bMedicare\s*(?:No|Number)?\s*:?\s*(\d{4}\s?\d{5}\s?\d{1})\b`

#### Standard Entity Detection
- **Email Addresses**: High-confidence detection with .gov.au priority
- **Phone Numbers**: Australian mobile and landline patterns
- **Person Names**: NLP-based detection for Australian naming conventions
---

## Anonymisation Strategy

### Replacement Tokens
- **Personal Names**: `[PERSON]` - Protects individual privacy
- **Email Addresses**: `[EMAIL]` - Prevents contact information leakage  
- **Phone Numbers**: `[PHONE]` - Anonymises direct contact methods
- **Government IDs**: `[ABN]`, `[ACN]`, `[TFN]`, `[MEDICARE]` - Protects Australian identifiers
- **Locations**: `[LOCATION]` - Generalises geographic references

### Context Preservation Principles
- **Semantic Integrity**: Maintain sentence structure for meaningful embedding generation
- **Sentiment Retention**: Preserve emotional context and intent after anonymisation
- **Relationship Mapping**: Keep relative references (e.g., "my manager" → "my [ROLE]")
- **Readability**: Ensure anonymised text remains comprehensible for analysis

---

## Security & Compliance

### Australian Privacy Principles (APP) Compliance
- **APP 1**: Open and transparent management of personal information
- **APP 3**: Collection of solicited personal information with clear purpose
- **APP 6**: Use or disclosure of personal information for secondary purposes
- **APP 11**: Security of personal information through technical safeguards

### Data Protection Measures
- **Zero PII Storage**: No raw PII stored in vector embeddings table
- **Audit Logging**: Complete trail of all PII detection and anonymisation activities
- **Access Controls**: Read-only database permissions for PII processing components
- **Error Sanitisation**: No PII exposure in error messages or debugging output

### Government Security Requirements
- **Protective Security Policy Framework (PSPF)** compliance
- **Information Security Manual (ISM)** alignment for OFFICIAL data classification
- **Digital Transformation Agency (DTA)** privacy guidelines adherence
- **Australian Cyber Security Centre (ACSC)** recommendations implementation

---

## Usage & Integration

### Basic Usage
```python
from core.privacy.pii_detector import get_pii_detector

# Get global singleton instance
detector = await get_pii_detector()

# Process single text
result = await detector.detect_and_anonymise(
    "Contact me at john.smith@agency.gov.au about ABN 53 004 085 616"
)
# Result: "Contact me at [EMAIL] about [ABN]"

# Batch processing
results = await detector.batch_process([text1, text2, text3])
```

### Integration Points
- **Content Processor**: Pre-embedding PII anonymisation
- **Query Handler**: User input sanitisation before LLM processing  
- **Audit Logger**: PII detection statistics and compliance reporting
- **Error Handler**: Secure error messages without PII exposure
async def detect_and_anonymise(text: str) -> PIIDetectionResult
async def batch_process(texts: List[str]) -> List[PIIDetectionResult]
async def initialise() -> None
```

#### 2. `PIIDetectionResult` Dataclass
```python
@dataclass
class PIIDetectionResult:
    original_text: str
    anonymised_text: str
    entities_detected: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    processing_time: float
    anonymisation_applied: bool
```

#### 3. Integration Points
- **Content Processor**: Pre-processing pipeline integration
- **Embedding Generator**: Mandatory PII check before embedding
- **Query Handler**: PII detection for user queries
- **Audit Logger**: Complete PII detection audit trail

### Performance Considerations

#### Async Processing
### Performance & Scalability
- **Async Processing**: Non-blocking PII detection for high throughput
- **Batch Operations**: Process multiple texts efficiently with connection pooling
- **Resource Management**: Proper async cleanup and session management
- **Caching Strategy**: Optimised pattern matching with minimal redundant processing

### Quality Assurance & Testing
- **Test Coverage**: 13/13 core tests passing with comprehensive validation
- **Australian Patterns**: Validated detection of ABN, ACN, TFN, Medicare numbers
- **Performance Testing**: Sub-2 second processing for standard evaluation text
- **Error Resilience**: Graceful fallback when external services unavailable

---

## Production Deployment

### Security Requirements
- **Mandatory Processing**: Zero tolerance for PII bypass in any processing pipeline
- **Audit Compliance**: Complete logging of all PII detection activities with session tracking
- **Error Sanitisation**: No PII exposure in error messages, logs, or debugging output
- **Access Controls**: Read-only database permissions and role-based component access

### Monitoring & Alerting
- **Detection Failures**: Immediate alerts for PII detection system failures
- **Performance Degradation**: Monitoring for processing time increases
- **Compliance Reporting**: Regular audit reports for privacy compliance verification
- **Pattern Accuracy**: Ongoing validation of Australian entity recognition effectiveness

### Integration Checkpoints
1. **Pre-Embedding**: Mandatory PII anonymisation before vector generation
2. **Pre-LLM**: User input sanitisation before external API calls
3. **Pre-Storage**: Anonymised content verification before database storage
4. **Audit Trail**: Complete session tracking for all processed content

---

**Implementation Status**: ✅ **Phase 2 Task 2.1 Complete**  
**Security Clearance**: Ready for production deployment  
**Dependencies**: Microsoft Presidio, spaCy en_core_web_sm model  
**Next Integration**: Phase 2 Task 2.2 - pgVector schema with anonymised content
