# Query Routing & Classification Module

**Purpose**: Intelligent multi-stage query classification for optimal RAG processing  
**Implementation**: LLM-powered classification with rule-based fallbacks and Australian PII protection  
**Security Status**: Production-ready with comprehensive privacy controls

---

## Module Overview

The `src/rag/core/routing` module implements **sophisticated query classification** to determine optimal processing strategies for user queries. This module ensures accurate routing between SQL analysis, vector search, and hybrid processing while maintaining strict Australian data governance and mandatory PII protection throughout the classification process.

### Classification Architecture

#### Multi-Stage Classification Pipeline
- **Rule-Based Pre-Filter**: Fast pattern matching for obvious query types
- **LLM-Based Analysis**: Sophisticated natural language understanding with confidence scoring
- **Fallback Mechanisms**: Robust error handling with graceful degradation
- **Privacy Protection**: Mandatory PII anonymisation before LLM processing

#### Implementation Status
- **Production Ready**: Full async implementation with comprehensive error handling
- **Privacy Compliant**: Australian PII detection integrated at all processing stages
- **Performance Optimised**: Sub-second classification for most queries
- **Audit Ready**: Complete logging with privacy-safe audit trails

---

## Core Components

### `query_classifier.py` - Intelligent Query Classification Engine

**Primary Class**: `QueryClassifier`
- **Async Architecture**: Non-blocking processing with configurable timeouts
- **Multi-Provider Support**: Compatible with OpenAI, Anthropic, and Google LLMs
- **Confidence Scoring**: Quantitative assessment of classification accuracy
- **PII Protection**: Mandatory anonymisation before external LLM calls

**Key Methods**:
```python
async def classify_query(query: str, session_id: str) -> ClassificationResult
async def _rule_based_classification(query: str) -> Optional[ClassificationResult]
async def _llm_based_classification(query: str, session_id: str) -> ClassificationResult
async def _get_fallback_classification(query: str) -> ClassificationResult
```

### Classification Types & Routing Logic

#### Primary Classifications
- **SQL**: Queries requiring statistical analysis or numerical breakdowns
- **VECTOR**: Queries seeking qualitative feedback or user experiences  
- **HYBRID**: Complex queries requiring both statistical and qualitative insights
- **CLARIFICATION_NEEDED**: Ambiguous queries requiring user input

#### Confidence Levels
- **HIGH (0.8-1.0)**: Clear classification with strong confidence
- **MEDIUM (0.5-0.79)**: Reasonable classification with moderate confidence
- **LOW (0.0-0.49)**: Uncertain classification requiring user clarification

---

## Enhanced Data Governance Framework

### Australian Privacy Principles (APP) Compliance

#### APP 3 (Collection of Personal Information)
- **Purpose-Limited Collection**: Query text collected only for classification processing
- **Minimal Data Processing**: Only classification-relevant features extracted
- **Temporary Processing**: Query text not stored after classification completion

#### APP 6 (Use or Disclosure of Personal Information)
- **Internal Processing**: Classification used exclusively within RAG system boundaries
- **PII Anonymisation**: All personal data anonymised before LLM classification
- **No Secondary Use**: Classification data not used for profiling or analytics

#### APP 8 (Cross-border Disclosure of Personal Information)
- **Anonymised-Only Transmission**: Only PII-anonymised query text sent to LLM providers
- **Australian Data Sovereignty**: All personal data remains within Australian jurisdiction
- **Compliance Monitoring**: Complete audit trail of all cross-border interactions

#### APP 11 (Security of Personal Information)
- **Encrypted Transmission**: All LLM API communications use TLS encryption
- **Secure Session Management**: Temporary processing with automatic cleanup
- **Access Controls**: Classification limited to authorised RAG system components

### Privacy Protection Implementation

#### Mandatory PII Detection
- **Pre-Classification Scanning**: All queries scanned for Australian PII before processing
- **Automatic Anonymisation**: ABN, ACN, TFN, Medicare numbers automatically masked
- **Safe LLM Processing**: Only anonymised text transmitted to external LLM providers
- **Audit Compliance**: Complete logging of PII detection and anonymisation activities

#### Query Sanitisation Pipeline
1. **Input Validation**: Query length and format validation
2. **PII Detection**: Australian entity recognition and anonymisation
3. **Safe Classification**: Privacy-protected LLM analysis
4. **Result Validation**: Classification result security verification
5. **Audit Logging**: Privacy-safe activity recording

---

## Rule-Based Classification Patterns

### SQL Query Indicators
- **Statistical Keywords**: `count`, `how many`, `average`, `percentage`, `total`
- **Analytical Phrases**: `breakdown by`, `compare numbers`, `statistics for`
- **Temporal Analysis**: `over time`, `trends in`, `changes since`
- **Quantitative Terms**: `metrics`, `KPIs`, `performance indicators`

### Vector Search Indicators  
- **Feedback Keywords**: `what did people say`, `user feedback`, `comments about`
- **Experience Terms**: `experiences with`, `opinions on`, `satisfaction levels`
- **Qualitative Phrases**: `how do users feel`, `feedback themes`, `user sentiment`
- **Testimonial Requests**: `examples of`, `specific feedback`, `user stories`

### Hybrid Query Indicators
- **Combined Analysis**: `analyse satisfaction with numbers and feedback`
- **Comprehensive Requests**: `complete picture of`, `both data and opinions`
- **Multi-Modal Questions**: `statistics supported by user comments`
- **Comparative Analysis**: `compare performance with user experiences`

---

## Classification Confidence & Fallback Strategy

### Confidence Calculation Methodology

#### High Confidence (0.8-1.0)
- **Strong Pattern Match**: Multiple rule-based indicators present
- **LLM Agreement**: Rule-based and LLM classifications align
- **Clear Intent**: Unambiguous query with specific objectives
- **Historical Success**: Similar queries classified successfully

#### Medium Confidence (0.5-0.79)
- **Partial Indicators**: Some classification signals present
- **LLM Uncertainty**: Classification confidence below optimal threshold
- **Moderate Ambiguity**: Query could reasonably fit multiple categories
- **Context Dependent**: Classification requires additional information

#### Low Confidence (0.0-0.49)
- **Minimal Indicators**: Few or conflicting classification signals
- **High Ambiguity**: Query meaning unclear or overly broad
- **Classification Conflict**: Rule-based and LLM results disagree
- **Insufficient Context**: Requires user clarification for accurate routing

### Fallback Hierarchy

1. **Primary**: LLM-based classification with confidence scoring
2. **Secondary**: Rule-based classification using pattern matching
3. **Tertiary**: Default classification based on system configuration
4. **Final**: Route to clarification node for user input

---

## Integration & Usage

### RAG Agent Integration
The query classifier integrates seamlessly with the main RAG agent through:
- **Initialisation**: Automatic setup during agent initialisation
- **State Management**: Direct integration with LangGraph state flow
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Audit Integration**: Privacy-safe logging throughout classification process

### Performance Characteristics
- **Rule-Based Speed**: < 100ms for pattern-based classification
- **LLM Classification**: < 3 seconds for complex natural language analysis
- **Memory Efficiency**: Stateless processing with minimal resource usage
- **Concurrent Processing**: Multiple queries processed simultaneously

### Configuration Options
- **Classification Timeout**: Configurable LLM response timeout (default: 5 seconds)
- **Confidence Thresholds**: Adjustable confidence levels for routing decisions
- **Fallback Behaviour**: Customisable fallback classification strategies
- **Retry Logic**: Configurable retry attempts for LLM classification failures

---

## Testing & Quality Assurance

### Classification Accuracy Metrics
- **SQL Query Recognition**: Target ≥95% accuracy for statistical queries
- **Vector Search Detection**: Target ≥95% accuracy for feedback queries
- **Hybrid Identification**: Target ≥90% accuracy for complex multi-modal queries
- **Ambiguity Detection**: Target ≥85% accuracy for clarification-needed queries

### Privacy Compliance Testing
- **PII Detection Validation**: Comprehensive testing of Australian entity recognition
- **Anonymisation Verification**: Validation of complete PII removal before LLM processing
- **Audit Trail Testing**: Verification of privacy-safe logging throughout classification
- **Cross-Border Compliance**: Validation of data sovereignty controls

### Performance Benchmarks
- **Classification Latency**: Sub-second response for 95% of queries
- **Throughput**: Support for concurrent classification requests
- **Memory Usage**: Efficient processing with minimal resource consumption
- **Error Recovery**: Graceful handling of all failure scenarios

---

## Future Enhancements

### Planned Improvements
- **Adaptive Learning**: Classification accuracy improvement through usage patterns
- **Context Awareness**: Enhanced classification using conversation history
- **Domain Specialisation**: Government-specific query pattern recognition
- **Advanced Confidence**: Multi-factor confidence scoring with uncertainty quantification

### Scalability Considerations
- **Caching Strategy**: Classification result caching for improved performance
- **Load Balancing**: Distribution of classification requests across LLM providers
- **Rate Limiting**: Intelligent request throttling for API cost optimisation
- **Monitoring Integration**: Real-time classification performance monitoring

This query routing module forms the intelligence foundation of the RAG system, ensuring accurate query classification while maintaining the highest standards of Australian data governance and privacy protection.


**Last Updated**: 22 June 2025