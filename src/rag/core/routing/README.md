# Query Routing & Classification Module

**Purpose**: Intelligent multi-stage query classification for optimal RAG processing  
**Implementation**: Modular LLM-powered classification with sophisticated fallback mechanisms and Australian PII protection  
**Security Status**: Production-ready with comprehensive privacy controls and resilience patterns

---

## Module Overview

The `src/rag/core/routing` module implements **sophisticated modular query classification** to determine optimal processing strategies for user queries. This refactored module ensures accurate routing between SQL analysis, vector search, and hybrid processing while maintaining strict Australian data governance and mandatory PII protection throughout the classification process.

### Modular Classification Architecture

#### Multi-Stage Classification Pipeline
- **Rule-Based Pre-Filter**: Fast APS-specific pattern matching with weighted confidence scoring
- **LLM-Based Analysis**: Sophisticated natural language understanding with structured prompts
- **Confidence Calibration**: Dynamic confidence adjustment based on query complexity and historical accuracy
- **Circuit Breaker Protection**: Resilience patterns with exponential backoff and graceful degradation
- **Privacy Protection**: Mandatory PII anonymisation before LLM processing

#### Implementation Status
- **Production Ready**: Fully modular async implementation with comprehensive error handling
- **Privacy Compliant**: Australian PII detection integrated at all processing stages
- **Performance Optimised**: Sub-second classification with intelligent caching and fallback
- **Audit Ready**: Complete logging with privacy-safe audit trails and metrics collection

---

## Core Components

### Modular System Architecture

The routing system has been **refactored into 8 specialized modules** for enhanced maintainability, testability, and extensibility:

#### 1. `query_classifier.py` - Main Orchestrator & Entry Point

**Primary Class**: `QueryClassifier`
- **Async Architecture**: Non-blocking processing with configurable timeouts
- **Modular Integration**: Coordinates between all specialized components
- **Multi-Provider Support**: Compatible with OpenAI, Anthropic, and Google LLMs
- **PII Protection**: Mandatory anonymisation before external LLM calls

**Key Methods**:
```python
async def classify_query(query: str, session_id: str, anonymize_query: bool) -> ClassificationResult
async def initialize() -> None
def get_classification_stats() -> Dict[str, Any]
def get_fallback_metrics() -> Dict[str, Any]
```

#### 2. `pattern_matcher.py` - APS-Specific Rule-Based Classification

**Primary Class**: `PatternMatcher`
- **Weighted Regex Patterns**: Australian Public Service domain-specific patterns
- **Fast Path Classification**: Sub-100ms processing for obvious queries
- **Confidence Scoring**: Pattern strength analysis with reliability metrics
- **Usage Statistics**: Pattern match tracking and optimization

**Key Methods**:
```python
def classify_query(query: str) -> Optional[ClassificationResult]
def get_pattern_stats() -> Dict[str, int]
```

#### 3. `llm_classifier.py` - LLM-Based Semantic Classification

**Primary Class**: `LLMClassifier`
- **Structured Prompts**: Sophisticated prompt engineering for APS domain
- **Response Parsing**: Robust LLM response validation and extraction
- **Error Handling**: Comprehensive failure recovery and fallback
- **Performance Tracking**: Processing time and success rate monitoring

**Key Methods**:
```python
async def classify_query(query: str) -> ClassificationResult
async def initialize() -> None
```

#### 4. `confidence_calibrator.py` - Dynamic Confidence System

**Primary Class**: `ConfidenceCalibrator`
- **Multi-Dimensional Analysis**: Query complexity, pattern strength, and historical accuracy
- **Adaptive Thresholds**: Dynamic confidence adjustment based on system performance
- **Learning System**: Classification outcome tracking for continuous improvement
- **APS Domain Awareness**: Australian Public Service terminology recognition

**Key Methods**:
```python
def calibrate_confidence(raw_confidence: str, classification: str, query: str) -> ConfidenceCalibrationResult
def record_classification_outcome(classification: str, was_correct: bool, confidence_score: float) -> None
def get_calibration_stats() -> Dict[str, Any]
```

#### 5. `circuit_breaker.py` - Resilience & Fallback Protection

**Primary Classes**: `CircuitBreaker`, `RetryConfig`, `FallbackMetrics`
- **Circuit Breaker Pattern**: Protection against LLM service failures
- **Exponential Backoff**: Intelligent retry logic with jitter
- **Metrics Collection**: Real-time performance and failure tracking
- **Graceful Degradation**: Automatic fallback to rule-based classification

**Key Methods**:
```python
def can_execute() -> bool
def record_success() -> None
def record_failure() -> None
def get_delay(attempt: int) -> float
```

#### 6. `data_structures.py` - Type Definitions & Data Classes

**Core Data Structures**:
- `ClassificationResult`: Complete classification outcome with metadata
- `QueryComplexityAnalysis`: Multi-dimensional query complexity metrics
- `ConfidenceCalibrationResult`: Detailed confidence adjustment information
- `ClassificationStatistics`: System performance and accuracy tracking

#### 7. `aps_patterns.py` - Domain-Specific Pattern Library

**Primary Classes**: `APSPatterns`, `APSPatternWeights`
- **APS Terminology**: Australian Public Service specific patterns
- **Weighted Classification**: High/medium/low confidence pattern matching
- **Executive Levels**: EL1, EL2, APS Level 1-6 recognition
- **Training Context**: Learning analytics and professional development patterns

#### 8. `confidence_calibration.py` - Advanced Calibration Algorithms

**Core Functions**:
- `analyze_query_complexity()`: Structural and semantic complexity analysis
- `calibrate_confidence_score()`: Multi-factor confidence adjustment
- Domain knowledge integration for APS-specific confidence scoring

### Enhanced Classification Types & Routing Logic

#### Primary Classifications
- **SQL**: Queries requiring statistical analysis, aggregations, or structured data retrieval
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
- **Modular Privacy**: Each component implements privacy-by-design principles

#### APP 6 (Use or Disclosure of Personal Information)
- **Internal Processing**: Classification used exclusively within RAG system boundaries
- **PII Anonymisation**: All personal data anonymised before LLM classification
- **No Secondary Use**: Classification data not used for profiling or analytics
- **Component Isolation**: Privacy controls enforced at each modular boundary

#### APP 8 (Cross-border Disclosure of Personal Information)
- **Anonymised-Only Transmission**: Only PII-anonymised query text sent to LLM providers
- **Australian Data Sovereignty**: All personal data remains within Australian jurisdiction
- **Compliance Monitoring**: Complete audit trail of all cross-border interactions
- **Circuit Breaker Protection**: Automatic blocking of failed privacy checks

#### APP 11 (Security of Personal Information)
- **Encrypted Transmission**: All LLM API communications use TLS encryption
- **Secure Session Management**: Temporary processing with automatic cleanup
- **Access Controls**: Classification limited to authorised RAG system components
- **Resilience Patterns**: Circuit breaker protection against data exposure risks

### Privacy Protection Implementation

#### Mandatory PII Detection (Integrated Across All Modules)
- **Pre-Classification Scanning**: All queries scanned for Australian PII before processing
- **Automatic Anonymisation**: ABN, ACN, TFN, Medicare numbers automatically masked
- **Safe LLM Processing**: Only anonymised text transmitted to external LLM providers
- **Audit Compliance**: Complete logging of PII detection and anonymisation activities

#### Enhanced Query Sanitisation Pipeline
1. **Input Validation**: Query length and format validation (QueryClassifier)
2. **PII Detection**: Australian entity recognition and anonymisation (QueryClassifier)
3. **Pattern Pre-Filter**: Privacy-safe rule-based classification (PatternMatcher)
4. **Confidence Calibration**: Privacy-aware confidence adjustment (ConfidenceCalibrator)
5. **Safe LLM Classification**: Privacy-protected LLM analysis (LLMClassifier)
6. **Circuit Protection**: Failure isolation and recovery (CircuitBreaker)
7. **Result Validation**: Classification result security verification (QueryClassifier)
8. **Audit Logging**: Privacy-safe activity recording (All Components)

---

## Modular Pattern Recognition & Classification Logic

### Enhanced APS-Specific Patterns (APSPatterns)

#### SQL Query Indicators with Weighted Confidence
- **High-Weight Patterns**: `Level 6 users`, `Executive Level 1`, `completion rate by agency`
- **Statistical Keywords**: `count`, `how many`, `average`, `percentage`, `total`
- **Analytical Phrases**: `breakdown by`, `compare numbers`, `statistics for`
- **Temporal Analysis**: `over time`, `trends in`, `changes since`
- **APS Terminology**: `EL1`, `EL2`, `APS Level 4-6`, `department`, `portfolio`

#### Vector Search Indicators with Context Awareness
- **High-Weight Patterns**: `participant feedback`, `delegate experiences`, `course quality`
- **Feedback Keywords**: `what did people say`, `user feedback`, `comments about`
- **Experience Terms**: `experiences with`, `opinions on`, `satisfaction levels`
- **Qualitative Phrases**: `how do users feel`, `feedback themes`, `user sentiment`
- **APS Context**: `facilitator effectiveness`, `virtual learning experience`, `accessibility concerns`

#### Hybrid Query Indicators with Complexity Recognition  
- **High-Weight Patterns**: `satisfaction trends with supporting feedback`, `ROI analysis with participant comments`
- **Combined Analysis**: `analyse satisfaction with numbers and feedback`
- **Comprehensive Requests**: `complete picture of`, `both data and opinions`
- **Multi-Modal Questions**: `statistics supported by user comments`
- **APS Integration**: `agency performance with stakeholder feedback`, `cost-benefit with user experience`

---

## Advanced Confidence & Fallback Strategy

### Multi-Dimensional Confidence Calculation (ConfidenceCalibrator)

#### High Confidence (0.8-1.0)
- **Strong Pattern Match**: Multiple high-weight APS patterns present
- **LLM Agreement**: Rule-based and LLM classifications align with high certainty
- **Low Complexity**: Clear query structure with minimal ambiguity indicators
- **Historical Success**: Similar queries classified successfully with good accuracy
- **Domain Specificity**: High APS terminology density

#### Medium Confidence (0.5-0.79)
- **Partial Indicators**: Some classification signals present with medium weights
- **LLM Uncertainty**: Classification confidence below optimal threshold
- **Moderate Complexity**: Query complexity within acceptable range
- **Context Dependent**: Classification requires additional contextual information
- **Mixed Signals**: Some conflicting pattern matches resolved through calibration

#### Low Confidence (0.0-0.49)
- **Minimal Indicators**: Few or conflicting classification signals
- **High Complexity**: Complex query structure with multiple ambiguity markers
- **Classification Conflict**: Rule-based and LLM results disagree significantly
- **Insufficient Context**: Requires user clarification for accurate routing
- **Poor Historical Performance**: Similar queries have shown low accuracy

### Enhanced Fallback Hierarchy with Resilience Patterns

#### Circuit Breaker Integration
1. **Primary**: LLM-based classification with circuit breaker protection
2. **Circuit Open**: Automatic bypass to rule-based classification
3. **Half-Open Testing**: Gradual LLM service recovery with limited calls
4. **Metrics-Driven**: Real-time performance monitoring and threshold adjustment

#### Sophisticated Retry Logic
1. **Exponential Backoff**: Intelligent delay calculation with jitter
2. **Failure Classification**: Different retry strategies for different error types
3. **Resource Protection**: Automatic throttling to prevent cascade failures
4. **Recovery Monitoring**: Success rate tracking for adaptive behavior

#### Enhanced Fallback Classification
1. **Weighted Pattern Analysis**: Multi-dimensional keyword scoring
2. **Query Complexity Assessment**: Structural and semantic complexity analysis
3. **Contextual Pattern Recognition**: APS domain-specific fallback logic
4. **Confidence Degradation**: Appropriate confidence reduction for fallback results

---

## Integration & Usage

### Modular RAG Agent Integration
The refactored query classifier integrates seamlessly with the main RAG agent through:
- **Factory Pattern**: `create_query_classifier()` for easy initialization
- **Dependency Injection**: Modular components can be individually configured or replaced
- **State Management**: Direct integration with LangGraph state flow
- **Error Isolation**: Component-level error handling with graceful degradation
- **Metrics Collection**: Comprehensive monitoring across all modules

### Enhanced Performance Characteristics
- **Rule-Based Speed**: < 50ms for pattern-based classification with weighted scoring
- **LLM Classification**: < 2 seconds for complex analysis with circuit protection
- **Memory Efficiency**: Modular design with optimized resource usage
- **Concurrent Processing**: Thread-safe components support parallel query processing
- **Adaptive Performance**: Dynamic optimization based on usage patterns

### Advanced Configuration Options
- **Component-Level Configuration**: Individual module settings and thresholds
- **Circuit Breaker Tuning**: Customizable failure thresholds and recovery timeouts
- **Confidence Calibration**: Adjustable weights for different calibration factors
- **Pattern Weighting**: Configurable pattern importance and confidence mapping
- **Retry Strategies**: Customizable exponential backoff and jitter parameters

---

## Testing & Quality Assurance

### Modular Testing Strategy
- **Component Isolation**: Individual module testing with comprehensive unit tests
- **Integration Testing**: End-to-end workflow validation across all components
- **Mock-Based Testing**: LLM-independent testing with configurable mock responses
- **Circuit Breaker Testing**: Failure scenario validation and recovery testing
- **Performance Testing**: Load testing and latency validation for each component

### Enhanced Classification Accuracy Metrics
- **SQL Query Recognition**: Target ≥98% accuracy for statistical queries (improved with APS patterns)
- **Vector Search Detection**: Target ≥97% accuracy for feedback queries (enhanced context awareness)
- **Hybrid Identification**: Target ≥93% accuracy for complex multi-modal queries (better complexity analysis)
- **Ambiguity Detection**: Target ≥90% accuracy for clarification-needed queries (improved confidence calibration)

### Comprehensive Privacy Compliance Testing
- **Modular PII Detection**: Validation across all component boundaries
- **Anonymisation Verification**: Complete PII removal validation before LLM processing
- **Circuit Breaker Privacy**: Failure scenario privacy protection testing
- **Audit Trail Testing**: Privacy-safe logging verification across all modules
- **Cross-Border Compliance**: Enhanced data sovereignty controls validation

### Advanced Performance Benchmarks
- **Component Latency**: Sub-component response time optimization
- **Circuit Breaker Efficiency**: Failure detection and recovery time validation
- **Confidence Calibration Speed**: Real-time calibration performance testing
- **Memory Usage**: Modular memory efficiency and resource optimization
- **Throughput**: Concurrent classification capacity with circuit protection

---

## Future Enhancements

### Planned Modular Improvements
- **Machine Learning Integration**: Pattern learning and optimization within PatternMatcher
- **Advanced Calibration**: Enhanced confidence scoring with uncertainty quantification
- **Contextual Memory**: Classification history integration for improved accuracy
- **Dynamic Pattern Updates**: Real-time pattern weight adjustment based on performance
- **Multi-Model Ensemble**: Multiple LLM provider integration with consensus classification

### Scalability & Resilience Enhancements
- **Distributed Circuit Breakers**: Cross-instance failure coordination
- **Advanced Caching**: Classification result caching with intelligent invalidation
- **Load Balancing**: Multi-provider LLM distribution with health checking
- **Predictive Scaling**: Usage pattern-based resource optimization
- **Real-Time Monitoring**: Enhanced metrics collection and alerting

This modular query routing system provides a robust, maintainable, and extensible foundation for the RAG system, ensuring accurate query classification while maintaining the highest standards of Australian data governance, privacy protection, and system resilience.


**Last Updated**: 24 June 2025  
**Architecture Version**: 2.0 (Modular Refactor)