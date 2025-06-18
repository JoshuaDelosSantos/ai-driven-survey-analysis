# Vector Search Module

[![Status](https://img.shields.io/badge/Status-Phase%202%20Task%202.5%20Complete-green)](https://shields.io/)
[![Implementation](https://img.shields.io/badge/Implementation-Vector%20Search%20Tool%20Ready-blue)](https://shields.io/)
[![Models](https://img.shields.io/badge/Models-Sentence--BERT%20%2B%20OpenAI-orange)](https://shields.io/)

## Overview

The Vector Search module provides comprehensive semantic search capabilities for the RAG system. It implements efficient, async-first embedding generation with support for multiple providers, batch processing, and a complete LangChain-compatible search tool for agent orchestration.

## Current Implementation

### ✅ Embedder (`embedder.py`)

**Status: Implemented**

Clean async embedding generation service that provides a focused interface for embedding generation, separated from storage operations.

#### Key Features:
- **Async Interface**: Full async support with batch processing capabilities
- **Provider Flexibility**: Supports OpenAI and Sentence Transformer models
- **Configurable Batching**: Automatic batching with configurable batch sizes
- **Error Handling**: Robust error handling with retry logic
- **Performance Monitoring**: Built-in metrics collection and performance tracking
- **Model Versioning**: Support for future model upgrades

#### Supported Models:
- **OpenAI**: `text-embedding-ada-002` (1536 dimensions)
- **Sentence Transformers**: `all-MiniLM-L6-v2` (384 dimensions) - **Local Priority**

### ✅ Vector Search Tool (`vector_search_tool.py`) **NEW - Phase 2 Task 2.5**

**Status: Implemented**

Privacy-compliant, async LangChain tool for semantic search over evaluation feedback with automatic PII protection and rich metadata filtering.

#### Key Features:
- **LangChain Integration**: Fully compatible `BaseTool` for agent orchestration
- **Privacy-First Design**: Automatic query anonymization using Australian PII detection
- **Rich Metadata Filtering**: Filter by user level, agency, sentiment, delivery type
- **Performance Monitoring**: Built-in metrics and audit logging
- **Relevance Categorization**: Automatic result quality classification
- **Configurable Thresholds**: Adjustable similarity thresholds for different use cases

#### Metadata Filtering Capabilities:
- **High Priority**: `user_level`, `agency`, `sentiment_scores`
- **Medium Priority**: `course_delivery_type`, `knowledge_level_prior`
- **Field Filtering**: Target specific feedback fields (`general_feedback`, `did_experience_issue_detail`, `course_application_other`)

### ✅ Search Result Structures (`search_result.py`) **NEW - Phase 2 Task 2.5**

**Status: Implemented**

Comprehensive data structures for vector search results with rich metadata access and performance tracking.

#### Key Features:
- **Type-Safe Results**: Structured result containers with metadata access
- **Relevance Classification**: Automatic categorization (High/Medium/Low/Weak)
- **Performance Metrics**: Processing time tracking and optimization data
- **Serialization Support**: JSON-compatible for API responses
- **User Context**: Human-readable summaries of user and sentiment context

## Usage Examples

### Basic Vector Search Tool Usage

```python
from src.rag.core.vector_search import VectorSearchTool

# Initialize and use vector search tool
tool = VectorSearchTool()
await tool.initialize()

# Basic semantic search
response = await tool.search(
    query="feedback about technical issues",
    max_results=10,
    similarity_threshold=0.75
)

print(f"Found {response.result_count} relevant feedback items")
for result in response.results:
    print(f"[{result.relevance_category.value}] {result.user_context}")
    print(f"Feedback: {result.chunk_text[:200]}...")
```

### Advanced Metadata Filtering

```python
# Search with comprehensive filtering
response = await tool.search(
    query="course effectiveness feedback",
    filters={
        "user_level": ["Level 5", "Level 6", "Exec Level 1"],
        "agency": ["Department of Finance", "Australian Taxation Office"],
        "sentiment": {"type": "negative", "min_score": 0.6},
        "course_delivery_type": ["Virtual", "Blended"],
        "field_name": ["general_feedback", "did_experience_issue_detail"]
    },
    max_results=20,
    similarity_threshold=0.65
)

# Analyze results by relevance
distribution = response.relevance_distribution
print(f"High relevance: {distribution['High']} results")
print(f"Average similarity: {response.average_similarity:.3f}")
```
])

print(f"Processed {results.success_count}/{len(results.texts)} texts")
print(f"Total time: {results.total_processing_time:.2f}s")

# Custom configuration
embedder = Embedder(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
    batch_size=50
)

# Context manager usage
async with Embedder(provider="openai") as embedder:
    result = await embedder.embed_text("context managed embedding")
```

#### Performance Features:
- **Automatic Batching**: Efficiently processes large text collections
- **Metrics Collection**: Tracks performance statistics
- **Provider Testing**: Validates providers on initialization
- **Error Recovery**: Handles partial batch failures gracefully

#### Configuration:
Uses RAGSettings for configuration with override support:
```python
# Via environment variables
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=100

# Via direct instantiation
embedder = Embedder(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
    batch_size=50
)
```

## Planned Components

### 🔄 Chunk Processor (`chunk_processor.py`)

**Status: Planned for Phase 2.3 Integration**

Advanced text chunking strategies for optimal embedding generation and retrieval.

#### Planned Features:
- **Sentence-Level Chunking**: Intelligent sentence boundary detection
- **Semantic Chunking**: Content-aware chunking using semantic similarity
- **Overlap Strategies**: Configurable text overlap for context preservation
- **Length Optimization**: Dynamic chunk sizing based on content type
- **Metadata Preservation**: Maintain source context through chunking process

#### Future Enhancement Ideas:
- **Sliding Window Chunking**: For long documents with context preservation
- **Hierarchical Chunking**: Multi-level chunking for complex documents
- **Content-Type Specific**: Different strategies for feedback vs. technical content
- **Language-Aware Chunking**: Handle multilingual content appropriately

#### Configuration Options:
```python
chunk_processor = ChunkProcessor(
    strategy="sentence",  # sentence, semantic, fixed_length
    max_chunk_size=500,
    overlap_size=50,
    preserve_sentence_boundaries=True,
    min_chunk_size=100
)
```

### 🔍 Retriever (`retriever.py`)

**Status: Planned for Phase 3**

Semantic retrieval engine for finding relevant content using vector similarity.

#### Planned Features:
- **Similarity Search**: Vector cosine similarity with configurable thresholds
- **Metadata Filtering**: Filter by user level, agency, sentiment, date ranges
- **Hybrid Ranking**: Combine vector similarity with metadata relevance
- **Result Aggregation**: Merge and rank results from multiple searches
- **Contextual Expansion**: Expand search with related terms and synonyms

#### Advanced Capabilities:
- **Multi-Modal Search**: Support for different content types (feedback, issues, applications)
- **Temporal Filtering**: Time-based relevance scoring
- **User Context**: Personalized results based on user characteristics
- **Explanation Generation**: Provide reasoning for search results

#### Example API:
```python
retriever = Retriever(embedder=embedder)

results = await retriever.search(
    query="course satisfaction issues",
    field_names=["general_feedback", "did_experience_issue_detail"],
    filters={
        "user_level": [5, 6],
        "agency": "ATO",
        "sentiment_negative": {"gte": 0.7}
    },
    limit=10,
    similarity_threshold=0.75
)
```

### 📊 Indexer (`indexer.py`)

**Status: Planned for Phase 3+**

Vector indexing management for efficient large-scale retrieval.

#### Planned Features:
- **Index Management**: Create, update, and optimize vector indexes
- **Efficient Storage**: Compress and organize embeddings for fast retrieval
- **Batch Operations**: Efficient bulk index updates and rebuilds
- **Index Versioning**: Support for model upgrades and index migrations
- **Performance Optimization**: Automatic index tuning based on usage patterns

#### Advanced Indexing:
- **Approximate Nearest Neighbor (ANN)**: FAISS, Annoy, or similar integration
- **Distributed Indexing**: Support for larger datasets across multiple nodes
- **Incremental Updates**: Efficient updates without full rebuilds
- **Index Analytics**: Usage statistics and optimization recommendations

#### Configuration:
```python
indexer = Indexer(
    index_type="ivfflat",  # ivfflat, hnsw, flat
    lists=100,             # for IVFFlat
    probes=10,            # for search
    maintenance_schedule="daily"
)
```

## Integration Architecture

### Current Integration Points:

1. **EmbeddingsManager Integration**: 
   - EmbeddingsManager can optionally use Embedder for generation
   - Maintains backward compatibility with existing storage operations
   - Clean separation of embedding generation from storage

2. **Content Processor Integration** (Phase 2.3):
   - Content processor will use Embedder directly for text processing
   - Embedder handles PII-anonymized content from privacy pipeline
   - Results passed to EmbeddingsManager for storage

3. **Configuration Integration**:
   - Uses RAGSettings for consistent configuration
   - Supports environment variable overrides
   - Maintains compatibility with existing configuration

### Future Integration (Phase 3):

1. **Query Router Integration**:
   - Vector search tool will use Retriever for semantic queries
   - Hybrid queries will combine SQL and vector search results
   - LangGraph nodes will orchestrate vector search operations

2. **Answer Synthesis Integration**:
   - Retrieved vector results will be synthesized with SQL results
   - Context ranking and relevance scoring
   - Source attribution and explanation generation

## Testing Strategy

### Current Testing (Planned):
- **Unit Tests**: Individual component testing with mocked providers
- **Integration Tests**: Full pipeline testing with real embeddings
- **Performance Tests**: Batch processing and latency benchmarks
- **Provider Tests**: Validation across OpenAI and Sentence Transformer models

### Test Files:
```
src/rag/tests/
├── test_embedder.py              # Unit tests for Embedder class
├── test_embedder_integration.py  # Integration tests with real models
├── test_embedder_performance.py  # Performance and batch testing
└── manual_test_embedder.py       # Manual testing script
```

### Test Coverage Areas:
- ✅ Provider initialization and configuration
- ✅ Single text embedding generation
- ✅ Batch processing with various sizes
- ✅ Error handling and recovery
- ✅ Performance metrics collection
- ✅ Model versioning and compatibility

## Future Enhancements

### Short-term (Phase 2 Completion):
1. **Enhanced Error Handling**: Exponential backoff, circuit breakers
2. **Caching Layer**: Cache frequently embedded texts
3. **Model Validation**: Embedding quality checks and validation
4. **Configuration Validation**: Enhanced settings validation

### Medium-term (Phase 3):
1. **New Model Support**: OpenAI text-embedding-3-small/large
2. **Provider Expansion**: Cohere, HuggingFace Hub integration
3. **Adaptive Batching**: Dynamic batch size based on performance
4. **Rate Limiting**: API rate limiting for external providers

### Long-term (Phase 4+):
1. **Model Fine-tuning**: Custom model training on domain data
2. **Multi-modal Embeddings**: Support for different content types
3. **Embedding Compression**: Reduce storage requirements
4. **Distributed Processing**: Scale across multiple nodes

## Configuration Reference

### Environment Variables:
```bash
# Embedding Provider Configuration
EMBEDDING_PROVIDER=sentence_transformers    # openai, sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2      # Model name
EMBEDDING_DIMENSION=384                     # Vector dimension
EMBEDDING_BATCH_SIZE=100                   # Batch processing size
EMBEDDING_API_KEY=sk-...                   # API key (if needed)

# Performance Tuning
EMBEDDING_MAX_RETRIES=3                    # Future: Retry attempts
EMBEDDING_TIMEOUT=30                       # Future: Request timeout
EMBEDDING_CACHE_SIZE=1000                  # Future: Cache size
```

### Model Specifications:

| Provider | Model | Dimension | Use Case | Performance |
|----------|-------|-----------|----------|-------------|
| Sentence Transformers | all-MiniLM-L6-v2 | 384 | Local, Fast | ⭐⭐⭐⭐⭐ |
| OpenAI | text-embedding-ada-002 | 1536 | High Quality | ⭐⭐⭐⭐ |
| Future: OpenAI | text-embedding-3-small | 1536 | Improved Quality | ⭐⭐⭐⭐⭐ |
| Future: OpenAI | text-embedding-3-large | 3072 | Best Quality | ⭐⭐⭐⭐⭐ |

## Dependencies

### Required:
- `asyncio`: Async processing
- `sentence-transformers`: Local embedding models
- `openai`: OpenAI API integration
- `numpy`: Vector operations
- `pydantic`: Configuration management

### Optional (Future):
- `faiss-cpu`: Efficient vector indexing
- `transformers`: HuggingFace model support
- `cohere`: Cohere embedding API
- `redis`: Caching layer

## Contributing

When extending this module:

1. **Maintain Async Patterns**: All new components should be fully async
2. **Follow Configuration Patterns**: Use RAGSettings and environment variables
3. **Error Handling**: Implement robust error handling with logging
4. **Testing**: Include comprehensive tests for new functionality
5. **Documentation**: Update this README with new features and usage examples

## Performance Considerations

### Current Optimizations:
- Async batch processing for improved throughput
- Connection pooling for database operations (via EmbeddingsManager)
- Lazy model loading for memory efficiency
- Performance metrics for monitoring

### Future Optimizations:
- Embedding caching for frequently used texts
- Model quantization for faster inference
- Distributed processing for large-scale operations
- Adaptive batching based on system resources

## Security Considerations

### Current Security:
- No sensitive data in embeddings (PII pre-filtered)
- Secure API key management via environment variables
- Input validation and sanitization

### Future Security:
- Embedding content validation
- Rate limiting for API abuse prevention
- Audit logging for embedding operations
- Model integrity verification
