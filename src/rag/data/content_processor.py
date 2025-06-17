"""
Unified Text Processing & Ingestion Pipeline for RAG System

This module implements the core content processing pipeline that orchestrates
the sequential workflow for evaluation free-text fields:

1. Extract: Retrieve free-text fields from evaluation records
2. Anonymise: Remove/replace PII using Australian-specific detection
3. Analyse: Generate sentiment scores using local RoBERTa model
4. Chunk: Split text into sentence-level chunks for embedding
5. Store: Generate embeddings and persist with rich metadata

The pipeline ensures data privacy compliance while maintaining processing efficiency
through async operations and batch processing capabilities.

Classes:
- ContentProcessor: Main orchestrator for the unified pipeline
- TextChunker: Handles text segmentation with future extensibility
- ProcessingResult: Result container for individual record processing
- ProcessingConfig: Configuration management for processing parameters

Example Usage:
    # Process all evaluation records
    processor = ContentProcessor()
    await processor.initialize()
    results = await processor.process_all_evaluations()
    
    # Process specific records with custom batch size
    record_ids = [1, 2, 3, 100, 150]
    results = await processor.process_evaluation_records(
        record_ids, 
        batch_size=10
    )
    
    # Process single record with detailed metadata
    result = await processor.process_single_evaluation(
        response_id=123,
        include_metadata=True,
        chunk_strategy="sentence"
    )
    
    # Batch process with error recovery and resumption
    results = await processor.process_all_evaluations(
        batch_size=50,
        resume_from=500,  # Resume from response_id 500
        max_retries=3
    )

Architecture:
    The ContentProcessor integrates existing components:
    - AustralianPIIDetector: Mandatory PII anonymisation
    - SentimentAnalyser: Local RoBERTa sentiment analysis
    - EmbeddingsManager: Vector storage and retrieval
    - DatabaseManager: Secure read-only database access
"""

import asyncio
import logging
import time
import re
import importlib.util
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

# Import existing RAG components
from ..core.privacy.pii_detector import AustralianPIIDetector, PIIDetectionResult
from ..data.embeddings_manager import EmbeddingsManager
from ..utils.db_utils import DatabaseManager
from ..utils.logging_utils import get_logger
from ..config.settings import get_settings

logger = get_logger(__name__)

# Import sentiment analysis component with proper error handling
try:
    # Direct approach: temporarily modify sys.modules to provide the config
    import importlib.util
    
    sentiment_analysis_path = Path(__file__).parent.parent.parent / "sentiment-analysis"
    
    # Load sentiment config
    config_spec = importlib.util.spec_from_file_location(
        "sentiment_config_module", 
        sentiment_analysis_path / "config.py"
    )
    sentiment_config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(sentiment_config)
    
    # Temporarily place config in sys.modules for analyser import
    original_config = sys.modules.get('config')
    sys.modules['config'] = sentiment_config
    
    # Temporarily add sentiment analysis path
    sys.path.insert(0, str(sentiment_analysis_path))
    
    try:
        # Now import the analyser - it should find our config
        from analyser import SentimentAnalyser as _SentimentAnalyser  # type: ignore
        
        # Create a wrapper class
        class SentimentAnalyser:
            """Wrapper for the sentiment analysis component."""
            def __init__(self):
                self._analyser = _SentimentAnalyser()
            
            def analyse(self, text: str) -> dict:
                """Analyse text and return sentiment scores."""
                return self._analyser.analyse(text)
        
        logger.info("Successfully imported real SentimentAnalyser with transformer model")
        
    finally:
        # Clean up: restore original config and remove path
        if original_config is not None:
            sys.modules['config'] = original_config
        elif 'config' in sys.modules:
            del sys.modules['config']
        
        sys.path.remove(str(sentiment_analysis_path))
    
except Exception as e:
    logger.error(f"Failed to import SentimentAnalyser: {e}")
    
    # Create a mock sentiment analyser for testing
    class SentimentAnalyser:
        """Mock sentiment analyser for testing purposes."""
        def __init__(self):
            pass
        
        def analyse(self, text: str) -> dict:
            """Return mock sentiment scores."""
            return {"neg": 0.1, "neu": 0.8, "pos": 0.1}
    
    logger.warning("Using mock SentimentAnalyser - sentiment analysis will return dummy values")


@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata."""
    text: str
    index: int
    original_length: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result container for individual record processing."""
    response_id: int
    success: bool
    field_results: Dict[str, Any] = field(default_factory=dict)
    chunks_processed: int = 0
    embeddings_stored: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    pii_detected: bool = False
    sentiment_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    

@dataclass
class ProcessingConfig:
    """Configuration for content processing pipeline."""
    
    # Text fields to process
    text_fields: List[str] = field(default_factory=lambda: [
        "general_feedback", 
        "did_experience_issue_detail", 
        "course_application_other"
    ])
    
    # Chunking configuration
    chunk_strategy: str = "sentence"
    max_chunk_size: int = 500
    min_chunk_size: int = 50
    chunk_overlap: int = 0  # Future enhancement
    
    # Processing configuration
    batch_size: int = 50
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_pii_detection: bool = True
    enable_sentiment_analysis: bool = True
    
    # Storage configuration
    embedding_model_version: str = "text-embedding-ada-002-v1"
    include_metadata: bool = True
    store_original_text: bool = False  # Security: don't store original PII-containing text
    
    # Performance configuration
    concurrent_processing: int = 5
    enable_progress_logging: bool = True
    log_interval: int = 100


class TextChunker:
    """
    Simple sentence-based text chunking with future extensibility.
    
    Provides basic sentence-level text segmentation while maintaining
    a design that can accommodate more sophisticated chunking strategies
    in future enhancements.
    """
    
    def __init__(self, strategy: str = "sentence", max_chunk_size: int = 500, min_chunk_size: int = 50):
        """
        Initialize text chunker with specified strategy.
        
        Args:
            strategy: Chunking strategy ("sentence" for now, extensible)
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Simple sentence boundary patterns
        self.sentence_patterns = [
            r'[.!?]+\s+',  # Standard sentence endings with whitespace
            r'[.!?]+$',    # Sentence endings at end of text
        ]
    
    async def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk text based on configured strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of TextChunk objects with metadata
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        chunks = []
        
        if self.strategy == "sentence":
            chunks = await self._sentence_chunking(text)
        else:
            # Future: Add other chunking strategies
            logger.warning(f"Unknown chunking strategy '{self.strategy}', falling back to sentence chunking")
            chunks = await self._sentence_chunking(text)
        
        return chunks
    
    async def _sentence_chunking(self, text: str) -> List[TextChunk]:
        """
        Split text into sentence-based chunks.
        
        Uses regex patterns to identify sentence boundaries and creates
        chunks that respect both sentence boundaries and size limits.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Find sentence boundaries
        sentences = []
        current_pos = 0
        
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                if match.start() > current_pos:
                    sentence = text[current_pos:match.end()].strip()
                    if sentence:
                        sentences.append((sentence, current_pos, match.end()))
                    current_pos = match.end()
        
        # Handle remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append((remaining, current_pos, len(text)))
        
        # If no sentences found, treat entire text as one sentence
        if not sentences:
            sentences = [(text, 0, len(text))]
        
        # Create chunks respecting size limits
        chunks = []
        current_chunk = ""
        chunk_start = 0
        chunk_index = 0
        
        for sentence, start_pos, end_pos in sentences:
            # If adding this sentence would exceed max size, finalize current chunk
            if current_chunk and len(current_chunk) + len(sentence) > self.max_chunk_size:
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        index=chunk_index,
                        original_length=len(current_chunk.strip()),
                        start_position=chunk_start,
                        end_position=chunk_start + len(current_chunk),
                        metadata={"chunking_strategy": self.strategy}
                    ))
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence
                chunk_start = start_pos
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    chunk_start = start_pos
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                index=chunk_index,
                original_length=len(current_chunk.strip()),
                start_position=chunk_start,
                end_position=chunk_start + len(current_chunk),
                metadata={"chunking_strategy": self.strategy}
            ))
        
        logger.debug(f"Text chunking completed: {len(chunks)} chunks from {len(text)} characters")
        return chunks


class ContentProcessor:
    """
    Unified text processing and ingestion pipeline for evaluation free-text fields.
    
    Orchestrates the five-stage sequential workflow:
    1. Extract: Get free-text fields from evaluation records
    2. Anonymise: Apply mandatory PII detection and anonymisation
    3. Analyse: Generate sentiment scores using local model
    4. Chunk: Split anonymised text into processable segments
    5. Store: Generate embeddings and persist with rich metadata
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize content processor with configuration.
        
        Args:
            config: Processing configuration, uses defaults if None
        """
        self.config = config or ProcessingConfig()
        self.settings = get_settings()
        
        # Component instances
        self.db_manager: Optional[DatabaseManager] = None
        self.pii_detector: Optional[AustralianPIIDetector] = None
        self.sentiment_analyser: Optional[SentimentAnalyser] = None
        self.embeddings_manager: Optional[EmbeddingsManager] = None
        self.text_chunker: Optional[TextChunker] = None
        
        # State tracking
        self._initialized = False
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "pii_detections": 0
        }
    
    async def initialize(self) -> None:
        """
        Initialize all components and verify system readiness.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return
        
        try:
            logger.info("Initializing ContentProcessor components...")
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            # Verify database connection by getting pool
            await self.db_manager.get_pool()
            logger.info("Database manager initialized")
            
            # Initialize PII detector
            if self.config.enable_pii_detection:
                self.pii_detector = AustralianPIIDetector()
                await self.pii_detector.initialise()
                logger.info("PII detector initialized")
            
            # Initialize sentiment analyser
            if self.config.enable_sentiment_analysis:
                self.sentiment_analyser = SentimentAnalyser()
                logger.info("Sentiment analyser initialized")
            
            # Initialize embeddings manager
            self.embeddings_manager = EmbeddingsManager()
            await self.embeddings_manager.initialize()
            logger.info("Embeddings manager initialized")
            
            # Initialize text chunker
            self.text_chunker = TextChunker(
                strategy=self.config.chunk_strategy,
                max_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size
            )
            logger.info("Text chunker initialized")
            
            self._initialized = True
            logger.info("ContentProcessor initialization completed successfully")
            
        except Exception as e:
            logger.error(f"ContentProcessor initialization failed: {e}")
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize ContentProcessor: {e}")
    
    async def process_all_evaluations(
        self, 
        batch_size: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> List[ProcessingResult]:
        """
        Process all evaluation records with resumption capability.
        
        Args:
            batch_size: Override default batch size
            resume_from: Resume processing from specific response_id
            
        Returns:
            List of ProcessingResult objects
        """
        if not self._initialized:
            await self.initialize()
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Get all evaluation record IDs
            query = """
                SELECT response_id 
                FROM "Learning Content Evaluation" 
                WHERE response_id >= $1
                ORDER BY response_id
            """
            
            pool = await self.db_manager.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, resume_from or 0)
                response_ids = [row['response_id'] for row in rows]
            
            logger.info(f"Processing {len(response_ids)} evaluation records (batch_size={batch_size})")
            
            # Process in batches
            all_results = []
            for i in range(0, len(response_ids), batch_size):
                batch_ids = response_ids[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}: records {i + 1}-{min(i + batch_size, len(response_ids))}")
                
                batch_results = await self.process_evaluation_records(batch_ids)
                all_results.extend(batch_results)
                
                # Progress logging
                if self.config.enable_progress_logging:
                    successful = sum(1 for r in batch_results if r.success)
                    logger.info(f"Batch completed: {successful}/{len(batch_results)} successful")
            
            # Final statistics
            total_successful = sum(1 for r in all_results if r.success)
            logger.info(f"Processing completed: {total_successful}/{len(all_results)} records successful")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Failed to process all evaluations: {e}")
            raise
    
    async def process_evaluation_records(
        self, 
        response_ids: List[int], 
        batch_size: Optional[int] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple evaluation records efficiently.
        
        Args:
            response_ids: List of response IDs to process
            batch_size: Concurrent processing batch size
            
        Returns:
            List of ProcessingResult objects
        """
        if not self._initialized:
            await self.initialize()
        
        batch_size = batch_size or self.config.concurrent_processing
        
        # Process in concurrent batches
        results = []
        for i in range(0, len(response_ids), batch_size):
            batch_ids = response_ids[i:i + batch_size]
            
            # Create concurrent tasks
            tasks = [
                self.process_single_evaluation(response_id) 
                for response_id in batch_ids
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for response_id, result in zip(batch_ids, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Processing failed for response_id {response_id}: {result}")
                    results.append(ProcessingResult(
                        response_id=response_id,
                        success=False,
                        errors=[str(result)]
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def process_single_evaluation(
        self,
        response_id: int,
        include_metadata: Optional[bool] = None
    ) -> ProcessingResult:
        """
        Process a single evaluation record through the complete pipeline.
        
        Args:
            response_id: Evaluation record ID to process
            include_metadata: Override config for metadata inclusion
            
        Returns:
            ProcessingResult with detailed processing information
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        result = ProcessingResult(response_id=response_id, success=False)
        include_metadata = include_metadata if include_metadata is not None else self.config.include_metadata
        
        try:
            # Stage 1: Extract evaluation data
            evaluation_data = await self._extract_evaluation_data(response_id)
            if not evaluation_data:
                result.errors.append(f"No evaluation data found for response_id {response_id}")
                return result
            
            logger.debug(f"Processing response_id {response_id}: {len(evaluation_data)} fields found")
            
            # Stage 2-5: Process each text field through the pipeline
            total_chunks = 0
            total_embeddings = 0
            pii_detected = False
            
            for field_name in self.config.text_fields:
                if field_name not in evaluation_data or not evaluation_data[field_name]:
                    continue
                
                field_result = await self._process_text_field(
                    response_id=response_id,
                    field_name=field_name,
                    text=evaluation_data[field_name],
                    evaluation_metadata=evaluation_data if include_metadata else {}
                )
                
                result.field_results[field_name] = field_result
                
                if field_result.get('success', False):
                    total_chunks += field_result.get('chunks_processed', 0)
                    total_embeddings += field_result.get('embeddings_stored', 0)
                    if field_result.get('pii_detected', False):
                        pii_detected = True
                    
                    # Store sentiment scores
                    if 'sentiment_scores' in field_result:
                        result.sentiment_scores[field_name] = field_result['sentiment_scores']
                else:
                    result.errors.extend(field_result.get('errors', []))
                    result.warnings.extend(field_result.get('warnings', []))
            
            # Update result summary
            result.chunks_processed = total_chunks
            result.embeddings_stored = total_embeddings
            result.pii_detected = pii_detected
            result.success = total_embeddings > 0
            result.processing_time = time.time() - start_time
            
            # Update statistics
            self._processing_stats["total_processed"] += 1
            if result.success:
                self._processing_stats["successful"] += 1
                self._processing_stats["total_chunks"] += total_chunks
                self._processing_stats["total_embeddings"] += total_embeddings
                if pii_detected:
                    self._processing_stats["pii_detections"] += 1
            else:
                self._processing_stats["failed"] += 1
            
            logger.debug(f"Completed response_id {response_id}: {total_embeddings} embeddings stored")
            return result
            
        except Exception as e:
            result.errors.append(f"Processing failed: {str(e)}")
            result.processing_time = time.time() - start_time
            logger.error(f"Failed to process response_id {response_id}: {e}")
            return result
    
    async def _extract_evaluation_data(self, response_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract evaluation record data including text fields and metadata.
        
        Args:
            response_id: Evaluation record ID
            
        Returns:
            Dictionary with evaluation data or None if not found
        """
        try:
            # Build query to get evaluation data
            text_fields_sql = ', '.join([f'"{field}"' for field in self.config.text_fields])
            
            query = f"""
                SELECT 
                    response_id,
                    {text_fields_sql},
                    course_end_date,
                    user_id,
                    course_delivery_type,
                    agency
                FROM "Learning Content Evaluation"
                WHERE response_id = $1
            """
            
            pool = await self.db_manager.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, response_id)
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract evaluation data for response_id {response_id}: {e}")
            return None
    
    async def _process_text_field(
        self,
        response_id: int,
        field_name: str,
        text: str,
        evaluation_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single text field through the complete pipeline.
        
        Args:
            response_id: Evaluation record ID
            field_name: Name of the text field being processed
            text: Raw text content
            evaluation_metadata: Additional evaluation metadata
            
        Returns:
            Dictionary with processing results
        """
        field_result = {
            "success": False,
            "chunks_processed": 0,
            "embeddings_stored": 0,
            "pii_detected": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Stage 2: PII Detection and Anonymisation
            anonymised_text = text
            if self.config.enable_pii_detection and self.pii_detector:
                pii_result = await self.pii_detector.detect_and_anonymise(text)
                anonymised_text = pii_result.anonymised_text
                field_result["pii_detected"] = pii_result.anonymisation_applied
                
                if pii_result.entities_detected:
                    logger.info(f"PII detected and anonymised in {field_name} for response_id {response_id}")
            
            # Stage 3: Sentiment Analysis
            sentiment_scores = {}
            if self.config.enable_sentiment_analysis and self.sentiment_analyser:
                sentiment_scores = self.sentiment_analyser.analyse(anonymised_text)
                field_result["sentiment_scores"] = sentiment_scores
            
            # Stage 4: Text Chunking
            chunks = await self.text_chunker.chunk_text(anonymised_text)
            field_result["chunks_processed"] = len(chunks)
            
            if not chunks:
                field_result["warnings"].append(f"No valid chunks generated for field {field_name}")
                return field_result
            
            # Stage 5: Embedding Generation and Storage
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Prepare metadata for storage
            storage_metadata = {
                "user_id": evaluation_metadata.get("user_id"),
                "agency": evaluation_metadata.get("agency"),
                "course_delivery_type": evaluation_metadata.get("course_delivery_type"),
                "course_end_date": str(evaluation_metadata.get("course_end_date", "")),
                "field_name": field_name,
                "sentiment_scores": sentiment_scores,
                "original_length": len(text),
                "anonymised_length": len(anonymised_text),
                "chunk_count": len(chunks),
                "pii_detected": field_result["pii_detected"],
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            # Store embeddings
            embedding_ids = await self.embeddings_manager.store_embeddings(
                response_id=response_id,
                field_name=field_name,
                text_chunks=chunk_texts,
                metadata=storage_metadata
            )
            
            embeddings_stored = len(embedding_ids)
            field_result["embeddings_stored"] = embeddings_stored
            field_result["success"] = embeddings_stored > 0
            
            logger.debug(f"Field {field_name} processed: {len(chunks)} chunks, {embeddings_stored} embeddings stored")
            return field_result
            
        except Exception as e:
            field_result["errors"].append(f"Field processing failed: {str(e)}")
            logger.error(f"Failed to process field {field_name} for response_id {response_id}: {e}")
            return field_result
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            **self._processing_stats,
            "success_rate": (
                self._processing_stats["successful"] / max(1, self._processing_stats["total_processed"])
            ) * 100,
            "avg_chunks_per_record": (
                self._processing_stats["total_chunks"] / max(1, self._processing_stats["successful"])
            ),
            "avg_embeddings_per_record": (
                self._processing_stats["total_embeddings"] / max(1, self._processing_stats["successful"])
            )
        }
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if self.embeddings_manager:
                await self.embeddings_manager.close()
            
            if self.pii_detector:
                await self.pii_detector.close()
            
            if self.db_manager:
                await self.db_manager.close()
            
            logger.info("ContentProcessor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ContentProcessor cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Convenience functions for common operations

async def process_evaluation_batch(
    response_ids: List[int],
    config: Optional[ProcessingConfig] = None
) -> List[ProcessingResult]:
    """
    Convenience function to process a batch of evaluation records.
    
    Args:
        response_ids: List of evaluation record IDs
        config: Optional processing configuration
        
    Returns:
        List of ProcessingResult objects
    """
    async with ContentProcessor(config) as processor:
        return await processor.process_evaluation_records(response_ids)


async def process_single_record(
    response_id: int,
    config: Optional[ProcessingConfig] = None
) -> ProcessingResult:
    """
    Convenience function to process a single evaluation record.
    
    Args:
        response_id: Evaluation record ID
        config: Optional processing configuration
        
    Returns:
        ProcessingResult object
    """
    async with ContentProcessor(config) as processor:
        return await processor.process_single_evaluation(response_id)
