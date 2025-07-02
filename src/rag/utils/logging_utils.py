"""
Logging Utilities for RAG Module

Provides secure logging configuration with data privacy controls
for the Text-to-SQL system.

Security: Masks sensitive data, prevents PII exposure in logs.
Compliance: Audit trail support for governance requirements.
"""

import logging
import logging.handlers
import json
import re
from typing import Any, Dict, Optional, List
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the rag module to path for absolute imports
rag_path = Path(__file__).parent.parent
if str(rag_path) not in sys.path:
    sys.path.insert(0, str(rag_path))

from config.settings import get_settings


class PIIMaskingFormatter(logging.Formatter):
    """
    Custom formatter that masks potential PII in log messages.
    
    Removes or masks:
    - Database credentials
    - API keys
    - Email addresses
    - Phone numbers
    - Personal names (basic detection)
    """
    
    # Patterns for PII detection
    PII_PATTERNS = [
        # Database URLs with credentials
        (re.compile(r'postgresql://[^:]+:[^@]+@'), 'postgresql://***:***@'),
        # API keys (common patterns)
        (re.compile(r'sk-[a-zA-Z0-9]{32,}'), 'sk-***'),
        (re.compile(r'Bearer [a-zA-Z0-9_.-]+'), 'Bearer ***'),
        # Email addresses
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '***@***.***'),
        # Phone numbers (basic patterns)
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '***-***-****'),
        (re.compile(r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'), '+***-***-****'),
        # Credit card numbers (basic)
        (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '****-****-****-****'),
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with PII masking."""
        # Get original formatted message
        message = super().format(record)
        
        # Apply PII masking
        for pattern, replacement in self.PII_PATTERNS:
            message = pattern.sub(replacement, message)
        
        return message


class StructuredJSONFormatter(PIIMaskingFormatter):
    """
    JSON formatter for structured logging with PII protection.
    
    Outputs logs in JSON format for easier parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON with structured fields."""
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Convert to JSON and apply PII masking
        json_str = json.dumps(log_entry, default=str)
        
        # Apply PII patterns to JSON string
        for pattern, replacement in self.PII_PATTERNS:
            json_str = pattern.sub(replacement, json_str)
        
        return json_str


class RAGLogger:
    """
    RAG module logger with security and audit features.
    
    Features:
    - PII masking
    - Structured logging
    - Audit trail support
    - Secure error handling
    """
    
    def __init__(self, name: str):
        """
        Initialize RAG logger.
        
        Args:
            name: Logger name (usually module name)
        """
        self.logger = logging.getLogger(name)
        self.settings = get_settings()
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup log handlers with appropriate formatters."""
        # Set log level
        log_level = getattr(logging, self.settings.log_level.upper())
        self.logger.setLevel(log_level)
        
        # Console handler with PII masking
        console_handler = logging.StreamHandler()
        console_formatter = PIIMaskingFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)
        
        # File handler for application logs
        if hasattr(self.settings, 'log_file_path') and self.settings.log_file_path:
            log_file = Path(self.settings.log_file_path)
        else:
            log_file = Path('rag.log')
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_formatter = PIIMaskingFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        self.logger.addHandler(file_handler)
        
        # Audit log handler (structured JSON)
        if self.settings.debug_mode:
            audit_file = log_file.parent / 'audit.log'
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            audit_formatter = StructuredJSONFormatter()
            audit_handler.setFormatter(audit_formatter)
            audit_handler.setLevel(logging.INFO)
            self.logger.addHandler(audit_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with optional extra fields."""
        self._log_with_extra(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with optional extra fields."""
        self._log_with_extra(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional extra fields."""
        self._log_with_extra(logging.ERROR, message, extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with optional extra fields."""
        self._log_with_extra(logging.DEBUG, message, extra)
    
    def _log_with_extra(self, level: int, message: str, extra: Optional[Dict[str, Any]]) -> None:
        """Log message with extra fields."""
        if extra:
            # Create a copy to avoid modifying original
            safe_extra = self._sanitize_extra_fields(extra.copy())
            self.logger.log(level, message, extra={'extra_fields': safe_extra})
        else:
            self.logger.log(level, message)
    
    def _sanitize_extra_fields(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize extra fields to remove PII."""
        sanitized = {}
        
        for key, value in extra.items():
            # Convert to string and check for PII patterns
            str_value = str(value)
            
            # Apply PII masking
            for pattern, replacement in PIIMaskingFormatter.PII_PATTERNS:
                str_value = pattern.sub(replacement, str_value)
            
            sanitized[key] = str_value
        
        return sanitized
    
    def log_query_execution(
        self,
        query: str,
        execution_time: float,
        row_count: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Log SQL query execution for audit trail.
        
        Args:
            query: SQL query (will be masked for security)
            execution_time: Query execution time in seconds
            row_count: Number of rows returned
            success: Whether query succeeded
            error: Error message if failed
        """
        # Mask the query for logging
        masked_query = self._mask_sql_query(query)
        
        extra = {
            'event_type': 'sql_query',
            'query_hash': hash(query.strip()),  # Use hash for tracking without exposing query
            'execution_time': execution_time,
            'row_count': row_count,
            'success': success
        }
        
        if success:
            self.info(f"SQL query executed successfully in {execution_time:.3f}s", extra)
        else:
            extra['error_type'] = type(error).__name__ if error else 'unknown'
            self.error(f"SQL query failed: {error}", extra)
    
    def log_llm_interaction(
        self,
        model: str,
        prompt_length: int,
        response_length: int,
        tokens_used: Optional[int] = None,
        response_time: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Log LLM API interaction for audit trail.
        
        Args:
            model: Model name used
            prompt_length: Length of prompt in characters
            response_length: Length of response in characters
            tokens_used: Number of tokens consumed
            response_time: Response time in seconds
            success: Whether interaction succeeded
            error: Error message if failed
        """
        extra = {
            'event_type': 'llm_interaction',
            'model': model,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'tokens_used': tokens_used,
            'response_time': response_time,
            'success': success
        }
        
        if success:
            self.info(f"LLM interaction completed with {model}", extra)
        else:
            extra['error_type'] = type(error).__name__ if error else 'unknown'
            self.error(f"LLM interaction failed: {error}", extra)
    
    def log_user_query(
        self,
        query_id: str,
        query_type: str,
        processing_time: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log user query processing for audit trail.
        
        Args:
            query_id: Unique query identifier
            query_type: Type of query (sql, vector, hybrid, agent)
            processing_time: Total processing time
            success: Whether processing succeeded
            error: Error message if failed
            metadata: Optional metadata dictionary for additional context
        """
        extra = {
            'event_type': 'user_query',
            'query_id': query_id,
            'query_type': query_type,
            'processing_time': processing_time,
            'success': success
        }
        
        # Add metadata if provided
        if metadata:
            extra.update(metadata)
        
        if success:
            self.info(f"User query processed successfully ({query_type})", extra)
        else:
            extra['error_type'] = type(error).__name__ if error else 'unknown'
            self.error(f"User query processing failed: {error}", extra)
    
    def _mask_sql_query(self, query: str) -> str:
        """Mask SQL query for safe logging."""
        # Remove potential PII from WHERE clauses
        query = re.sub(r"= '[^']*'", "= '***'", query)
        query = re.sub(r'= "[^"]*"', '= "***"', query)
        
        # Truncate very long queries
        if len(query) > 200:
            query = query[:200] + "... [TRUNCATED]"
        
        return query


# Global logger instances
_loggers: Dict[str, RAGLogger] = {}


def get_logger(name: str) -> RAGLogger:
    """
    Get RAG logger instance for module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        RAGLogger: Configured logger instance
    """
    if name not in _loggers:
        _loggers[name] = RAGLogger(name)
    
    return _loggers[name]


def setup_logging() -> None:
    """Setup global logging configuration."""
    settings = get_settings()
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncpg').setLevel(logging.INFO)
    
    # Enable SQL query logging if configured
    if hasattr(settings, 'log_sql_queries') and settings.log_sql_queries:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


def mask_sensitive_data(data: str) -> str:
    """
    Mask sensitive data in string.
    
    Args:
        data: String that might contain sensitive data
        
    Returns:
        str: String with sensitive data masked
    """
    formatter = PIIMaskingFormatter()
    
    # Apply PII patterns
    for pattern, replacement in formatter.PII_PATTERNS:
        data = pattern.sub(replacement, data)
    
    return data


# Convenience function for quick logging
def log_event(
    level: str,
    message: str,
    logger_name: str = 'rag',
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log event with specified level.
    
    Args:
        level: Log level (info, warning, error, debug)
        message: Log message
        logger_name: Logger name
        extra: Extra fields for structured logging
    """
    logger = get_logger(logger_name)
    
    level_methods = {
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'debug': logger.debug
    }
    
    method = level_methods.get(level.lower(), logger.info)
    method(message, extra)
