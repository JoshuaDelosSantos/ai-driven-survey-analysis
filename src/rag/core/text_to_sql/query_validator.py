"""
Query Logic Validator for Phase 3 Enhancement

This module implements validation and auto-correction for SQL queries to detect
and fix common logic errors, particularly those related to incorrect table joins
and semantic mismatches.

Key Features:
- Pattern-based error detection for problematic SQL constructs
- Auto-correction suggestions for common mistakes
- Integration with classification results for context-aware validation
- Detailed logging and metrics for continuous improvement

Example Usage:
    # Initialize validator
    validator = QueryLogicValidator()
    
    # Validate a query
    result = await validator.validate_and_correct(
        sql_query="SELECT * FROM rag_user_feedback JOIN learning_content...",
        original_question="What feedback about courses?",
        classification_result=classification_result
    )
    
    if not result["valid"]:
        print(f"Issues found: {result['issues']}")
        print(f"Suggested fix: {result['corrected_query']}")
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a SQL query."""
    pattern: str
    issue_type: str
    description: str
    suggestion: str
    severity: str  # "critical", "warning", "info"


@dataclass
class ValidationResult:
    """Result of SQL query validation."""
    valid: bool
    issues: List[ValidationIssue]
    corrected_query: Optional[str] = None
    should_retry: bool = False
    confidence: float = 0.0
    reasoning: str = ""


class QueryLogicValidator:
    """
    Validates SQL queries for logical correctness and suggests corrections.
    
    Focuses on detecting and correcting the specific issues identified in Phase 1/2:
    - Incorrect joins between rag_user_feedback and learning_content
    - Semantic mismatches in query logic
    - Empty result patterns
    """
    
    def __init__(self):
        """Initialize validator with problematic patterns and correction rules."""
        
        # Critical patterns that should never appear
        self.critical_patterns = [
            ValidationIssue(
                pattern=r"rag_user_feedback[\s\S]*?JOIN[\s\S]*?learning_content",
                issue_type="incorrect_table_join",
                description="Incorrect join: rag_user_feedback is for system feedback, not content feedback",
                suggestion="Use evaluation table for learning content feedback",
                severity="critical"
            ),
            ValidationIssue(
                pattern=r"learning_content[\s\S]*?JOIN[\s\S]*?rag_user_feedback",
                issue_type="incorrect_table_join", 
                description="Incorrect join: learning_content should not join with rag_user_feedback",
                suggestion="Use evaluation table for learning content feedback",
                severity="critical"
            ),
            ValidationIssue(
                pattern=r"LIKE.*\|\|.*name.*\|\|",
                issue_type="semantic_mismatch",
                description="Semantic mismatch: joining query text with content names using LIKE",
                suggestion="Use proper foreign key relationships instead of text matching",
                severity="critical"
            ),
            ValidationIssue(
                pattern=r"query_text.*LIKE.*learning_content",
                issue_type="semantic_mismatch",
                description="Semantic mismatch: matching RAG queries against course names",
                suggestion="Use evaluation table for course feedback instead",
                severity="critical"
            )
        ]
        
        # Warning patterns that indicate potential issues
        self.warning_patterns = [
            ValidationIssue(
                pattern=r"rag_user_feedback[\s\S]*?WHERE[\s\S]*?content_type",
                issue_type="logical_inconsistency",
                description="RAG user feedback table doesn't contain content_type information",
                suggestion="Use evaluation and learning_content tables for content-related queries",
                severity="warning"
            ),
            ValidationIssue(
                pattern=r"SELECT[\s\S]*?FROM rag_user_feedback[\s\S]*?(?!.*rag_|.*system|.*platform|.*search|.*query)",
                issue_type="potential_misuse",
                description="rag_user_feedback query without system-related context",
                suggestion="Verify this is actually about RAG system feedback",
                severity="warning"
            )
        ]
        
        # Statistics tracking
        self.validation_stats = {
            "total_validations": 0,
            "issues_found": 0,
            "auto_corrections": 0,
            "critical_issues": 0,
            "warning_issues": 0
        }
    
    async def validate_and_correct(
        self, 
        sql_query: str, 
        original_question: str,
        classification_result: Optional[Any] = None
    ) -> ValidationResult:
        """
        Validate SQL query and suggest corrections if needed.
        
        Args:
            sql_query: The generated SQL query to validate
            original_question: The original natural language question
            classification_result: Optional classification result with table guidance
            
        Returns:
            ValidationResult with validation status and corrections
        """
        self.validation_stats["total_validations"] += 1
        
        # Find all issues
        all_issues = []
        
        # Check critical patterns
        for issue_template in self.critical_patterns:
            if re.search(issue_template.pattern, sql_query, re.IGNORECASE):
                all_issues.append(issue_template)
                self.validation_stats["critical_issues"] += 1
                logger.warning(f"Critical issue detected: {issue_template.description}")
        
        # Check warning patterns  
        for issue_template in self.warning_patterns:
            if re.search(issue_template.pattern, sql_query, re.IGNORECASE):
                all_issues.append(issue_template)
                self.validation_stats["warning_issues"] += 1
                logger.info(f"Warning issue detected: {issue_template.description}")
        
        # Determine if query is valid
        critical_issues = [issue for issue in all_issues if issue.severity == "critical"]
        is_valid = len(critical_issues) == 0
        
        if not is_valid:
            self.validation_stats["issues_found"] += 1
            
            # Attempt auto-correction for critical issues
            corrected_query = await self._generate_corrected_query(
                sql_query, original_question, critical_issues, classification_result
            )
            
            should_retry = corrected_query is not None
            if should_retry:
                self.validation_stats["auto_corrections"] += 1
            
            return ValidationResult(
                valid=False,
                issues=all_issues,
                corrected_query=corrected_query,
                should_retry=should_retry,
                confidence=0.8 if should_retry else 0.0,
                reasoning=f"Found {len(critical_issues)} critical issues, {len(all_issues) - len(critical_issues)} warnings"
            )
        
        # Query is valid but might have warnings
        return ValidationResult(
            valid=True,
            issues=all_issues,
            confidence=1.0 if len(all_issues) == 0 else 0.9,
            reasoning=f"Query valid with {len(all_issues)} warnings" if all_issues else "Query valid"
        )
    
    async def _generate_corrected_query(
        self,
        original_query: str,
        original_question: str,
        issues: List[ValidationIssue],
        classification_result: Optional[Any] = None
    ) -> Optional[str]:
        """
        Generate a corrected version of the query based on identified issues.
        
        Args:
            original_query: The problematic SQL query
            original_question: The original natural language question
            issues: List of validation issues found
            classification_result: Optional classification result with table guidance
            
        Returns:
            Corrected SQL query string or None if correction not possible
        """
        try:
            corrected_query = original_query
            
            for issue in issues:
                if issue.issue_type == "incorrect_table_join":
                    corrected_query = await self._fix_incorrect_table_join(
                        corrected_query, original_question, classification_result
                    )
                elif issue.issue_type == "semantic_mismatch":
                    corrected_query = await self._fix_semantic_mismatch(
                        corrected_query, original_question, classification_result
                    )
            
            # Validate the corrected query doesn't have the same issues
            if corrected_query != original_query:
                validation_check = await self.validate_and_correct(
                    corrected_query, original_question, classification_result
                )
                if validation_check.valid or len([i for i in validation_check.issues if i.severity == "critical"]) == 0:
                    logger.info(f"Successfully generated corrected query")
                    return corrected_query
                else:
                    logger.warning(f"Corrected query still has issues: {validation_check.issues}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating corrected query: {e}")
            return None
    
    async def _fix_incorrect_table_join(
        self,
        query: str,
        original_question: str,
        classification_result: Optional[Any] = None
    ) -> str:
        """Fix incorrect joins between rag_user_feedback and learning_content."""
        
        # If we have classification guidance, use it
        if (classification_result and 
            hasattr(classification_result, 'feedback_table_suggestion') and
            classification_result.feedback_table_suggestion == 'evaluation'):
            
            # Replace rag_user_feedback with evaluation in feedback queries
            if "feedback" in original_question.lower():
                corrected = re.sub(
                    r'\brag_user_feedback\b',
                    'evaluation',
                    query,
                    flags=re.IGNORECASE
                )
                
                # Fix the join condition if it exists
                corrected = re.sub(
                    r'ON\s+\w+\.query_text\s+LIKE.*',
                    'ON evaluation.learning_content_surrogate_key = learning_content.surrogate_key',
                    corrected,
                    flags=re.IGNORECASE | re.DOTALL
                )
                
                # Update column references
                corrected = re.sub(r'\bruf\b', 'e', corrected)
                corrected = re.sub(r'\brag_user_feedback\b', 'evaluation e', corrected)
                
                return corrected
        
        # Default correction for content feedback queries
        if any(keyword in original_question.lower() for keyword in ['course', 'training', 'learning', 'content']):
            # This appears to be content feedback, use evaluation table
            corrected = query.replace('rag_user_feedback', 'evaluation')
            corrected = re.sub(
                r'ON\s+.*\.query_text.*LIKE.*',
                'ON evaluation.learning_content_surrogate_key = learning_content.surrogate_key',
                corrected,
                flags=re.IGNORECASE | re.DOTALL
            )
            return corrected
        
        return query
    
    async def _fix_semantic_mismatch(
        self,
        query: str,
        original_question: str,
        classification_result: Optional[Any] = None
    ) -> str:
        """Fix semantic mismatches like LIKE joins with unrelated text."""
        
        # Fix LIKE patterns that join query_text with names
        corrected = re.sub(
            r'ON\s+\w+\.query_text\s+LIKE\s+.*name.*',
            'ON evaluation.learning_content_surrogate_key = learning_content.surrogate_key',
            query,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        # Fix other LIKE concatenation patterns
        corrected = re.sub(
            r"LIKE\s+'%'\s*\|\|\s*\w+\.\w+\s*\|\|\s*'%'",
            '= learning_content.surrogate_key',
            corrected,
            flags=re.IGNORECASE
        )
        
        return corrected
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        stats = self.validation_stats.copy()
        if stats["total_validations"] > 0:
            stats["issue_rate"] = stats["issues_found"] / stats["total_validations"]
            stats["correction_rate"] = stats["auto_corrections"] / max(stats["issues_found"], 1)
        else:
            stats["issue_rate"] = 0.0
            stats["correction_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics."""
        for key in self.validation_stats:
            self.validation_stats[key] = 0
