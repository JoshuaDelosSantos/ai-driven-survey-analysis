"""
LLM Utilities for RAG Module

Provides utilities for Language Model interactions, prompt management,
and response processing for the Text-to-SQL system.

Security: No sensitive data in prompts, secure error handling.
Performance: Async operations with retry logic.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import time
import json

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM interaction."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    response_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class LLMManager:
    """
    Manages Language Model interactions for RAG system.
    
    Features:
    - Multiple provider support (OpenAI, Anthropic, Gemini)
    - Async operations with retry logic
    - Token usage tracking
    - Secure prompt handling
    """
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM manager.
        
        Args:
            model_name: Override default model name
            api_key: Override default API key
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.llm_model_name
        self.api_key = api_key or self.settings.llm_api_key
        
        self._llm: Optional[BaseLanguageModel] = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            if self.model_name.startswith('gpt-'):
                # OpenAI models
                self._llm = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=0.1,  # Low temperature for deterministic SQL generation
                    max_tokens=1000,
                    timeout=30
                )
                logger.info(f"Initialized OpenAI model: {self.model_name}")
                
            elif self.model_name.startswith('claude-'):
                # Anthropic models
                self._llm = ChatAnthropic(
                    model=self.model_name,
                    anthropic_api_key=self.api_key,
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=30
                )
                logger.info(f"Initialized Anthropic model: {self.model_name}")
                
            elif self.model_name.startswith(('gemini-', 'models/gemini-')):
                # Google Gemini models
                self._llm = self._create_gemini_llm()
                logger.info(f"Initialized Gemini model: {self.model_name}")
                
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    def _create_gemini_llm(self) -> ChatGoogleGenerativeAI:
        """
        Create and configure Gemini LLM instance.
        
        Returns:
            ChatGoogleGenerativeAI: Configured Gemini model instance
        """
        # Handle different Gemini model name formats
        model_name = self.model_name
        if not model_name.startswith('models/'):
            if model_name.startswith('gemini-'):
                model_name = f"models/{model_name}"
            else:
                model_name = f"models/gemini-{model_name}"
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.1,  # Low temperature for deterministic SQL generation
            max_output_tokens=1000,
            timeout=30
            # Note: Removed safety_settings due to enum validation issues
            # These can be configured per request if needed
        )
    
    @property
    def llm(self) -> BaseLanguageModel:
        """Get the language model instance."""
        if self._llm is None:
            self._initialize_llm()
        return self._llm
    
    async def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        max_retries: int = 3
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            max_retries: Maximum retry attempts
            
        Returns:
            LLMResponse: LLM response with metadata
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Prepare messages
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                
                # Generate response asynchronously
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm.invoke(messages)
                )
                
                response_time = time.time() - start_time
                
                # Extract content
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Try to extract token usage if available
                tokens_used = None
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    tokens_used = usage.get('total_tokens')
                
                logger.debug(f"LLM response generated in {response_time:.3f}s, tokens: {tokens_used}")
                
                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    return LLMResponse(
                        content="",
                        model=self.model_name,
                        response_time=0.0,
                        success=False,
                        error=str(e)
                    )
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def generate_sql(
        self, 
        question: str, 
        schema_description: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> LLMResponse:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            schema_description: Database schema description
            examples: Optional example question-SQL pairs
            
        Returns:
            LLMResponse: Generated SQL response
        """
        system_message = """You are an expert SQL query generator for an Australian Public Service learning analytics database.

Generate valid PostgreSQL SELECT queries only. Follow these rules:
1. Use only SELECT statements - no INSERT, UPDATE, DELETE, or DDL
2. Use proper JOINs to combine tables when needed
3. Include appropriate WHERE clauses for filtering
4. Use meaningful column aliases for clarity
5. Return only the SQL query, no explanations or markdown
6. Ensure queries are safe and read-only"""

        # Build prompt with schema and examples
        prompt_parts = [
            "# Database Schema",
            schema_description,
            "",
            "# Question",
            question,
            ""
        ]
        
        if examples:
            prompt_parts.extend([
                "# Example Queries",
                ""
            ])
            for i, example in enumerate(examples, 1):
                prompt_parts.extend([
                    f"**Example {i}:**",
                    f"Question: {example['question']}",
                    f"SQL: {example['sql']}",
                    ""
                ])
        
        prompt_parts.extend([
            "# Instructions",
            "Generate a PostgreSQL SELECT query for the question above.",
            "Return only the SQL query without any explanations or formatting.",
            "",
            "SQL Query:"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        return await self.generate(prompt, system_message)
    
    async def validate_sql_response(self, response: str, question: str) -> LLMResponse:
        """
        Validate and improve SQL response.
        
        Args:
            response: Generated SQL response
            question: Original question
            
        Returns:
            LLMResponse: Validated/improved SQL
        """
        system_message = """You are a SQL validation expert. Review the provided SQL query and ensure it:
1. Is a valid PostgreSQL SELECT query
2. Correctly answers the given question
3. Uses proper table aliases and JOINs
4. Contains no security vulnerabilities
5. Is optimized for performance

If the query is correct, return it as-is. If it needs improvements, return the corrected version.
Return only the SQL query, no explanations."""

        prompt = f"""Original Question: {question}

SQL Query to Validate:
{response}

Validated SQL Query:"""

        return await self.generate(prompt, system_message)
    
    def get_example_queries(self) -> List[Dict[str, str]]:
        """Get example question-SQL pairs for few-shot prompting."""
        return [
            {
                "question": "How many users completed courses in each agency?",
                "sql": """SELECT u.agency, COUNT(*) as completed_users
FROM users u
JOIN attendance a ON u.user_id = a.user_id
WHERE a.status = 'Completed'
GROUP BY u.agency
ORDER BY completed_users DESC"""
            },
            {
                "question": "Show attendance status breakdown by user level",
                "sql": """SELECT u.user_level, a.status, COUNT(*) as count
FROM users u
JOIN attendance a ON u.user_id = a.user_id
GROUP BY u.user_level, a.status
ORDER BY u.user_level, a.status"""
            },
            {
                "question": "Which courses have the highest enrollment?",
                "sql": """SELECT lc.name, lc.content_type, COUNT(*) as enrollment_count
FROM learning_content lc
JOIN attendance a ON lc.surrogate_key = a.learning_content_surrogate_key
GROUP BY lc.surrogate_key, lc.name, lc.content_type
ORDER BY enrollment_count DESC
LIMIT 10"""
            }
        ]


# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def get_llm() -> BaseLanguageModel:
    """Get configured language model instance."""
    manager = get_llm_manager()
    return manager.llm


# Convenience functions
async def generate_sql_query(question: str, schema_description: str) -> str:
    """
    Generate SQL query from natural language question.
    
    Args:
        question: Natural language question
        schema_description: Database schema description
        
    Returns:
        str: Generated SQL query
        
    Raises:
        ValueError: If SQL generation fails
    """
    manager = get_llm_manager()
    examples = manager.get_example_queries()
    
    response = await manager.generate_sql(question, schema_description, examples)
    
    if not response.success:
        raise ValueError(f"SQL generation failed: {response.error}")
    
    return response.content.strip()


async def validate_and_improve_sql(sql_query: str, question: str) -> str:
    """
    Validate and potentially improve SQL query.
    
    Args:
        sql_query: SQL query to validate
        question: Original question
        
    Returns:
        str: Validated/improved SQL query
    """
    manager = get_llm_manager()
    
    response = await manager.validate_sql_response(sql_query, question)
    
    if response.success:
        return response.content.strip()
    else:
        logger.warning(f"SQL validation failed: {response.error}")
        return sql_query  # Return original if validation fails


class PromptTemplate:
    """Template for consistent prompt formatting."""
    
    def __init__(self, template: str, required_vars: List[str]):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with {variable} placeholders
            required_vars: List of required variable names
        """
        self.template = template
        self.required_vars = required_vars
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            str: Formatted prompt
            
        Raises:
            ValueError: If required variables missing
        """
        missing_vars = set(self.required_vars) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)


# Common prompt templates
SQL_GENERATION_TEMPLATE = PromptTemplate(
    template="""Generate a PostgreSQL SELECT query for the following question:

Database Schema:
{schema}

Question: {question}

Requirements:
- Use only SELECT statements
- Include proper JOINs when needed
- Use meaningful aliases
- Return only the SQL query

SQL Query:""",
    required_vars=['schema', 'question']
)

SQL_VALIDATION_TEMPLATE = PromptTemplate(
    template="""Review and validate this SQL query:

Original Question: {question}
SQL Query: {sql_query}

Check for:
- Correct syntax
- Proper table relationships  
- Security (read-only)
- Performance optimization

Return the corrected SQL query or the original if it's correct:""",
    required_vars=['question', 'sql_query']
)
