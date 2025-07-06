"""
Conversational Intelligence Module for RAG Agent

This module provides conversational capabilities for the RAG system, enabling
natural interactions with users through greetings, system questions, and
off-topic query handling with Australian-friendly responses.

The conversational handler uses pattern recognition, Australian context, and
simple pattern learning to provide friendly, professional responses while
maintaining focus on the system's data analysis capabilities.
"""

from .handler import ConversationalHandler, ConversationalResponse, ConversationalPattern

__all__ = ['ConversationalHandler', 'ConversationalResponse', 'ConversationalPattern']

__all__ = ['ConversationalHandler', 'ConversationalPattern', 'ConversationalResponse']
