# CSV Data Directory

This directory contains source CSV files and metadata for the AI-driven analysis project, supporting database population and machine learning workflows.

## Current Files

- **user.csv**: User metadata including user_id, user_level, and agency information
- **learning_content.csv**: Learning module data with content details, types, and governing bodies
- **evaluation.csv**: Survey evaluation responses containing both structured and free-text feedback
- **attendance.csv**: Attendance logs for learning sessions and events
- **data-dictionary.json**: Comprehensive schema definitions and column descriptions for all CSV files

## Key Features

### Free-Text Fields for RAG Analysis
The evaluation dataset contains three key free-text fields used for semantic analysis:
- **`did_experience_issue_detail`**: Detailed technical issue descriptions
- **`course_application_other`**: Application plans and implementation strategies
- **`general_feedback`**: Overall course feedback and suggestions

These fields are processed by the RAG (Retrieval-Augmented Generation) module for embedding generation and semantic search capabilities.

### Data Dictionary Integration
The `data-dictionary.json` provides:
- Complete column specifications for all datasets
- Data type definitions and constraints
- Foreign key relationships between tables
- Free-text field identification for ML processing

## Usage Guidelines

### For Database Scripts
- Scripts in `../db/` reference these CSV files for table population
- Column names must match corresponding table creation scripts
- Foreign key relationships are defined in the data dictionary

### For Analysis Modules
- `../sentiment-analysis/` processes evaluation free-text content
- `../rag/` creates embeddings from specified free-text fields
- Maintain consistent field names across all processing scripts

## Best Practices

- **File Integrity**: Do not modify CSV filenames without updating dependent scripts
- **Schema Synchronisation**: Keep `data-dictionary.json` aligned with actual CSV headers
- **Data Governance**: Follow Australian Privacy Principles when handling personal information
- **Version Control**: Commit data responsibly, considering file size limitations

---
**Last Updated**: 17 June 2025
