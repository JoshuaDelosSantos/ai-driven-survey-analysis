"""
Mock responses for testing external dependencies.

Contains predefined responses for mocking database operations,
model predictions, and other external service calls.
"""

# Mock responses for transformers model
MOCK_MODEL_RESPONSES = {
    'positive_text': {
        'logits': [[[-2.5, -1.0, 2.1]]],  # Raw logits: neg, neu, pos
        'softmax_scores': {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}
    },
    'negative_text': {
        'logits': [[[2.1, -1.0, -2.5]]],  # Raw logits: neg, neu, pos
        'softmax_scores': {'neg': 0.7, 'neu': 0.2, 'pos': 0.1}
    },
    'neutral_text': {
        'logits': [[[0.0, 1.0, 0.0]]],    # Raw logits: neg, neu, pos
        'softmax_scores': {'neg': 0.3, 'neu': 0.4, 'pos': 0.3}
    },
    'edge_case_positive': {
        'logits': [[[−3.0, -2.0, 3.0]]],  # Strong positive
        'softmax_scores': {'neg': 0.05, 'neu': 0.15, 'pos': 0.8}
    },
    'edge_case_negative': {
        'logits': [[[3.0, -2.0, -3.0]]],  # Strong negative
        'softmax_scores': {'neg': 0.8, 'neu': 0.15, 'pos': 0.05}
    }
}

# Mock responses for database operations
MOCK_DATABASE_RESPONSES = {
    'successful_insert': {
        'rows_affected': 1,
        'status': 'success',
        'message': 'Record inserted successfully'
    },
    'successful_update': {
        'rows_affected': 1,
        'status': 'success',
        'message': 'Record updated successfully'
    },
    'no_rows_affected': {
        'rows_affected': 0,
        'status': 'warning',
        'message': 'No rows were affected'
    },
    'connection_error': {
        'error': 'ConnectionError',
        'message': 'Unable to connect to database'
    },
    'constraint_violation': {
        'error': 'IntegrityError',
        'message': 'Foreign key constraint violation'
    },
    'table_not_found': {
        'error': 'OperationalError',
        'message': 'Table evaluation_sentiment does not exist'
    }
}

# Mock responses for fetch_data operations
MOCK_FETCH_RESPONSES = {
    'normal_evaluation_data': [
        (1, 'Great course content and delivery!', 'No technical issues encountered', 'Highly recommend to others'),
        (2, 'Course was okay, met expectations', 'Some minor loading delays', 'Average experience overall'),
        (3, 'Excellent learning materials provided', 'Platform worked perfectly', 'Five star experience'),
        (4, 'Poor quality content and presentation', 'Multiple system crashes', 'Would not recommend'),
        (5, 'Good course with practical examples', 'No issues to report', 'Valuable learning experience')
    ],
    'mixed_data_with_nulls': [
        (1, 'Good content', None, 'Recommended'),
        (2, '', 'No issues', ''),
        (3, 'Excellent!', '   ', 'Great course'),
        (4, None, None, None),
        (5, 'Average content', 'Some problems', 'Okay overall')
    ],
    'empty_dataset': [],
    'single_record': [
        (42, 'Single evaluation response', 'No issues reported', 'Good overall experience')
    ],
    'large_dataset': [
        (i, f'Evaluation feedback {i}', f'Issue details {i}', f'General comments {i}')
        for i in range(1, 1001)
    ]
}

# Mock tokenizer responses
MOCK_TOKENIZER_RESPONSES = {
    'short_text': {
        'input_ids': [[101, 2023, 2003, 3191, 102]],
        'attention_mask': [[1, 1, 1, 1, 1]],
        'token_count': 5
    },
    'medium_text': {
        'input_ids': [[101] + list(range(2000, 2050)) + [102]],
        'attention_mask': [[1] * 52],
        'token_count': 52
    },
    'long_text': {
        'input_ids': [[101] + list(range(2000, 2511)) + [102]],  # Max length 512
        'attention_mask': [[1] * 512],
        'token_count': 512,
        'truncated': True
    },
    'empty_text': {
        'input_ids': [[101, 102]],  # Just [CLS] and [SEP] tokens
        'attention_mask': [[1, 1]],
        'token_count': 2
    }
}

# Mock configuration responses
MOCK_CONFIG_RESPONSES = {
    'default_config': {
        'EVALUATION_TABLE': 'evaluation',
        'FREE_TEXT_COLUMNS': ['did_experience_issue_detail', 'course_application_other', 'general_feedback'],
        'MODEL_NAME': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'SENTIMENT_TABLE': 'evaluation_sentiment'
    },
    'test_config': {
        'EVALUATION_TABLE': 'test_evaluation',
        'FREE_TEXT_COLUMNS': ['feedback_1', 'feedback_2'],
        'MODEL_NAME': 'distilbert-base-uncased',
        'SENTIMENT_TABLE': 'test_sentiment'
    }
}

# Mock error responses
MOCK_ERROR_RESPONSES = {
    'model_loading_error': {
        'error_type': 'OSError',
        'message': 'Unable to load model from Hugging Face'
    },
    'tokenizer_error': {
        'error_type': 'TokenizerError',
        'message': 'Invalid tokenizer configuration'
    },
    'analysis_error': {
        'error_type': 'RuntimeError',
        'message': 'Error during sentiment analysis'
    },
    'database_timeout': {
        'error_type': 'TimeoutError',
        'message': 'Database operation timed out'
    },
    'memory_error': {
        'error_type': 'MemoryError',
        'message': 'Insufficient memory for model inference'
    }
}

# Performance benchmarks
MOCK_PERFORMANCE_DATA = {
    'analysis_times': {
        'short_text': 0.05,    # 50ms
        'medium_text': 0.15,   # 150ms
        'long_text': 0.45,     # 450ms
        'batch_10': 0.8,       # 800ms for 10 texts
        'batch_100': 7.5       # 7.5s for 100 texts
    },
    'memory_usage': {
        'model_loading': 512,   # MB
        'single_analysis': 50,  # MB
        'batch_analysis': 200   # MB
    },
    'throughput': {
        'texts_per_second': 12,
        'words_per_second': 150,
        'characters_per_second': 800
    }
}
