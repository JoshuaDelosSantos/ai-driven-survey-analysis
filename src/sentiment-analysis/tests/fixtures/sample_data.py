"""
Sample data for testing the sentiment analysis module.

Contains various test datasets for different testing scenarios.
"""

# Sample texts for sentiment analysis testing
SAMPLE_TEXTS = {
    'positive': [
        "This course was absolutely fantastic! I learned so much.",
        "Excellent content and great instructor. Highly recommended!",
        "Amazing experience, would definitely take again.",
        "Outstanding course material and engaging presentations.",
        "Perfect balance of theory and practical examples."
    ],
    'negative': [
        "This course was terrible and a waste of time.",
        "Poor quality content with confusing explanations.",
        "Disappointing experience, would not recommend.",
        "Horrible course structure and unhelpful materials.",
        "Worst online course I've ever taken."
    ],
    'neutral': [
        "The course covered the expected topics adequately.",
        "Standard content delivery with average engagement.",
        "Course met basic expectations, nothing extraordinary.",
        "Typical online course format with regular assignments.",
        "Course content was as described in the syllabus."
    ],
    'mixed': [
        "Great content but poor delivery methods.",
        "Interesting topics but technical issues were frustrating.",
        "Good instructor but outdated course materials.",
        "Valuable information but difficult to navigate platform.",
        "Helpful examples but assignments were too easy."
    ]
}

# Edge cases for testing
EDGE_CASES = {
    'empty_strings': ['', '   ', '\n\n', '\t\t'],
    'very_short': ['Ok', 'No', 'Yes', 'Bad', 'Good'],
    'very_long': [
        "This is an extremely long text that goes on and on about various aspects of the course including detailed explanations of every single topic covered in multiple sessions with extensive examples and practical applications that demonstrate the concepts in real-world scenarios while also providing comprehensive background information and historical context that helps students understand the evolution of the subject matter and its current relevance in today's rapidly changing technological landscape.",
        "Another very long review that discusses multiple aspects of the learning experience including the quality of instruction, the relevance of course materials, the effectiveness of teaching methods, the appropriateness of assessment techniques, the usefulness of supplementary resources, the clarity of communication, the responsiveness of support staff, and the overall value proposition of the educational program."
    ],
    'special_characters': [
        "Great course! 👍 Highly recommend 🌟",
        "Poor experience :( Not worth it 👎",
        "Course content: 50% theory + 50% practice = 100% value!",
        "Rating: ★★★★☆ (4/5 stars)",
        "Feedback: A+ content, B- delivery, C+ platform"
    ],
    'multilingual': [
        "Excellent course! Excelente curso! 优秀的课程!",
        "Good content but some parts in español were confusing.",
        "The course covered both English and français concepts well."
    ]
}

# Database test data
DATABASE_SAMPLE_DATA = {
    'evaluation_responses': [
        {
            'response_id': 1,
            'did_experience_issue_detail': 'No major issues, just minor login delays occasionally.',
            'course_application_other': 'Course material was relevant to my current job role.',
            'general_feedback': 'Overall excellent experience with comprehensive content.'
        },
        {
            'response_id': 2,
            'did_experience_issue_detail': 'System crashed multiple times during video playback.',
            'course_application_other': 'Difficult to apply concepts without better examples.',
            'general_feedback': 'Disappointing quality and technical problems throughout.'
        },
        {
            'response_id': 3,
            'did_experience_issue_detail': '',  # Empty response
            'course_application_other': None,   # None response
            'general_feedback': 'Standard course content with average delivery.'
        },
        {
            'response_id': 4,
            'did_experience_issue_detail': 'Excellent platform performance with no technical issues.',
            'course_application_other': 'Highly applicable to my professional development goals.',
            'general_feedback': 'Exceptional learning experience with practical value.'
        }
    ],
    'sentiment_scores': [
        {'neg': 0.1, 'neu': 0.2, 'pos': 0.7},  # Positive
        {'neg': 0.7, 'neu': 0.2, 'pos': 0.1},  # Negative
        {'neg': 0.3, 'neu': 0.4, 'pos': 0.3},  # Neutral
        {'neg': 0.05, 'neu': 0.15, 'pos': 0.8}, # Very positive
        {'neg': 0.85, 'neu': 0.1, 'pos': 0.05}  # Very negative
    ]
}

# Performance test data
PERFORMANCE_TEST_DATA = {
    'small_dataset': [(i, f'Text {i}', f'Issue {i}', f'Feedback {i}') for i in range(1, 11)],
    'medium_dataset': [(i, f'Text {i}', f'Issue {i}', f'Feedback {i}') for i in range(1, 101)],
    'large_dataset': [(i, f'Text {i}', f'Issue {i}', f'Feedback {i}') for i in range(1, 1001)],
}

# Error scenarios
ERROR_SCENARIOS = {
    'invalid_response_ids': [None, 'string', -1, 0, 99999999],
    'malformed_texts': [
        {'text': None, 'expected_error': 'NoneType'},
        {'text': 123, 'expected_error': 'TypeError'},
        {'text': [], 'expected_error': 'TypeError'},
        {'text': {}, 'expected_error': 'TypeError'}
    ],
    'database_errors': [
        'Connection timeout',
        'Table does not exist',
        'Foreign key constraint violation',
        'Duplicate entry error'
    ]
}
