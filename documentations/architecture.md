# MVP 1

## 1. Data
- Mock data is created with the help of Google Gemini 2.5 Pro.
- Mock data are stored in google sheets.

## 1.1 DB Schema
### User Table:
- user_id
- user_level (Level 1-6, Exec Level 1-2)
- agency (Australian Public Service agencies)

### Learning Content Table:
- name
- content_id (course id are not unique to each course, a 'Video' and 'Course' learning content can share the same content id)
- content_type (Live Learning, Video, Course)
- target_level (learning content's intended target audience)
- governing_bodies (learning content's governing bodies)
- surrogate_key (from natural key of content_id and content_type)

### Attendance Table:
- user_id
- learning_content.surrgorate_key
- date_start
- date_end
- status (Enrolled, In-progress, Completed, Withdrew)

### Learning Content Evaluation Table:
- response_id
- course_end_date
- learning_content.surrogate_key
- user_id
- course_delivery_type (single choice)
    -  In-person
    -  Virtual
    -  Blended
- agency (an Australian Public Service agency)
- attendance_motivation (multi choice)
    1. To improve my performance in my current role
    2. To develop for a future role or career progression
    3. Recommendation from others who have attended
    4. My supervisor/manager encouraged me to attend
    5. My attendance was mandatory
    6. This course is a part of a broader learning pathway or program I am participating in
- positive_learning_experience (likert scale)
- effective_use_of_time (likert scale)
- relevant_to_work (likert scale)
- did_experience_issue (multi choice)
    1. Technical difficulties negatively impacted my participation
    2. Learning resources, such as workbooks or slides, were poorly designed or inconsistent with each other
    3. Some of the course content was factually incorrect or out of date
    4. One or more elements of the course did not meet my accessibility needs and no suitable alternative was provided
    5. None of the above
- did_experience_issue_detail (free text - if not choice #5 is selected)
- facilitator_skills (multi choice)
    1. Displayed strong knowledge of the subject matter
    2. Communicated clearly
    3. Encouraged participation and discussion
    4. Provided clear responses to learner questions
    5. Managed time and pacing
    6. Used examples that are relevant to my context
    7. None of the above
- had_guest_speakers
    - Yes
    - No
- guest_contribution (multi choice - if yes to guest speakers)
    1. Strengthened my understanding of they key concepts
    2. Enhanced my understanding of how learning is relevant to my work context
    3. Provided insights into the challenges and barriers I may face
    4. Gave me confidence I will be able to successfully apply the learning
    5. Brought specialist knowledge or expertise
    6. The contributions of the guest speakers or presenters did not enhance my learning
- knowledge_level_prior
    1. Novice
    2. Beginner
    3. Early Practitioner
    4. Experienced Practitioner
    5. Expert
- course_aaplication (multi choice)
    1. During the course I had the opportunity to practice new skills
    2. The course has prompted me to reflect in how I will do things differently
    3. I have taken away useful tools or resources
    4. I have increased my confidence in my ability to use what I have learned
    5. I can put what I have learned into my own words
    6. I have a plan on how I will apply what I have learned
    7. I am still unsure how to apply what I have learned
    8. Other
- course_application_other (free text - if 'other' is selected)
- course_application_timeframe (single choice)
    1. Immediately
    2. Within the next month
    3. Within the next 1-3 months
    4. Longer term
    5. I am not sure
    6. I do not intent to apply anything covered in this course
    7. I will not have the opportunity to apply anything covered in this course
- general_feedback (free text)


## 2. Local setup
- Python virtual environment is used to contain dependencies
- Dependencies are stored in requirements.txt
- Docker compose is used for services, currently it has these services:
    - Database: pgvector
- PostgreSQL for database client (VS Code extension)

## 3. Sentiment Analysis
### Scripts:
- db_connector.py         # Handles PostgreSQL connection and operations
- data_loader.py          # Extracts data from PostgreSQL
- text_preprocessor.py    # (Optional) Cleans text before tokenization
- sentiment_analyzer.py   # Performs sentiment analysis with RoBERTa
- results_saver.py        # Saves sentiment results to PostgreSQL
- main_workflow.py        # Orchestrates the entire process

#### 1. scripts/db_connector.py (Database Utility)
Purpose: 
- Provides reusable functions to connect to and interact with your PostgreSQL database.
Key Functions:
- get_db_connection(): Reads credentials and establishes a connection to PostgreSQL. Returns a connection object.
- close_db_connection(connection, cursor=None): Closes the database connection and cursor.
- fetch_data(query, params=None, connection=None): Executes a SELECT query and returns the results (e.g., as a list of tuples or dictionaries).
- execute_query(query, params=None, connection=None): Executes INSERT, UPDATE, or DELETE queries.
- batch_insert_data(query, data_list, connection=None): Efficiently inserts multiple rows of data.

#### 2. scripts/data_loader.py (Data Extraction)
Purpose: 
- Fetches the relevant free-text data from the evaluation table in PostgreSQL.
Logic:
- Imports functions from db_connector.py.
- Establishes a database connection.
- Constructs a SQL query to select:
    - response_id (or your primary key for the evaluation table).
    - The free-text columns you want to analyze (e.g., did_experience_issue_detail, course_application_other, general_feedback).
    - Optionally, add a condition to select only rows where sentiment has not yet been analyzed (e.g., WHERE sentiment_analyzed_timestamp IS NULL).
- Uses db_connector.fetch_data() to retrieve the data, preferably into a Pandas DataFrame for ease of handling.
- Closes the database connection.
- Returns the DataFrame.
- Example Function: load_evaluation_texts_for_analysis()

#### 3. scripts/text_preprocessor.py (Optional Text Cleaning)
Purpose: 
- Performs any necessary text cleaning before RoBERTa's tokenizer. Often, RoBERTa's tokenizer is robust, but you might want to include steps like:
    - Normalizing whitespace.
    - Removing or replacing specific custom artifacts not well-handled by the tokenizer.
Note: 
- Extensive preprocessing like stop-word removal or stemming is generally NOT recommended for transformer models like RoBERTa.
Logic:
- Takes a DataFrame with text columns as input.
- Applies cleaning functions to the specified text columns.
- Returns the DataFrame with cleaned text.
- Example Function: clean_text_column(text_series)

#### 4. scripts/sentiment_analyzer.py (Sentiment Analysis Core)
Purpose: 
- Loads the RoBERTa model and tokenizer, and performs sentiment analysis on the input texts.
Key Components & Logic:
- Model & Tokenizer Loading:
    - Imports RobertaTokenizer, RobertaForSequenceClassification from transformers (or pipeline for a higher-level abstraction).
    - Specifies the RoBERTa model. A good starting point for general sentiment is "cardiffnlp/twitter-roberta-base-sentiment-latest" or "siebert/sentiment-roberta-large-english".
    - Loads the pre-trained model and its corresponding tokenizer.
Sentiment Prediction:
- Takes a list of texts (or a DataFrame column) as input.
Tokenization: 
- Converts texts into token IDs that RoBERTa understands, handling padding and truncation. RoBERTa has a maximum sequence length (e.g., 512 tokens). Texts longer than this need a strategy (truncation, summarization, or chunking).
Inference: 
- Passes the tokenized input to the model for prediction. Process in batches for efficiency.
Output Processing: 
- Converts model outputs (logits) into human-readable sentiment labels (e.g., "Positive", "Negative", "Neutral") and confidence scores. The cardiffnlp model, for instance, outputs these directly.
- Returns a list of dictionaries or a DataFrame containing the original identifier, the text column name, the predicted sentiment label, and the sentiment score.
Example Functions:
- load_roberta_sentiment_pipeline(model_name): Loads a Hugging Face sentiment analysis pipeline.
- predict_sentiment(texts, pipeline): Uses the pipeline to predict sentiment for a batch of texts.

#### 5. scripts/results_saver.py (Storing Results)
Purpose: 
- Saves the sentiment analysis results back into PostgreSQL.
Logic:
- Imports functions from db_connector.py.
- Establishes a database connection.
- Takes the sentiment results (e.g., DataFrame with response_id, text_column_name, sentiment_label, sentiment_score) as input.
Recommended Approach: 
- Insert results into a new dedicated table, e.g., evaluation_sentiments.
- evaluation_sentiments table schema example:
    - sentiment_id (SERIAL PRIMARY KEY)
    - evaluation_response_id (INTEGER, FOREIGN KEY to your evaluation table)
    - text_column_evaluated (TEXT, e.g., 'general_feedback')
    - sentiment_label (TEXT, e.g., 'Positive', 'Negative', 'Neutral')
    - sentiment_score (FLOAT)
    - analyzed_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
Constructs the SQL INSERT query.
Uses db_connector.batch_insert_data() for efficient insertion.
Closes the database connection.
Example Function: save_sentiment_results_to_db(sentiment_data)

#### 6. scripts/main_workflow.py (Orchestrator)
Purpose: 
- Coordinates the execution of the entire workflow from data loading to saving results.
Logic:
- Imports necessary functions from all other scripts (data_loader, text_preprocessor (if used), sentiment_analyzer, results_saver).
Configuration: 
- Loads database and model configurations.
Load Data: 
- Calls data_loader.load_evaluation_texts_for_analysis().
Iterate through Free-Text Columns: 
- For each specified free-text column (e.g., did_experience_issue_detail, general_feedback):
- Extract the relevant text series and their response_ids.
- (Optional) Preprocess texts using text_preprocessor.
- Perform sentiment analysis using sentiment_analyzer.predict_sentiment().
- Structure the results with response_id and the name of the column analyzed.
- Save the results for this column using results_saver.save_sentiment_results_to_db().
Includes logging for each step (e.g., number of rows processed, time taken).
Error handling (try-except blocks).