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
