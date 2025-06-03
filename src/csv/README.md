# CSV Data Directory

This folder holds all source CSV files and related metadata for the AI-Driven Analysis project.

Files:
- **user.csv** : User metadata (user_id, user_level, agency).
- **learning_content.csv** : Learning modules data (surrogate_key, name, content_id, content_type, target_level, governing_bodies).
- **evaluation.csv** : Survey evaluation responses for sentiment analysis.
- **attendance.csv** : Attendance logs for sessions or events.
- **data-dictionary.json** : Schema definitions and column descriptions for all CSV files.

Usage:
- Scripts in `../db/` and `../sentiment-analysis/` reference these CSV files via relative paths.
- Update or add new CSVs here; ensure column names match the corresponding table-creation scripts.

Best Practices:
- Do not modify CSV file names without updating script paths.
- Keep `data-dictionary.json` in sync with actual headers.
- Commit raw data responsibly; consider large-file storage if needed.
