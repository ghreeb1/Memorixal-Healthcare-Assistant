How to test feature usage tracking:

1) Start the app:
   python -m uvicorn main:app --host localhost --port 8000 --reload

2) Use the system (dashboard, chatbot, games). Each should call /api/v1/feature-usage/start and /end.

3) Inspect SQLite directly (optional):
   - SELECT feature_name, user_id, start_time, end_time, duration_seconds FROM feature_usage ORDER BY id DESC LIMIT 10;

4) API checks:
   - POST /api/v1/feature-usage/start {"user_id":1,"feature_name":"x"}
   - POST /api/v1/feature-usage/end {"usage_id":<id>}
