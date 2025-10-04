How to test dashboard fixes:

1) Start the app:
   python -m uvicorn main:app --host localhost --port 8000 --reload

2) Open http://localhost:8000/dashboard?patientId=1
   - Activity cards, recent activities, and AI insights should load.
   - Empty states and spinners appear while loading.

3) Patient-aware endpoints:
   - GET http://localhost:8000/api/v1/patients/1/dashboard
   - GET http://localhost:8000/api/v1/patients/1/ai-insights
   - GET http://localhost:8000/api/v1/patients/1/activities

4) Feature usage tracking:
   - Open dashboard and then close tab.
   - Check SQLite: SELECT * FROM feature_usage ORDER BY id DESC LIMIT 5; You should see entries for feature_name = 'dashboard'.

5) Frontend fetch URLs use the v1 routes (inspect Network tab).
