How to test games integration:

1) Start the app:
   python -m uvicorn main:app --host localhost --port 8000 --reload

2) Open http://localhost:8000/games/memory_card?patientId=1
   - Play until you win.
   - On the win modal, results are automatically POSTed to /api/v1/patients/1/activities.
   - Retry logic handles transient failures (check console for warnings).

3) Verify log:
   - GET http://localhost:8000/api/v1/patients/1/activities
   - The 'games' activity should reflect increased sessions.

4) Payload includes rows/cols/moves/time in `results` JSON.
