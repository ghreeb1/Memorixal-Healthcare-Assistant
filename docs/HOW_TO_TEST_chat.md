How to test chat stability fixes:

1) Start the app:
   - Windows PowerShell:
     python -m uvicorn main:app --host localhost --port 8000 --reload

2) Verify safe fallback when RAG is not initialized:
   - POST http://localhost:8000/api/chat with JSON {"message":"hello"}
   - Expect a streamed response containing a supportive, non-medical reply.

3) Verify temperature clamping:
   - POST /api/chat with {"message":"hello","temperature":0.9}
   - Should clamp to 0.6 internally and still return a proper response (or the safe fallback if no model).

4) Audio chat fallback:
   - POST /api/audio-chat with a small audio file. If STT/LLM not available, expect a friendly safe reply JSON instead of a 500.

5) Run unit tests:
   - pytest -q
