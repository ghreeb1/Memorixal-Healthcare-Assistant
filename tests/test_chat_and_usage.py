import pytest
from fastapi.testclient import TestClient

# Robust import of main
try:
    import main  # type: ignore
except Exception:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import main  # type: ignore

client = TestClient(main.app)


def read_streaming_text(resp):
    chunks = []
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk.decode('utf-8', errors='ignore'))
    return ''.join(chunks)


def test_chat_fallback_streaming_safe_reply():
    resp = client.post('/api/chat', json={'message': 'Hello there'})
    assert resp.status_code == 200
    # Because it's StreamingResponse, collect body
    text = read_streaming_text(resp)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "help" in text.lower()


def test_feature_usage_start_end():
    # start
    start_resp = client.post('/api/v1/feature-usage/start', json={
        'user_id': 1,
        'feature_name': 'test_feature',
        'metadata': {'case': 'unit'}
    })
    assert start_resp.status_code == 200
    usage_id = start_resp.json().get('usage_id')
    assert isinstance(usage_id, int)

    # end
    end_resp = client.post('/api/v1/feature-usage/end', json={
        'usage_id': usage_id,
        'metadata': {'finished': True}
    })
    assert end_resp.status_code == 200
    data = end_resp.json().get('usage')
    assert data is not None
    assert data.get('duration_seconds') is not None
