import pytest
from httpx import AsyncClient

from api.stt import app as stt_app
from api.orchestrate import app as orchestrate_app
from api.tts import app as tts_app
from api.mcps import app as mcps_app

@pytest.mark.asyncio
async def test_health_mcps():
    async with AsyncClient(app=mcps_app, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "mcps running"}

@pytest.mark.asyncio
async def test_stt_empty_file():
    async with AsyncClient(app=stt_app, base_url="http://test") as ac:
        files = {"file": ("empty.wav", b"")}
        response = await ac.post("/api/stt", files=files)
        assert response.status_code == 200
        json_data = response.json()
        assert "text" in json_data

@pytest.mark.asyncio
async def test_orchestrate_unknown():
    async with AsyncClient(app=orchestrate_app, base_url="http://test") as ac:
        payload = {"user_id": "u1", "text": "", "emotion_data": {"dominant_emotion": "neutral"}}
        response = await ac.post("/api/orchestrate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data and "entities" in data and "next_specialist" in data

@pytest.mark.asyncio
async def test_tts_simulated():
    async with AsyncClient(app=tts_app, base_url="http://test") as ac:
        payload = {"text": "Hello", "voice_profile": "default", "emotion_hint": "neutral"}
        response = await ac.post("/api/tts", json=payload)
        assert response.status_code == 200
        content = response.content
        # Either real audio or simulated bytes
        assert content.startswith(b"simulated_audio_bytes_for_[Hello") or response.headers.get("content-type", "").startswith("audio/") 