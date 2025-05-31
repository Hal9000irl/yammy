from fastapi import FastAPI, Request, Response
import yaml
from response_generation_services import TextToSpeechService

app = FastAPI()

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
tts_service = TextToSpeechService(config)

@app.post("/api/tts")
async def tts(request: Request):
    payload = await request.json()
    text = payload.get("text", "")
    voice = payload.get("voice_profile", "default_professional")
    emotion = payload.get("emotion_hint")
    audio = tts_service.synthesize_speech(text, voice_profile=voice, emotion_hint=emotion)
    return Response(content=audio, media_type="audio/mpeg") 