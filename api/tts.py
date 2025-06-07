from fastapi import FastAPI, Request, Response, HTTPException
import yaml
from response_generation_services import TextToSpeechService, TTSError
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys # For stderr
# config_utils might not be needed if TextToSpeechService handles defaults for voice/emotion robustly
# from config_utils import resolve_config_value

app = FastAPI()

# Basic logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("FATAL: config.yml not found. TTS API cannot start correctly.")
    config = {}
except Exception as e:
    logger.error(f"FATAL: Error loading config.yml: {e}. TTS API cannot start correctly.")
    config = {}

# Initialize TTS Service
# TextToSpeechService expects the full config object.
try:
    tts_service = TextToSpeechService(config=config)
    logger.info("TextToSpeechService initialized successfully for /api/tts.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize TextToSpeechService: {e}. /api/tts will not work.", exc_info=True)
    tts_service = None # Make it unavailable if init fails

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/tts")
async def tts(request: Request):
    if not tts_service:
        logger.error("/api/tts called but TTS service is not available due to initialization failure.")
        raise HTTPException(status_code=503, detail="TTS Service is unavailable.")

    try:
        payload = await request.json()
        text = payload.get("text")

        if text is None: # Ensure 'text' field is present
            logger.warning("/api/tts request missing 'text' field in payload.")
            raise HTTPException(status_code=400, detail="Missing 'text' field in payload.")

        # Extract voice_profile and emotion_hint.
        # The TextToSpeechService is designed to handle None for these and use its internal defaults.
        voice_profile = payload.get("voice_profile")
        emotion_hint = payload.get("emotion_hint")

        logger.info(f"/api/tts: Received request: text snippet='{text[:50]}...', "
                    f"voice_profile='{voice_profile}', emotion_hint='{emotion_hint}'")

        audio = tts_service.synthesize_speech(
            text_input=text,
            voice_profile=voice_profile,
            emotion_hint=emotion_hint
        )

        if not audio: # Should be handled by TTSError, but as a safeguard
            logger.error("/api/tts: TTS service returned empty audio without raising TTSError.")
            raise HTTPException(status_code=500, detail="TTS Service generated empty audio.")

        logger.info(f"/api/tts: Successfully synthesized audio. Audio size: {len(audio)} bytes.")
        return Response(content=audio, media_type="audio/mpeg") # Assuming MPEG, adjust if different (e.g. audio/wav)
    except TTSError as e:
        logger.error(f"TTSError in /api/tts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS Service Error: {str(e)}")
    except HTTPException as he: # Re-raise HTTPExceptions we threw (e.g. missing text)
        raise he
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in /api/tts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run this app (for testing this file directly):
# uvicorn api.tts:app --reload --port 8002
# Then send a POST request like:
# curl -X POST http://localhost:8002/api/tts -H "Content-Type: application/json" -d '{"text": "Hello world"}'
# curl -X POST http://localhost:8002/api/tts -H "Content-Type: application/json" -d '{"text": "Hello world", "voice_profile": "some_voice_if_needed", "emotion_hint": "neutral"}'
