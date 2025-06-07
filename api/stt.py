from fastapi import FastAPI, UploadFile, File, HTTPException
import yaml
from input_processing_services import SpeechToTextService, STTError
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys # For stderr

app = FastAPI()

# Basic logging configuration
# In a production app, you'd likely use a more robust logging setup (e.g., from main app)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load configuration
try:
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("FATAL: config.yml not found. STT API cannot start correctly.")
    # In a real scenario, you might want the app to not start or have very limited functionality.
    config = {} # Provide an empty config to allow app to load but services might fail.
except Exception as e:
    logger.error(f"FATAL: Error loading config.yml: {e}. STT API cannot start correctly.")
    config = {}

# Initialize STT Service
# SpeechToTextService expects the full config object, it will extract its own section.
try:
    stt_service = SpeechToTextService(config=config)
    logger.info("SpeechToTextService initialized successfully for /api/stt.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize SpeechToTextService: {e}. /api/stt will not work.", exc_info=True)
    # Define a fallback stt_service or ensure endpoints handle its absence.
    # For now, if it fails, endpoints will likely raise errors.
    stt_service = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

@app.post("/api/stt")
async def transcribe(file: UploadFile = File(...)):
    if not stt_service:
        logger.error("/api/stt called but STT service is not available due to initialization failure.")
        raise HTTPException(status_code=503, detail="STT Service is unavailable.")

    try:
        audio_content = await file.read()
        if not audio_content:
            logger.warning("/api/stt received an empty audio file.")
            raise HTTPException(status_code=400, detail="Empty audio file received.")

        logger.info(f"/api/stt: Received audio file of size {len(audio_content)} bytes.")
        text = stt_service.transcribe_audio_chunk(audio_content)
        logger.info(f"/api/stt: Successfully transcribed audio. Text snippet: '{text[:50]}...'")
        return {"text": text}
    except STTError as e:
        logger.error(f"STTError in /api/stt for file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT Service Error: {str(e)}")
    except HTTPException as he: # Re-raise HTTPExceptions we threw (e.g. empty file)
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in /api/stt for file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run this app (for testing this file directly):
# uvicorn api.stt:app --reload --port 8001
# Then send a POST request with a file to http://localhost:8001/api/stt
