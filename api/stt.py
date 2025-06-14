from fastapi import FastAPI, UploadFile, File
import yaml
from input_processing_services import SpeechToTextService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
stt_service = SpeechToTextService(config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/stt")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    text = stt_service.transcribe_audio_chunk(audio)
    return {"text": text} 