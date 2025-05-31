from fastapi import FastAPI, UploadFile, File
import yaml
from input_processing_services import SpeechToTextService

app = FastAPI()

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
stt_service = SpeechToTextService(config)

@app.post("/api/stt")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    text = stt_service.transcribe_audio_chunk(audio)
    return {"text": text} 