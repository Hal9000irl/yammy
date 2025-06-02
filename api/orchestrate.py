from fastapi import FastAPI, Request
import yaml
from dialogue_manager_service import RasaService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
rasa_service = RasaService(config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/orchestrate")
async def orchestrate(request: Request):
    payload = await request.json()
    user_id = payload.get("user_id")
    text = payload.get("text")
    emotion_data = payload.get("emotion_data")
    result = rasa_service.process_user_message(user_id, text, emotion_data)
    return result 