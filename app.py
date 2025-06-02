from fastapi import FastAPI
from api import stt, orchestrate, tts, mcps

app = FastAPI()

app.mount("/api/stt", stt.app)
app.mount("/api/orchestrate", orchestrate.app)
app.mount("/api/tts", tts.app)
app.mount("/api/mcps", mcps.app)