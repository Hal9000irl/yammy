# YAMMY Voice Agent

This repository contains the **YAMMY** orchestrator for the Real Estate Voice Agent, integrating:

- Twilio Voice Streams
- Deepgram Speech-to-Text
- Rasa Dialogue Management
- MoEL Empathy Specialist (optional)
- Sesame CSM Text-to-Speech
- MCPS (ElevenLabs Agent) Platform Integration


## Project Structure

```
YAMMY/
├── api/                 # Vercel serverless functions
│   ├── stt.py           # /api/stt → transcribe audio
│   ├── orchestrate.py   # /api/orchestrate → Rasa NLU & Core
│   ├── tts.py           # /api/tts → Sesame CSM TTS
│   └── mcps.py          # /api/mcps → MCPS agent startup & health
├── run_mcps_agent.py    # Starts MCPS core services (Ingestion + Dispatch)
├── main.py              # Local orchestrator (Twilio → STT → Rasa → Specialist → TTS), with optional MCPS
├── config.yml           # Application configuration (providers, endpoints, feature flags)
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel function build & route configuration
└── README.md            # This file
```


## Environment Variables

Create a `.env` file in this folder (and add to `.gitignore`) with:

```
DEEPGRAM_API_KEY=your_deepgram_api_key
SESAME_CSM_URL=http://localhost:5001
ELEVENLABS_API_KEY=your_elevenlabs_api_key
RASA_URL=http://localhost:5005
TWILIO_ACCOUNT_SID=ACxxx
TWILIO_AUTH_TOKEN=xxx
```

Any other secrets (e.g. GCP, AssemblyAI) should also be added here.


## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Load environment variables:
   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

3. Run MCPS pipeline (optional):
   ```bash
   python run_mcps_agent.py
   ```

4. Start local API endpoints using Vercel CLI (recommended):
   ```bash
   npm install -g vercel
   vercel dev
   ```

   Alternatively, run individual endpoints with Uvicorn:
   ```bash
   uvicorn api.stt:app --port 8001
   uvicorn api.orchestrate:app --port 8002
   uvicorn api.tts:app --port 8003
   uvicorn api.mcps:app --port 8004
   ```


## Testing Endpoints

- **Health**: `GET http://localhost:3000/api/mcps` should return `{ "status": "mcps running" }`
- **STT**: 
  ```bash
  curl -X POST -F "file=@path/to/audio.wav" http://localhost:3000/api/stt
  ```
- **Orchestrate**:
  ```bash
  curl -X POST http://localhost:3000/api/orchestrate \
    -H "Content-Type: application/json" \
    -d '{"user_id":"u1","text":"Hello","emotion_data":{"dominant_emotion":"neutral"}}'
  ```
- **TTS**:
  ```bash
  curl -X POST http://localhost:3000/api/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello world","voice_profile":"default_professional"}' --output out.mp3
  ```


## Deployment to Vercel

1. Push this repo to GitHub.
2. In Vercel dashboard, import the GitHub project.
3. Set environment variables in Vercel project settings (same keys as .env).
4. Vercel will automatically detect the `api/` functions and deploy them.
5. Your endpoints will be live at `https://<your-project>.vercel.app/api/...`.


---

Feel free to extend or customize any service providers or pipelines—YAMMY is fully modular! Let me know if you need anything else. 