# YAMMY Voice Agent

This repository contains the **YAMMY** orchestrator for the Real Estate Voice Agent, integrating:

- Twilio Voice Streams
- Deepgram Speech-to-Text / AssemblyAI Speech-to-Text
- Rasa Dialogue Management
- MoEL Empathy Specialist (optional)
- Sesame CSM Text-to-Speech / ElevenLabs Text-to-Speech
- MCPS (ElevenLabs Agent) Platform Integration


## Project Structure

```
YAMMY/
├── api/                 # FastAPI endpoints, typically aggregated in app.py
│   ├── stt.py           # /api/stt → transcribe audio
│   ├── orchestrate.py   # /api/orchestrate → Rasa NLU & Core
│   ├── tts.py           # /api/tts → TTS service
│   └── mcps.py          # /api/mcps → MCPS agent startup & health (if applicable)
├── app.py               # Main FastAPI application that aggregates API endpoints from api/
├── run_mcps_agent.py    # Script to start MCPS core services (if used)
├── main.py              # Local orchestrator for simulations (Twilio → STT → Rasa → Specialist → TTS), with optional MCPS
├── config.yml           # Application configuration (providers, endpoints, feature flags)
├── requirements.txt     # Python dependencies
├── env.example          # Example environment variables template
├── vercel.json          # Vercel build & route configuration (if deploying to Vercel as serverless functions)
└── README.md            # This file
```


## Environment Variables

Create a `.env` file in the project root directory by copying the template `env.example`.
Populate this `.env` file with your actual API keys and configuration values as needed by the services you intend to use (defined in `config.yml`).
This `.env` file is gitignored by default. The application uses the `python-dotenv` library (make sure it's in `requirements.txt`) to automatically load these variables when it starts, making them available as environment variables.

Example variables from `env.example` (refer to the file for a comprehensive list):
```
DEEPGRAM_API_KEY=
ASSEMBLYAI_API_KEY=
ELEVENLABS_API_KEY=
# ... and many more
```
Ensure all necessary variables required by your chosen services in `config.yml` are set.


## Local Development

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up Environment Variables:**
    Create a `.env` file from `env.example` as described above and fill in your API keys and desired configurations.

3.  **Run MCPS Pipeline (Optional):**
    If you intend to use the MCPS platform features:
    ```bash
    python run_mcps_agent.py
    ```

4.  **Start Local API Server:**
    The main FastAPI application is defined in `app.py`, which typically includes routers for endpoints defined in the `api/` directory. To run it:
    ```bash
    uvicorn app:app --reload --port 8000
    ```
    This makes the API endpoints (like `/api/stt`, `/api/tts`) available at `http://localhost:8000`.

5.  **Run Local Simulations (Optional):**
    To test the core agent logic via simulated calls without an HTTP server:
    ```bash
    python main.py
    ```
    This script runs various agent simulations (e.g., empathetic agent, sales agent) and also uses the environment variables from your `.env` file for service configurations. It will also attempt to start the MCPS agent in a background thread if enabled in `config.yml`.

### Note on STT Provider
The default Speech-to-Text (STT) provider in `config.yml` is currently set to "assemblyai". You can change this to "deepgram" or other configured providers by modifying `config.yml` and ensuring the relevant API keys are set in your `.env` file.


## Testing Endpoints

(Assuming the API server is running on `http://localhost:8000` as per the `uvicorn app:app` command)

- **Health (Example - if `api/mcps.py` is part of `app.py` and has a health check):** `GET http://localhost:8000/api/mcps/health`
- **STT**:
  ```bash
  curl -X POST -F "file=@path/to/audio.wav" http://localhost:8000/api/stt
  ```
- **Orchestrate (Rasa proxy)**:
  ```bash
  curl -X POST http://localhost:8000/api/orchestrate \
    -H "Content-Type: application/json" \
    -d '{"user_id":"u1","text":"Hello","emotion_data":{"dominant_emotion":"neutral"}}'
  ```
- **TTS**:
  ```bash
  curl -X POST http://localhost:8000/api/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello world","voice_profile":"default_professional"}' --output out.mp3
  ```


## Deployment to Vercel

(This section might be outdated if the project is primarily a backend FastAPI app run with Uvicorn/Docker, rather than as individual serverless functions.)

1. Push this repo to GitHub.
2. In Vercel dashboard, import the GitHub project.
3. Set environment variables in Vercel project settings (same keys as .env).
4. Vercel deployment behavior depends on `vercel.json`. If configured for serverless functions in `api/`, they will be deployed. If `app.py` is the main entry point, Vercel might deploy it as a single service.
5. Your endpoints would typically be live at `https://<your-project>.vercel.app/api/...` (for serverless) or `https://<your-project>.vercel.app/...` (for a single app).


---

Feel free to extend or customize any service providers or pipelines—YAMMY is fully modular! Let me know if you need anything else.
