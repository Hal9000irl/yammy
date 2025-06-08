import os
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Attempt to import RasaService, handle if not available for robustness
try:
    from dialogue_manager_service import RasaService
except ImportError:
    RasaService = None
    print("ERROR: api/orchestrate.py: dialogue_manager_service.py not found or RasaService cannot be imported. Orchestration will likely fail.")

app = FastAPI()

# Determine project root assuming this script is in api/
# and config.yml is in the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yml")

config = {} # Default to empty config
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    if config is None: # Handles empty config file
        print(f"WARNING: api/orchestrate.py: Configuration file {CONFIG_PATH} is empty. Using default empty config.")
        config = {}
    else:
        print(f"INFO: api/orchestrate.py: Configuration loaded successfully from {CONFIG_PATH}")
except FileNotFoundError:
    print(f"ERROR: api/orchestrate.py: Configuration file not found at {CONFIG_PATH}. Using default empty config.")
    # config remains {}
except Exception as e:
    print(f"ERROR: api/orchestrate.py: Could not load or parse configuration file {CONFIG_PATH}: {e}. Using default empty config.")
    # config remains {}

# Initialize RasaService, handling potential failure if class wasn't imported or config is bad
if RasaService:
    try:
        rasa_service = RasaService(config=config if config else {}) # Ensure config is a dict
    except Exception as e:
        print(f"ERROR: api/orchestrate.py: Failed to initialize RasaService: {e}. Endpoint will not function correctly.")
        rasa_service = None # Set to None if initialization fails
else:
    rasa_service = None # Set to None if RasaService class itself is None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/") # Corrected route
async def orchestrate(request: Request):
    if not rasa_service:
        print("ERROR: api/orchestrate.py: RasaService not initialized. Cannot process request.")
        # Consider returning a 503 Service Unavailable status code
        return {"error": "Service unavailable: Dialogue manager not configured or failed to initialize.", "action_plan": {}}

    payload = await request.json()
    user_id = payload.get("user_id")
    text = payload.get("text")
    emotion_data = payload.get("emotion_data")

    try:
        result = rasa_service.process_user_message(user_id, text, emotion_data)
        return result
    except Exception as e:
        print(f"ERROR: api/orchestrate.py: Error during RasaService.process_user_message: {e}")
        # Consider returning a 500 Internal Server Error status code
        return {"error": "An internal error occurred while processing the message.", "action_plan": {}}
