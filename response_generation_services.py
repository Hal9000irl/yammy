# response_generation_services.py
# Contains services for NLG and TTS.

import time # For simulation
import os
import requests
import logging

from config_utils import resolve_config_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSError(Exception):
    """Custom exception for TextToSpeechService errors."""
    pass

class NaturalLanguageGenerationService:
    """
    Generates human-like text responses, often using an LLM.
    """
    def __init__(self, config: dict): # Expects global config
        self.service_config = config.get('natural_language_generation_service', {})
        self.provider = self.service_config.get('provider', 'local_llama')
        # Get the settings for the chosen provider
        provider_settings_key = f"{self.provider}_settings"
        self.settings = self.service_config.get(provider_settings_key, {})

        # Resolve provider-specific settings using resolve_config_value
        if self.provider == "local_llama":
            raw_model_path = self.settings.get('model_path') # Path from config.yml
            # Default path if placeholder or config value is missing/unresolved
            default_llama_path = "/path/to/your/default_llama_model.gguf"
            self.settings['model_path'] = resolve_config_value(raw_model_path, default_llama_path)
        elif self.provider == "openai_gpt":
            raw_api_key = self.settings.get('api_key')
            self.settings['api_key'] = resolve_config_value(raw_api_key, "")
            self.settings['model_name'] = self.settings.get('model_name', 'gpt-3.5-turbo')
        elif self.provider == "anthropic_claude":
            raw_api_key = self.settings.get('api_key')
            self.settings['api_key'] = resolve_config_value(raw_api_key, "")
            self.settings['model_name'] = self.settings.get('model_name', 'claude-2')

        logger.info(f"NaturalLanguageGenerationService Initialized (Provider: {self.provider}, Settings: {self.settings})")

    def generate_text_response(self, prompt: str, context_data: dict = None) -> str:
        logger.info(f"NLGService ({self.provider}): Generating text for prompt: '{prompt[:100]}...'")
        if "clarify" in prompt.lower():
            return f"NLG Simulated Response: I understand you mentioned '{context_data.get('last_user_utterance', 'something') if context_data else 'something'}'. Could you please elaborate a bit more on that?"
        elif "summarize" in prompt.lower():
            return f"NLG Simulated Response: To summarize, we discussed {context_data.get('key_topics', ['several important points']) if context_data else ['several important points']}."
        return f"NLG Simulated Response: Based on your query about '{prompt[:30]}...', I'd be happy to provide more information. What specifically are you interested in?"

class TextToSpeechService:
    def __init__(self, config: dict): # Expects global config
        self.service_config = config.get('text_to_speech_service', {})
        self.provider = self.service_config.get('provider', 'sesame_csm')
        self.settings = self.service_config.get(f"{self.provider}_settings", {})
        logger.info(f"TextToSpeechService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        self.resolved_elevenlabs_api_key = None
        self.resolved_sesame_csm_url = None

        if self.provider == "elevenlabs":
            self._initialize_elevenlabs_key()
        elif self.provider == "sesame_csm":
            self._initialize_sesame_csm_url()

    def _initialize_elevenlabs_key(self):
        config_api_key = self.settings.get("api_key")
        # Use resolve_config_value, which handles placeholder and env var.
        # The refined warning logic from subtask 12 can be mostly superseded by resolve_config_value's behavior
        # if we ensure resolve_config_value itself logs appropriately or if we add specific warnings here.
        # For now, relying on resolve_config_value's core logic.
        self.resolved_elevenlabs_api_key = resolve_config_value(config_api_key, "") # Default to empty if not found

        if not self.resolved_elevenlabs_api_key:
             # Try standard env var as a final fallback if placeholder didn't resolve and no direct value
            standard_env_key = "ELEVENLABS_API_KEY"
            env_api_key_value = os.getenv(standard_env_key)
            if env_api_key_value:
                logger.info(f"Using ElevenLabs API key from environment variable {standard_env_key} as placeholder/config was not resolved.")
                self.resolved_elevenlabs_api_key = env_api_key_value
            else:
                logger.error("ElevenLabs API key could not be resolved from config, placeholder, or standard environment variable.")

        if self.resolved_elevenlabs_api_key and not (config_api_key and config_api_key.startswith("${")):
            # This condition means a key was resolved, but it wasn't from a placeholder in the config.
            # It could be a hardcoded key from config or from the standard ELEVENLABS_API_KEY env var
            # when no placeholder was in config.
            if config_api_key == self.resolved_elevenlabs_api_key: # Key came from non-placeholder config
                 logger.warning("Using a non-placeholder API key found in the configuration for ElevenLabs. Prefer placeholders.")
            # If resolved_elevenlabs_api_key came from standard env var (because config_api_key was empty), that's fine.


    def _initialize_sesame_csm_url(self):
        config_service_url = self.settings.get("service_url")
        self.resolved_sesame_csm_url = resolve_config_value(config_service_url, "")

        if not self.resolved_sesame_csm_url:
            standard_env_key = "SESAME_CSM_URL"
            env_url_value = os.getenv(standard_env_key)
            if env_url_value:
                logger.info(f"Using Sesame CSM URL from environment variable {standard_env_key} as placeholder/config was not resolved.")
                self.resolved_sesame_csm_url = env_url_value
            else:
                logger.error("Sesame CSM Service URL could not be resolved from config, placeholder, or standard environment variable.")

        if self.resolved_sesame_csm_url and not (config_service_url and config_service_url.startswith("${")):
            if config_service_url == self.resolved_sesame_csm_url:
                 logger.warning("Using a non-placeholder service URL found in the configuration for Sesame CSM. Prefer placeholders.")


    def synthesize_speech(self, text_input: str, voice_profile: str = "default_professional", emotion_hint: str = None) -> bytes:
        effective_emotion = emotion_hint if emotion_hint else "neutral"
        logger.info(f"TTSService ({self.provider}): Synthesizing speech for: '{text_input}' (Voice: {voice_profile}, Emotion Hint: {effective_emotion})")

        if self.provider == "elevenlabs":
            if not self.resolved_elevenlabs_api_key:
                logger.error("ElevenLabs API key not found/resolved. Cannot synthesize speech.")
                raise TTSError("ElevenLabs API key is not configured.")
            voice_id = self.settings.get("default_voice_id", voice_profile)
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": self.resolved_elevenlabs_api_key}
            payload = {"text": text_input, "voice_settings": self.settings.get("voice_settings")}
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.settings.get("timeout", 10))
                response.raise_for_status()
                return response.content
            except requests.exceptions.HTTPError as e:
                logger.error(f"ElevenLabs API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise TTSError(f"ElevenLabs API request failed with status {e.response.status_code}: {e.response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"ElevenLabs API request error: {e}", exc_info=True)
                raise TTSError(f"Failed to connect to ElevenLabs API: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during ElevenLabs TTS synthesis: {e}", exc_info=True)
                raise TTSError(f"An unexpected error occurred with ElevenLabs TTS: {e}")

        elif self.provider == "sesame_csm":
            if not self.resolved_sesame_csm_url:
                logger.error("Sesame CSM service URL not found/resolved. Cannot synthesize speech.")
                raise TTSError("Sesame CSM service URL is not configured.")
            url = f"{self.resolved_sesame_csm_url}/generate-speech"
            logger.info(f"TextToSpeechService (SesameCSM): Calling {url}")
            payload = {"text": text_input, "voice_profile": voice_profile, "emotion_hint": effective_emotion}
            try:
                response = requests.post(url, json=payload, timeout=self.settings.get("timeout", 10))
                response.raise_for_status()
                return response.content
            except requests.exceptions.HTTPError as e:
                logger.error(f"Sesame CSM API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise TTSError(f"Sesame CSM API request failed with status {e.response.status_code}: {e.response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Sesame CSM API request error: {e}", exc_info=True)
                raise TTSError(f"Failed to connect to Sesame CSM API: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during Sesame CSM TTS synthesis: {e}", exc_info=True)
                raise TTSError(f"An unexpected error occurred with Sesame CSM TTS: {e}")
        else:
            logger.info(f"Using simulated TTS for provider: {self.provider}")
            return f"simulated_audio_bytes_for_[{text_input.replace(' ','_')[:30]}]_emotion_{effective_emotion}".encode('utf-8')

if __name__ == '__main__':
    # This sys.path manipulation is for allowing direct execution of the service file
    # if config_utils is in the parent directory.
    if "config_utils" not in sys.modules:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try:
            from config_utils import resolve_config_value # Use if available
        except ImportError:
            print("Warning: Could not import resolve_config_value from config_utils. Using local fallback for __main__.")
            # Define the fallback resolve_config_value if it's not available for __main__
            def resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
                if isinstance(value_from_config, str) and value_from_config.startswith("${") and value_from_config.endswith("}"):
                    var_name = value_from_config.strip("${}")
                    val = os.getenv(var_name, default_if_placeholder_not_set)
                    if target_type == int and val is not None: return int(val)
                    return val if target_type == str else None
                if target_type == int and value_from_config is not None: return int(value_from_config)
                return value_from_config


    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger()

    nlg_config_test = {
        "natural_language_generation_service": {
            "provider": "local_llama",
            "local_llama_settings": {"model_path": "${LLAMA_MODEL_PATH_TEST:-/default/llama.gguf}"}
        }
    }
    os.environ["LLAMA_MODEL_PATH_TEST"] = "env_llama_model.gguf"
    nlg_service = NaturalLanguageGenerationService(config=nlg_config_test)
    main_logger.info(f"NLG Model Path: {nlg_service.settings.get('model_path')}")
    del os.environ["LLAMA_MODEL_PATH_TEST"]

    # ... (other __main__ tests) ...
