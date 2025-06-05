# response_generation_services.py
# Contains services for NLG and TTS.

import time # For simulation
import os
import requests
import logging

logging.basicConfig(level=logging.INFO) # Ensure basicConfig is called early
logger = logging.getLogger(__name__)

class TTSError(Exception):
    """Custom exception for TextToSpeechService errors."""
    pass

class NaturalLanguageGenerationService:
    """
    Generates human-like text responses, often using an LLM.
    Used when a specialist doesn't provide the full response or for general replies.
    """
    def __init__(self, config: dict):
        self.config = config.get('natural_language_generation_service', {})
        self.provider = self.config.get('provider', 'local_llama')
        self.settings = self.config.get(f"{self.provider}_settings", {})
        logger.info(f"NaturalLanguageGenerationService Initialized (Provider: {self.provider}, Settings: {self.settings})")

    def generate_text_response(self, prompt: str, context_data: dict = None) -> str:
        logger.info(f"NLGService ({self.provider}): Generating text for prompt: '{prompt[:100]}...'")
        if "clarify" in prompt.lower():
            return f"NLG Simulated Response: I understand you mentioned '{context_data.get('last_user_utterance', 'something') if context_data else 'something'}'. Could you please elaborate a bit more on that?"
        elif "summarize" in prompt.lower():
            return f"NLG Simulated Response: To summarize, we discussed {context_data.get('key_topics', ['several important points']) if context_data else ['several important points']}."
        return f"NLG Simulated Response: Based on your query about '{prompt[:30]}...', I'd be happy to provide more information. What specifically are you interested in?"

class TextToSpeechService:
    def __init__(self, config: dict):
        self.config = config.get('text_to_speech_service', {})
        self.provider = self.config.get('provider', 'sesame_csm')
        self.settings = self.config.get(f"{self.provider}_settings", {})
        logger.info(f"TextToSpeechService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        # Resolved API keys/URLs are stored here after __init__ logic
        self.resolved_elevenlabs_api_key = None
        self.resolved_sesame_csm_url = None

        if self.provider == "elevenlabs":
            self._initialize_elevenlabs_key()
        elif self.provider == "sesame_csm":
            self._initialize_sesame_csm_url()

    def _initialize_elevenlabs_key(self):
        config_api_key = self.settings.get("api_key")
        env_api_key_value = os.getenv("ELEVENLABS_API_KEY") # Standard env var name
        final_api_key = None

        if config_api_key and config_api_key.startswith("${") and config_api_key.endswith("}"):
            placeholder = config_api_key.strip("${}")
            final_api_key = os.getenv(placeholder)
            if not final_api_key:
                logger.warning(f"Environment variable {placeholder} for ElevenLabs API key not found.")
        elif config_api_key: # Non-empty, but not a placeholder (potentially hardcoded)
            logger.warning(
                "A non-placeholder API key was found in the configuration for ElevenLabs. "
                "For security, prefer using placeholders like ${ELEVENLABS_API_KEY}."
            )
            if env_api_key_value:
                logger.warning(
                    "Environment variable ELEVENLABS_API_KEY is overriding the non-placeholder API key "
                    "found in configuration for ElevenLabs."
                )
                final_api_key = env_api_key_value
            else:
                logger.warning(
                    "Using the non-placeholder API key from configuration for ElevenLabs as environment "
                    "variable ELEVENLABS_API_KEY is not set. This is not recommended."
                )
                final_api_key = config_api_key
        else: # config_api_key is None or empty
            if env_api_key_value:
                logger.info("Using ElevenLabs API key from environment variable ELEVENLABS_API_KEY.")
                final_api_key = env_api_key_value
            else:
                logger.error("No ElevenLabs API key configured via placeholder, direct config, or environment variable ELEVENLABS_API_KEY.")

        if not final_api_key:
            # This will be caught before actual API call in synthesize_speech
            logger.error("ElevenLabs API key could not be resolved.")
            # No TTSError raised here, will be raised at time of use if key is still None
        self.resolved_elevenlabs_api_key = final_api_key


    def _initialize_sesame_csm_url(self):
        config_service_url = self.settings.get("service_url")
        env_service_url = os.getenv("SESAME_CSM_URL") # Standard env var name
        final_url = None

        if config_service_url and config_service_url.startswith("${") and config_service_url.endswith("}"):
            placeholder = config_service_url.strip("${}")
            final_url = os.getenv(placeholder)
            if not final_url:
                logger.warning(f"Environment variable {placeholder} for Sesame CSM URL not found.")
        elif config_service_url: # Non-empty, but not a placeholder
            logger.warning(
                "A non-placeholder service URL was found in the configuration for Sesame CSM. "
                "Consider using a placeholder like ${SESAME_CSM_URL}."
            )
            if env_service_url:
                 logger.warning(
                    "Environment variable SESAME_CSM_URL is overriding the non-placeholder service URL "
                    "found in configuration for Sesame CSM."
                )
                 final_url = env_service_url
            else:
                logger.info( # Info level, as URL might not always be secret, but placeholder is good practice
                    "Using the non-placeholder service URL from configuration for Sesame CSM as environment variable "
                    "SESAME_CSM_URL is not set."
                )
                final_url = config_service_url
        else: # config_service_url is None or empty
            if env_service_url:
                logger.info("Using Sesame CSM service URL from environment variable SESAME_CSM_URL.")
                final_url = env_service_url
            else:
                 logger.error("No Sesame CSM Service URL configured via placeholder, direct config, or environment variable SESAME_CSM_URL.")

        if not final_url:
            logger.error("Sesame CSM Service URL could not be resolved.")
        self.resolved_sesame_csm_url = final_url


    def synthesize_speech(self, text_input: str, voice_profile: str = "default_professional", emotion_hint: str = None) -> bytes:
        effective_emotion = emotion_hint if emotion_hint else "neutral"
        logger.info(f"TTSService ({self.provider}): Synthesizing speech for: '{text_input}' (Voice: {voice_profile}, Emotion Hint: {effective_emotion})")

        if self.provider == "elevenlabs":
            if not self.resolved_elevenlabs_api_key: # Check resolved key
                logger.error("ElevenLabs API key not found/resolved. Cannot synthesize speech.")
                raise TTSError("ElevenLabs API key is not configured.")

            voice_id = self.settings.get("default_voice_id", voice_profile)
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.resolved_elevenlabs_api_key
            }
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
            if not self.resolved_sesame_csm_url: # Check resolved URL
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
    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger(__name__) # Use __name__ for the logger in __main__

    # Test ElevenLabs with placeholder (requires ELEVENLABS_API_KEY_MAIN_TEST to be set)
    os.environ["ELEVENLABS_API_KEY_MAIN_TEST"] = "actual_env_key_for_elevenlabs" # Simulate env var
    dummy_config_el_placeholder = {
        "text_to_speech_service": {
            "provider": "elevenlabs",
            "elevenlabs_settings": {"api_key": "${ELEVENLABS_API_KEY_MAIN_TEST}", "default_voice_id": "vid1"}
        }
    }
    tts_el_placeholder = TextToSpeechService(config=dummy_config_el_placeholder)
    main_logger.info(f"ElevenLabs API Key (Placeholder Test): {tts_el_placeholder.resolved_elevenlabs_api_key}")
    # To actually call synthesize_speech, you'd need a running ElevenLabs or mock requests.post

    # Test Sesame with direct URL and env var override
    os.environ["SESAME_CSM_URL_MAIN_TEST_ENV"] = "http://env-sesame.url"
    dummy_config_sesame_direct_override = {
        "text_to_speech_service": {
            "provider": "sesame_csm",
            "sesame_csm_settings": {"service_url": "http://config-sesame.url"} # Non-placeholder
        }
    }
    # Temporarily set SESAME_CSM_URL to test override warning for non-placeholder
    os.environ["SESAME_CSM_URL"] = os.environ["SESAME_CSM_URL_MAIN_TEST_ENV"]
    tts_sesame_override = TextToSpeechService(config=dummy_config_sesame_direct_override)
    main_logger.info(f"Sesame URL (Direct Config with Env Override Test): {tts_sesame_override.resolved_sesame_csm_url}")
    del os.environ["SESAME_CSM_URL"] # Clean up env var

    # Test simulation
    dummy_config_sim = {"text_to_speech_service": {"provider": "simulation"}}
    tts_sim = TextToSpeechService(config=dummy_config_sim)
    audio_data_sim = tts_sim.synthesize_speech("Hello from simulation.")
    main_logger.info(f"Simulated TTS audio: {audio_data_sim[:50]}...")

    # Clean up test env var
    del os.environ["ELEVENLABS_API_KEY_MAIN_TEST"]
