# input_processing_services.py
# Contains services for STT and Acoustic Emotion Analysis.

import time # For simulation
import os
import asyncio
import logging
# Corrected import for Deepgram v3 and its options classes
from deepgram import DeepgramClient, DeepgramClientOptions
from deepgram.clients.prerecorded.v1 import PrerecordedOptions

from config_utils import resolve_config_value


# Custom Exception for STT errors
class STTError(Exception):
    pass

class SpeechToTextService:
    """
    Transcribes speech to text via Deepgram (or fallback simulation).
    """
    def __init__(self, config: dict): # config is the global app_config
        self.service_config = config.get('speech_to_text_service', {})
        self.provider = self.service_config.get('provider', 'deepgram').lower()
        self.settings = self.service_config.get(f"{self.provider}_settings", {})
        logger = logging.getLogger(__name__)
        logger.info(f"SpeechToTextService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        if self.provider == "deepgram":
            config_api_key_val = self.settings.get("api_key", "${DEEPGRAM_API_KEY}")

            # Using resolve_config_value for Deepgram API key
            # The refined logic from subtask 12 about warnings for hardcoded keys etc.
            # could be integrated into resolve_config_value or handled here if needed.
            # For now, direct use of resolve_config_value for simplicity as per current task scope.
            final_api_key = resolve_config_value(config_api_key_val, None) # No specific default here, relies on placeholder or env

            if not final_api_key: # If placeholder didn't resolve and no direct value
                 # Attempt to get from a standard environment variable as a last resort
                standard_env_key = "DEEPGRAM_API_KEY"
                final_api_key = os.getenv(standard_env_key)
                if final_api_key:
                    logger.info(f"Used Deepgram API key from environment variable {standard_env_key} as placeholder/config value was not found/resolved.")
                else:
                    logger.warning(f"No Deepgram API key configured via placeholder '{config_api_key_val}', direct config, or default environment variable {standard_env_key}.")


            if not final_api_key:
                logger.warning("SpeechToTextService: No valid Deepgram API key resolved, falling back to simulation mode.")
                self.provider = "simulation"
                self.dg_client = None
                return
            try:
                client_options = DeepgramClientOptions(api_key=final_api_key)
                self.dg_client = DeepgramClient(client_options)
                logger.info("Deepgram client initialized successfully.")
            except Exception as e:
                logger.error(f"SpeechToTextService: Deepgram client initialization failed: {e}", exc_info=True)
                self.provider = "simulation"
                self.dg_client = None
            return

    def transcribe_audio_chunk(self, audio_chunk: bytes) -> str:
        logger = logging.getLogger(__name__)
        if not audio_chunk: return ""
        if self.provider == "deepgram":
            if not self.dg_client:
                logger.error("Deepgram provider selected, but client not initialized.")
                raise STTError("Deepgram client not initialized. Cannot transcribe.")
            try:
                source = {'buffer': audio_chunk, 'mimetype': 'audio/wav'}
                options_dict = self.settings.get("options", {"model": "nova-2", "smart_format": True})
                sdk_options = PrerecordedOptions(**options_dict)
                logger.debug(f"Calling Deepgram Prerecorded API with options: {sdk_options}")
                response = self.dg_client.listen.prerecorded.v("1").transcribe_file(source, sdk_options)
                logger.debug("Deepgram Prerecorded API response received")
                if response and response.results and response.results.channels:
                    first_channel = response.results.channels[0]
                    if first_channel.alternatives:
                        transcript = first_channel.alternatives[0].transcript
                        if transcript:
                            logger.info(f"Deepgram transcription successful: \"{transcript[:50]}...\"")
                            return transcript.strip()
                        else: logger.info("Deepgram transcription resulted in an empty transcript.")
                    else: logger.warning("Deepgram transcription returned no alternatives in the first channel.")
                else: logger.warning("Deepgram transcription response was missing expected results, channels, or alternatives.")
                return ""
            except Exception as e:
                error_name = e.__class__.__name__
                logger.error(f"Error during Deepgram transcription: {error_name} - {e}", exc_info=True)
                raise STTError(f"An unexpected error occurred during transcription: {e}")

        # Simulation mode
        if not hasattr(self, "_sim_counter"): self._sim_counter = 0
        sim_texts = ["Hello, who is this?", "I'm interested in selling my house.", "What's the market like in the downtown area?", "That sounds a bit too expensive for me.", "Okay, tell me more about the property on Elm Street.", "Thank you, that was very helpful.", "Goodbye."]
        text = sim_texts[self._sim_counter % len(sim_texts)]
        self._sim_counter = (self._sim_counter + 1) % len(sim_texts)
        logger.info(f"Using simulated transcription: \"{text}\"")
        return text

class AcousticEmotionAnalyzerService:
    def __init__(self, service_config: dict):
        self.config = service_config

        raw_model_path = self.config.get('model_path', 'default_emotion_model.pkl')
        self.model_path = resolve_config_value(raw_model_path, default_if_placeholder_not_set='path/to/acoustic_emotion_model.pkl')

        self.sample_rate = self.config.get('sample_rate', 22050)

        logger = logging.getLogger(__name__)
        logger.info(f"AcousticEmotionAnalyzerService Initialized (Model: {self.model_path}, SampleRate: {self.sample_rate})")

    def analyze_emotion_from_audio(self, audio_chunk: bytes) -> dict:
        logger = logging.getLogger(__name__)
        if not audio_chunk: return {"dominant_emotion": "neutral", "probabilities": {"neutral": 1.0}}
        logger.debug(f"AcousticEmotionAnalyzerService: Analyzing emotion from {len(audio_chunk)} bytes.")
        if not hasattr(self, "_emo_sim_counter"): self._emo_sim_counter = 0
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
        dominant_emotion = emotions[self._emo_sim_counter % len(emotions)]
        self._emo_sim_counter = (self._emo_sim_counter + 1) % len(emotions)
        probabilities = {emo: 0.1 for emo in emotions}
        probabilities[dominant_emotion] = 0.6
        total_prob = sum(probabilities.values())
        normalized_probabilities = {emo: prob/total_prob for emo, prob in probabilities.items()}
        logger.debug(f"Simulated emotion: {dominant_emotion}")
        return {"dominant_emotion": dominant_emotion, "probabilities": normalized_probabilities}

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger(__name__)

    # Temporarily add config_utils to sys.path for direct script execution if needed
    # This is only for the __main__ block. Services should import it directly.
    if "config_utils" not in sys.modules:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try:
            from config_utils import resolve_config_value as main_resolve_config_value
        except ImportError:
             main_resolve_config_value = resolve_config_value # Use local fallback if still not found

    dummy_global_config = {
        "speech_to_text_service": {
            "provider": "deepgram",
            "deepgram_settings": { "api_key": "${DEEPGRAM_API_KEY_TEST}" }
        },
        "acoustic_emotion_analyzer_service": {
             "model_path": "${ACOUSTIC_MODEL_PATH_TEST:-./fallback_model.pkl}",
             "sample_rate": 16000
        }
    }
    os.environ["DEEPGRAM_API_KEY_TEST"] = "dummy_env_key_for_direct_run"

    stt_service = SpeechToTextService(config=dummy_global_config)
    # Use the main_resolve_config_value for the __main__ block if it was imported
    # This ensures the __main__ block tests with the "correct" resolver if possible
    aea_service_config = dummy_global_config['acoustic_emotion_analyzer_service']
    aea_service_config['model_path'] = main_resolve_config_value(aea_service_config['model_path'], './fallback_model.pkl')
    aea_service = AcousticEmotionAnalyzerService(service_config=aea_service_config) # Pass resolved for test

    main_logger.info(f"AEA Model Path from test: {aea_service.model_path}")

    sample_audio = b"short_audio_data_for_testing_123"
    try:
        main_logger.info(f"Attempting transcription with provider: {stt_service.provider} (using __main__)")
        text_result_dg = stt_service.transcribe_audio_chunk(sample_audio)
        main_logger.info(f"STT Result (provider: {stt_service.provider}): {text_result_dg}")
    except STTError as e:
        main_logger.error(f"STTError caught during transcription test in __main__: {e}")

    del os.environ["DEEPGRAM_API_KEY_TEST"]
