# input_processing_services.py
# Contains services for STT and Acoustic Emotion Analysis.

import time # For simulation
import os
import asyncio
import logging
# Corrected import for Deepgram v3 and its options classes
from deepgram import DeepgramClient, DeepgramClientOptions
from deepgram.clients.prerecorded.v1 import PrerecordedOptions
# For specific errors, e.g., from deepgram.errors import DeepgramError, DeepgramApiError

# Custom Exception for STT errors
class STTError(Exception):
    pass

class SpeechToTextService:
    """
    Transcribes speech to text via Deepgram (or fallback simulation).
    """
    def __init__(self, config: dict):
        self.config = config.get('speech_to_text_service', {})
        self.provider = self.config.get('provider', 'deepgram').lower()
        self.settings = self.config.get(f"{self.provider}_settings", {})
        logger = logging.getLogger(__name__)
        logger.info(f"SpeechToTextService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        if self.provider == "deepgram":
            config_api_key = self.settings.get("api_key", "")
            final_api_key = None

            if config_api_key.startswith("${") and config_api_key.endswith("}"):
                placeholder = config_api_key.strip("${}")
                final_api_key = os.getenv(placeholder)
                if not final_api_key:
                    logger.warning(f"Environment variable {placeholder} for Deepgram API key not found.")
            elif config_api_key: # Non-empty, but not a placeholder
                logger.warning(
                    "A non-placeholder API key was found in the configuration for Deepgram. "
                    "For security, prefer using placeholders like ${DEEPGRAM_API_KEY} linked to environment variables."
                )
                # Check if a standard environment variable (e.g., DEEPGRAM_API_KEY) exists and prioritize it.
                standard_env_var = "DEEPGRAM_API_KEY" # Default environment variable name for Deepgram
                env_api_key = os.getenv(standard_env_var)
                if env_api_key:
                    logger.warning(
                        f"Using API key from environment variable {standard_env_var} "
                        "instead of the non-placeholder key found in configuration."
                    )
                    final_api_key = env_api_key
                else:
                    logger.warning(
                        "Using the non-placeholder API key from configuration as "
                        f"environment variable {standard_env_var} was not found. This is not recommended."
                    )
                    final_api_key = config_api_key # Use hardcoded key as last resort, with warning
            else: # api_key in config is empty or not present
                # Attempt to get from a default environment variable if no specific placeholder was used
                standard_env_var = "DEEPGRAM_API_KEY"
                final_api_key = os.getenv(standard_env_var)
                if final_api_key:
                     logger.info(f"Using Deepgram API key from environment variable {standard_env_var}.")
                else:
                    logger.warning(f"No Deepgram API key configured via placeholder, direct config, or default environment variable {standard_env_var}.")


            if not final_api_key: # If key is still not found after all checks
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
        if not audio_chunk:
            return ""

        if self.provider == "deepgram":
            if not self.dg_client:
                logger.error("Deepgram provider selected, but client not initialized (likely API key issue).")
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
                        else:
                            logger.info("Deepgram transcription resulted in an empty transcript.")
                            return ""
                    else:
                        logger.warning("Deepgram transcription returned no alternatives in the first channel.")
                        return ""
                else:
                    logger.warning("Deepgram transcription response was missing expected results, channels, or alternatives.")
                    return ""
            except Exception as e:
                error_name = e.__class__.__name__
                logger.error(f"Error during Deepgram transcription: {error_name} - {e}", exc_info=True)
                raise STTError(f"An unexpected error occurred during transcription: {e}")

        # Simulation mode
        if not hasattr(self, "_sim_counter"):
            self._sim_counter = 0

        sim_texts = [
            "Hello, who is this?", "I'm interested in selling my house.",
            "What's the market like in the downtown area?", "That sounds a bit too expensive for me.",
            "Okay, tell me more about the property on Elm Street.", "Thank you, that was very helpful.", "Goodbye."
        ]
        text = sim_texts[self._sim_counter % len(sim_texts)]
        self._sim_counter = (self._sim_counter + 1) % len(sim_texts)
        logger.info(f"Using simulated transcription: \"{text}\"")
        return text

class AcousticEmotionAnalyzerService:
    def __init__(self, config: dict):
        self.config = config.get('acoustic_emotion_analyzer_service', {})
        self.model_path = self.config.get('model_path', 'default_emotion_model.pkl')
        self.sample_rate = self.config.get('sample_rate', 22050)
        logger = logging.getLogger(__name__)
        logger.info(f"AcousticEmotionAnalyzerService Initialized (Model: {self.model_path}, SampleRate: {self.sample_rate})")

    def analyze_emotion_from_audio(self, audio_chunk: bytes) -> dict:
        logger = logging.getLogger(__name__)
        if not audio_chunk:
            return {"dominant_emotion": "neutral", "probabilities": {"neutral": 1.0}}

        logger.debug(f"AcousticEmotionAnalyzerService: Analyzing emotion from {len(audio_chunk)} bytes.")
        if not hasattr(self, "_emo_sim_counter"):
            self._emo_sim_counter = 0
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

    dummy_config_deepgram_test = {
        "speech_to_text_service": {
            "provider": "deepgram",
            "deepgram_settings": {
                "api_key": "YOUR_DG_API_KEY_NEEDS_TO_BE_REAL_FOR_THIS_TEST_TO_PASS_DEEPGRAM",
                "options": {"model": "nova-2", "language": "en-US"}
            }
        },
    }
    stt_dg_test = SpeechToTextService(config=dummy_config_deepgram_test)
    sample_audio = b"short_audio_data_for_testing_123"

    try:
        main_logger.info(f"Attempting transcription with provider: {stt_dg_test.provider} (using __main__)")
        text_result_dg = stt_dg_test.transcribe_audio_chunk(sample_audio)
        main_logger.info(f"STT Result (provider: {stt_dg_test.provider}): {text_result_dg}")
    except STTError as e:
        main_logger.error(f"STTError caught during transcription test in __main__: {e}")
    except Exception as e:
        main_logger.error(f"Unexpected error during transcription test in __main__: {e}", exc_info=True)

    dummy_config_sim_test = { "speech_to_text_service": {"provider": "simulation"} }
    stt_sim_test = SpeechToTextService(config=dummy_config_sim_test)
    try:
        main_logger.info(f"Attempting transcription with provider: {stt_sim_test.provider} (using __main__)")
        text_result_sim = stt_sim_test.transcribe_audio_chunk(sample_audio)
        main_logger.info(f"STT Result (Simulation): {text_result_sim}")
    except STTError as e:
        main_logger.error(f"STTError caught during simulation test in __main__: {e}")

    dummy_config_aea = {"acoustic_emotion_analyzer_service": {}}
    emotion_analyzer = AcousticEmotionAnalyzerService(config=dummy_config_aea)
    emotion_result = emotion_analyzer.analyze_emotion_from_audio(sample_audio)
    main_logger.info(f"Emotion Result: {emotion_result}")
