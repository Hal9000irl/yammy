# input_processing_services.py
# Contains services for STT and Acoustic Emotion Analysis.

import time # For simulation
import os
import asyncio
import logging
import tempfile # Added for AssemblyAI
import assemblyai # Added for AssemblyAI

# Corrected import for Deepgram v3 and its options classes
from deepgram import DeepgramClient, DeepgramClientOptions
from deepgram.clients.prerecorded.v1 import PrerecordedOptions

from config_utils import resolve_config_value


# Custom Exception for STT errors
class STTError(Exception):
    pass

class SpeechToTextService:
    """
    Transcribes speech to text via Deepgram or AssemblyAI (or fallback simulation).
    """
    def __init__(self, config: dict): # config is the global app_config
        self.service_config = config.get('speech_to_text_service', {})
        self.provider = self.service_config.get('provider', 'deepgram').lower()
        self.settings = self.service_config.get(f"{self.provider}_settings", {})
        self.logger = logging.getLogger(__name__) # Use self.logger for instance-wide logging
        self.logger.info(f"SpeechToTextService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        self.dg_client = None
        self.assemblyai_client = None

        if self.provider == "deepgram":
            config_api_key_val = self.settings.get("api_key", "${DEEPGRAM_API_KEY}")
            final_api_key = resolve_config_value(config_api_key_val, None)

            if not final_api_key:
                standard_env_key = "DEEPGRAM_API_KEY"
                final_api_key = os.getenv(standard_env_key)
                if final_api_key:
                    self.logger.info(f"Used Deepgram API key from environment variable {standard_env_key}.")
                else:
                    self.logger.warning(f"No Deepgram API key configured via placeholder '{config_api_key_val}', direct config, or default environment variable {standard_env_key}.")

            if not final_api_key:
                self.logger.warning("SpeechToTextService: No valid Deepgram API key resolved, falling back to simulation mode.")
                self.provider = "simulation"
                return
            try:
                client_options = DeepgramClientOptions(api_key=final_api_key)
                self.dg_client = DeepgramClient(client_options)
                self.logger.info("Deepgram client initialized successfully.")
            except Exception as e:
                self.logger.error(f"SpeechToTextService: Deepgram client initialization failed: {e}", exc_info=True)
                self.provider = "simulation"

        elif self.provider == "assemblyai":
            config_api_key_val = self.settings.get("api_key", "${ASSEMBLYAI_API_KEY}")
            api_key = resolve_config_value(config_api_key_val, None)

            if not api_key:
                standard_env_key = "ASSEMBLYAI_API_KEY" # Common env var for AssemblyAI
                api_key = os.getenv(standard_env_key)
                if api_key:
                    self.logger.info(f"Used AssemblyAI API key from environment variable {standard_env_key}.")

            if not api_key:
                self.logger.warning("SpeechToTextService: No valid AssemblyAI API key provided, falling back to simulation mode.")
                self.provider = "simulation"
            else:
                try:
                    assemblyai.settings.api_key = api_key
                    self.assemblyai_client = assemblyai.Transcriber()
                    self.logger.info("AssemblyAI client initialized successfully.")
                except Exception as e:
                    self.logger.error(f"SpeechToTextService: AssemblyAI client initialization failed: {e}", exc_info=True)
                    self.provider = "simulation"
            return # Added return

    def transcribe_audio_chunk(self, audio_chunk: bytes) -> str:
        if not audio_chunk: return ""

        if self.provider == "deepgram":
            if not self.dg_client:
                self.logger.error("Deepgram provider selected, but client not initialized.")
                raise STTError("Deepgram client not initialized. Cannot transcribe.")
            try:
                source = {'buffer': audio_chunk, 'mimetype': 'audio/wav'}
                options_dict = self.settings.get("options", {"model": "nova-2", "smart_format": True})
                sdk_options = PrerecordedOptions(**options_dict)
                self.logger.debug(f"Calling Deepgram Prerecorded API with options: {sdk_options}")
                response = self.dg_client.listen.prerecorded.v("1").transcribe_file(source, sdk_options)
                self.logger.debug("Deepgram Prerecorded API response received")
                if response and response.results and response.results.channels:
                    first_channel = response.results.channels[0]
                    if first_channel.alternatives:
                        transcript = first_channel.alternatives[0].transcript
                        if transcript:
                            self.logger.info(f"Deepgram transcription successful: \"{transcript[:50]}...\"")
                            return transcript.strip()
                        else: self.logger.info("Deepgram transcription resulted in an empty transcript.")
                    else: self.logger.warning("Deepgram transcription returned no alternatives in the first channel.")
                else: self.logger.warning("Deepgram transcription response was missing expected results, channels, or alternatives.")
                return ""
            except Exception as e:
                error_name = e.__class__.__name__
                self.logger.error(f"Error during Deepgram transcription: {error_name} - {e}", exc_info=True)
                raise STTError(f"An unexpected error occurred during transcription: {e}")

        elif self.provider == "assemblyai":
            if not self.assemblyai_client:
                self.logger.error("AssemblyAI provider selected, but client not initialized.")
                raise STTError("AssemblyAI client not initialized. Cannot transcribe.")
            self.logger.debug(f"Attempting transcription with AssemblyAI for {len(audio_chunk)} bytes of audio.")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                    tmp_audio_file.write(audio_chunk)
                    tmp_audio_file_path = tmp_audio_file.name

                transcript = self.assemblyai_client.transcribe(tmp_audio_file_path)
                os.remove(tmp_audio_file_path)

                if transcript.status == assemblyai.TranscriptStatus.error:
                    self.logger.error(f"AssemblyAI transcription error: {transcript.error}")
                    return ""
                if transcript.text:
                    self.logger.info(f"AssemblyAI transcription successful: \"{transcript.text[:50]}...\"")
                    return transcript.text.strip()
                else:
                    self.logger.info("AssemblyAI transcription resulted in an empty transcript.")
                    return ""
            except Exception as e:
                self.logger.error(f"Error during AssemblyAI transcription: {e}", exc_info=True)
                # Clean up temp file if it still exists and an error occurred after its creation
                if 'tmp_audio_file_path' in locals() and os.path.exists(tmp_audio_file_path):
                    os.remove(tmp_audio_file_path)
                raise STTError(f"An error occurred during AssemblyAI transcription: {e}")

        # Simulation mode (default or if other providers failed to init)
        if not hasattr(self, "_sim_counter"): self._sim_counter = 0
        sim_texts = ["Hello, who is this?", "I'm interested in selling my house.", "What's the market like in the downtown area?", "That sounds a bit too expensive for me.", "Okay, tell me more about the property on Elm Street.", "Thank you, that was very helpful.", "Goodbye."]
        text = sim_texts[self._sim_counter % len(sim_texts)]
        self._sim_counter = (self._sim_counter + 1) % len(sim_texts)
        self.logger.info(f"Using simulated transcription: \"{text}\"")
        return text

class AcousticEmotionAnalyzerService:
    def __init__(self, service_config: dict):
        self.config = service_config
        raw_model_path = self.config.get('model_path', 'default_emotion_model.pkl')
        self.model_path = resolve_config_value(raw_model_path, default_if_placeholder_not_set='path/to/acoustic_emotion_model.pkl')
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AcousticEmotionAnalyzerService Initialized (Model: {self.model_path}, SampleRate: {self.sample_rate})")

    def analyze_emotion_from_audio(self, audio_chunk: bytes) -> dict:
        if not audio_chunk: return {"dominant_emotion": "neutral", "probabilities": {"neutral": 1.0}}
        self.logger.debug(f"AcousticEmotionAnalyzerService: Analyzing emotion from {len(audio_chunk)} bytes.")
        if not hasattr(self, "_emo_sim_counter"): self._emo_sim_counter = 0
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
        dominant_emotion = emotions[self._emo_sim_counter % len(emotions)]
        self._emo_sim_counter = (self._emo_sim_counter + 1) % len(emotions)
        probabilities = {emo: 0.1 for emo in emotions}
        probabilities[dominant_emotion] = 0.6
        total_prob = sum(probabilities.values())
        normalized_probabilities = {emo: prob/total_prob for emo, prob in probabilities.items()}
        self.logger.debug(f"Simulated emotion: {dominant_emotion}")
        return {"dominant_emotion": dominant_emotion, "probabilities": normalized_probabilities}

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger(__name__)

    if "config_utils" not in sys.modules:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try: from config_utils import resolve_config_value as main_resolve_config_value
        except ImportError: main_resolve_config_value = resolve_config_value

    # Test AssemblyAI provider (will likely fallback to simulation without key)
    dummy_assemblyai_config = {
        "speech_to_text_service": {
            "provider": "assemblyai",
            "assembly_ai_settings": { "api_key": "${ASSEMBLYAI_API_KEY_TEST:-}" }
        },
         "acoustic_emotion_analyzer_service": { # Keep other services for full config structure
             "model_path": "path/to/model.pkl"
        }
    }
    # To test actual AssemblyAI, set ASSEMBLYAI_API_KEY_TEST environment variable
    # os.environ["ASSEMBLYAI_API_KEY_TEST"] = "your_real_assemblyai_key"

    stt_assembly = SpeechToTextService(config=dummy_assemblyai_config)
    main_logger.info(f"STT AssemblyAI provider type after init: {stt_assembly.provider}")

    sample_audio = b"short_audio_data_for_testing_123" * 100 # Longer for some STTs
    try:
        main_logger.info(f"Attempting transcription with provider: {stt_assembly.provider} (using __main__)")
        text_result = stt_assembly.transcribe_audio_chunk(sample_audio)
        main_logger.info(f"STT Result (provider: {stt_assembly.provider}): {text_result}")
    except STTError as e:
        main_logger.error(f"STTError caught during AssemblyAI transcription test in __main__: {e}")

    # if "ASSEMBLYAI_API_KEY_TEST" in os.environ: del os.environ["ASSEMBLYAI_API_KEY_TEST"]
