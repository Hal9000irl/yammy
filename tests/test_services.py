import unittest
from unittest.mock import patch, MagicMock
import logging
import os
import sys
import requests # Needed for requests.exceptions
import re # Ensure re is imported for the fallback resolver

# Adjust sys.path to include the parent directory (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from input_processing_services import SpeechToTextService, STTError, AcousticEmotionAnalyzerService
from response_generation_services import TextToSpeechService, TTSError, NaturalLanguageGenerationService
from dialogue_manager_service import RasaService
from specialist_empathy_service import EmpathySpecialistService
from specialist_sales_services import GenericSalesSkillService, RealEstateKnowledgeService, SalesAgentService # Added SalesAgentService
from deepgram.clients.prerecorded.v1 import PrerecordedOptions
import numpy as np # For RealEstateKnowledgeService tests

# Attempt to import resolve_config_value from config_utils
try:
    from config_utils import resolve_config_value, PLACEHOLDER_REGEX
except ImportError:
    print("Warning: Could not import from config_utils. Using fallback resolve_config_value in tests/test_services.py")
    PLACEHOLDER_REGEX = re.compile(r"\$\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::-([^}]*))?\s*\}")
    def resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
        if isinstance(value_from_config, str):
            match = PLACEHOLDER_REGEX.fullmatch(value_from_config)
            if match:
                var_name = match.group(1)
                placeholder_default = match.group(2)
                env_value = os.getenv(var_name)
                if env_value is not None:
                    resolved_value = env_value
                elif placeholder_default is not None:
                    resolved_value = placeholder_default
                else:
                    resolved_value = default_if_placeholder_not_set
            else:
                resolved_value = value_from_config
        else:
            resolved_value = value_from_config

        if resolved_value is None:
            if target_type is bool: return False
            return None
        try:
            if target_type == int: return int(resolved_value)
            elif target_type == bool:
                if isinstance(resolved_value, str):
                    return resolved_value.lower() in ('true', 'yes', '1', 'on')
                return bool(resolved_value)
            return target_type(resolved_value)
        except ValueError:
            return str(resolved_value) if resolved_value is not None else None


class TestSpeechToTextService(unittest.TestCase):
    # ... (Content of TestSpeechToTextService remains the same - verified as passing) ...
    def setUp(self):
        self.base_config = {
            "speech_to_text_service": {
                "provider": "deepgram",
                "deepgram_settings": {
                    "api_key": "test_deepgram_api_key",
                    "options": {"model": "nova-2", "smart_format": True}
                }
            }
        }
        self.sample_audio_chunk = b"sample_audio_data"
        self.logger_name_stt = 'input_processing_services'


    @patch('input_processing_services.DeepgramClient')
    def test_transcribe_deepgram_success(self, MockDeepgramClient):
        mock_dg_instance = MockDeepgramClient.return_value
        mock_transcription_result = MagicMock()
        mock_transcription_result.results.channels = [
            MagicMock(alternatives=[MagicMock(transcript="Hello world")])
        ]
        mock_dg_instance.listen.prerecorded.v("1").transcribe_file.return_value = mock_transcription_result

        service = SpeechToTextService(config=self.base_config)
        with self.assertLogs(self.logger_name_stt, level='INFO') as cm:
            transcript = service.transcribe_audio_chunk(self.sample_audio_chunk)

        self.assertEqual(transcript, "Hello world")
        mock_dg_instance.listen.prerecorded.v("1").transcribe_file.assert_called_once()
        call_args = mock_dg_instance.listen.prerecorded.v("1").transcribe_file.call_args
        self.assertEqual(call_args[0][0], {'buffer': self.sample_audio_chunk, 'mimetype': 'audio/wav'})
        self.assertIsInstance(call_args[0][1], PrerecordedOptions)
        self.assertEqual(call_args[0][1].model, "nova-2")
        self.assertTrue(any('Deepgram transcription successful: "Hello world..."' in log_msg for log_msg in cm.output))


    @patch('input_processing_services.DeepgramClient')
    def test_transcribe_deepgram_api_error(self, MockDeepgramClient):
        mock_dg_instance = MockDeepgramClient.return_value
        mock_dg_instance.listen.prerecorded.v("1").transcribe_file.side_effect = Exception("Simulated API Error")

        service = SpeechToTextService(config=self.base_config)

        with self.assertRaises(STTError) as context, \
             self.assertLogs(self.logger_name_stt, level='ERROR') as cm:
            service.transcribe_audio_chunk(self.sample_audio_chunk)

        self.assertIn("Simulated API Error", str(context.exception))
        self.assertTrue(any("Error during Deepgram transcription: Exception - Simulated API Error" in log_msg for log_msg in cm.output))


    @patch('input_processing_services.os.getenv')
    @patch('input_processing_services.DeepgramClient')
    def test_deepgram_client_initialization_failure_fallback(self, MockDeepgramClient, mock_getenv):
        mock_getenv.return_value = "fake_key_for_init_attempt"
        MockDeepgramClient.side_effect = Exception("Simulated DG Client Init Error")

        config_with_deepgram = {
            "speech_to_text_service": {
                "provider": "deepgram",
                "deepgram_settings": {"api_key": "try_this_key_first"}
            }
        }
        with self.assertLogs(self.logger_name_stt, level='ERROR') as cm:
            service = SpeechToTextService(config=config_with_deepgram)

        self.assertEqual(service.provider, "simulation")
        self.assertTrue(any("SpeechToTextService: Deepgram client initialization failed: Simulated DG Client Init Error" in log_msg for log_msg in cm.output))

        if hasattr(service, '_sim_counter'): service._sim_counter = 0
        transcript = service.transcribe_audio_chunk(self.sample_audio_chunk)
        self.assertIn(transcript, ["Hello, who is this?", "I'm interested in selling my house."])


    def test_transcribe_simulation_provider(self):
        sim_config = {"speech_to_text_service": {"provider": "simulation"}}
        service = SpeechToTextService(config=sim_config)
        if hasattr(service, '_sim_counter'): service._sim_counter = 0

        with self.assertLogs(self.logger_name_stt, level='INFO') as cm:
            transcript1 = service.transcribe_audio_chunk(self.sample_audio_chunk)
        self.assertEqual(transcript1, "Hello, who is this?")
        self.assertTrue(any('Using simulated transcription: "Hello, who is this?"' in log_msg for log_msg in cm.output))

        with self.assertLogs(self.logger_name_stt, level='INFO') as cm2:
            transcript2 = service.transcribe_audio_chunk(self.sample_audio_chunk)
        self.assertEqual(transcript2, "I'm interested in selling my house.")
        self.assertTrue(any('Using simulated transcription: "I\'m interested in selling my house."' in log_msg for log_msg in cm2.output))


    def test_transcribe_empty_audio_chunk(self):
        sim_config = {"speech_to_text_service": {"provider": "simulation"}}
        service = SpeechToTextService(config=sim_config)
        transcript = service.transcribe_audio_chunk(b"")
        self.assertEqual(transcript, "")

    @patch('input_processing_services.DeepgramClient')
    def test_deepgram_response_issues(self, MockDeepgramClient):
        mock_dg_instance = MockDeepgramClient.return_value
        service = SpeechToTextService(config=self.base_config)
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm:
            mock_dg_instance.listen.prerecorded.v("1").transcribe_file.return_value = MagicMock(results=MagicMock(channels=[]))
            transcript_no_channels = service.transcribe_audio_chunk(self.sample_audio_chunk)
            self.assertEqual(transcript_no_channels, "")
        self.assertTrue(any("Deepgram transcription response was missing expected results, channels, or alternatives." in log_msg for log_msg in cm.output))
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm2:
            no_alt_response = MagicMock(); no_alt_response.results.channels = [MagicMock(alternatives=[])]
            mock_dg_instance.listen.prerecorded.v("1").transcribe_file.return_value = no_alt_response
            transcript_no_alt = service.transcribe_audio_chunk(self.sample_audio_chunk)
            self.assertEqual(transcript_no_alt, "")
        self.assertTrue(any("Deepgram transcription returned no alternatives in the first channel." in log_msg for log_msg in cm2.output))
        mock_dg_instance.listen.prerecorded.v("1").transcribe_file.return_value = MagicMock(results=None)
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm3:
            transcript_no_results_attr = service.transcribe_audio_chunk(self.sample_audio_chunk)
            self.assertEqual(transcript_no_results_attr, "")
        self.assertTrue(any("Deepgram transcription response was missing expected results, channels, or alternatives." in log_msg for log_msg in cm3.output))
        mock_dg_instance.listen.prerecorded.v("1").transcribe_file.return_value = None
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm4:
            transcript_none_response = service.transcribe_audio_chunk(self.sample_audio_chunk)
            self.assertEqual(transcript_none_response, "")
        self.assertTrue(any("Deepgram transcription response was missing expected results, channels, or alternatives." in log_msg for log_msg in cm4.output))

    @patch('input_processing_services.os.getenv')
    @patch('input_processing_services.DeepgramClient')
    def test_deepgram_api_key_missing_fallback_to_simulation(self, MockDeepgramClient, mock_getenv):
        config_no_api_key = {"speech_to_text_service": {"provider": "deepgram", "deepgram_settings": {"api_key": ""}}}
        mock_getenv.return_value = None
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm:
            service = SpeechToTextService(config=config_no_api_key)
        self.assertEqual(service.provider, "simulation")
        self.assertTrue(any("SpeechToTextService: No valid Deepgram API key resolved, falling back to simulation mode." in log_msg for log_msg in cm.output) or \
                        any("No Deepgram API key configured via placeholder '', direct config, or default environment variable DEEPGRAM_API_KEY." in log_msg for log_msg in cm.output) )
        if hasattr(service, '_sim_counter'): service._sim_counter = 0
        transcript = service.transcribe_audio_chunk(self.sample_audio_chunk)
        self.assertEqual(transcript, "Hello, who is this?")

    def test_deepgram_client_none_raises_stterror_if_provider_deepgram(self):
        with patch('input_processing_services.DeepgramClientOptions'), \
             patch('input_processing_services.DeepgramClient'):
            service = SpeechToTextService(config=self.base_config)
        service.provider = "deepgram"
        service.dg_client = None
        with self.assertRaises(STTError) as context, \
             self.assertLogs(self.logger_name_stt, level='ERROR') as cm:
            service.transcribe_audio_chunk(self.sample_audio_chunk)
        self.assertIn("Deepgram client not initialized. Cannot transcribe.", str(context.exception))
        self.assertTrue(any("Deepgram provider selected, but client not initialized." in log_msg for log_msg in cm.output))

class TestTextToSpeechService(unittest.TestCase):
    # ... (Content remains the same)
    def setUp(self):
        self.sample_text = "Hello, this is a test."
        self.elevenlabs_config_ok = {
            "text_to_speech_service": {
                "provider": "elevenlabs",
                "elevenlabs_settings": {
                    "api_key": "test_el_api_key",
                    "default_voice_id": "voice_id_123",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
                }
            }
        }
        self.sesame_csm_config_ok = {
            "text_to_speech_service": {
                "provider": "sesame_csm",
                "sesame_csm_settings": {"service_url": "http://fake-sesame.url"}
            }
        }
        self.simulation_config = {"text_to_speech_service": {"provider": "simulation"}}
        self.logger_name_tts = 'response_generation_services'

    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"elevenlabs_audio_bytes"
        mock_post.return_value = mock_response
        service = TextToSpeechService(config=self.elevenlabs_config_ok)
        service.resolved_elevenlabs_api_key = "test_el_api_key"

        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech(self.sample_text)

        self.assertEqual(audio, b"elevenlabs_audio_bytes")
        expected_url = "https://api.elevenlabs.io/v1/text-to-speech/voice_id_123"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], expected_url)
        self.assertEqual(call_args[1]['json']['text'], self.sample_text)
        self.assertEqual(call_args[1]['headers']['xi-api-key'], "test_el_api_key")
        self.assertTrue(any(f"TTSService (elevenlabs): Synthesizing speech for: '{self.sample_text}'" in log for log in cm.output))

    def test_synthesize_elevenlabs_api_key_missing(self):
        config_no_key = {
            "text_to_speech_service": {
                "provider": "elevenlabs",
                "elevenlabs_settings": {"default_voice_id": "voice_id_123", "api_key": ""}
            }
        }
        with patch('response_generation_services.os.getenv', return_value=None), \
             self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service = TextToSpeechService(config=config_no_key)
            service.synthesize_speech(self.sample_text)

        self.assertIn("ElevenLabs API key is not configured.", str(context.exception))
        self.assertTrue(any("ElevenLabs API key could not be resolved" in log for log in cm.output))
        self.assertTrue(any("ElevenLabs API key not found/resolved. Cannot synthesize speech." in log for log in cm.output))


    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.elevenlabs_config_ok)
        service.resolved_elevenlabs_api_key = "test_el_api_key"
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)

        self.assertIn("ElevenLabs API request failed with status 401: Invalid API key", str(context.exception))
        self.assertTrue(any("ElevenLabs API HTTP error: 401 - Invalid API key" in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        service = TextToSpeechService(config=self.elevenlabs_config_ok)
        service.resolved_elevenlabs_api_key = "test_el_api_key"
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)

        self.assertIn("Failed to connect to ElevenLabs API: Failed to connect", str(context.exception))
        self.assertTrue(any("ElevenLabs API request error: Failed to connect" in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_sesame_csm_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"sesame_audio_bytes"
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.sesame_csm_config_ok)
        service.resolved_sesame_csm_url = "http://fake-sesame.url"
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech(self.sample_text)

        self.assertEqual(audio, b"sesame_audio_bytes")
        expected_url = "http://fake-sesame.url/generate-speech"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], expected_url)
        self.assertEqual(call_args[1]['json']['text'], self.sample_text)
        self.assertTrue(any(f"TTSService (sesame_csm): Synthesizing speech for: '{self.sample_text}'" in log for log in cm.output))


    def test_synthesize_sesame_csm_url_missing(self):
        config_no_url = {"text_to_speech_service": {"provider": "sesame_csm", "sesame_csm_settings": {"service_url":""}}}
        with patch('response_generation_services.os.getenv', return_value=None), \
             self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service = TextToSpeechService(config=config_no_url)
            service.synthesize_speech(self.sample_text)

        self.assertIn("Sesame CSM service URL is not configured.", str(context.exception))
        self.assertTrue(any("Sesame CSM Service URL could not be resolved" in log for log in cm.output))
        self.assertTrue(any("Sesame CSM service URL not found/resolved. Cannot synthesize speech." in log for log in cm.output))


    @patch('response_generation_services.requests.post')
    def test_synthesize_sesame_csm_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.sesame_csm_config_ok)
        service.resolved_sesame_csm_url = "http://fake-sesame.url"
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)

        self.assertIn("Sesame CSM API request failed with status 500: Server error", str(context.exception))
        self.assertTrue(any("Sesame CSM API HTTP error: 500 - Server error" in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_sesame_csm_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        service = TextToSpeechService(config=self.sesame_csm_config_ok)
        service.resolved_sesame_csm_url = "http://fake-sesame.url"
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)
        self.assertIn("Failed to connect to Sesame CSM API: Request timed out", str(context.exception))
        self.assertTrue(any("Sesame CSM API request error: Request timed out" in log for log in cm.output))

    def test_synthesize_simulation_provider(self):
        service = TextToSpeechService(config=self.simulation_config)
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech(self.sample_text)

        expected_sim_audio = f"simulated_audio_bytes_for_[{self.sample_text.replace(' ','_')[:30]}]_emotion_neutral".encode('utf-8')
        self.assertEqual(audio, expected_sim_audio)
        self.assertTrue(any(f"Using simulated TTS for provider: simulation" in log_msg for log_msg in cm.output))

    def test_synthesize_empty_text_simulation(self):
        service = TextToSpeechService(config=self.simulation_config)
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech("")

        expected_sim_audio = f"simulated_audio_bytes_for_[]_emotion_neutral".encode('utf-8')
        self.assertEqual(audio, expected_sim_audio)
        self.assertTrue(any("TTSService (simulation): Synthesizing speech for: ''" in log for log in cm.output))


class TestAcousticEmotionAnalyzerService(unittest.TestCase):
    def setUp(self):
        self.logger_name_aea = "input_processing_services"
        self.sample_audio_chunk = b"sample_audio_data_for_emotion"

    def test_initialization_defaults(self):
        service = AcousticEmotionAnalyzerService(service_config={})
        self.assertEqual(service.model_path, "default_emotion_model.pkl")
        self.assertEqual(service.sample_rate, 22050)

    def test_initialization_with_config(self):
        config = {
            "model_path": "${CUSTOM_EMOTION_MODEL:-custom/model.pkl}",
            "sample_rate": 16000
        }

        original_env_val = os.environ.get("CUSTOM_EMOTION_MODEL")
        os.environ["CUSTOM_EMOTION_MODEL"] = "env_model.pkl"
        try:
            service = AcousticEmotionAnalyzerService(service_config=config)
            self.assertEqual(service.model_path, "env_model.pkl")
            self.assertEqual(service.sample_rate, 16000)
        finally:
            if original_env_val is None:
                del os.environ["CUSTOM_EMOTION_MODEL"]
            else:
                os.environ["CUSTOM_EMOTION_MODEL"] = original_env_val

        if "CUSTOM_EMOTION_MODEL" in os.environ:
            original_env_val_for_default_test = os.environ.get("CUSTOM_EMOTION_MODEL")
            del os.environ["CUSTOM_EMOTION_MODEL"]
        else:
            original_env_val_for_default_test = None

        try:
             service_placeholder_default = AcousticEmotionAnalyzerService(service_config=config)
             self.assertEqual(service_placeholder_default.model_path, "custom/model.pkl")
        finally:
            if original_env_val_for_default_test is not None:
                 os.environ["CUSTOM_EMOTION_MODEL"] = original_env_val_for_default_test


    def test_analyze_emotion_from_audio_non_empty(self):
        service = AcousticEmotionAnalyzerService(service_config={})
        initial_emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]

        first_result = service.analyze_emotion_from_audio(self.sample_audio_chunk)
        self.assertIn("dominant_emotion", first_result)
        self.assertIn("probabilities", first_result)
        self.assertIsInstance(first_result["dominant_emotion"], str)
        self.assertIsInstance(first_result["probabilities"], dict)
        self.assertAlmostEqual(sum(first_result["probabilities"].values()), 1.0, places=5)
        self.assertIn(first_result["dominant_emotion"], initial_emotions)

        seen_emotions = set()
        for _ in range(len(initial_emotions) + 2):
            result = service.analyze_emotion_from_audio(self.sample_audio_chunk)
            seen_emotions.add(result["dominant_emotion"])
        self.assertEqual(seen_emotions, set(initial_emotions))


    def test_analyze_emotion_from_audio_empty(self):
        service = AcousticEmotionAnalyzerService(service_config={})
        result = service.analyze_emotion_from_audio(b"")
        self.assertEqual(result["dominant_emotion"], "neutral")
        self.assertEqual(result["probabilities"], {"neutral": 1.0})


class TestNaturalLanguageGenerationService(unittest.TestCase):
    def setUp(self):
        self.logger_name_nlg = "response_generation_services"
        self.base_config_nlg = {
            "natural_language_generation_service": {
                "provider": "local_llama",
                "local_llama_settings": {"model_path": "${LLAMA_MODEL_TEST:-/default/path.gguf}"}
            }
        }
        self.sample_prompt = "Tell me a joke."
        self.env_var_name = "LLAMA_MODEL_TEST"

    def tearDown(self): # Corrected: This will hold the original value for the specific test method context
        if hasattr(self, f"original_env_val_{self.env_var_name}"): # Check for specific test method's backup
            original_val = getattr(self, f"original_env_val_{self.env_var_name}")
            if original_val is None:
                if self.env_var_name in os.environ:
                    del os.environ[self.env_var_name]
            else:
                os.environ[self.env_var_name] = original_val


    def test_initialization_with_env_var(self):
        env_value = "env_llama.gguf"
        # Store original env value if it exists, for this specific var name
        setattr(self, f"original_env_val_{self.env_var_name}", os.environ.get(self.env_var_name))
        os.environ[self.env_var_name] = env_value

        service_env = NaturalLanguageGenerationService(config=self.base_config_nlg)
        self.assertEqual(service_env.provider, "local_llama")
        self.assertEqual(service_env.settings.get("model_path"), env_value)
        # tearDown will restore/delete

    def test_initialization_with_placeholder_default(self):
        placeholder_default = "/default/path.gguf"
        # Store original env value if it exists, for this specific var name
        setattr(self, f"original_env_val_{self.env_var_name}", os.environ.get(self.env_var_name))
        if self.env_var_name in os.environ:
            del os.environ[self.env_var_name] # Ensure it's not set for this test

        self.assertIsNone(os.getenv(self.env_var_name))
        service_default = NaturalLanguageGenerationService(config=self.base_config_nlg)
        self.assertEqual(service_default.settings.get("model_path"), placeholder_default)
        # tearDown will restore original (which was None or some value if we didn't del it here properly)

    def test_generate_text_response_clarify(self):
        service = NaturalLanguageGenerationService(config=self.base_config_nlg)
        prompt = "Can you clarify that for me?"
        context = {"last_user_utterance": "your previous point"}
        with self.assertLogs(self.logger_name_nlg, level='INFO'):
            response = service.generate_text_response(prompt, context_data=context)
        self.assertIn("Could you please elaborate a bit more on that?", response)
        self.assertIn(context["last_user_utterance"], response)

    def test_generate_text_response_summarize(self):
        service = NaturalLanguageGenerationService(config=self.base_config_nlg)
        prompt = "Summarize our discussion."
        context = {"key_topics": ["topic A", "topic B"]}
        with self.assertLogs(self.logger_name_nlg, level='INFO'):
            response = service.generate_text_response(prompt, context_data=context)
        self.assertIn("To summarize, we discussed", response)
        self.assertIn("topic A", response)
        self.assertIn("topic B", response)

    def test_generate_text_response_generic(self):
        service = NaturalLanguageGenerationService(config=self.base_config_nlg)
        with self.assertLogs(self.logger_name_nlg, level='INFO'):
            response = service.generate_text_response(self.sample_prompt)
        self.assertIn(f"Based on your query about '{self.sample_prompt[:30]}...'", response)

    def test_generate_text_response_no_context(self):
        service = NaturalLanguageGenerationService(config=self.base_config_nlg)
        prompt_clarify = "Please clarify."
        prompt_summarize = "Summarize that."

        response_clarify = service.generate_text_response(prompt_clarify)
        self.assertIn("'something'. Could you please elaborate", response_clarify)

        response_summarize = service.generate_text_response(prompt_summarize)
        self.assertIn("['several important points']", response_summarize)


class TestRasaService(unittest.TestCase):
    def setUp(self):
        self.logger_name_rasa = "dialogue_manager_service"
        self.base_rasa_config = {
            "rasa_service": {
                "server_url": "${RASA_SERVER_URL_TEST:-http://default-rasa.com:5005}"
            }
        }
        self.sample_user_id = "test_user_rasa"
        self.sample_text_input = "hello there"
        self.sample_emotion_input = {"dominant_emotion": "neutral"}
        self.env_var_name_rasa = "RASA_SERVER_URL_TEST"

    def tearDown(self):
        if hasattr(self, 'original_env_val_rasa'):
            if self.original_env_val_rasa is None:
                if self.env_var_name_rasa in os.environ:
                    del os.environ[self.env_var_name_rasa]
            else:
                os.environ[self.env_var_name_rasa] = self.original_env_val_rasa

    def test_initialization(self):
        self.original_env_val_rasa = os.environ.get(self.env_var_name_rasa)

        os.environ[self.env_var_name_rasa] = "http://env-rasa.com:5005"
        service_env = RasaService(service_config=self.base_rasa_config["rasa_service"])
        self.assertEqual(service_env.server_url, "http://env-rasa.com:5005")

        del os.environ[self.env_var_name_rasa]
        self.assertIsNone(os.getenv(self.env_var_name_rasa))
        service_default = RasaService(service_config=self.base_rasa_config["rasa_service"])
        self.assertEqual(service_default.server_url, "http://default-rasa.com:5005")

    @patch('dialogue_manager_service.requests.post')
    @patch('dialogue_manager_service.print')
    def test_process_user_message_success(self, mock_print, mock_post):
        mock_rasa_response = MagicMock()
        mock_rasa_response.status_code = 200
        mock_rasa_response.json.return_value = {
            "intent": {"name": "greet", "confidence": 0.95},
            "entities": [{"entity": "name", "value": "Alex"}]
        }
        mock_post.return_value = mock_rasa_response

        service = RasaService(service_config=self.base_rasa_config["rasa_service"])
        service.server_url = "http://default-rasa.com:5005"

        action_plan = service.process_user_message(
            self.sample_user_id, self.sample_text_input, self.sample_emotion_input
        )

        mock_post.assert_called_once_with(
            f"{service.server_url}/model/parse",
            json={"text": self.sample_text_input}
        )
        self.assertEqual(action_plan["intent"], "greet")
        self.assertEqual(action_plan["entities"].get("name"), "Alex")
        self.assertEqual(action_plan["next_specialist"], "empathy_specialist")
        self.assertEqual(action_plan["response_emotion_hint"], "friendly")
        self.assertTrue(any("RasaService: Parsed intent='greet'" in call_args[0][0] for call_args in mock_print.call_args_list))


    @patch('dialogue_manager_service.requests.post')
    @patch('dialogue_manager_service.print')
    def test_process_user_message_http_error(self, mock_print, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_post.return_value = mock_response

        service = RasaService(service_config=self.base_rasa_config["rasa_service"])
        service.server_url = "http://default-rasa.com:5005"

        action_plan = service.process_user_message(
            self.sample_user_id, "any text", self.sample_emotion_input
        )

        self.assertEqual(action_plan["intent"], "rasa_communication_error")
        self.assertEqual(action_plan["next_specialist"], "nlg_service")
        self.assertTrue(any("RasaService: Communication error with Rasa server" in call_args[0][0] for call_args in mock_print.call_args_list))

    @patch('dialogue_manager_service.requests.post')
    @patch('dialogue_manager_service.print')
    def test_process_user_message_connection_error(self, mock_print, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        service = RasaService(service_config=self.base_rasa_config["rasa_service"])
        service.server_url = "http://another-rasa.com:5005"

        action_plan = service.process_user_message(
            self.sample_user_id, "any text", self.sample_emotion_input
        )

        self.assertEqual(action_plan["intent"], "rasa_communication_error")
        self.assertEqual(action_plan["next_specialist"], "nlg_service")
        self.assertTrue(any("RasaService: Communication error with Rasa server (http://another-rasa.com:5005): Connection failed" in call_args[0][0] for call_args in mock_print.call_args_list))

    @patch('dialogue_manager_service.print')
    def test_process_user_message_invalid_url(self, mock_print):
        invalid_url_config = {"rasa_service": {"server_url": "rasa_server_no_http"}}
        service = RasaService(service_config=invalid_url_config["rasa_service"])

        action_plan = service.process_user_message(
            self.sample_user_id, "any text", self.sample_emotion_input
        )
        self.assertEqual(action_plan["intent"], "rasa_configuration_error")
        self.assertEqual(action_plan["next_specialist"], "nlg_service")
        self.assertTrue(any("RasaService: Configuration error: Invalid Rasa server URL: rasa_server_no_http" in call_args[0][0] for call_args in mock_print.call_args_list))


# No module-level patches for tf, sqlite3, psycopg2 needed for current RealEstateKnowledgeService implementation
# as it doesn't import or use them directly at module level or load models/DBs in __init__.
class TestRealEstateKnowledgeService(unittest.TestCase):
    def setUp(self):
        self.sample_re_config = {
            "tf_model_base_path": "${RE_TF_MODEL_BASE_PATH:-/app/ml_models/}",
            "property_embedding_model_weights": "prop_embed.h5", # Not used in current init
            "prospect_embedding_model_weights": "prospect_embed.h5", # Not used in current init
            "matching_model_weights": "matching_model.h5", # Not used in current init
            "database_connections": {
                "mls_db": {
                    "type": "postgresql",
                    "host": "${MLS_DB_HOST:-localhost}",
                    "port": "${MLS_DB_PORT:-5432}",
                    "user": "${MLS_DB_USER:-user}",
                    "password": "${MLS_DB_PASS:-pass}",
                    "database_name": "${MLS_DB_NAME:-db}"
                },
                "regional_metrics_db": {
                    "type": "sqlite",
                    "path": "${RE_REGIONAL_DB_PATH:-/regional.db}"
                }
            },
            "real_estate_glossary_path": "${RE_GLOSSARY_PATH:-glossary.json}",
            "legal_document_templates_path": "${RE_LEGAL_DOCS_PATH:-legal_docs/}"
        }
        self.logger_name_reks = "specialist_sales_services"

        # These mocks are not needed for the current __init__ test, but might be useful later
        # self.mock_prop_embed_model = MagicMock()
        # self.mock_prospect_embed_model = MagicMock()
        # self.mock_matching_model = MagicMock()

    @patch('specialist_sales_services.resolve_config_value') # Patched where it's looked up by the service
    # No os.path.exists or tf.keras.models.load_model or db.connect patches needed for current __init__
    def test_initialization_config_resolution_and_mocked_loading(
        self, mock_resolve_config_value # Only mock_resolve_config_value is needed now
    ):
        resolved_tf_base_path = "/resolved/app/ml_models/"
        resolved_regional_db_path = "/resolved/regional.db"
        resolved_glossary_path = "/resolved/glossary.json"
        resolved_legal_docs_path = "/resolved/legal_docs/"
        resolved_db_host = "resolved_host"
        resolved_db_port_str = "1234"
        resolved_db_port_int = 1234
        resolved_db_user = "resolved_user"
        resolved_db_pass = "resolved_pass"
        resolved_db_name = "resolved_dbname"

        # CORRECTED SIGNATURE HERE:
        def resolve_side_effect(value_from_config, default_if_placeholder_not_set=None, target_type=str):
            # Debug print to see call arguments:
            # print(f"resolve_config_value CALLED WITH: ('{value_from_config}', default='{default_if_placeholder_not_set}', type={target_type})")
            if value_from_config == self.sample_re_config["tf_model_base_path"]:
                return resolved_tf_base_path
            elif value_from_config == self.sample_re_config["database_connections"]["mls_db"]["host"]:
                return resolved_db_host
            elif value_from_config == self.sample_re_config["database_connections"]["mls_db"]["port"]:
                return resolved_db_port_int if target_type is int else resolved_db_port_str
            elif value_from_config == self.sample_re_config["database_connections"]["mls_db"]["user"]:
                return resolved_db_user
            elif value_from_config == self.sample_re_config["database_connections"]["mls_db"]["password"]:
                return resolved_db_pass
            elif value_from_config == self.sample_re_config["database_connections"]["mls_db"]["database_name"]:
                return resolved_db_name
            elif value_from_config == self.sample_re_config["database_connections"]["regional_metrics_db"]["path"]:
                return resolved_regional_db_path
            elif value_from_config == self.sample_re_config["real_estate_glossary_path"]:
                return resolved_glossary_path
            elif value_from_config == self.sample_re_config["legal_document_templates_path"]:
                return resolved_legal_docs_path
            # Fallback for any other unexpected call to resolve_config_value
            return f"UNMOCKED_CONFIG_VALUE_PASSED_TO_RESOLVE:_{value_from_config}"


        mock_resolve_config_value.side_effect = resolve_side_effect

        service = RealEstateKnowledgeService(service_config=self.sample_re_config)

        self.assertEqual(service.tf_model_base_path, resolved_tf_base_path)
        # Corrected attribute names based on RealEstateKnowledgeService.__init__
        self.assertEqual(service.regional_db_path, resolved_regional_db_path)
        self.assertEqual(service.real_estate_glossary_path, resolved_glossary_path)
        self.assertEqual(service.legal_document_templates_path, resolved_legal_docs_path)

        expected_mls_db_settings = {
            "type": "postgresql", # This is a default in the service's __init__
            "host": resolved_db_host,
            "port": resolved_db_port_int, # Stored as int
            "user": resolved_db_user,
            "password": resolved_db_pass,
            "database_name": resolved_db_name
        }
        self.assertEqual(service.mls_db_settings, expected_mls_db_settings)

        # Assertions for resolve_config_value calls based on RealEstateKnowledgeService.__init__
        # The order of these calls in the __init__ method matters for assert_has_calls.
        # CORRECTED unittest.mock.call arguments
        calls = [
            unittest.mock.call(self.sample_re_config['tf_model_base_path'], default_if_placeholder_not_set='/app/ml_models/'),
            unittest.mock.call(self.sample_re_config['database_connections']['mls_db']['host'], "localhost"),
            unittest.mock.call(self.sample_re_config['database_connections']['mls_db']['port'], 5432, target_type=int),
            unittest.mock.call(self.sample_re_config['database_connections']['mls_db']['user'], ""),
            unittest.mock.call(self.sample_re_config['database_connections']['mls_db']['password'], ""),
            unittest.mock.call(self.sample_re_config['database_connections']['mls_db']['database_name'], "mls_data"),
            unittest.mock.call(self.sample_re_config['database_connections']['regional_metrics_db']['path'], "/path/to/data/real_estate/regional_metrics.db"),
            unittest.mock.call(self.sample_re_config['real_estate_glossary_path'], "path/to/data/real_estate/glossary.json"),
            unittest.mock.call(self.sample_re_config['legal_document_templates_path'], "path/to/data/real_estate/legal_docs/")
        ]
        mock_resolve_config_value.assert_has_calls(calls, any_order=False)
        self.assertEqual(mock_resolve_config_value.call_count, len(calls))

    def test_get_property_details_simulated_response(self):
        """
        Tests the get_property_details method for its current simulated response.
        """
        service = RealEstateKnowledgeService(service_config=self.sample_re_config)
        address_input = "123 Main St"
        expected_output = {
            "address": address_input,
            "price": 500000,
            "beds": 3,
            "baths": 2,
            "sqft": 1800,
            "status": "Available",
            "description": "Charming colonial in a quiet neighborhood."
        }

        # Patch print to check log messages if necessary
        with patch('specialist_sales_services.print') as mock_print:
            actual_output = service.get_property_details(address_input)

        self.assertEqual(actual_output, expected_output)
        mock_print.assert_any_call(f"RealEstateKnowledge: Getting details for '{address_input}'.")

    def test_find_matching_properties_simulated_response(self):
        """
        Tests the find_matching_properties method for its current simulated response.
        """
        service = RealEstateKnowledgeService(service_config=self.sample_re_config)
        prospect_profile_input = {'name': 'Test Prospect', 'budget': 750000}
        expected_output = [
            {"address": "123 Elm Street, Springfield", "price": 760000, "beds": 4, "baths": 3, "sqft": 2500, "match_score": 0.92, "reason": "Great location fit and features, slightly above ideal budget but high overall match."},
            {"address": "456 Oak Avenue, Springfield", "price": 720000, "beds": 3, "baths": 2.5, "sqft": 2200, "match_score": 0.88, "reason": "Excellent price fit, good style match, meets all core needs."},
        ]

        with patch('specialist_sales_services.print') as mock_print:
            actual_output = service.find_matching_properties(prospect_profile_input)

        self.assertEqual(actual_output, expected_output)
        mock_print.assert_any_call(f"RealEstateKnowledge: Finding matches for prospect: {prospect_profile_input['name']}")

    def test_get_market_analysis_simulated_response(self):
        """
        Tests the get_market_analysis method for its current simulated response.
        """
        service = RealEstateKnowledgeService(service_config=self.sample_re_config)
        area_criteria_input = {'location': 'Springfield Downtown'}
        expected_output = {
            "area": area_criteria_input['location'],
            "avg_price": 650000,
            "trend": "stable",
            "days_on_market": 35,
            "advice": "A steady market."
        }

        with patch('specialist_sales_services.print') as mock_print:
            actual_output = service.get_market_analysis(area_criteria_input)

        self.assertEqual(actual_output, expected_output)
        mock_print.assert_any_call(f"RealEstateKnowledge: Getting market analysis for area: {area_criteria_input['location']}")


class TestSalesAgentService(unittest.TestCase):
    def setUp(self):
        self.mock_app_config = { # Simulating the structure of the main app_config
            "sales_agent_service": {
                "default_sales_stage": "${DEFAULT_STAGE:-greeting}"
            }
        }
        self.mock_generic_sales_service = MagicMock(spec=GenericSalesSkillService)
        self.mock_real_estate_service = MagicMock(spec=RealEstateKnowledgeService)

        # We need to resolve the config for SalesAgentService before passing it
        # For the test, we can manually provide the resolved part or mock resolve_config_value here too
        # Assuming 'greeting' is the expected resolved default for this test setup

        # Patch resolve_config_value just for the instantiation of SalesAgentService in setUp
        # to ensure 'default_sales_stage' is resolved predictably.
        with patch('specialist_sales_services.resolve_config_value', return_value="greeting") as mock_resolve_for_setup:
            self.sales_agent_service = SalesAgentService(
                config=self.mock_app_config, # Pass the whole app_config
                generic_sales_service=self.mock_generic_sales_service,
                real_estate_service=self.mock_real_estate_service
            )
            # Check if resolve_config_value was called during setup as expected
            mock_resolve_for_setup.assert_called_once_with(
                self.mock_app_config['sales_agent_service'].get('default_sales_stage', "greeting"), "greeting"
            )

        self.logger_name_sas = "specialist_sales_services" # Assuming logger name from service file

    def test_initialization(self):
        """
        Tests that the SalesAgentService initializes with its dependencies and default stage.
        """
        self.assertIs(self.sales_agent_service.generic_sales, self.mock_generic_sales_service)
        self.assertIs(self.sales_agent_service.real_estate, self.mock_real_estate_service)
        # The default_sales_stage is resolved during __init__
        # Based on the setup patch, it should be "greeting"
        self.assertEqual(self.sales_agent_service.default_sales_stage, "greeting")
        # If current_stage is set directly from default_sales_stage in init without further logic:
        # self.assertEqual(self.sales_agent_service.current_stage, "greeting")
        # However, SalesAgentService.__init__ does not set current_stage. It's set in generate_sales_response.

    def test_generate_sales_response_greeting_stage(self):
        """
        Tests response generation when in the 'greeting' stage.
        """
        sales_context = {"stage": "greeting", "prospect_profile": {"type": "warm_lead"}}
        user_input = {"text": "Hello", "intent": "greet"}
        expected_opening_line = "Mocked opening line for warm lead."

        self.mock_generic_sales_service.get_opening_line.return_value = expected_opening_line

        # The method modifies sales_context in-place and returns only the response string.
        response_text = self.sales_agent_service.generate_sales_response(
            sales_context, user_input, {} # empty dict for emotion_data
        )

        self.mock_generic_sales_service.get_opening_line.assert_called_once_with(
            sales_context.get("prospect_profile")
        )
        self.assertEqual(response_text, expected_opening_line)
        self.assertEqual(sales_context["stage"], "discovery_initial") # Check context modification

    def test_generate_sales_response_handles_objection(self):
        """
        Tests response generation when handling a user objection.
        Assumes SalesAgentService delegates to GenericSalesSkillService.
        """
        sales_context = {"stage": "presentation", "some_key": "some_value"}
        user_input = {"text": "It's too expensive", "intent": "inform_objection"}
        expected_objection_response = "Mocked objection response to price."

        self.mock_generic_sales_service.handle_objection.return_value = expected_objection_response

        response_text = self.sales_agent_service.generate_sales_response(
            sales_context, user_input, {}
        )

        # This assertion will likely fail with current SalesAgentService implementation
        self.mock_generic_sales_service.handle_objection.assert_called_once_with(
            objection_text=user_input["text"],
            sales_context=sales_context # The service passes the whole context
        )
        self.assertEqual(response_text, expected_objection_response)
        # Optionally, assert if the stage should change or remain the same
        # self.assertEqual(sales_context["stage"], "presentation_objection_handled")

    def test_generate_sales_response_property_query_uses_re_service(self):
        """
        Tests response generation for a property query, expecting delegation
        to RealEstateKnowledgeService.
        """
        sales_context = {"stage": "discovery", "current_property_discussion": None}
        user_input = {
            "text": "Tell me about 123 Main St",
            "intent": "query_property",
            "entities": {"property_id": "123 Main St"}
        }
        expected_property_details = "Details for 123 Main St: It's lovely and spacious."

        self.mock_real_estate_service.get_property_details.return_value = expected_property_details

        response_text = self.sales_agent_service.generate_sales_response(
            sales_context, user_input, {}
        )

        # This assertion will likely fail with current SalesAgentService implementation
        self.mock_real_estate_service.get_property_details.assert_called_once_with(
            address_or_mls=user_input["entities"]["property_id"]
        )
        self.assertEqual(response_text, expected_property_details)
        # Assuming the service updates the context with the property details it fetched (or the fact it discussed it)
        # This part of the assertion depends on the exact design of context update logic.
        # For example, it might store the returned string or a structured version:
        # self.assertEqual(sales_context["current_property_discussion"], expected_property_details)
        # Or simply:
        self.assertIsNotNone(sales_context.get("current_property_discussion"), "Context should be updated after property query.")
        # Let's refine this based on actual (future) implementation if it stores something specific.
        # For now, just checking it's not None might be too vague if it doesn't store anything.
        # If the service is designed to put the *result* of get_property_details into context:
        # self.assertEqual(sales_context["current_property_discussion"], expected_property_details)

    def test_generate_sales_response_closing_stage(self):
        """
        Tests response generation during the closing stage, expecting delegation
        to GenericSalesSkillService for a closing technique.
        """
        sales_context = {"stage": "closing_arguments", "strong_interest_expressed": True}
        user_input = {"text": "I'm ready to proceed", "intent": "affirm"}
        expected_closing_response = "Mocked closing technique: Let's sign the papers!"

        self.mock_generic_sales_service.suggest_closing_technique.return_value = expected_closing_response

        response_text = self.sales_agent_service.generate_sales_response(
            sales_context, user_input, {}
        )

        # This assertion will likely fail with current SalesAgentService implementation
        self.mock_generic_sales_service.suggest_closing_technique.assert_called_once_with(
            sales_context=sales_context
        )
        self.assertEqual(response_text, expected_closing_response)
        # Optionally, assert context changes, e.g., stage moving to "deal_closed"
        # self.assertEqual(sales_context["stage"], "deal_closed")

    def test_generate_sales_response_fallback_if_no_specific_handler(self):
        """
        Tests the fallback response when no specific handler for the intent/stage is met.
        """
        sales_context = {"stage": "some_undefined_stage", "some_key": "some_value"}
        user_input = {"text": "Random comment", "intent": "inform"}

        # Since this is testing fallback, we ensure specific handlers are NOT called.
        # The mocks are already configured in setUp to be MagicMocks.

        response_text = self.sales_agent_service.generate_sales_response(
            sales_context, user_input, {}
        )

        # Assert that specific delegation methods were NOT called
        self.mock_generic_sales_service.get_opening_line.assert_not_called()
        self.mock_generic_sales_service.handle_objection.assert_not_called()
        self.mock_generic_sales_service.suggest_closing_technique.assert_not_called()
        self.mock_real_estate_service.get_property_details.assert_not_called()
        self.mock_real_estate_service.find_matching_properties.assert_not_called()
        self.mock_real_estate_service.get_market_analysis.assert_not_called()

        # Assert the expected fallback response
        self.assertEqual(response_text, "Sales response based on intent and stage.")
        # Context stage should not have been modified from "some_undefined_stage" by the fallback
        self.assertEqual(sales_context["stage"], "some_undefined_stage")


if __name__ == '__main__':
    unittest.main()
