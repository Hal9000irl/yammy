import unittest
from unittest.mock import patch, MagicMock
import logging
import os
import sys
import requests # Needed for requests.exceptions

# Adjust sys.path to include the parent directory (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from input_processing_services import SpeechToTextService, STTError
from response_generation_services import TextToSpeechService, TTSError
# Import PrerecordedOptions for type checking if needed, or mocking its creation
from deepgram.clients.prerecorded.v1 import PrerecordedOptions


class TestSpeechToTextService(unittest.TestCase):

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
        service = SpeechToTextService(config=self.base_config)
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
            no_alt_response = MagicMock()
            no_alt_response.results.channels = [MagicMock(alternatives=[])]
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
        config_no_api_key = {
            "speech_to_text_service": {
                "provider": "deepgram", "deepgram_settings": {"api_key": ""}
            }
        }
        mock_getenv.return_value = None
        with self.assertLogs(self.logger_name_stt, level='WARNING') as cm:
            service = SpeechToTextService(config=config_no_api_key)

        self.assertEqual(service.provider, "simulation")
        self.assertTrue(any("SpeechToTextService: No valid Deepgram API key provided, falling back to simulation mode." in log_msg for log_msg in cm.output))

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
        self.assertTrue(any("Deepgram provider selected, but client not initialized (likely API key issue)." in log_msg for log_msg in cm.output))


class TestTextToSpeechService(unittest.TestCase):
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
        self.logger_name_tts = 'response_generation_services' # Logger used in TextToSpeechService

    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"elevenlabs_audio_bytes"
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.elevenlabs_config_ok)
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech(self.sample_text)

        self.assertEqual(audio, b"elevenlabs_audio_bytes")
        expected_url = "https://api.elevenlabs.io/v1/text-to-speech/voice_id_123"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], expected_url) # URL is the first positional arg
        self.assertEqual(call_args[1]['json']['text'], self.sample_text)
        self.assertEqual(call_args[1]['headers']['xi-api-key'], "test_el_api_key")
        self.assertTrue(any(f"TTSService (elevenlabs): Synthesizing speech for: '{self.sample_text}'" in log for log in cm.output))

    def test_synthesize_elevenlabs_api_key_missing(self):
        config_no_key = {
            "text_to_speech_service": {
                "provider": "elevenlabs",
                "elevenlabs_settings": {"default_voice_id": "voice_id_123"} # API key missing
            }
        }
        # Patch os.getenv to ensure it doesn't provide a fallback key
        with patch('response_generation_services.os.getenv', return_value=None), \
             self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service = TextToSpeechService(config=config_no_key)
            service.synthesize_speech(self.sample_text) # This call will trigger the error

        self.assertIn("ElevenLabs API key is not configured.", str(context.exception))
        self.assertTrue(any("ElevenLabs API key not found. Cannot synthesize speech." in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401 # Unauthorized
        mock_response.text = "Invalid API key"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.elevenlabs_config_ok)
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)

        self.assertIn("ElevenLabs API request failed with status 401: Invalid API key", str(context.exception))
        self.assertTrue(any("ElevenLabs API HTTP error: 401 - Invalid API key" in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_elevenlabs_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        service = TextToSpeechService(config=self.elevenlabs_config_ok)
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
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech(self.sample_text)

        self.assertEqual(audio, b"sesame_audio_bytes")
        expected_url = "http://fake-sesame.url/generate-speech"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], expected_url)
        self.assertEqual(call_args[1]['json']['text'], self.sample_text)
        self.assertTrue(any(f"TTSService (sesame_csm): Synthesizing speech for: '{self.sample_text}'" in log for log in cm.output))
        self.assertTrue(any("TextToSpeechService (SesameCSM): Calling http://fake-sesame.url/generate-speech" in log for log in cm.output))


    def test_synthesize_sesame_csm_url_missing(self):
        config_no_url = {"text_to_speech_service": {"provider": "sesame_csm", "sesame_csm_settings": {}}}
        with patch('response_generation_services.os.getenv', return_value=None), \
             self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service = TextToSpeechService(config=config_no_url)
            service.synthesize_speech(self.sample_text)

        self.assertIn("Sesame CSM service URL is not configured.", str(context.exception))
        self.assertTrue(any("Sesame CSM service URL not found. Cannot synthesize speech." in log for log in cm.output))


    @patch('response_generation_services.requests.post')
    def test_synthesize_sesame_csm_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        service = TextToSpeechService(config=self.sesame_csm_config_ok)
        with self.assertRaises(TTSError) as context, \
             self.assertLogs(self.logger_name_tts, level='ERROR') as cm:
            service.synthesize_speech(self.sample_text)

        self.assertIn("Sesame CSM API request failed with status 500: Server error", str(context.exception))
        self.assertTrue(any("Sesame CSM API HTTP error: 500 - Server error" in log for log in cm.output))

    @patch('response_generation_services.requests.post')
    def test_synthesize_sesame_csm_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        service = TextToSpeechService(config=self.sesame_csm_config_ok)
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
        """ Test TTS with empty text using simulation provider. """
        service = TextToSpeechService(config=self.simulation_config)
        with self.assertLogs(self.logger_name_tts, level='INFO') as cm:
            audio = service.synthesize_speech("") # Empty text

        expected_sim_audio = f"simulated_audio_bytes_for_[]_emotion_neutral".encode('utf-8')
        self.assertEqual(audio, expected_sim_audio)
        self.assertTrue(any("TTSService (simulation): Synthesizing speech for: ''" in log for log in cm.output))


if __name__ == '__main__':
    # If you want to run both test classes when executing this file directly:
    # Create a TestSuite
    # suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(TestSpeechToTextService))
    # suite.addTest(unittest.makeSuite(TestTextToSpeechService))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    # For simplicity, unittest.main() will discover both if run from command line as module
    unittest.main()
