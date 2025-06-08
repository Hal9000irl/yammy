import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import requests # Required for requests.exceptions

# Add project root to sys.path to allow importing dialogue_manager_service
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dialogue_manager_service import RasaService

class TestRasaServiceFallback(unittest.TestCase):

    def test_rasa_server_unavailable_fallback(self):
        # Configuration that points to a non-existent server
        unavailable_config = {
            'rasa_service': {
                'server_url': 'http://localhost:12345' # Unlikely to be running
            }
        }
        # Suppress print output from RasaService during this test for cleaner test logs
        with patch('builtins.print') as mock_print:
            rasa_service_instance = RasaService(config=unavailable_config)

            # Mock requests.post to simulate a connection error
            with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Test connection error")):
                action_plan = rasa_service_instance.process_user_message(
                    user_id="test_user",
                    text_input="Hello there", # This should match keyword "hello"
                    acoustic_emotion_input={"dominant_emotion": "neutral"}
                )

        # Check that it fell back to keyword-based greeting
        self.assertEqual(action_plan.get("intent"), "greet")
        self.assertEqual(action_plan.get("next_specialist"), "empathy_specialist")
        self.assertEqual(action_plan.get("response_emotion_hint"), "friendly")

    def test_rasa_server_error_fallback_to_unknown(self):
        # Configuration that points to a non-existent server
        error_config = {
            'rasa_service': {
                'server_url': 'http://localhost:12346'
            }
        }
        with patch('builtins.print') as mock_print: # Suppress prints
            rasa_service_instance = RasaService(config=error_config)

            # Mock requests.post to simulate a generic server error (e.g., 500)
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test HTTP error")
            # If raise_for_status throws, the parsed json won't be accessed.
            # The intent will be None, then action_plan["intent"] becomes "unknown"

            with patch('requests.post', return_value=mock_response):
                 action_plan = rasa_service_instance.process_user_message(
                     user_id="test_user",
                     text_input="some unrecognized input string", # Should not match keywords
                     acoustic_emotion_input={"dominant_emotion": "neutral"}
                 )

        # With NLU failed and no keyword match, it should go to the final 'else'
        # The initial intent from failed NLU is "unknown".
        # The final 'else' block changes intent to "fallback".
        self.assertEqual(action_plan.get("intent"), "fallback")
        self.assertEqual(action_plan.get("next_specialist"), "empathy_specialist")
        self.assertEqual(action_plan.get("response_emotion_hint"), "neutral_helpful")

    def test_empty_input_silence_intent(self):
        default_config = {'rasa_service': {}} # Use default config
        with patch('builtins.print') as mock_print: # Suppress prints
            rasa_service_instance = RasaService(config=default_config)

            # No need to mock requests.post if text_input is empty, as it should be handled before the call
            action_plan = rasa_service_instance.process_user_message(
                user_id="test_user",
                text_input="", # Empty input
                acoustic_emotion_input={"dominant_emotion": "neutral"}
            )

        self.assertEqual(action_plan.get("intent"), "silence")
        self.assertEqual(action_plan.get("next_specialist"), "empathy_specialist")
        self.assertEqual(action_plan.get("response_emotion_hint"), "gentle_query")


if __name__ == '__main__':
    unittest.main()
