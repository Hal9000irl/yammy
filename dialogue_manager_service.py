# dialogue_manager_service.py
# Contains the RasaService for NLU and dialogue management.

import time # For simulation
import requests
import os # For os.getenv

# Attempt to import resolve_config_value from main.
try:
    from main import resolve_config_value
except ImportError:
    # Fallback basic version if direct import fails (e.g., module run standalone)
    def resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
        if isinstance(value_from_config, str) and value_from_config.startswith("${") and value_from_config.endswith("}"):
            var_name = value_from_config.strip("${}")
            # Basic resolution without default-in-placeholder support from pattern
            val = os.getenv(var_name, default_if_placeholder_not_set)
            return target_type(val) if val is not None and target_type is not None else val
        return target_type(value_from_config) if value_from_config is not None and target_type is not None else value_from_config


class RasaService:
    """
    Handles Natural Language Understanding (NLU) and Dialogue Management.
    This would interact with a trained Rasa model/server.
    """
    def __init__(self, service_config: dict): # Changed to accept service_config directly
        self.config = service_config # Use the passed service_config

        raw_server_url = self.config.get('server_url', 'http://localhost:5005') # Default if not in config
        self.server_url = resolve_config_value(raw_server_url, default_if_placeholder_not_set='http://localhost:5005')

        print(f"RasaService Initialized (Server URL: {self.server_url})")

    def process_user_message(self, user_id: str, text_input: str, acoustic_emotion_input: dict = None) -> dict:
        """
        Processes user input and decides on the agent's next high-level action or specialist.
        Returns an action plan.
        """
        try:
            # Ensure server_url is valid before making a request
            if not self.server_url or not (self.server_url.startswith("http://") or self.server_url.startswith("https://")):
                 raise ValueError(f"Invalid Rasa server URL: {self.server_url}")

            resp = requests.post(f"{self.server_url}/model/parse", json={"text": text_input})
            resp.raise_for_status()
            parsed = resp.json()
            intent = parsed.get("intent", {}).get("name")
            entities = {e.get("entity"): e.get("value") for e in parsed.get("entities", [])}
            print(f"RasaService: Parsed intent='{intent}', entities={entities}")
        except requests.exceptions.RequestException as e: # Catches network errors, HTTP errors via raise_for_status
            print(f"RasaService: Communication error with Rasa server ({self.server_url}): {e}")
            intent = "rasa_communication_error" # Custom intent for this case
            entities = {}
        except ValueError as e: # Catches invalid URL
             print(f"RasaService: Configuration error: {e}")
             intent = "rasa_configuration_error"
             entities = {}
        except Exception as e: # Other unexpected errors
            print(f"RasaService: Error calling Rasa NLU parse: {e}")
            intent = None # Fallback to default handling
            entities = {}

        current_emotion = acoustic_emotion_input.get('dominant_emotion', 'N/A') if acoustic_emotion_input else 'N/A'
        print(f"RasaService: Processing for user {user_id}: '{text_input}' (Emotion: {current_emotion})")

        action_plan = {"intent": intent or "unknown", "entities": entities, "next_specialist": "nlg_direct_answer", "response_emotion_hint": "neutral"}

        if not text_input:
            action_plan["intent"] = "silence"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "gentle_query"
            print(f"RasaService: Action Plan (Silence): {action_plan}")
            return action_plan

        lower_text = text_input.lower()

        # Simplified rule-based dialogue logic based on intent or keywords
        if intent == "greet" or "hello" in lower_text or "hi" in lower_text:
            action_plan["intent"] = "greet" # Ensure intent is set if matched by keyword
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "friendly"
        elif intent == "rasa_communication_error" or intent == "rasa_configuration_error":
            # If Rasa itself is down or misconfigured, use a safe fallback
            action_plan["next_specialist"] = "nlg_service" # Generic NLG
            action_plan["response_emotion_hint"] = "neutral_apologetic"
            # Potentially set a specific message like "I'm having trouble understanding right now."
        elif current_emotion == "sad" or intent == "expresses_sadness":
            action_plan["intent"] = "expresses_sadness"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "empathetic"
        elif current_emotion == "angry" or intent == "expresses_anger":
            action_plan["intent"] = "expresses_anger"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "calming_empathetic"
        elif intent == "inquire_real_estate" or "sell my house" in lower_text or "property" in lower_text:
            action_plan["intent"] = "inquire_real_estate" # Ensure intent
            # Example entity extraction (can be improved or rely on Rasa's if working)
            if not entities.get("topic") and "sell" in lower_text: entities["topic"] = "selling"
            if not entities.get("topic") and "buy" in lower_text: entities["topic"] = "buying"
            action_plan["entities"] = entities
            action_plan["next_specialist"] = "sales_agent"
            action_plan["response_emotion_hint"] = "professional_helpful"
        elif intent == "price_objection":
            action_plan["next_specialist"] = "sales_agent"
            action_plan["response_emotion_hint"] = "understanding_persuasive"
        elif intent == "thank_you":
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "warm"
        elif intent == "goodbye":
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "polite_farewell"
            action_plan["end_call"] = True
        else: # Default fallback
            action_plan["intent"] = intent if intent else "fallback" # Use parsed intent if available
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "neutral_helpful"

        print(f"RasaService: Action Plan: {action_plan}")
        return action_plan

if __name__ == '__main__':
    # Example of how config would be structured in the main app_config
    dummy_app_config = {
        "rasa_service": {
            "server_url": "${RASA_URL_TEST:-http://simulated-rasa-server:5005}"
        }
    }
    # Simulate setting an environment variable for testing
    os.environ["RASA_URL_TEST"] = "http://env-rasa-server:5005"

    # Pass only the relevant part of the config to the service
    rasa = RasaService(service_config=dummy_app_config['rasa_service'])

    # Test with a mock for requests.post if server isn't running
    # For direct execution, we need to import patch from unittest.mock
    from unittest.mock import patch
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
        def json(self): return self.json_data
        def raise_for_status(self):
            if self.status_code >= 400: raise requests.exceptions.HTTPError(f"Error {self.status_code}")

    def mock_requests_post(url, json):
        if "parse" in url:
            if json['text'] == "Hello there!":
                return MockResponse({"intent": {"name": "greet"}, "entities": []}, 200)
            if "sell my house" in json['text']:
                 return MockResponse({"intent": {"name": "inquire_real_estate"}, "entities": [{"entity":"topic", "value":"selling"}]}, 200)
        return MockResponse({}, 404) # Default mock response

    with patch('requests.post', side_effect=mock_requests_post):
        plan1 = rasa.process_user_message("user123", "Hello there!", {"dominant_emotion": "neutral"})
        plan2 = rasa.process_user_message("user123", "I want to sell my house in the suburbs.", {"dominant_emotion": "excited"})

    print("\nTest Plan 1 (Greet):", plan1) # Should use intent from mock
    print("Test Plan 2 (Real Estate Inquiry):", plan2) # Should use intent from mock

    del os.environ["RASA_URL_TEST"] # Clean up
# Removed the stray ``` marker from the end of the file.
