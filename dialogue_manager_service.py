# dialogue_manager_service.py
# Contains the RasaService for NLU and dialogue management.

import time # For simulation
import requests
import os # For os.getenv

from config_utils import resolve_config_value


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
            if not self.server_url or not (self.server_url.startswith("http://") or self.server_url.startswith("https://")):
                 raise ValueError(f"Invalid Rasa server URL: {self.server_url}")

            resp = requests.post(f"{self.server_url}/model/parse", json={"text": text_input})
            resp.raise_for_status()
            parsed = resp.json()
            intent = parsed.get("intent", {}).get("name")
            entities = {e.get("entity"): e.get("value") for e in parsed.get("entities", [])}
            print(f"RasaService: Parsed intent='{intent}', entities={entities}")
        except requests.exceptions.RequestException as e:
            print(f"RasaService: Communication error with Rasa server ({self.server_url}): {e}")
            intent = "rasa_communication_error"
            entities = {}
        except ValueError as e:
             print(f"RasaService: Configuration error: {e}")
             intent = "rasa_configuration_error"
             entities = {}
        except Exception as e:
            print(f"RasaService: Error calling Rasa NLU parse: {e}")
            intent = None
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

        if intent == "greet" or "hello" in lower_text or "hi" in lower_text:
            action_plan["intent"] = "greet"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "friendly"
        elif intent == "rasa_communication_error" or intent == "rasa_configuration_error":
            action_plan["next_specialist"] = "nlg_service"
            action_plan["response_emotion_hint"] = "neutral_apologetic"
        elif current_emotion == "sad" or intent == "expresses_sadness":
            action_plan["intent"] = "expresses_sadness"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "empathetic"
        elif current_emotion == "angry" or intent == "expresses_anger":
            action_plan["intent"] = "expresses_anger"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "calming_empathetic"
        elif intent == "inquire_real_estate" or "sell my house" in lower_text or "property" in lower_text:
            action_plan["intent"] = "inquire_real_estate"
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
        else:
            action_plan["intent"] = intent if intent else "fallback"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "neutral_helpful"

        print(f"RasaService: Action Plan: {action_plan}")
        return action_plan

if __name__ == '__main__':
    # This sys.path manipulation is for allowing direct execution of the service file
    # if config_utils is in the parent directory.
    # It needs to be done before attempting to import from config_utils
    import sys # Ensure sys is imported if not already
    if "config_utils" not in sys.modules:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try:
            from config_utils import resolve_config_value as main_resolve_config_value # Use if available
        except ImportError:
            # Define the fallback resolve_config_value if it's not available for __main__
            def main_resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
                if isinstance(value_from_config, str) and value_from_config.startswith("${") and value_from_config.endswith("}"):
                    var_name = value_from_config.strip("${}")
                    val = os.getenv(var_name, default_if_placeholder_not_set)
                    if target_type == int and val is not None: return int(val)
                    return val if target_type == str else None
                if target_type == int and value_from_config is not None: return int(value_from_config)
                return value_from_config
            print("Warning: Could not import resolve_config_value from config_utils. Using local fallback for __main__.")


    dummy_app_config = {
        "rasa_service": {
            "server_url": "${RASA_URL_TEST:-http://simulated-rasa-server:5005}"
        }
    }
    os.environ["RASA_URL_TEST"] = "http://env-rasa-server:5005"

    rasa_service_cfg = dummy_app_config['rasa_service']

    rasa = RasaService(service_config=rasa_service_cfg)
    print(f"Rasa server URL in __main__: {rasa.server_url}")

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
        return MockResponse({}, 404)

    with patch('requests.post', side_effect=mock_requests_post):
        plan1 = rasa.process_user_message("user123", "Hello there!", {"dominant_emotion": "neutral"})
        plan2 = rasa.process_user_message("user123", "I want to sell my house in the suburbs.", {"dominant_emotion": "excited"})

    print("\nTest Plan 1 (Greet):", plan1)
    print("Test Plan 2 (Real Estate Inquiry):", plan2)

    del os.environ["RASA_URL_TEST"]
