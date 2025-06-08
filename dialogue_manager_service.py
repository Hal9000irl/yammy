# dialogue_manager_service.py
# Contains the RasaService for NLU and dialogue management.

import time # For simulation
import requests

class RasaService:
    """
    Handles Natural Language Understanding (NLU) and Dialogue Management.
    This would interact with a trained Rasa model/server.
    """
    def __init__(self, config: dict):
        self.config = config.get('rasa_service', {})
        self.server_url = self.config.get('server_url', 'http://localhost:5005')
        # self.model_path = self.config.get('model_path') # If loading model directly
        if self.config.get('server_url') is None: # Or if it's the default
            print(f"INFO: RasaService: No specific server_url found in config, using default: {self.server_url}. Ensure Rasa server is running at this base URL.")
        else:
            print(f"RasaService Initialized (Server Base URL: {self.server_url})")
        # Real: Load Rasa agent or configure API client to Rasa server
        # Example: from rasa.core.agent import Agent
        # if self.model_path: self.agent = Agent.load(self.model_path)

    def process_user_message(self, user_id: str, text_input: str, acoustic_emotion_input: dict = None) -> dict:
        """
        Processes user input and decides on the agent's next high-level action or specialist.
        Returns an action plan.
        """
        # Real: parse user message via Rasa NLU
        try:
            resp = requests.post(f"{self.server_url}/model/parse", json={"text": text_input})
            resp.raise_for_status()
            parsed = resp.json()
            intent = parsed.get("intent", {}).get("name")
            entities = {e.get("entity"): e.get("value") for e in parsed.get("entities", [])}
            print(f"RasaService: Parsed intent='{intent}', entities={entities}")
        except Exception as e:
            print(f"RasaService: Error calling Rasa NLU parse: {e}")
            intent = None
            entities = {}
        
        current_emotion = acoustic_emotion_input.get('dominant_emotion', 'N/A') if acoustic_emotion_input else 'N/A'
        print(f"RasaService: Processing for user {user_id}: '{text_input}' (Emotion: {current_emotion})")
        # Initialize action plan with parsed values (fallback to simulation values)
        action_plan = {"intent": intent or "unknown", "entities": entities, "next_specialist": "nlg_direct_answer", "response_emotion_hint": "neutral"}

        if not text_input: # Handle cases where STT might return empty string
            action_plan["intent"] = "silence"
            action_plan["next_specialist"] = "empathy_specialist"  # Maybe ask if everything is okay
            action_plan["response_emotion_hint"] = "gentle_query"
            print(f"RasaService: Action Plan (Silence): {action_plan}")
            return action_plan

        lower_text = text_input.lower()

        if intent == "greet" or "hello" in lower_text or "hi" in lower_text:
            action_plan["intent"] = "greet"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "friendly"
        elif "sad" == current_emotion or "not happy" in lower_text or "depressed" in lower_text:
            action_plan["intent"] = "expresses_sadness"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "empathetic"
        elif "angry" == current_emotion or "frustrated" in lower_text or "upset" in lower_text:
            action_plan["intent"] = "expresses_anger"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "calming_empathetic"
        elif "sell my house" in lower_text or "property" in lower_text or "market" in lower_text or "real estate" in lower_text:
            action_plan["intent"] = "inquire_real_estate"
            action_plan["entities"] = {"topic": "selling"} # Simplified entity extraction
            if "downtown" in lower_text:
                action_plan["entities"]["location"] = "downtown area"
            elif "suburbs" in lower_text:
                action_plan["entities"]["location"] = "suburban area"
            action_plan["next_specialist"] = "sales_agent"
            action_plan["response_emotion_hint"] = "professional_helpful"
        elif "expensive" in lower_text or "cost too much" in lower_text:
            action_plan["intent"] = "price_objection"
            action_plan["next_specialist"] = "sales_agent"
            action_plan["response_emotion_hint"] = "understanding_persuasive"
        elif "thank you" in lower_text or "thanks" in lower_text:
            action_plan["intent"] = "thank_you"
            action_plan["next_specialist"] = "empathy_specialist"
            action_plan["response_emotion_hint"] = "warm"
        elif "goodbye" in lower_text or "bye" in lower_text:
            action_plan["intent"] = "goodbye"
            action_plan["next_specialist"] = "empathy_specialist" # For a polite closing
            action_plan["response_emotion_hint"] = "polite_farewell"
            action_plan["end_call"] = True # Signal to end the interaction
        else: # Default fallback for unrecognized input
            action_plan["intent"] = "fallback"
            action_plan["next_specialist"] = "empathy_specialist" # Default to empathy for unknown
            action_plan["response_emotion_hint"] = "neutral_helpful"

        print(f"RasaService: Action Plan: {action_plan}")
        return action_plan

if __name__ == '__main__':
    dummy_config = {
        "rasa_service": {
            "server_url": "http://simulated-rasa-server:5005"
        }
    }
    rasa = RasaService(config=dummy_config)
    plan1 = rasa.process_user_message("user123", "Hello there!", {"dominant_emotion": "neutral"})
    plan2 = rasa.process_user_message("user123", "I want to sell my house in the suburbs.", {"dominant_emotion": "excited"})
    plan3 = rasa.process_user_message("user123", "That's too expensive.", {"dominant_emotion": "annoyed"})
    plan4 = rasa.process_user_message("user123", "Goodbye", {"dominant_emotion": "neutral"})

    print("\nTest Plan 1 (Greet):", plan1)
    print("Test Plan 2 (Real Estate Inquiry):", plan2)
    print("Test Plan 3 (Objection):", plan3)
    print("Test Plan 4 (Goodbye):", plan4)
