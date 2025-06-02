# main_agent.py
# Main orchestrator for the Voice Agent, bringing all services together.

import yaml
import time # For simulation delays

# Import services from their respective files
from infrastructure_services import TwilioService, SimulatedAudioStream
from input_processing_services import SpeechToTextService, AcousticEmotionAnalyzerService
from dialogue_manager_service import RasaService
from specialist_empathy_service import EmpathySpecialistService
from specialist_sales_services import GenericSalesSkillService, RealEstateKnowledgeService, SalesAgentService
from response_generation_services import NaturalLanguageGenerationService, TextToSpeechService

def load_config(config_path="config.yml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}. Please create it.")
        return None
    except Exception as e:
        print(f"ERROR: Could not load or parse configuration file {config_path}: {e}")
        return None

class VoiceAgent:
    """
    The main orchestrator for the voice agent.
    It initializes and coordinates all other services based on loaded configuration.
    """
    def __init__(self, config: dict, agent_type_override: str = None):
        if not config:
            raise ValueError("Configuration is required to initialize VoiceAgent.")
        self.config = config
        
        # Determine agent type: override > config > default
        self.agent_type = agent_type_override if agent_type_override else self.config.get('application', {}).get('default_agent_type', "empathy_base")
        
        print(f"\nInitializing VoiceAgent of type: {self.agent_type}...")

        # Infrastructure
        self.twilio_service = TwilioService(config=self.config)

        # Input Processing
        self.stt_service = SpeechToTextService(config=self.config)
        self.acoustic_analyzer_service = AcousticEmotionAnalyzerService(config=self.config)

        # Core Logic
        self.rasa_service = RasaService(config=self.config)

        # Response Generation
        self.nlg_service = NaturalLanguageGenerationService(config=self.config)
        self.tts_service = TextToSpeechService(config=self.config)

        # Base Specialist
        self.empathy_specialist_service = EmpathySpecialistService(config=self.config)

        # Sales Components (loaded based on agent_type)
        self.generic_sales_skill_service = None
        self.real_estate_knowledge_service = None
        self.sales_agent_specialist_service = None

        if "sales" in self.agent_type.lower():
            self.generic_sales_skill_service = GenericSalesSkillService(config=self.config)
            if "real_estate" in self.agent_type.lower(): # Specific check for real estate niche
                self.real_estate_knowledge_service = RealEstateKnowledgeService(config=self.config)
            
            self.sales_agent_specialist_service = SalesAgentService(
                config=self.config, # Pass config to SalesAgentService
                generic_sales_service=self.generic_sales_skill_service,
                real_estate_service=self.real_estate_knowledge_service
            )
        
        print(f"VoiceAgent ({self.agent_type}) initialization complete.\n")

    def simulate_call_interaction(self, call_sid: str, num_turns: int = 3):
        """Simulates a call interaction for demonstration."""
        print(f"--- Simulating Call {call_sid} with {self.agent_type} Agent ---")
        
        audio_stream = self.twilio_service.start_audio_stream(call_sid)
        user_id = f"user_sim_{call_sid}" # In real scenario, this might be the caller's number
        conversation_history = []
        
        sales_context = {} # Initialize sales_context
        if self.sales_agent_specialist_service:
            sales_context = {
                "stage": self.config.get('sales_agent_service',{}).get('default_sales_stage', "greeting"), 
                "prospect_profile": {}, 
                "current_property_discussion": None,
                "call_history": [] # To store summary of interactions in this call
            }

        for turn in range(num_turns):
            print(f"\n[Turn {turn + 1}]")
            
            user_audio_chunk = self.twilio_service.receive_audio_from_caller(audio_stream)
            if not user_audio_chunk and turn > 0 : # If user hangs up or stream ends mid-conversation
                print("User audio stream ended or user hung up.")
                break
            
            user_text = self.stt_service.transcribe_audio_chunk(user_audio_chunk)
            user_emotion_data = self.acoustic_analyzer_service.analyze_emotion_from_audio(user_audio_chunk)
            
            print(f"User (Emotion: {user_emotion_data.get('dominant_emotion')}): {user_text}")
            conversation_history.append({"speaker": "user", "text": user_text, "emotion": user_emotion_data.get('dominant_emotion')})
            if self.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "user", "text": user_text})


            action_plan = self.rasa_service.process_user_message(user_id, user_text, user_emotion_data)
            response_text = ""
            specialist_used = "N/A"

            if action_plan.get("next_specialist") == "empathy_specialist":
                specialist_used = "EmpathySpecialist"
                response_text = self.empathy_specialist_service.generate_empathetic_response(
                    context={"history": conversation_history, "user_intent": action_plan.get("intent")},
                    emotion_data=user_emotion_data
                )
            elif action_plan.get("next_specialist") == "sales_agent" and self.sales_agent_specialist_service:
                specialist_used = "SalesAgentSpecialist"
                response_text = self.sales_agent_specialist_service.generate_sales_response(
                    sales_context=sales_context,
                    user_input_details={"text": user_text, "intent": action_plan.get("intent"), "entities": action_plan.get("entities")},
                    emotion_data=user_emotion_data
                )
            elif action_plan.get("next_specialist") == "nlg_direct_answer":
                specialist_used = "NLGService (Direct)"
                prompt = f"User said: '{user_text}'. Dominant emotion: {user_emotion_data.get('dominant_emotion')}. Recent history: {conversation_history[-3:]}. Provide a concise, helpful response."
                response_text = self.nlg_service.generate_text_response(prompt, context_data={"last_user_utterance": user_text, "key_topics": action_plan.get("entities", {}).values()})
            else: 
                specialist_used = "NLGService (Fallback)"
                response_text = self.nlg_service.generate_text_response(f"I received: '{user_text}'. How can I further assist?", context_data={"last_user_utterance": user_text})

            print(f"Agent (using {specialist_used}, emotion hint: {action_plan.get('response_emotion_hint')}): {response_text}")
            conversation_history.append({"speaker": "agent", "text": response_text, "specialist": specialist_used})
            if self.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "agent", "text": response_text})


            agent_audio_chunk = self.tts_service.synthesize_speech(
                response_text,
                voice_profile=self.config.get('text_to_speech_service',{}).get('sesame_csm_settings',{}).get('default_voice_profile','professional_warm'),
                emotion_hint=action_plan.get("response_emotion_hint")
            )
            self.twilio_service.send_audio_to_caller(audio_stream, agent_audio_chunk)

            if action_plan.get("end_call"):
                print("\nAgent determined it's time to end the call.")
                break
            
            time.sleep(0.5) # Small delay to make simulation readable

        audio_stream.close()
        print(f"--- Call Simulation {call_sid} Ended ---")
        if self.sales_agent_specialist_service:
            print(f"Final Sales Context for {call_sid}: {sales_context}")


if __name__ == "__main__":
    app_config = load_config()

    if app_config:
        # --- Demo 1: Base Empathetic Voice Agent ---
        print("*"*10 + " DEMO: Base Empathetic Agent " + "*"*10)
        # You can override agent_type here if needed, or set a default in config.yml
        base_empathy_agent = VoiceAgent(config=app_config, agent_type_override="empathy_base")
        base_empathy_agent.simulate_call_interaction(call_sid="empathy_call_001", num_turns=3)

        print("\n\n" + "="*40 + "\n\n")

        # --- Demo 2: Real Estate Sales Voice Agent ---
        print("*"*10 + " DEMO: Real Estate Sales Agent " + "*"*10)
        real_estate_sales_agent = VoiceAgent(config=app_config, agent_type_override="real_estate_sales")
        real_estate_sales_agent.simulate_call_interaction(call_sid="sales_call_001", num_turns=5)
    else:
        print("Could not run demos due to configuration loading issues.")

