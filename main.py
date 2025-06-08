# main_agent.py
# Main orchestrator for the Voice Agent, bringing all services together.

import yaml
import time # For simulation delays
import threading
import asyncio

# Attempt to import run_mcps, handle if not available
try:
    from run_mcps_agent import main as run_mcps
except ImportError:
    run_mcps = None
    print("WARNING: run_mcps_agent.py not found or cannot be imported. MCPS integration will be disabled.")

# Import services from their respective files
from infrastructure_services import TwilioService, SimulatedAudioStream
from input_processing_services import SpeechToTextService, AcousticEmotionAnalyzerService
from dialogue_manager_service import RasaService
from specialist_empathy_services import EmpathySpecialistService
from specialist_sales_services import GenericSalesSkillService, RealEstateKnowledgeService, SalesAgentService
from response_generation_services import NaturalLanguageGenerationService, TextToSpeechService

def load_config(config_path="config.yml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty config file
            print(f"WARNING: Configuration file {config_path} is empty. Using default empty config.")
            return {}
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}. Please create it. Returning empty config.")
        return {} # Return empty dict to allow services to potentially use defaults
    except Exception as e:
        print(f"ERROR: Could not load or parse configuration file {config_path}: {e}. Returning empty config.")
        return {} # Return empty dict

class VoiceAgent:
    """
    The main orchestrator for the voice agent.
    It initializes and coordinates all other services based on loaded configuration.
    """
    def __init__(self, config: dict, agent_type_override: str = None):
        if not isinstance(config, dict): # Check if config is a dict
            print("ERROR: Invalid configuration provided to VoiceAgent (not a dict). Raising ValueError.")
            raise ValueError("Configuration must be a dictionary.")
        self.config = config

        self.agent_type = agent_type_override if agent_type_override else self.config.get('application', {}).get('default_agent_type', "empathy_base")
        print(f"\nInitializing VoiceAgent of type: {self.agent_type}...")

        # Helper to initialize services
        def _initialize_service(service_class, service_name, init_params_dict):
            if not service_class:
                print(f"WARNING: Service class for {service_name} is None (likely import error). Service cannot be initialized.")
                return None
            try:
                return service_class(**init_params_dict)
            except Exception as e:
                print(f"WARNING: Failed to initialize {service_name}. Service will be unavailable. Error: {e}")
                return None

        # Infrastructure
        self.twilio_service = _initialize_service(TwilioService, "TwilioService", {"config": self.config})

        # Input Processing
        self.stt_service = _initialize_service(SpeechToTextService, "SpeechToTextService", {"config": self.config})
        self.acoustic_analyzer_service = _initialize_service(AcousticEmotionAnalyzerService, "AcousticEmotionAnalyzerService", {"config": self.config})

        # Core Logic
        self.rasa_service = _initialize_service(RasaService, "RasaService", {"config": self.config})

        # Response Generation
        self.nlg_service = _initialize_service(NaturalLanguageGenerationService, "NaturalLanguageGenerationService", {"config": self.config})
        self.tts_service = _initialize_service(TextToSpeechService, "TextToSpeechService", {"config": self.config})

        # Base Specialist
        self.empathy_specialist_service = _initialize_service(EmpathySpecialistService, "EmpathySpecialistService", {"config": self.config})

        # Sales Components
        self.generic_sales_skill_service = None
        self.real_estate_knowledge_service = None
        self.sales_agent_specialist_service = None

        if "sales" in self.agent_type.lower():
            self.generic_sales_skill_service = _initialize_service(GenericSalesSkillService, "GenericSalesSkillService", {"config": self.config})

            if "real_estate" in self.agent_type.lower():
                if self.generic_sales_skill_service: # Dependency check
                    self.real_estate_knowledge_service = _initialize_service(RealEstateKnowledgeService, "RealEstateKnowledgeService", {"config": self.config})
                else:
                    print("WARNING: RealEstateKnowledgeService not initialized because GenericSalesSkillService is unavailable.")

            if self.generic_sales_skill_service: # Dependency check
                # Pass real_estate_knowledge_service, even if it's None
                self.sales_agent_specialist_service = _initialize_service(
                    SalesAgentService,
                    "SalesAgentService",
                    {
                        "config": self.config,
                        "generic_sales_service": self.generic_sales_skill_service,
                        "real_estate_service": self.real_estate_knowledge_service
                    }
                )
            else:
                print("WARNING: SalesAgentService not initialized because GenericSalesSkillService is unavailable.")

        print(f"VoiceAgent ({self.agent_type}) initialization process complete.\n")

    def simulate_call_interaction(self, call_sid: str, num_turns: int = 3):
        """Simulates a call interaction for demonstration."""
        print(f"--- Simulating Call {call_sid} with {self.agent_type} Agent ---")

        if not self.twilio_service:
            print("ERROR: TwilioService is not available. Cannot simulate call.")
            return

        audio_stream = self.twilio_service.start_audio_stream(call_sid)
        if not audio_stream:
            print("ERROR: Failed to start audio stream via TwilioService. Cannot simulate call.")
            return

        user_id = f"user_sim_{call_sid}"
        conversation_history = []
        sales_context = {}

        if self.sales_agent_specialist_service:
            sales_context = {
                "stage": self.config.get('sales_agent_service',{}).get('default_sales_stage', "greeting"),
                "prospect_profile": {},
                "current_property_discussion": None,
                "call_history": []
            }

        for turn in range(num_turns):
            print(f"\n[Turn {turn + 1}]")

            user_audio_chunk = self.twilio_service.receive_audio_from_caller(audio_stream)
            if not user_audio_chunk and turn > 0:
                print("User audio stream ended or user hung up.")
                break

            user_text = "[STT_UNAVAILABLE]"
            if self.stt_service:
                user_text = self.stt_service.transcribe_audio_chunk(user_audio_chunk)
            else:
                print("WARNING: STT service not available, cannot transcribe audio. Using placeholder text.")

            user_emotion_data = {"dominant_emotion": "neutral", "error": "AcousticEmotionAnalyzerService unavailable"}
            if self.acoustic_analyzer_service:
                user_emotion_data = self.acoustic_analyzer_service.analyze_emotion_from_audio(user_audio_chunk)
            else:
                print("WARNING: AcousticEmotionAnalyzerService not available. Using neutral emotion.")

            print(f"User (Emotion: {user_emotion_data.get('dominant_emotion')}): {user_text}")
            conversation_history.append({"speaker": "user", "text": user_text, "emotion": user_emotion_data.get('dominant_emotion')})
            if self.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "user", "text": user_text})

            if not self.rasa_service:
                print("ERROR: RasaService is not available. Cannot determine action plan. Ending call simulation.")
                break
            action_plan = self.rasa_service.process_user_message(user_id, user_text, user_emotion_data)

            response_text = "I apologize, I am currently unable to process your request."
            specialist_used = "N/A"

            if action_plan.get("next_specialist") == "empathy_specialist" and self.empathy_specialist_service:
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
            elif action_plan.get("next_specialist") == "nlg_direct_answer" and self.nlg_service:
                specialist_used = "NLGService (Direct)"
                prompt = f"User said: '{user_text}'. Dominant emotion: {user_emotion_data.get('dominant_emotion')}. Recent history: {conversation_history[-3:]}. Provide a concise, helpful response."
                response_text = self.nlg_service.generate_text_response(prompt, context_data={"last_user_utterance": user_text, "key_topics": action_plan.get("entities", {}).values()})
            elif self.nlg_service: # Fallback to NLG if available and no other specialist was chosen or available
                specialist_used = "NLGService (Fallback)"
                response_text = self.nlg_service.generate_text_response(f"I received: '{user_text}'. How can I further assist?", context_data={"last_user_utterance": user_text})
            else:
                print("WARNING: NLGService is not available. Using placeholder fallback response.")
                # response_text is already set to a generic apology
                if not self.empathy_specialist_service and not self.sales_agent_specialist_service:
                     print("ERROR: No primary specialist services (Empathy, Sales, NLG) available. Cannot generate meaningful response.")

            print(f"Agent (using {specialist_used}, emotion hint: {action_plan.get('response_emotion_hint')}): {response_text}")
            conversation_history.append({"speaker": "agent", "text": response_text, "specialist": specialist_used})
            if self.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "agent", "text": response_text})

            agent_audio_chunk = None
            if self.tts_service:
                agent_audio_chunk = self.tts_service.synthesize_speech(
                    response_text,
                    voice_profile=self.config.get('text_to_speech_service',{}).get('sesame_csm_settings',{}).get('default_voice_profile','professional_warm'),
                    emotion_hint=action_plan.get("response_emotion_hint")
                )
            else:
                print("WARNING: TTS service not available, cannot synthesize agent speech.")

            if agent_audio_chunk and self.twilio_service: # Ensure twilio_service is still there
                self.twilio_service.send_audio_to_caller(audio_stream, agent_audio_chunk)
            elif not agent_audio_chunk:
                print("INFO: Skipping sending audio to caller as no audio was synthesized.")
            # If twilio_service is None, already handled at start of function

            if action_plan.get("end_call"):
                print("\nAgent determined it's time to end the call.")
                break

            time.sleep(0.5)

        if hasattr(audio_stream, 'close'):
            audio_stream.close()
        print(f"--- Call Simulation {call_sid} Ended ---")
        if self.sales_agent_specialist_service:
            print(f"Final Sales Context for {call_sid}: {sales_context}")


if __name__ == "__main__":
    app_config = load_config()
    if not app_config: # If config loading failed and returned empty dict (or some other falsy value)
        print("CRITICAL: Configuration could not be loaded. VoiceAgent demos cannot run.")
    else:
        # Optional MCPS integration
        if app_config.get('mcps_integration', {}).get('enabled'):
            if run_mcps: # Check if import was successful
                 threading.Thread(target=lambda: asyncio.run(run_mcps()), daemon=True).start()
                 print("MCPS integration enabled: starting MCPS agent in background.")
            else:
                print("WARNING: MCPS integration was enabled in config, but run_mcps could not be imported. MCPS will not run.")

        # --- Demo 1: Base Empathetic Voice Agent ---
        print("*"*10 + " DEMO: Base Empathetic Agent " + "*"*10)
        base_empathy_agent = VoiceAgent(config=app_config, agent_type_override="empathy_base")
        base_empathy_agent.simulate_call_interaction(call_sid="empathy_call_001", num_turns=3)

        print("\n\n" + "="*40 + "\n\n")

        # --- Demo 2: Real Estate Sales Voice Agent ---
        print("*"*10 + " DEMO: Real Estate Sales Agent " + "*"*10)
        real_estate_sales_agent = VoiceAgent(config=app_config, agent_type_override="real_estate_sales")
        real_estate_sales_agent.simulate_call_interaction(call_sid="sales_call_001", num_turns=5)
