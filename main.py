# main_agent.py
# Main orchestrator for the Voice Agent, bringing all services together.

import yaml
import time # For simulation delays
import threading
import asyncio
import os # For os.getenv
import sys # Added for stderr printing
# import re # For regex parsing of placeholders - Handled by config_utils # re is used by resolve_config_value if it were here

from dotenv import load_dotenv

# Load environment variables from .env file
# This should be one of the first things to do
load_dotenv()

# Import the resolver utility
from config_utils import resolve_config_value

# from run_mcps_agent import main as run_mcps # Temporarily commented out

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

class ServiceManager:
    def __init__(self, config: dict, agent_type: str):
        if not config:
            raise ValueError("Configuration is required to initialize ServiceManager.")
        self.config = config
        self.agent_type = agent_type

        print(f"\nServiceManager: Initializing services for agent type: {self.agent_type}...")

        # Pass the entire global config to services that manage their own sub-config directly
        # or where sub-config extraction is straightforward and doesn't need resolver yet.
        self.twilio_service = TwilioService(config=self.config)
        self.stt_service = SpeechToTextService(config=self.config)
        self.nlg_service = NaturalLanguageGenerationService(config=self.config)
        self.tts_service = TextToSpeechService(config=self.config)
        self.empathy_specialist_service = EmpathySpecialistService(config=self.config)

        # Pass specific sub-configs to services that expect it and use resolver internally
        self.acoustic_analyzer_service = AcousticEmotionAnalyzerService(service_config=self.config.get('acoustic_emotion_analyzer_service', {}))
        self.rasa_service = RasaService(service_config=self.config.get('rasa_service', {}))

        self.generic_sales_skill_service = None
        self.real_estate_knowledge_service = None
        self.sales_agent_specialist_service = None

        if "sales" in self.agent_type.lower():
            self.generic_sales_skill_service = GenericSalesSkillService(service_config=self.config.get('generic_sales_skill_service', {}))
            if "real_estate" in self.agent_type.lower():
                self.real_estate_knowledge_service = RealEstateKnowledgeService(service_config=self.config.get('real_estate_knowledge_service', {}))

            self.sales_agent_specialist_service = SalesAgentService(
                config=self.config,
                generic_sales_service=self.generic_sales_skill_service,
                real_estate_service=self.real_estate_knowledge_service
            )
        print(f"ServiceManager: Services initialization complete for {self.agent_type}.\n")

class SpecialistDispatcher:
    def __init__(self, service_manager: ServiceManager):
        self.sm = service_manager

    def dispatch(self, action_plan: dict, conversation_history: list,
                 user_emotion_data: dict, sales_context: dict = None) -> tuple[str, str]:
        response_text = ""
        specialist_used = "N/A"
        last_user_text = conversation_history[-1]['text'] if conversation_history and conversation_history[-1]['speaker'] == 'user' else ""

        if action_plan.get("next_specialist") == "empathy_specialist":
            specialist_used = "EmpathySpecialist"
            response_text = self.sm.empathy_specialist_service.generate_empathetic_response(
                context={"history": conversation_history, "user_intent": action_plan.get("intent")},
                emotion_data=user_emotion_data
            )
        elif action_plan.get("next_specialist") == "sales_agent" and self.sm.sales_agent_specialist_service:
            specialist_used = "SalesAgentSpecialist"
            response_text = self.sm.sales_agent_specialist_service.generate_sales_response(
                sales_context=sales_context,
                user_input_details={"text": last_user_text,
                                    "intent": action_plan.get("intent"),
                                    "entities": action_plan.get("entities")},
                emotion_data=user_emotion_data
            )
        elif action_plan.get("next_specialist") == "nlg_direct_answer":
            specialist_used = "NLGService (Direct)"
            prompt = f"User said: '{last_user_text}'. Dominant emotion: {user_emotion_data.get('dominant_emotion')}. Recent history: {conversation_history[-3:]}. Provide a concise, helpful response."
            response_text = self.sm.nlg_service.generate_text_response(
                prompt,
                context_data={"last_user_utterance": last_user_text,
                              "key_topics": action_plan.get("entities", {}).values()}
            )
        else:
            specialist_used = "NLGService (Fallback)"
            response_text = self.sm.nlg_service.generate_text_response(
                f"I received: '{last_user_text}'. How can I further assist?",
                context_data={"last_user_utterance": last_user_text}
            )
        return response_text, specialist_used

class CallHandler:
    def __init__(self, config: dict, service_manager: ServiceManager):
        self.config = config
        self.sm = service_manager
        self.specialist_dispatcher = SpecialistDispatcher(service_manager)

    def run_simulation(self, call_sid: str, num_turns: int = 3):
        print(f"--- Simulating Call {call_sid} (Handler) ---")
        audio_stream = self.sm.twilio_service.start_audio_stream(call_sid)
        user_id = f"user_sim_{call_sid}"
        conversation_history = []
        sales_context = {}

        if self.sm.sales_agent_specialist_service:
            sales_agent_service_config = self.config.get('sales_agent_service', {})
            default_stage_raw = sales_agent_service_config.get('default_sales_stage', "greeting")
            sales_context = {
                "stage": resolve_config_value(default_stage_raw, "greeting"),
                "prospect_profile": {},
                "current_property_discussion": None,
                "call_history": []
            }

        for turn in range(num_turns):
            print(f"\n[Turn {turn + 1}]")
            user_audio_chunk = self.sm.twilio_service.receive_audio_from_caller(audio_stream)
            if not user_audio_chunk and turn > 0 :
                print("User audio stream ended or user hung up.")
                break

            user_text = self.sm.stt_service.transcribe_audio_chunk(user_audio_chunk)
            user_emotion_data = self.sm.acoustic_analyzer_service.analyze_emotion_from_audio(user_audio_chunk)

            print(f"User (Emotion: {user_emotion_data.get('dominant_emotion')}): {user_text}")
            conversation_history.append({"speaker": "user", "text": user_text, "emotion": user_emotion_data.get('dominant_emotion')})
            if self.sm.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "user", "text": user_text})

            action_plan = self.sm.rasa_service.process_user_message(user_id, user_text, user_emotion_data)
            response_text, specialist_used = self.specialist_dispatcher.dispatch(
                action_plan, conversation_history, user_emotion_data, sales_context
            )

            print(f"Agent (using {specialist_used}, emotion hint: {action_plan.get('response_emotion_hint')}): {response_text}")
            conversation_history.append({"speaker": "agent", "text": response_text, "specialist": specialist_used})
            if self.sm.sales_agent_specialist_service:
                 sales_context["call_history"].append({"speaker": "agent", "text": response_text})

            tts_service_config = self.config.get('text_to_speech_service', {})
            sesame_settings = tts_service_config.get('sesame_csm_settings', {})
            default_voice = "professional_warm"
            voice_profile_from_config = resolve_config_value(sesame_settings.get('default_voice_profile'), default_voice)


            agent_audio_chunk = self.sm.tts_service.synthesize_speech(
                response_text,
                voice_profile=voice_profile_from_config,
                emotion_hint=action_plan.get("response_emotion_hint")
            )
            self.sm.twilio_service.send_audio_to_caller(audio_stream, agent_audio_chunk)

            if action_plan.get("end_call"):
                print("\nAgent determined it's time to end the call.")
                break
            time.sleep(0.5)

        audio_stream.close()
        print(f"--- Call Simulation {call_sid} Ended (Handler) ---")
        if self.sm.sales_agent_specialist_service:
            print(f"Final Sales Context for {call_sid}: {sales_context}")

class VoiceAgent:
    def __init__(self, config: dict, agent_type_override: str = None):
        if not config:
            raise ValueError("Configuration is required to initialize VoiceAgent.")
        self.config = config
        application_config = self.config.get('application', {})
        default_agent_type_raw = application_config.get('default_agent_type', "empathy_base")
        default_agent_type = resolve_config_value(default_agent_type_raw, "empathy_base")
        self.agent_type = agent_type_override if agent_type_override else default_agent_type

        print(f"\nInitializing VoiceAgent of type: {self.agent_type}...")
        self.service_manager = ServiceManager(config=self.config, agent_type=self.agent_type)
        print(f"VoiceAgent ({self.agent_type}) initialization complete.\n")

    def simulate_call_interaction(self, call_sid: str, num_turns: int = 3):
        print(f"--- VoiceAgent delegating call {call_sid} for agent type {self.agent_type} ---")
        call_handler = CallHandler(config=self.config, service_manager=self.service_manager)
        call_handler.run_simulation(call_sid, num_turns)

if __name__ == "__main__":
    app_config = load_config()

    if app_config:
        # Optional MCPS integration
        mcps_config = app_config.get('mcps_integration', {})
        mcps_enabled = resolve_config_value(mcps_config.get('enabled'), target_type=bool, default_if_placeholder_not_set=False)
        mcps_server_url_resolved = resolve_config_value(mcps_config.get('mcps_server_url'), default_if_placeholder_not_set="local_url_not_set")

        if mcps_enabled: # Check mcps_enabled directly, app_config is already confirmed
            print(f"MCPS integration is enabled. Attempting to start MCPS agent. (MCPS Server URL from config: {mcps_server_url_resolved})")
            try:
                import run_mcps_agent # Test simple import first
                run_mcps = run_mcps_agent.main # Then access main
                # Ensure threading and asyncio are imported (already done at top of main.py)

                # Define a wrapper for the thread target to handle asyncio loop
                def mcps_thread_target():
                    try:
                        asyncio.run(run_mcps())
                    except Exception as e:
                        print(f"Error in MCPS thread: {e}", file=sys.stderr)

                threading.Thread(target=mcps_thread_target, daemon=True).start()
                print("MCPS agent thread started.")
            except ImportError as e:
                print(f"Failed to import or start MCPS agent: {e}", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred while trying to start MCPS agent: {e}", file=sys.stderr)
        else:
            print("MCPS integration is not enabled in the configuration.")

        print("\n" + "*"*10 + " DEMO: Base Empathetic Agent " + "*"*10)
        base_empathy_agent = VoiceAgent(config=app_config, agent_type_override="empathy_base")
        base_empathy_agent.simulate_call_interaction(call_sid="empathy_call_001", num_turns=3)

        print("\n\n" + "="*40 + "\n\n")

        print("*"*10 + " DEMO: Real Estate Sales Agent " + "*"*10)
        real_estate_sales_agent = VoiceAgent(config=app_config, agent_type_override="real_estate_sales")
        real_estate_sales_agent.simulate_call_interaction(call_sid="sales_call_001", num_turns=5)
    else:
        print("Could not run demos due to configuration loading issues.")
