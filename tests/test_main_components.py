import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Adjust sys.path to include the parent directory (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ServiceManager, SpecialistDispatcher, CallHandler # Classes to test

from infrastructure_services import TwilioService
from input_processing_services import SpeechToTextService, AcousticEmotionAnalyzerService
from dialogue_manager_service import RasaService
from response_generation_services import NaturalLanguageGenerationService, TextToSpeechService
from specialist_empathy_service import EmpathySpecialistService
from specialist_sales_services import (
    GenericSalesSkillService,
    RealEstateKnowledgeService,
    SalesAgentService
)

class TestServiceManager(unittest.TestCase):
    def setUp(self):
        self.sample_base_config = {
            "application": {"default_agent_type": "empathy_base"},
            "twilio_service": {"account_sid": "ACxxxx", "auth_token": "testtoken"},
            "speech_to_text_service": {"provider": "deepgram"},
            "acoustic_emotion_analyzer_service": {"model_path": "emotion_model.pkl"},
            "rasa_service": {"server_url": "http://rasa:5005"},
            "natural_language_generation_service": {"provider": "openai"},
            "text_to_speech_service": {"provider": "elevenlabs"},
            "empathy_specialist_service": {"model_path": "empathy_model.pkl"},
            "generic_sales_skill_service": {"model_path": "generic_sales_model.pkl"},
            "real_estate_knowledge_service": {"tf_model_base_path": "/models/"},
            "sales_agent_service": {"default_sales_stage": "greeting"}
        }
        self.logger_name_sm = "main"

    @patch('main.SalesAgentService')
    @patch('main.RealEstateKnowledgeService')
    @patch('main.GenericSalesSkillService')
    @patch('main.EmpathySpecialistService')
    @patch('main.TextToSpeechService')
    @patch('main.NaturalLanguageGenerationService')
    @patch('main.RasaService')
    @patch('main.AcousticEmotionAnalyzerService')
    @patch('main.SpeechToTextService')
    @patch('main.TwilioService')
    def test_initialization_empathy_base_agent(
        self, MockTwilioService, MockSpeechToTextService, MockAcousticEmotionAnalyzerService,
        MockRasaService, MockNaturalLanguageGenerationService, MockTextToSpeechService,
        MockEmpathySpecialistService, MockGenericSalesSkillService, MockRealEstateKnowledgeService,
        MockSalesAgentService
    ):
        agent_type = "empathy_base"
        manager = ServiceManager(config=self.sample_base_config, agent_type=agent_type)
        MockTwilioService.assert_called_once_with(config=self.sample_base_config)
        MockSpeechToTextService.assert_called_once_with(config=self.sample_base_config)
        MockAcousticEmotionAnalyzerService.assert_called_once_with(
            service_config=self.sample_base_config.get('acoustic_emotion_analyzer_service', {})
        )
        MockRasaService.assert_called_once_with(
            service_config=self.sample_base_config.get('rasa_service', {})
        )
        MockNaturalLanguageGenerationService.assert_called_once_with(config=self.sample_base_config)
        MockTextToSpeechService.assert_called_once_with(config=self.sample_base_config)
        MockEmpathySpecialistService.assert_called_once_with(config=self.sample_base_config)
        MockGenericSalesSkillService.assert_not_called()
        MockRealEstateKnowledgeService.assert_not_called()
        MockSalesAgentService.assert_not_called()
        self.assertIs(manager.twilio_service, MockTwilioService.return_value)
        self.assertIs(manager.stt_service, MockSpeechToTextService.return_value)
        self.assertIs(manager.acoustic_analyzer_service, MockAcousticEmotionAnalyzerService.return_value)
        self.assertIs(manager.rasa_service, MockRasaService.return_value)
        self.assertIs(manager.nlg_service, MockNaturalLanguageGenerationService.return_value)
        self.assertIs(manager.tts_service, MockTextToSpeechService.return_value)
        self.assertIs(manager.empathy_specialist_service, MockEmpathySpecialistService.return_value)
        self.assertIsNone(manager.generic_sales_skill_service)
        self.assertIsNone(manager.real_estate_knowledge_service)
        self.assertIsNone(manager.sales_agent_specialist_service)

    @patch('main.SalesAgentService')
    @patch('main.RealEstateKnowledgeService')
    @patch('main.GenericSalesSkillService')
    @patch('main.EmpathySpecialistService')
    @patch('main.TextToSpeechService')
    @patch('main.NaturalLanguageGenerationService')
    @patch('main.RasaService')
    @patch('main.AcousticEmotionAnalyzerService')
    @patch('main.SpeechToTextService')
    @patch('main.TwilioService')
    def test_initialization_real_estate_sales_agent(
        self, MockTwilioService, MockSpeechToTextService, MockAcousticEmotionAnalyzerService,
        MockRasaService, MockNaturalLanguageGenerationService, MockTextToSpeechService,
        MockEmpathySpecialistService, MockGenericSalesSkillService, MockRealEstateKnowledgeService,
        MockSalesAgentService
    ):
        agent_type = "real_estate_sales"
        manager = ServiceManager(config=self.sample_base_config, agent_type=agent_type)
        MockTwilioService.assert_called_once_with(config=self.sample_base_config)
        MockSpeechToTextService.assert_called_once_with(config=self.sample_base_config)
        MockAcousticEmotionAnalyzerService.assert_called_once_with(service_config=self.sample_base_config.get('acoustic_emotion_analyzer_service', {}))
        MockRasaService.assert_called_once_with(service_config=self.sample_base_config.get('rasa_service', {}))
        MockNaturalLanguageGenerationService.assert_called_once_with(config=self.sample_base_config)
        MockTextToSpeechService.assert_called_once_with(config=self.sample_base_config)
        MockEmpathySpecialistService.assert_called_once_with(config=self.sample_base_config)
        MockGenericSalesSkillService.assert_called_once_with(service_config=self.sample_base_config.get('generic_sales_skill_service', {}))
        MockRealEstateKnowledgeService.assert_called_once_with(service_config=self.sample_base_config.get('real_estate_knowledge_service', {}))
        MockSalesAgentService.assert_called_once_with(
            config=self.sample_base_config,
            generic_sales_service=MockGenericSalesSkillService.return_value,
            real_estate_service=MockRealEstateKnowledgeService.return_value
        )
        self.assertIs(manager.twilio_service, MockTwilioService.return_value)
        self.assertIs(manager.stt_service, MockSpeechToTextService.return_value)
        self.assertIs(manager.acoustic_analyzer_service, MockAcousticEmotionAnalyzerService.return_value)
        self.assertIs(manager.rasa_service, MockRasaService.return_value)
        self.assertIs(manager.nlg_service, MockNaturalLanguageGenerationService.return_value)
        self.assertIs(manager.tts_service, MockTextToSpeechService.return_value)
        self.assertIs(manager.empathy_specialist_service, MockEmpathySpecialistService.return_value)
        self.assertIs(manager.generic_sales_skill_service, MockGenericSalesSkillService.return_value)
        self.assertIs(manager.real_estate_knowledge_service, MockRealEstateKnowledgeService.return_value)
        self.assertIs(manager.sales_agent_specialist_service, MockSalesAgentService.return_value)

    @patch('main.SalesAgentService')
    @patch('main.RealEstateKnowledgeService')
    @patch('main.GenericSalesSkillService')
    @patch('main.EmpathySpecialistService')
    @patch('main.TextToSpeechService')
    @patch('main.NaturalLanguageGenerationService')
    @patch('main.RasaService')
    @patch('main.AcousticEmotionAnalyzerService')
    @patch('main.SpeechToTextService')
    @patch('main.TwilioService')
    def test_initialization_uses_default_agent_type_from_config(
        self, MockTwilioService, MockSpeechToTextService, MockAcousticEmotionAnalyzerService,
        MockRasaService, MockNaturalLanguageGenerationService, MockTextToSpeechService,
        MockEmpathySpecialistService, MockGenericSalesSkillService, MockRealEstateKnowledgeService,
        MockSalesAgentService
    ):
        default_agent_type = "empathy_base"
        self.sample_base_config["application"]["default_agent_type"] = default_agent_type
        manager = ServiceManager(config=self.sample_base_config, agent_type=default_agent_type)
        MockTwilioService.assert_called_once_with(config=self.sample_base_config)
        MockSpeechToTextService.assert_called_once_with(config=self.sample_base_config)
        MockAcousticEmotionAnalyzerService.assert_called_once_with(service_config=self.sample_base_config.get('acoustic_emotion_analyzer_service', {}))
        MockRasaService.assert_called_once_with(service_config=self.sample_base_config.get('rasa_service', {}))
        MockNaturalLanguageGenerationService.assert_called_once_with(config=self.sample_base_config)
        MockTextToSpeechService.assert_called_once_with(config=self.sample_base_config)
        MockEmpathySpecialistService.assert_called_once_with(config=self.sample_base_config)
        MockGenericSalesSkillService.assert_not_called()
        MockRealEstateKnowledgeService.assert_not_called()
        MockSalesAgentService.assert_not_called()
        self.assertIs(manager.twilio_service, MockTwilioService.return_value)
        self.assertIs(manager.stt_service, MockSpeechToTextService.return_value)
        self.assertIs(manager.empathy_specialist_service, MockEmpathySpecialistService.return_value)
        self.assertIsNone(manager.generic_sales_skill_service)


class TestSpecialistDispatcher(unittest.TestCase):
    def setUp(self):
        self.mock_service_manager = MagicMock(spec=ServiceManager)
        self.mock_service_manager.empathy_specialist_service = MagicMock(spec=EmpathySpecialistService)
        self.mock_service_manager.sales_agent_specialist_service = MagicMock(spec=SalesAgentService)
        self.mock_service_manager.nlg_service = MagicMock(spec=NaturalLanguageGenerationService)
        self.dispatcher = SpecialistDispatcher(service_manager=self.mock_service_manager)
        self.conversation_history = [
            {"speaker": "user", "text": "Hello there."},
            {"speaker": "agent", "text": "Hi, how can I help?"},
            {"speaker": "user", "text": "I need some help with my account."}
        ]
        self.user_emotion_data = {"dominant_emotion": "neutral", "probabilities": {"neutral": 0.8, "happy": 0.2}}
        self.user_text = "I need some help with my account."
        self.user_intent = "request_assistance"
        self.user_entities = {"topic": "account"}
        self.sales_context = {"stage": "discovery", "lead_score": 7}
        self.logger_name_sd = "main"

    def test_dispatch_empathy_specialist(self):
        action_plan = {"next_specialist": "empathy_specialist", "intent": self.user_intent, "entities": self.user_entities}
        expected_response = "Mocked empathetic response."
        self.mock_service_manager.empathy_specialist_service.generate_empathetic_response.return_value = expected_response
        response_text, specialist_used = self.dispatcher.dispatch(
            action_plan, self.conversation_history, self.user_emotion_data, self.sales_context
        )
        expected_context_for_empathy = {"history": self.conversation_history, "user_intent": self.user_intent}
        self.mock_service_manager.empathy_specialist_service.generate_empathetic_response.assert_called_once_with(
            context=expected_context_for_empathy, emotion_data=self.user_emotion_data
        )
        self.assertEqual(response_text, expected_response)
        self.assertEqual(specialist_used, "EmpathySpecialist")

    def test_dispatch_sales_agent_specialist_available(self):
        action_plan = {"next_specialist": "sales_agent", "intent": self.user_intent, "entities": self.user_entities}
        expected_sales_response_text = "Mocked sales response."
        updated_stage_in_context = "discovery_clarification"
        def mock_generate_sales_response(sales_context, user_input_details, emotion_data):
            sales_context["stage"] = updated_stage_in_context
            sales_context["last_interaction_summary"] = "Discussed pricing."
            return expected_sales_response_text
        self.mock_service_manager.sales_agent_specialist_service.generate_sales_response.side_effect = mock_generate_sales_response
        self.assertIsNotNone(self.mock_service_manager.sales_agent_specialist_service)
        response_text, specialist_used = self.dispatcher.dispatch(
            action_plan, self.conversation_history, self.user_emotion_data, self.sales_context
        )
        expected_user_input_details = {"text": self.user_text, "intent": self.user_intent, "entities": self.user_entities}
        self.mock_service_manager.sales_agent_specialist_service.generate_sales_response.assert_called_once_with(
            sales_context=self.sales_context, user_input_details=expected_user_input_details, emotion_data=self.user_emotion_data
        )
        self.assertEqual(response_text, expected_sales_response_text)
        self.assertEqual(specialist_used, "SalesAgentSpecialist")
        self.assertEqual(self.sales_context["stage"], updated_stage_in_context)
        self.assertEqual(self.sales_context["last_interaction_summary"], "Discussed pricing.")

    def test_dispatch_sales_agent_specialist_not_available_falls_back_to_nlg(self):
        self.mock_service_manager.sales_agent_specialist_service = None
        action_plan = {"next_specialist": "sales_agent", "intent": self.user_intent, "entities": self.user_entities}
        expected_nlg_response = "NLG fallback for unavailable sales specialist."
        self.mock_service_manager.nlg_service.generate_text_response.return_value = expected_nlg_response
        response_text, specialist_used = self.dispatcher.dispatch(
            action_plan, self.conversation_history, self.user_emotion_data, self.sales_context
        )
        expected_fallback_prompt = f"I received: '{self.user_text}'. How can I further assist?"
        expected_context_data = {"last_user_utterance": self.user_text}
        self.mock_service_manager.nlg_service.generate_text_response.assert_called_once_with(
            expected_fallback_prompt, context_data=expected_context_data
        )
        self.assertEqual(response_text, expected_nlg_response)
        self.assertEqual(specialist_used, "NLGService (Fallback)")

    def test_dispatch_nlg_direct_answer(self):
        action_plan = {"next_specialist": "nlg_direct_answer", "intent": self.user_intent, "entities": self.user_entities}
        expected_nlg_response = "NLG direct answer about account."
        self.mock_service_manager.nlg_service.generate_text_response.return_value = expected_nlg_response
        response_text, specialist_used = self.dispatcher.dispatch(
            action_plan, self.conversation_history, self.user_emotion_data, self.sales_context
        )
        expected_prompt = (
            f"User said: '{self.user_text}'. Dominant emotion: {self.user_emotion_data.get('dominant_emotion')}. "
            f"Recent history: {self.conversation_history[-3:]}. Provide a concise, helpful response."
        )
        expected_context_data = {"last_user_utterance": self.user_text, "key_topics": self.user_entities.values()}
        self.mock_service_manager.nlg_service.generate_text_response.assert_called_once()
        called_args_tuple, called_kwargs_dict = self.mock_service_manager.nlg_service.generate_text_response.call_args
        self.assertEqual(called_args_tuple[0], expected_prompt)
        self.assertIn("context_data", called_kwargs_dict)
        self.assertEqual(called_kwargs_dict["context_data"]["last_user_utterance"], expected_context_data["last_user_utterance"])
        self.assertEqual(list(called_kwargs_dict["context_data"]["key_topics"]), list(expected_context_data["key_topics"]))
        self.assertEqual(response_text, expected_nlg_response)
        self.assertEqual(specialist_used, "NLGService (Direct)")

    def test_dispatch_nlg_fallback_default(self):
        action_plan = {"next_specialist": "some_unknown_specialist", "intent": self.user_intent, "entities": self.user_entities}
        expected_nlg_response = "NLG general fallback response."
        self.mock_service_manager.nlg_service.generate_text_response.return_value = expected_nlg_response
        response_text, specialist_used = self.dispatcher.dispatch(
            action_plan, self.conversation_history, self.user_emotion_data, self.sales_context
        )
        expected_fallback_prompt = f"I received: '{self.user_text}'. How can I further assist?"
        expected_context_data = {"last_user_utterance": self.user_text}
        self.mock_service_manager.nlg_service.generate_text_response.assert_called_once_with(
            expected_fallback_prompt, context_data=expected_context_data
        )
        self.assertEqual(response_text, expected_nlg_response)
        self.assertEqual(specialist_used, "NLGService (Fallback)")


class TestCallHandler(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "application": {"default_agent_type": "empathy_base"},
            "sales_agent_service": {"default_sales_stage": "greeting"},
            "text_to_speech_service": {
                "sesame_csm_settings": {"default_voice_profile": "test_voice"}
            }
        }
        self.mock_service_manager = MagicMock(spec=ServiceManager)
        self.mock_service_manager.twilio_service = MagicMock(spec=TwilioService)
        self.mock_service_manager.stt_service = MagicMock(spec=SpeechToTextService)
        self.mock_service_manager.acoustic_analyzer_service = MagicMock(spec=AcousticEmotionAnalyzerService)
        self.mock_service_manager.rasa_service = MagicMock(spec=RasaService)
        self.mock_service_manager.tts_service = MagicMock(spec=TextToSpeechService)
        self.mock_service_manager.nlg_service = MagicMock(spec=NaturalLanguageGenerationService)
        self.mock_service_manager.empathy_specialist_service = MagicMock(spec=EmpathySpecialistService)
        self.mock_service_manager.sales_agent_specialist_service = MagicMock(spec=SalesAgentService)

        self.specialist_dispatcher_patcher = patch('main.SpecialistDispatcher')
        self.MockSpecialistDispatcherClass = self.specialist_dispatcher_patcher.start()
        self.addCleanup(self.specialist_dispatcher_patcher.stop)

        self.mock_resolve_config_patcher = patch('main.resolve_config_value')
        self.mock_resolve_config = self.mock_resolve_config_patcher.start()
        self.addCleanup(self.mock_resolve_config_patcher.stop)

        def resolve_side_effect(raw_value, default_value, target_type=None):
            if raw_value == self.mock_config.get("sales_agent_service",{}).get("default_sales_stage"):
                return "greeting"
            elif raw_value == self.mock_config.get("text_to_speech_service",{}).get("sesame_csm_settings",{}).get("default_voice_profile"):
                return "test_voice"
            return default_value
        self.mock_resolve_config.side_effect = resolve_side_effect

        self.call_handler = CallHandler(config=self.mock_config, service_manager=self.mock_service_manager)
        self.logger_name_ch = "main"

    def tearDown(self): # Not strictly needed if only addCleanup is used for patchers
        pass # self.mock_resolve_config_patcher and self.specialist_dispatcher_patcher stopped by addCleanup

    def test_initialization(self):
        self.assertIs(self.call_handler.config, self.mock_config)
        self.assertIs(self.call_handler.sm, self.mock_service_manager)

        self.MockSpecialistDispatcherClass.assert_called_once_with(self.mock_service_manager)
        self.assertIs(self.call_handler.specialist_dispatcher, self.MockSpecialistDispatcherClass.return_value)

    def _configure_mocks_for_turn(
        self,
        user_audio=b"sample_user_audio",
        stt_text="Hello, world.",
        emotion_data={"dominant_emotion": "neutral"},
        rasa_plan={"intent": "greet", "next_specialist": "empathy_specialist", "entities": {}, "response_emotion_hint": "neutral", "end_call": False},
        dispatch_response_text="Hi there! How can I help you today?",
        dispatch_specialist_used="EmpathySpecialist",
        tts_audio=b"sample_agent_tts_audio"
    ):
        self.mock_service_manager.twilio_service.receive_audio_from_caller.return_value = user_audio
        self.mock_service_manager.stt_service.transcribe_audio_chunk.return_value = stt_text
        self.mock_service_manager.acoustic_analyzer_service.analyze_emotion_from_audio.return_value = emotion_data
        self.mock_service_manager.rasa_service.process_user_message.return_value = rasa_plan

        # self.call_handler.specialist_dispatcher is the instance of the mocked SpecialistDispatcher class
        # (which is self.MockSpecialistDispatcherClass.return_value)
        self.MockSpecialistDispatcherClass.return_value.dispatch.return_value = (dispatch_response_text, dispatch_specialist_used)

        self.mock_service_manager.tts_service.synthesize_speech.return_value = tts_audio

        mock_audio_stream = MagicMock()
        self.mock_service_manager.twilio_service.start_audio_stream.return_value = mock_audio_stream
        return mock_audio_stream

    @patch('main.print')
    def test_run_simulation_single_successful_turn(self, mock_print):
        call_sid = "test_call_001"
        num_turns = 1

        user_audio_data = b"test_user_audio_chunk"
        stt_result_text = "This is a test."
        emotion_result = {"dominant_emotion": "curious"}
        rasa_result_plan = {"intent": "query", "next_specialist": "nlg_direct_answer", "entities": {"item": "test"}, "response_emotion_hint": "neutral", "end_call": False}
        dispatch_result_text = "I understand you're asking about a test."
        dispatch_used_specialist = "NLGService (Direct)"
        tts_result_audio = b"test_agent_audio_response"

        mock_audio_stream_instance = self._configure_mocks_for_turn(
            user_audio=user_audio_data,
            stt_text=stt_result_text,
            emotion_data=emotion_result,
            rasa_plan=rasa_result_plan,
            dispatch_response_text=dispatch_result_text,
            dispatch_specialist_used=dispatch_used_specialist,
            tts_audio=tts_result_audio
        )

        self.call_handler.run_simulation(call_sid=call_sid, num_turns=num_turns)

        self.mock_service_manager.twilio_service.start_audio_stream.assert_called_once_with(call_sid)
        self.mock_service_manager.twilio_service.receive_audio_from_caller.assert_called_once_with(mock_audio_stream_instance)
        self.mock_service_manager.stt_service.transcribe_audio_chunk.assert_called_once_with(user_audio_data)
        self.mock_service_manager.acoustic_analyzer_service.analyze_emotion_from_audio.assert_called_once_with(user_audio_data)
        self.mock_service_manager.rasa_service.process_user_message.assert_called_once_with(
            f"user_sim_{call_sid}", stt_result_text, emotion_result
        )

        self.MockSpecialistDispatcherClass.return_value.dispatch.assert_called_once()
        dispatch_call_args = self.MockSpecialistDispatcherClass.return_value.dispatch.call_args[0]
        # For num_turns=1, conversation_history passed to dispatch has 1 item (the user's current utterance)
        # The mock captures a reference to this list. After dispatch returns, CallHandler appends agent's response.
        # So, when we inspect call_args later, call_args[1] (the history list) will have 2 items.
        # The user's utterance that dispatch processed is now at index 0.
        self.assertEqual(dispatch_call_args[0], rasa_result_plan)
        self.assertEqual(dispatch_call_args[1][0]['text'], stt_result_text)
        self.assertEqual(dispatch_call_args[2], emotion_result)

        self.mock_service_manager.tts_service.synthesize_speech.assert_called_once_with(
            dispatch_result_text,
            voice_profile="test_voice",
            emotion_hint=rasa_result_plan.get("response_emotion_hint")
        )
        self.mock_service_manager.twilio_service.send_audio_to_caller.assert_called_once_with(
            mock_audio_stream_instance, tts_result_audio
        )
        mock_audio_stream_instance.close.assert_called_once()

    @patch('main.print')
    def test_run_simulation_ends_call_based_on_action_plan(self, mock_print):
        """
        Tests that the simulation loop terminates early if action_plan contains 'end_call': True.
        """
        call_sid = "test_call_002"
        num_turns = 3 # Simulation should still only run for 1 turn

        rasa_plan_ends_call = {
            "intent": "goodbye",
            "next_specialist": "empathy_specialist",
            "entities": {},
            "response_emotion_hint": "neutral",
            "end_call": True # Key for this test
        }

        mock_audio_stream_instance = self._configure_mocks_for_turn(
            rasa_plan=rasa_plan_ends_call
            # Other params use defaults from _configure_mocks_for_turn
        )

        self.call_handler.run_simulation(call_sid=call_sid, num_turns=num_turns)

        # Assert that core methods for a turn were called only once
        self.mock_service_manager.twilio_service.receive_audio_from_caller.assert_called_once()
        self.mock_service_manager.stt_service.transcribe_audio_chunk.assert_called_once()
        self.mock_service_manager.acoustic_analyzer_service.analyze_emotion_from_audio.assert_called_once()
        self.mock_service_manager.rasa_service.process_user_message.assert_called_once()
        self.MockSpecialistDispatcherClass.return_value.dispatch.assert_called_once()
        self.mock_service_manager.tts_service.synthesize_speech.assert_called_once()
        self.mock_service_manager.twilio_service.send_audio_to_caller.assert_called_once()

        # Assert stream was closed
        mock_audio_stream_instance.close.assert_called_once()

    @patch('main.print')
    def test_run_simulation_user_hangs_up(self, mock_print):
        """
        Tests that the simulation loop terminates if user audio stream ends (hangs up).
        """
        call_sid = "test_call_003"
        num_turns = 3 # Simulation should stop after detecting empty audio on the 2nd turn's receive.

        first_audio_chunk = b"first_turn_audio"

        # Configure mocks for the first successful turn
        mock_audio_stream_instance = self._configure_mocks_for_turn(user_audio=first_audio_chunk)

        # Setup side_effect for receive_audio_from_caller:
        # first call returns audio, second call returns empty bytes (hang-up)
        self.mock_service_manager.twilio_service.receive_audio_from_caller.side_effect = [
            first_audio_chunk,
            b"" # Simulate hang-up on second attempt to receive
        ]

        self.call_handler.run_simulation(call_sid=call_sid, num_turns=num_turns)

        # Assertions
        # Methods related to a full turn should be called once for the first turn
        self.mock_service_manager.stt_service.transcribe_audio_chunk.assert_called_once_with(first_audio_chunk)
        self.mock_service_manager.acoustic_analyzer_service.analyze_emotion_from_audio.assert_called_once_with(first_audio_chunk)
        self.mock_service_manager.rasa_service.process_user_message.assert_called_once()
        self.MockSpecialistDispatcherClass.return_value.dispatch.assert_called_once()
        self.mock_service_manager.tts_service.synthesize_speech.assert_called_once()
        self.mock_service_manager.twilio_service.send_audio_to_caller.assert_called_once()

        # receive_audio_from_caller should have been called twice
        self.assertEqual(self.mock_service_manager.twilio_service.receive_audio_from_caller.call_count, 2)

        # Assert stream was closed
        mock_audio_stream_instance.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
