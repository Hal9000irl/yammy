# specialist_empathy_service.py
# Contains the EmpathySpecialistService.

import time # For simulation
import os
import sys
import torch
from collections import deque

class EmpathySpecialistService:
    """
    Generates empathetic responses.
    Could use "Topic based Empathetic Chatbot", MoEL, or a fine-tuned LLM.
    """
    def __init__(self, config: dict):
        self.config = config.get('empathy_specialist_service', {})
        self.provider = self.config.get('provider', 'default_empathy_logic')
        self.settings = self.config.get(f"{self.provider}_settings", {})
        print(f"EmpathySpecialistService Initialized (Provider: {self.provider}, Settings: {self.settings})")
        # Real: Initialize the chosen empathy model/logic based on provider and settings
        if self.provider == "moel":
            base_dir = os.path.dirname(__file__)
            moel_dir = os.path.abspath(os.path.join(base_dir, os.pardir, 'Stage 1', 'MoEL-master', 'MoEL-master'))
            sys.path.insert(0, moel_dir)
            from interact import model, vocab, make_batch, config as moel_config
            self.moel_model = model.eval()
            self.moel_vocab = vocab
            self.moel_make_batch = make_batch
            self.moel_config = moel_config
            self.moel_context = deque(["None"] * moel_config.DIALOG_SIZE, maxlen=moel_config.DIALOG_SIZE)
            print(f"EmpathySpecialistService: MoEL provider initialized with model path {moel_dir}")

    def generate_empathetic_response(self, context: dict, emotion_data: dict = None) -> str:
        """Generates an empathetic textual response."""
        # MoEL provider: use pretrained MoEL model for empathy
        if self.provider == "moel":
            # Append latest user utterances to MoEL context
            for turn in context.get("history", []):
                if turn.get("speaker") == "user":
                    self.moel_context.append(turn.get("text", ""))
            batch = self.moel_make_batch(self.moel_context, self.moel_vocab)
            sent = self.moel_model.decoder_greedy(batch, max_dec_step=30)
            response = sent[0]
            print(f"EmpathySpecialistService (MoEL): Generated '{response}'")
            return response
        else:
            user_emotion = emotion_data.get("dominant_emotion", "neutral") if emotion_data else "neutral"
            last_user_text = ""
            if context.get("history") and isinstance(context["history"], list) and len(context["history"]) > 0:
                # Find the last user utterance
                for i in range(len(context["history"]) - 1, -1, -1):
                    if context["history"][i].get("speaker") == "user":
                        last_user_text = context["history"][i].get("text", "")
                        break
            
            print(f"EmpathySpecialist ({self.provider}): Generating response for user emotion '{user_emotion}' and last user text: '{last_user_text}'")

            # Simplified logic based on emotion for simulation
            if self.provider == "topic_based_empathetic_chatbot" or self.provider == "default_empathy_logic":
                if user_emotion == "sad":
                    return "I hear that things might be tough right now, and I want you to know I'm here to listen. What's on your mind?"
                elif user_emotion == "angry":
                    return "It sounds like you're feeling quite frustrated, and that's completely understandable. Would you like to talk about what's causing it?"
                elif user_emotion == "happy" or user_emotion == "excited":
                    return "That's wonderful to hear! I'm glad things are going well for you."
                elif user_emotion == "fearful":
                    return "It sounds like that might be a bit unsettling. I'm here if you need to talk through it."
                elif "hello" in last_user_text.lower() or "hi" in last_user_text.lower():
                     return "Hello! It's nice to speak with you. How can I be of service today?"
                elif "thank you" in last_user_text.lower() or "thanks" in last_user_text.lower():
                    return "You're very welcome! Is there anything else I can help you with?"
                elif "goodbye" in last_user_text.lower() or "bye" in last_user_text.lower():
                    return "Goodbye for now! It was a pleasure speaking with you. Have a great day!"
                elif context.get("user_intent") == "silence":
                    return "Is everything alright? I'm here if you need anything."
                else: # Neutral or fallback
                    return "I understand. Please tell me more about how I can help you today."
            else:
                # Placeholder for other providers
                return f"Empathy (Provider: {self.provider}): I'm listening. How can I help?"

if __name__ == '__main__':
    dummy_config = {
        "empathy_specialist_service": {
            "provider": "topic_based_empathetic_chatbot",
            "topic_chatbot_settings": {
                "t5_chitchat_model_uri": "sim/t5_chitchat",
                "t5_empathetic_model_uri": "sim/t5_empathetic",
                "gpt2_topical_model_uri": "sim/gpt2_topical"
            }
        }
    }
    empathy_service = EmpathySpecialistService(config=dummy_config)

    history_greet = [{"speaker": "user", "text": "Hello"}]
    history_sad = [{"speaker": "user", "text": "I'm feeling down."}]
    history_angry = [{"speaker": "user", "text": "This is so frustrating!"}]
    history_fallback = [{"speaker": "user", "text": "Tell me about the weather."}]


    print("\nGreeting:")
    print(empathy_service.generate_empathetic_response(context={"history": history_greet}, emotion_data={"dominant_emotion": "neutral"}))
    
    print("\nSad User:")
    print(empathy_service.generate_empathetic_response(context={"history": history_sad}, emotion_data={"dominant_emotion": "sad"}))

    print("\nAngry User:")
    print(empathy_service.generate_empathetic_response(context={"history": history_angry}, emotion_data={"dominant_emotion": "angry"}))
    
    print("\nFallback/Neutral:")
    print(empathy_service.generate_empathetic_response(context={"history": history_fallback}, emotion_data={"dominant_emotion": "neutral"}))
    
    print("\nThank you:")
    print(empathy_service.generate_empathetic_response(context={"history": [{"speaker":"user", "text":"Thank you so much"}]}, emotion_data={"dominant_emotion": "happy"}))

    print("\nSilence:")
    print(empathy_service.generate_empathetic_response(context={"history": [{"speaker":"user", "text":""}], "user_intent": "silence"}, emotion_data={"dominant_emotion": "neutral"}))
