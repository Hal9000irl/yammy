# response_generation_services.py
# Contains services for NLG and TTS.

import time # For simulation
import os
import requests

class NaturalLanguageGenerationService:
    """
    Generates human-like text responses, often using an LLM.
    Used when a specialist doesn't provide the full response or for general replies.
    """
    def __init__(self, config: dict):
        self.config = config.get('natural_language_generation_service', {})
        self.provider = self.config.get('provider', 'local_llama')
        self.settings = self.config.get(f"{self.provider}_settings", {})
        print(f"NaturalLanguageGenerationService Initialized (Provider: {self.provider}, Settings: {self.settings})")
        # Real: Load LLM model or configure API client for the chosen provider

    def generate_text_response(self, prompt: str, context_data: dict = None) -> str:
        """Generates text from a prompt and optional context."""
        print(f"NLGService ({self.provider}): Generating text for prompt: '{prompt[:100]}...'") # Log truncated prompt
        # Real: Call LLM API or local model with the prompt and context_data
        # Example: if self.provider == 'openai_gpt': response = openai.Completion.create(...)
        
        # Simulated response
        if "clarify" in prompt.lower():
            return f"NLG Simulated Response: I understand you mentioned '{context_data.get('last_user_utterance', 'something') if context_data else 'something'}'. Could you please elaborate a bit more on that?"
        elif "summarize" in prompt.lower():
            return f"NLG Simulated Response: To summarize, we discussed {context_data.get('key_topics', ['several important points']) if context_data else ['several important points']}."
        return f"NLG Simulated Response: Based on your query about '{prompt[:30]}...', I'd be happy to provide more information. What specifically are you interested in?"

class TextToSpeechService:
    """
    Synthesizes text into speech.
    Could use Sesame CSM, ElevenLabs, etc.
    """
    def __init__(self, config: dict):
        self.config = config.get('text_to_speech_service', {})
        self.provider = self.config.get('provider', 'sesame_csm')
        self.settings = self.config.get(f"{self.provider}_settings", {})
        print(f"TextToSpeechService Initialized (Provider: {self.provider}, Settings: {self.settings})")
        # Real: Initialize TTS engine or API client

    def synthesize_speech(self, text_input: str, voice_profile: str = "default_professional", emotion_hint: str = None) -> bytes:
        """Synthesizes speech from text."""
        effective_emotion = emotion_hint if emotion_hint else "neutral"
        print(f"TTSService ({self.provider}): Synthesizing speech for: '{text_input}' (Voice: {voice_profile}, Emotion Hint: {effective_emotion})")
        
        if self.provider == "elevenlabs":
            # Determine which voice to use
            voice_id = self.settings.get("default_voice_id", voice_profile)
            # Grab API key from config or fallback to env var
            api_key = self.settings.get("api_key") or os.getenv("ELEVENLABS_API_KEY")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            payload = {"text": text_input}
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.content
            except Exception as e:
                print(f"TextToSpeechService: ElevenLabs error '{e}', falling back to simulation.")

        if self.provider == "sesame_csm":
            # Real: Call Sesame CSM API wrapper
            url = self.settings.get("service_url") or os.getenv("SESAME_CSM_URL")
            print(f"TextToSpeechService (SesameCSM): Calling {url}/generate-speech")
            payload = {
                "text": text_input,
                "voice_profile": voice_profile,
                "emotion_hint": effective_emotion
            }
            resp = requests.post(f"{url}/generate-speech", json=payload)
            resp.raise_for_status()
            return resp.content

        # Simulated audio bytes
        return f"simulated_audio_bytes_for_[{text_input.replace(' ','_')[:30]}]_emotion_{effective_emotion}".encode('utf-8')

if __name__ == '__main__':
    dummy_config = {
        "natural_language_generation_service": {
            "provider": "local_llama",
            "local_llama_settings": {"model_path": "sim/llama.gguf"}
        },
        "text_to_speech_service": {
            "provider": "sesame_csm",
            "sesame_csm_settings": {"service_url": "http://sim-csm-server:5001"}
        }
    }
    nlg = NaturalLanguageGenerationService(config=dummy_config)
    tts = TextToSpeechService(config=dummy_config)

    text_for_nlg = "User is asking about market trends in downtown."
    generated_text = nlg.generate_text_response(text_for_nlg, {"last_user_utterance": "downtown market"})
    print(f"NLG Generated: {generated_text}")

    audio_data = tts.synthesize_speech(generated_text, voice_profile="friendly_male", emotion_hint="helpful")
    print(f"TTS Generated Audio (simulated): {audio_data[:50]}...") # Print first 50 bytes
