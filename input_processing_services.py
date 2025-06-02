# input_processing_services.py
# Contains services for STT and Acoustic Emotion Analysis.

import time # For simulation
import os
import asyncio
from deepgram import Deepgram

class SpeechToTextService:
    """
    Transcribes speech to text via Deepgram (or fallback simulation).
    """
    def __init__(self, config: dict):
        self.config = config.get('speech_to_text_service', {})
        self.provider = self.config.get('provider', 'deepgram').lower()
        self.settings = self.config.get(f"{self.provider}_settings", {})
        print(f"SpeechToTextService Initialized (Provider: {self.provider}, Settings: {self.settings})")

        if self.provider == "deepgram":
            api_key = self.settings.get("api_key", "")
            # Expand env var placeholder if present
            if api_key.startswith("${") and api_key.endswith("}"):
                api_key = os.getenv(api_key.strip("${}"))
            if not api_key:
                print("SpeechToTextService: No valid Deepgram API key provided, falling back to simulation mode.")
                # Fallback to simulation by resetting provider
                self.provider = "simulation"
                return
            try:
                self.dg_client = Deepgram(api_key)
            except Exception as e:
                print(f"SpeechToTextService: Deepgram init error '{e}', falling back to simulation.")
                self.provider = "simulation"
            return
        # Real: Initialize actual STT client based on provider and settings

    def transcribe_audio_chunk(self, audio_chunk: bytes) -> str:
        """Transcribes a chunk of audio."""
        if not audio_chunk:
            return ""

        if self.provider == "deepgram":
            source = {"buffer": audio_chunk, "mimetype": "audio/wav"}
            response = asyncio.run(self.dg_client.transcription.sync_prerecorded(source))
            channels = response.get("results", {}).get("channels", [])
            if not channels or not channels[0].get("alternatives"):
                return ""
            return channels[0]["alternatives"][0].get("transcript", "").strip()
        if not hasattr(self, "_sim_counter"):
            self._sim_counter = 0
        
        sim_texts = [
            "Hello, who is this?",
            "I'm interested in selling my house.",
            "What's the market like in the downtown area?",
            "That sounds a bit too expensive for me.",
            "Okay, tell me more about the property on Elm Street.",
            "Thank you, that was very helpful.",
            "Goodbye."
        ]
        text = sim_texts[self._sim_counter % len(sim_texts)]
        self._sim_counter +=1
        return text

class AcousticEmotionAnalyzerService:
    """
    Analyzes acoustic features of speech to infer emotion.
    This would integrate your `acoustic_emotion_analyzer.py`.
    """
    def __init__(self, config: dict):
        self.config = config.get('acoustic_emotion_analyzer_service', {})
        self.model_path = self.config.get('model_path', 'default_emotion_model.pkl')
        self.sample_rate = self.config.get('sample_rate', 22050)
        # from acoustic_emotion_analyzer import AcousticEmotionAnalyzer # Assuming file is in path
        # self.analyzer = AcousticEmotionAnalyzer() # Real: Initialize your analyzer
        print(f"AcousticEmotionAnalyzerService Initialized (Model: {self.model_path}, SampleRate: {self.sample_rate})")

    def analyze_emotion_from_audio(self, audio_chunk: bytes) -> dict:
        """Analyzes emotion from an audio chunk."""
        if not audio_chunk: # Handle empty audio chunk
            return {"dominant_emotion": "neutral", "probabilities": {"neutral": 1.0}}
            
        print(f"AcousticEmotionAnalyzerService: Analyzing emotion from {len(audio_chunk)} bytes.")
        # Real: Process with self.analyzer.infer_emotion(audio_data, self.sample_rate)
        # For simulation:
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
        # Simple simulation: cycle through emotions based on time or a counter
        if not hasattr(self, "_emo_sim_counter"):
            self._emo_sim_counter = 0
        dominant_emotion = emotions[self._emo_sim_counter % len(emotions)]
        self._emo_sim_counter += 1
        
        probabilities = {emo: 0.1 for emo in emotions} # Base probability
        probabilities[dominant_emotion] = 0.6 # Higher for dominant
        # Normalize (simplified)
        total_prob = sum(probabilities.values())
        normalized_probabilities = {emo: prob/total_prob for emo, prob in probabilities.items()}

        return {
            "dominant_emotion": dominant_emotion,
            "probabilities": normalized_probabilities
        }

if __name__ == '__main__':
    dummy_config = {
        "speech_to_text_service": {
            "provider": "deepgram",
            "deepgram_settings": {"api_key": "your_deepgram_api_key"}
        },
        "acoustic_emotion_analyzer_service": {
            "model_path": "sim_emotion_model.pkl",
            "sample_rate": 16000
        }
    }
    stt = SpeechToTextService(config=dummy_config)
    emotion_analyzer = AcousticEmotionAnalyzerService(config=dummy_config)

    sample_audio = b"some_audio_data"
    text_result = stt.transcribe_audio_chunk(sample_audio)
    emotion_result = emotion_analyzer.analyze_emotion_from_audio(sample_audio)

    print(f"STT Result: {text_result}")
    print(f"Emotion Result: {emotion_result}")
