from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, base64, requests
from main import load_config, VoiceAgent

app = FastAPI()
cfg = load_config("config.yml")
agent = VoiceAgent(cfg)

STT_URL = os.getenv("STT_URL", "http://stage1:5000/stt")

class InteractReq(BaseModel):
    call_id: str
    audio_base64: str

class InteractRes(BaseModel):
    transcription: str
    response_audio_base64: str

@app.post("/interact", response_model=InteractRes)
def interact(req: InteractReq):
    # 1) decode audio, 2) STT
    audio_bytes = base64.b64decode(req.audio_base64)
    stt_resp = requests.post(STT_URL, data=audio_bytes)
    if stt_resp.status_code != 200:
        raise HTTPException(502, "STT service error")
    user_text = stt_resp.json().get("text", "")

    # 3) emotion
    emo = agent.acoustic_analyzer_service.analyze_emotion_from_audio(audio_bytes)

    # 4) NLU + planner
    plan = agent.rasa_service.process_user_message(req.call_id, user_text, emo)

    # 5) specialist dispatch (simplified from main.py)
    if plan.get("next_specialist") == "empathy_specialist":
        resp_text = agent.empathy_specialist_service.generate_empathetic_response(
            context={"history":[{"speaker":"user","text":user_text}]},
            emotion_data=emo
        )
    elif plan.get("next_specialist") == "sales_agent" and agent.sales_agent_specialist_service:
        resp_text = agent.sales_agent_specialist_service.generate_sales_response(
            sales_context={},
            user_input_details={"text": user_text, "intent": plan.get("intent"), "entities": plan.get("entities", {})},
            emotion_data=emo
        )
    else:
        resp_text = agent.nlg_service.generate_text_response(
            f"I received: '{user_text}'. How can I further assist?"
        )

    # 6) ElevenLabs TTS
    tts_bytes = agent.tts_service.synthesize_speech(resp_text)

    return InteractRes(
        transcription=user_text,
        response_audio_base64=base64.b64encode(tts_bytes).decode()
    ) 