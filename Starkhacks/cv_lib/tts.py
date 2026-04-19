import os
import threading
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

VOICE_ID = "XiPS9cXxAVbaIWtGDHDh"
MODEL_ID = "eleven_flash_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

_client: Optional[ElevenLabs] = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY missing from environment")
        _client = ElevenLabs(api_key=key)
    return _client


def talk(speech: str, block: bool = False) -> None:
    if not speech or not speech.strip():
        return

    def _run():
        audio = _get_client().text_to_speech.convert(
            text=speech,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT,
        )
        play(audio)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if block:
        t.join()
