import os
import time
from typing import Callable

import speech_recognition as sr


_recognizer = sr.Recognizer()


def take_picture():
    """Generates a timestamped filename and captures an image using the Brio."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    print(f"\n📸 Snapping photo with Brio: {filename}")
    os.system(
        f"fswebcam -d /dev/video4 -r 1920x1080 --no-banner {filename} > /dev/null 2>&1"
    )
    print("✅ Picture saved!\n")


def listen_transcripts(
    on_text: Callable[[str], None],
    stop_flag: Callable[[], bool],
    timeout: float = 2.0,
    phrase_time_limit: float = 6.0,
    ambient_duration: float = 2.0,
) -> None:
    """Blocking microphone loop. Calls on_text(str) for each recognized utterance.
    Exits when stop_flag() returns True (checked between listen windows)."""
    with sr.Microphone() as source:
        print("🎙️ Adjusting for ambient noise...")
        _recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)
        print("🎙️ Voice listener ready.")

        while not stop_flag():
            try:
                audio = _recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
                text = _recognizer.recognize_google(audio).lower().strip()
                if text:
                    on_text(text)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[voice] Google Speech API error: {e}")


if __name__ == "__main__":
    def _cb(text: str) -> None:
        print(f"Heard: '{text}'")
        if "take picture" in text or "cheese" in text:
            take_picture()

    try:
        listen_transcripts(_cb, stop_flag=lambda: False)
    except KeyboardInterrupt:
        print("\nExiting script...")
