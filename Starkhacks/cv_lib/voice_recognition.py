import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys

# Download a small model from alphacephei.com/vosk/models
model = Model("model-en-us") 
rec = KaldiRecognizer(model, 16000)

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    if rec.AcceptWaveform(bytes(indata)):
        print(rec.Result())
    else:
        print(rec.PartialResult())

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Listening for voice commands...")
    while True:
        pass