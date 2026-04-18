import speech_recognition as sr
import os
import time

# Initialize the speech recognizer
r = sr.Recognizer()

def take_picture():
    """Generates a timestamped filename and captures an image using the Brio."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    print(f"\n📸 Snapping photo with Brio: {filename}")
    
    # Executing fswebcam targeting /dev/video4
    os.system(f"fswebcam -d /dev/video4 -r 1920x1080 --no-banner {filename} > /dev/null 2>&1")
    print("✅ Picture saved!\n")

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    r.adjust_for_ambient_noise(source, duration=2)
    print("🎙️ Ready! Say 'take picture' or 'cheese' to snap a photo.")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            # Listen for audio, timing out after 5 seconds of silence
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
            # Using Google's free speech recognition (requires internet)
            command = r.recognize_google(audio).lower()
            print(f"Heard: '{command}'")

            # Check if our trigger words are in the recognized text
            if "take picture" in command or "cheese" in command:
                take_picture()

        except sr.WaitTimeoutError:
            # Re-loop if no speech is detected
            pass
        except sr.UnknownValueError:
            # Audio was detected, but couldn't be understood
            print("...") 
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition; {e}")
        except KeyboardInterrupt:
            print("\nExiting script...")
            break