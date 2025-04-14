import sounddevice as sd
import numpy as np
import wave
import time
import re
from faster_whisper import WhisperModel
from voice_leds import turn_on_led, turn_off_led

# --- Config ---
ACTIVATION_PHRASE = "wake up"
ENDING_PHRASE = "thank you"
SAMPLERATE = 44100
CHANNELS = 1
DURATION = 5  # seconds
TEMP_FILENAME = "temp_command.wav"

# Load whisper model
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def record_to_wav(filename, duration, samplerate, channels):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate,
                       channels=channels, dtype='int16')
    sd.wait()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

def normalize(text):
    return re.sub(r'[^a-z\s]', '', text.lower()).strip()

def listen_for_phrase(target_phrase):
    while True:
        record_to_wav(TEMP_FILENAME, DURATION, SAMPLERATE, CHANNELS)

        segments, _ = model.transcribe(TEMP_FILENAME, language="en", beam_size=2)

        found_phrase = False
        full_transcript = ""

        for segment in segments:
            raw_text = segment.text.strip()
            full_transcript += raw_text + " "

            normalized = normalize(raw_text)
            if target_phrase in normalized:
                turn_on_led()
                return raw_text  # ✅ Detected activation

        # If nothing matched, print what was heard
        print(f"❌ No activation phrase: {ACTIVATION_PHRASE} detected. Listening again...")
        print("📝 Transcript:", full_transcript.strip())


def listen_until_ending():
    print("🎧 Listening for command (say 'thank you' to stop)...")
    full_transcript = ""  # This will accumulate all transcribed text
    while True:
        # Record a new 5-second audio chunk
        record_to_wav(TEMP_FILENAME, DURATION, SAMPLERATE, CHANNELS)

        # Transcribe the current 5-second recording
        segments, _ = model.transcribe(TEMP_FILENAME, language="en", beam_size=5)

        for segment in segments:
            raw_text = segment.text.strip()

            # Accumulate all transcribed text
            full_transcript += raw_text + " "
            normalized = normalize(raw_text)

            # Print the command heard so far
            print("🗣️ Command Heard:", raw_text)

            # Check if the ending phrase is in the transcribed text
            if ENDING_PHRASE in normalized:
                print("🙏 Ending phrase detected. Exiting command mode.")
                turn_off_led()
                print("📝 Full Transcript:", full_transcript.strip())  # Print the full transcript
                return full_transcript.strip()  # Return the entire transcript up to "thank you"


# --- Main Loop ---

try:
    while True:
        print(f"🎙️ Waiting for activation phrase: {ACTIVATION_PHRASE}...")
        result = listen_for_phrase(ACTIVATION_PHRASE)
        print("✅ Activation Detected:", result)

        # Listen for commands until ending phrase
        listen_until_ending()
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Program stopped.")

