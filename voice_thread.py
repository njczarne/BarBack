import sounddevice as sd
import numpy as np
import wave
import time
import re
import threading
import queue
from faster_whisper import WhisperModel
from voice_leds import turn_on_led, turn_off_led # Assuming this module exists and works

# --- Config ---
ACTIVATION_PHRASE = "wake up"
ENDING_PHRASE = "thank you"
SAMPLERATE = 16000  # Whisper works best with 16kHz
CHANNELS = 1
CHUNK_DURATION = 0.5 # Process audio in smaller chunks (seconds)
PROCESS_DURATION = 5   # Transcribe chunks of this duration (seconds)
TEMP_FILENAME = "temp_command.wav" # Still needed for saving full command if desired
DEVICE = "cpu" # or "cuda" if you have Nvidia GPU + libraries
COMPUTE_TYPE = "int8" # or "float16" for GPU

# Calculate chunk sizes
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLERATE)
PROCESS_SAMPLES = int(PROCESS_DURATION * SAMPLERATE)

# --- Shared Resources ---
audio_queue = queue.Queue()
result_queue = queue.Queue() # To communicate results back to main thread
stop_event = threading.Event()
is_listening = threading.Event() # Flag to indicate if we are actively listening for commands

# --- Load whisper model ---
print("Loading Whisper model...")
# Using 'tiny.en' for speed, consider 'base.en' or 'small.en' for better accuracy
model = WhisperModel("tiny.en", device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")

# --- Helper Functions ---
def normalize(text):
    """Removes punctuation and converts to lowercase."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

# --- Thread Functions ---

def record_audio(stop_event, audio_queue):
    """Continuously records audio and puts chunks into the queue."""
    print("🎙️ Starting recording thread...")
    try:
        def audio_callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(f"Audio callback status: {status}", file=sys.stderr)
            # Ensure data is float32, required by Whisper
            audio_queue.put(indata.copy().astype(np.float32))

        with sd.InputStream(samplerate=SAMPLERATE,
                            channels=CHANNELS,
                            dtype='float32', # Use float32 directly
                            blocksize=CHUNK_SAMPLES, # Use smaller blocks for callback
                            callback=audio_callback):
            print("🎧 Recording started. Waiting for stop event...")
            stop_event.wait() # Keep recording until stop_event is set

    except Exception as e:
        print(f"Error in recording thread: {e}")
    finally:
        print("🛑 Recording thread stopped.")


def transcribe_audio(stop_event, audio_queue, result_queue, is_listening):
    """Continuously processes audio chunks from the queue and transcribes."""
    print("🧠 Starting transcription thread...")
    audio_buffer = np.array([], dtype=np.float32)
    full_command_transcript = "" # Accumulates transcript when listening

    try:
        while not stop_event.is_set():
            try:
                # Get audio data, wait max 1 second if queue is empty
                chunk = audio_queue.get(timeout=1.0)
                audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))

                # Process when buffer has enough data
                while len(audio_buffer) >= PROCESS_SAMPLES:
                    # Take the chunk to process
                    process_chunk = audio_buffer[:PROCESS_SAMPLES]
                    # Keep the remainder (overlap helps catch phrases at boundaries)
                    # Adjust overlap amount as needed, here keeping half
                    overlap_samples = PROCESS_SAMPLES // 2
                    audio_buffer = audio_buffer[PROCESS_SAMPLES - overlap_samples:]

                    # Transcribe the chunk
                    # Pass numpy array directly, it's more efficient
                    segments, info = model.transcribe(process_chunk,
                                                      language="en",
                                                      beam_size=5,
                                                      # word_timestamps=True, # Can be useful but slower
                                                      condition_on_previous_text=False) # Set True for potentially better coherence on longer audio

                    segment_text_list = []
                    found_activation = False
                    found_ending = False

                    for segment in segments:
                        raw_text = segment.text.strip()
                        normalized = normalize(raw_text)
                        segment_text_list.append(raw_text)
                        # print(f"Segment: {raw_text}") # Debug print

                        if not is_listening.is_set(): # Looking for activation phrase
                            if ACTIVATION_PHRASE in normalized:
                                print(f"👂 Heard potential activation: '{raw_text}'")
                                result_queue.put(("ACTIVATION_FOUND", raw_text))
                                found_activation = True
                                # Don't break here, finish processing the segment for context
                        else: # Actively listening for commands
                            full_command_transcript += raw_text + " "
                            result_queue.put(("PARTIAL_TRANSCRIPT", raw_text))
                            if ENDING_PHRASE in normalized:
                                print(f"👂 Heard potential ending: '{raw_text}'")
                                result_queue.put(("ENDING_FOUND", full_command_transcript.strip()))
                                found_ending = True
                                full_command_transcript = "" # Reset for next command
                                break # Stop processing segments if ending found

                    if found_ending: break # Exit outer loop if ending found in this chunk
                    # Debug: Print if nothing specific was found in this chunk
                    # if not found_activation and not is_listening.is_set() and segment_text_list:
                    #      print(f"❌ Heard (no activation): {' '.join(segment_text_list)}")


            except queue.Empty:
                # Queue was empty for the timeout duration, just loop again
                continue
            except Exception as e:
                print(f"Error in transcription loop: {e}")
                # Optional: add a small sleep to prevent tight loop on continuous errors
                time.sleep(0.1)

    finally:
        print("🛑 Transcription thread stopped.")


# --- Main Execution ---
def main():
    print(f"🎙️ Waiting for activation phrase: '{ACTIVATION_PHRASE}'...")

    # Start threads
    rec_thread = threading.Thread(target=record_audio, args=(stop_event, audio_queue))
    trans_thread = threading.Thread(target=transcribe_audio, args=(stop_event, audio_queue, result_queue, is_listening))

    rec_thread.start()
    trans_thread.start()

    try:
        while True:
            # State 1: Waiting for Activation
            if not is_listening.is_set():
                try:
                    result_type, text = result_queue.get(timeout=1.0) # Check for results
                    if result_type == "ACTIVATION_FOUND":
                        print(f"\n✅ Activation Detected: '{text}'")
                        turn_on_led() # Turn on LED upon activation
                        is_listening.set() # Set the flag
                        print("🎧 Listening for command (say 'thank you' to stop)...")
                        # Clear the queue slightly to avoid immediate processing of lingering activation phrase audio
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except queue.Empty: break
                        while not result_queue.empty():
                             try: result_queue.get_nowait()
                             except queue.Empty: break

                except queue.Empty:
                    # No activation detected yet, continue waiting
                    pass

            # State 2: Listening for Command / Ending Phrase
            if is_listening.is_set():
                try:
                    result_type, text = result_queue.get(timeout=1.0) # Check for results

                    if result_type == "PARTIAL_TRANSCRIPT":
                        print(f"🗣️ Heard: {text}") # Print intermediate results

                    elif result_type == "ENDING_FOUND":
                        print(f"\n🙏 Ending phrase detected.")
                        print(f"📝 Full Command Transcript: {text}")
                        turn_off_led() # Turn off LED
                        is_listening.clear() # Go back to waiting state
                        print(f"\n🎙️ Waiting for activation phrase: '{ACTIVATION_PHRASE}'...")
                        # Optional: Save the full transcript to a file here if needed

                    # Ignore any activation phrases heard while already listening
                    elif result_type == "ACTIVATION_FOUND":
                         pass # Already active, do nothing

                except queue.Empty:
                    # No new transcript part or ending phrase, continue listening
                    pass

            # Allow a small sleep to prevent high CPU usage in the main loop if queues are constantly empty
            time.sleep(0.05)


    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Stopping threads...")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        # Signal threads to stop and wait for them
        stop_event.set()
        if is_listening.is_set():
            turn_off_led() # Ensure LED is off on exit
        print("Waiting for recording thread to finish...")
        rec_thread.join()
        print("Waiting for transcription thread to finish...")
        trans_thread.join()
        print("Program finished.")

# --- Run Main ---
if __name__ == "__main__":
    # Add a check for microphone availability (optional but good practice)
    try:
        print("Available audio devices:", sd.query_devices())
        # You might want to set a specific device index using sd.default.device
        # sd.default.device = [input_device_index, output_device_index]
        print(f"Using default input device: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        print("Please ensure you have a microphone connected and configured.")
        exit()

    main()