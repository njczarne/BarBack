import sounddevice as sd
import numpy as np
# import wave # No longer needed if not saving temp WAV
import time
import re
import threading
import queue
from faster_whisper import WhisperModel
from voice_leds import turn_on_led, turn_off_led # Assuming this module exists and works
import sys

# --- Config ---
ACTIVATION_PHRASE = "wake up"
ENDING_PHRASE = "thank you"
SAMPLERATE = 16000  # Whisper works best with 16kHz
CHANNELS = 1
CHUNK_DURATION = 0.5
PROCESS_DURATION = 3   # Use tuned duration
# TEMP_FILENAME = "temp_command.wav" # Not used in this version
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = "tiny.en" # Faster, less accurate
# MODEL_NAME = "base.en" # Slower, more accurate
BUFFER_OVERLAP_SECONDS = PROCESS_DURATION / 2.0
IDLE_TIMEOUT_SECONDS = 10.0

# Calculate sample sizes
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLERATE)
PROCESS_SAMPLES = int(PROCESS_DURATION * SAMPLERATE)
OVERLAP_SAMPLES = int(BUFFER_OVERLAP_SECONDS * SAMPLERATE)
if OVERLAP_SAMPLES < 0: OVERLAP_SAMPLES = 0 # Ensure non-negative overlap

# --- Shared Resources ---
audio_queue = queue.Queue()
result_queue = queue.Queue()
stop_event = threading.Event()
is_listening = threading.Event() # Controls listen vs. wait state

# --- Load whisper model ---
print("Loading Whisper model...")
try:
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Whisper model loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    sys.exit(1)

# --- Helper Functions ---
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

# --- Thread Functions ---

def record_audio(stop_event, audio_queue):
    """Continuously records audio and puts chunks into the queue."""
    print("🎙️ Starting recording thread...")
    try:
        def audio_callback(indata, frames, time, status):
            if status: print(f"Audio callback status warning: {status}", file=sys.stderr)
            if not stop_event.is_set():
                audio_queue.put(indata.copy().astype(np.float32))

        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS,
                            dtype='float32', blocksize=CHUNK_SAMPLES,
                            callback=audio_callback):
            print("🎧 Recording stream active. Waiting for stop event.")
            stop_event.wait()
    except Exception as e:
        print(f"Error in recording thread: {e}")
    finally:
        print("🛑 Recording thread stopped.")


def transcribe_audio(stop_event, audio_queue, result_queue, is_listening):
    """Continuously processes audio chunks, transcribes, manages state and buffer."""
    print("🧠 Starting transcription thread...")
    audio_buffer = np.array([], dtype=np.float32)
    full_command_transcript = ""
    last_activity_time = time.monotonic()
    previous_segment_text = "" # For conditioning

    try:
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=1.0)
                audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                last_activity_time = time.monotonic()

                while len(audio_buffer) >= PROCESS_SAMPLES and not stop_event.is_set():
                    process_chunk = audio_buffer[:PROCESS_SAMPLES]
                    samples_to_keep_after = audio_buffer[PROCESS_SAMPLES - OVERLAP_SAMPLES:]

                    # --- Transcribe ---
                    # Use condition_on_previous_text=True for potentially better coherence
                    segments, info = model.transcribe(process_chunk,
                                                      language="en",
                                                      beam_size=2,
                                                      # Use previous text as prompt for the model
                                                      initial_prompt=previous_segment_text,
                                                      condition_on_previous_text=True)

                    # --- Process Segments ---
                    found_activation_in_chunk = False
                    found_ending_in_chunk = False
                    current_chunk_texts = [] # Store texts from this chunk

                    for segment in segments:
                        if stop_event.is_set(): break
                        raw_text = segment.text.strip()
                        if not raw_text: continue
                        normalized = normalize(raw_text)
                        current_chunk_texts.append(raw_text) # Keep track of last text in chunk

                        # Check state *before* processing
                        was_listening = is_listening.is_set()

                        if not was_listening: # Looking for activation phrase
                            if ACTIVATION_PHRASE in normalized:
                                print(f"[Transcribe] Heard potential activation: '{raw_text}'")
                                result_queue.put(("ACTIVATION_FOUND", raw_text))
                                is_listening.set() # <<< SET STATE IMMEDIATELY
                                found_activation_in_chunk = True
                                full_command_transcript = "" # Reset command transcript
                                break # Stop processing segments for this chunk
                        else: # Actively listening for commands
                            # Accumulate transcript
                            full_command_transcript += raw_text + " "
                            # Put partial result on queue for main loop to handle/print
                            result_queue.put(("PARTIAL_TRANSCRIPT", raw_text))

                            if ENDING_PHRASE in normalized:
                                print(f"[Transcribe] Heard potential ending: '{raw_text}'")
                                final_transcript = full_command_transcript.strip()
                                result_queue.put(("ENDING_FOUND", final_transcript))
                                is_listening.clear() # <<< CLEAR STATE IMMEDIATELY
                                found_ending_in_chunk = True
                                full_command_transcript = "" # Reset command transcript
                                break # Stop processing segments for this chunk

                    # Update previous_segment_text for conditioning the *next* transcription
                    if current_chunk_texts:
                         previous_segment_text = " ".join(current_chunk_texts)
                    else: # Reset if no text found (e.g. silence)
                         previous_segment_text = ""


                    # --- Buffer Management (After processing segments for a chunk) ---
                    if found_ending_in_chunk:
                        # Command finished, reset buffer completely
                        print("[Buffer Manage] Command ended. Resetting audio buffer.")
                        audio_buffer = np.array([], dtype=np.float32)
                        break # Exit chunk processing loop immediately
                    elif not is_listening.is_set():
                         # Still waiting for activation (or just finished command): keep only overlap
                         audio_buffer = samples_to_keep_after
                    else:
                         # Listening for command: keep overlap for context
                         audio_buffer = samples_to_keep_after


            except queue.Empty:
                # --- Handle Inactivity While Waiting for Activation ---
                current_time = time.monotonic()
                if not is_listening.is_set() and (current_time - last_activity_time > IDLE_TIMEOUT_SECONDS):
                    if len(audio_buffer) > 0:
                         print(f"[Buffer Manage] Idle timeout ({IDLE_TIMEOUT_SECONDS}s) while waiting. Clearing buffer.")
                         audio_buffer = np.array([], dtype=np.float32)
                         previous_segment_text = "" # Reset conditioning on idle clear
                    last_activity_time = current_time # Reset timer
                continue

            except Exception as e:
                print(f"Error in transcription loop: {e}")
                # Consider logging traceback here for debugging
                # import traceback; traceback.print_exc()
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

    last_partial_text = "" # Keep track to potentially reduce duplicate prints

    try:
        while True:
            try:
                # Check for results from the transcription thread
                result_type, text = result_queue.get(timeout=1.0)

                # --- State Machine based on Results ---
                if result_type == "ACTIVATION_FOUND":
                    # Check if we are *not already* listening to avoid redundant actions
                    if not is_listening.is_set():
                        print(f"\n✅ Activation Detected by Main: '{text}'")
                        turn_on_led()
                        is_listening.set() # Ensure state is set here too
                        print("🎧 Listening for command (say 'thank you' to stop)...")
                        # --- Clear queues to remove stale data ---
                        print("[Queue Clear] Clearing queues after activation...")
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except queue.Empty: break
                        while not result_queue.empty():
                             try: result_queue.get_nowait()
                             except queue.Empty: break
                        print("[Queue Clear] Queues cleared.")
                        last_partial_text = "" # Reset last printed text

                elif result_type == "PARTIAL_TRANSCRIPT":
                    # Only process/print if we are actually in listening state
                    if is_listening.is_set():
                        # Optional: Basic check to reduce printing exact duplicates consecutively
                        if text != last_partial_text:
                             print(f"🗣️ Heard: {text}")
                             last_partial_text = text

                elif result_type == "ENDING_FOUND":
                     # Only process if we were previously listening
                     if is_listening.is_set():
                        print(f"\n🙏 Ending phrase detected by Main.")
                        print(f"📝 Full Command Transcript: {text}")
                        turn_off_led()
                        is_listening.clear() # Ensure state is clear
                        # Buffer clearing now happens inside transcribe_audio
                        print(f"\n🎙️ Waiting for activation phrase: '{ACTIVATION_PHRASE}'...")
                        last_partial_text = "" # Reset last printed text

            except queue.Empty:
                # No results from queue, just loop again
                pass

            # Add a small sleep if result queue was empty to prevent busy-waiting
            # time.sleep(0.05) # Moved sleep to prevent delay after processing item

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Stopping threads...")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        # Consider logging traceback here
        # import traceback; traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("Initiating shutdown...")
        stop_event.set()
        if is_listening.is_set():
            turn_off_led() # Ensure LED is off

        print("Waiting for recording thread to finish...")
        if rec_thread and rec_thread.is_alive(): rec_thread.join(timeout=1.0) # Shorter timeout
        print("Waiting for transcription thread to finish...")
        if trans_thread and trans_thread.is_alive(): trans_thread.join(timeout=2.0)

        print("Program finished.")

# --- Run Main ---
if __name__ == "__main__":
    try:
        print("Available audio devices:", sd.query_devices())
        print(f"Using default input device: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        print("Please ensure you have a microphone connected and configured.")
        sys.exit(1)

    main()