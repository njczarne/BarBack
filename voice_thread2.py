import sounddevice as sd
import numpy as np
import wave
import time
import re
import threading
import queue
import os
import uuid
from faster_whisper import WhisperModel
from voice_leds import turn_on_led, turn_off_led
# -----------------------

# --- Config ---
SAMPLERATE = 44100
CHANNELS = 1
DURATION = 3  # seconds per chunk
TEMP_FILE_DIR = "temp_audio"
MODEL_SIZE = "tiny.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# --- Global Shared Resources ---
result_queue = queue.Queue(maxsize=1)
new_recording_event = threading.Event()
stop_event = threading.Event() # Used to signal threads to stop
latest_filename_for_worker = None
filename_lock = threading.Lock()
model = None # Global model instance
worker_thread = None # Global worker thread instance

# --- Initialization and Cleanup ---

def initialize_audio_system():
    """Loads model, ensures temp dir exists."""
    global model
    if model is not None:
        print("Audio system already initialized.")
        return True

    print(f"Initializing audio system...")
    print(f"Loading Whisper model ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return False # Indicate failure

    if not os.path.exists(TEMP_FILE_DIR):
        try:
            os.makedirs(TEMP_FILE_DIR)
        except OSError as e:
            print(f"Error creating temp directory {TEMP_FILE_DIR}: {e}")
            return False # Indicate failure
    print("Audio system initialized.")
    return True

def start_worker():
    """Starts the transcription worker thread if not already running."""
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        stop_event.clear() # Ensure stop event is clear before starting
        worker_thread = threading.Thread(target=transcription_worker, daemon=True)
        worker_thread.start()
        print("Transcription worker started.")
    else:
        print("Transcription worker already running.")

def stop_worker():
    """Signals the worker thread to stop and waits for it."""
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        print("Stopping transcription worker...")
        stop_event.set()
        new_recording_event.set() # Unblock worker's wait, if waiting
        worker_thread.join(timeout=3.0) # Wait for worker
        if worker_thread.is_alive():
            print("Warning: Worker thread did not stop cleanly.")
        print("Transcription worker stopped.")
        worker_thread = None
    else:
        print("Transcription worker not running.")

def cleanup_temp_files():
    """Removes remaining temp files and directory if empty."""
    print("Cleaning up temporary files...")
    if os.path.exists(TEMP_FILE_DIR):
        for f in os.listdir(TEMP_FILE_DIR):
            if f.startswith("rec_") and f.endswith(".wav"):
                try:
                    os.remove(os.path.join(TEMP_FILE_DIR, f))
                except OSError:
                    pass # Ignore errors during cleanup
        try:
            if not os.listdir(TEMP_FILE_DIR): # Check if empty
                os.rmdir(TEMP_FILE_DIR)
        except OSError:
             pass
        except FileNotFoundError:
            pass
    print("Cleanup finished.")


# --- Core Functions ---

def record_to_wav(duration, samplerate, channels):
    """Records audio and saves to a unique temp file."""
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate,
                           channels=channels, dtype='int16', blocking=True)
        filename = os.path.join(TEMP_FILE_DIR, f"rec_{uuid.uuid4()}.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(recording.tobytes())
        return filename
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def normalize(text):
    """Normalizes text."""
    return re.sub(r'[^a-z\s]', '', text.lower()).strip()

# --- Transcription Worker (Mostly Unchanged) ---

def transcription_worker():
    """Background thread worker."""
    global latest_filename_for_worker
    # No need to print "worker started" here, start_worker does it

    while not stop_event.is_set():
        if new_recording_event.wait(timeout=1.0):
            new_recording_event.clear()

            with filename_lock:
                filename_to_process = latest_filename_for_worker

            if stop_event.is_set() or filename_to_process is None:
                 continue

            try:
                if not os.path.exists(filename_to_process):
                    continue

                # Ensure model is loaded before transcribing
                if model is None:
                    print("Error: Model not loaded in worker thread.")
                    time.sleep(1) # Avoid busy loop
                    continue

                segments, _ = model.transcribe(
                    filename_to_process, language="en", beam_size=3, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                chunk_transcript = " ".join(segment.text.strip() for segment in segments)
                normalized_chunk = normalize(chunk_transcript)

                try:
                    result_queue.put((chunk_transcript, normalized_chunk), timeout=0.5)
                except queue.Full:
                    try: result_queue.get_nowait()
                    except queue.Empty: pass
                    try: result_queue.put((chunk_transcript, normalized_chunk), timeout=0.5)
                    except queue.Full: pass

            except Exception as e:
                print(f"Error during transcription for {filename_to_process}: {e}")
                try: result_queue.put(("", ""), timeout=0.1)
                except queue.Full: pass
            finally:
                if filename_to_process and os.path.exists(filename_to_process):
                     try: os.remove(filename_to_process)
                     except OSError: pass
    # print("Worker exiting loop.") # Optional debug


# --- Public Functions ---

def listen_for_phrase(activation_phrase):
    """
    Listens continuously until the activation phrase is detected.
    Assumes the worker thread is running.
    Returns True if phrase detected, False if interrupted.
    """
    global latest_filename_for_worker
    if worker_thread is None or not worker_thread.is_alive():
        print("Error: Worker thread not running. Call start_worker() first.")
        return False

    print(f"Listening for activation phrase: \"{activation_phrase}\"...")

    # Clear queue of any stale results
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except queue.Empty: break

    try:
        while not stop_event.is_set(): # Use global stop_event
            print(f"Recording (Listening for \"{activation_phrase}\")...") # Keep print inside loop
            current_filename = record_to_wav(DURATION, SAMPLERATE, CHANNELS)
            if not current_filename:
                time.sleep(0.5) # Wait on recording error
                continue

            # Signal Worker
            with filename_lock:
                latest_filename_for_worker = current_filename
            new_recording_event.set()

            # Process Results
            try:
                chunk_transcript, normalized_chunk = result_queue.get_nowait()
                has_content = chunk_transcript.strip() != ""

                if has_content and activation_phrase in normalized_chunk:
                    print("Activation heard.") # Keep simple confirmation
                    result_queue.task_done() # Mark as processed
                    return True # Phrase detected

                result_queue.task_done() # Mark as processed even if no match

            except queue.Empty:
                pass # No result ready, continue loop
            except Exception as e:
                print(f"Error processing result queue: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted while listening for phrase.")
        stop_event.set() # Signal stop globally
        return False

    return False # Should not be reached if loop broken by stop_event check

def listen_until_ending(ending_phrase):
    """
    Listens for commands after activation, accumulating transcript until
    the ending phrase is detected. Assumes worker thread is running.
    Returns the full transcript or None if interrupted.
    """
    global latest_filename_for_worker
    if worker_thread is None or not worker_thread.is_alive():
        print("Error: Worker thread not running. Call start_worker() first.")
        return None

    print(f"Listening for commands. End with \"{ending_phrase}\"")
    turn_on_led()
    accumulated_transcript = ""

    # Clear queue of any stale results from activation phase check
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except queue.Empty: break

    try:
        while not stop_event.is_set(): # Use global stop_event
             # No print before recording in this state per previous request
            current_filename = record_to_wav(DURATION, SAMPLERATE, CHANNELS)
            if not current_filename:
                time.sleep(0.5)
                continue

            # Signal Worker
            with filename_lock:
                latest_filename_for_worker = current_filename
            new_recording_event.set()

            # Process Results
            try:
                chunk_transcript, normalized_chunk = result_queue.get_nowait()
                has_content = chunk_transcript.strip() != ""

                if has_content:
                    # Accumulate transcript silently
                    accumulated_transcript += chunk_transcript.strip() + " "

                    # Check for ending phrase
                    if ending_phrase in normalized_chunk:
                        # print(f"Commands heard: {accumulated_transcript.strip()}") # Print handled by caller now
                        turn_off_led()
                        result_queue.task_done()
                        return accumulated_transcript.strip() # Return transcript on success

                result_queue.task_done()

            except queue.Empty:
                pass # No result ready, continue loop
            except Exception as e:
                print(f"Error processing result queue: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted while listening for commands.")
        stop_event.set() # Signal stop globally
        turn_off_led()
        # Return partial transcript on interrupt
        return accumulated_transcript.strip() if accumulated_transcript else None

    # If loop exits due to stop_event (not KeyboardInterrupt)
    turn_off_led()
    return accumulated_transcript.strip() if accumulated_transcript else None


# --- Example Usage ---

if __name__ == "__main__":
    # Define phrases for example
    ACT_PHRASE = "wake up"
    END_PHRASE = "thank you"

    if not initialize_audio_system():
        exit(1)

    start_worker()

    try:
        while True: # Main application loop
            # Wait for activation
            activated = listen_for_phrase(ACT_PHRASE)

            if not activated:
                # Interrupted or failed, exit main loop
                print("Activation listener interrupted or failed.")
                break

            # Listen for command until ending phrase
            full_command = listen_until_ending(END_PHRASE)

            if full_command is None:
                 # Interrupted during command listening
                 print("Command listener interrupted.")
                 break
            else:
                # Successfully got command
                print(f"\n--- Command Processing ---")
                print(f"Commands heard: {full_command}")
                print(f"--- Ready for next command ---")
                # Add a small delay before listening again (optional)
                time.sleep(0.5)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        # Ensure worker stops and files are cleaned
        stop_worker()
        cleanup_temp_files()
        print("Application finished.")