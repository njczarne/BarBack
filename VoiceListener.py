import sounddevice as sd
import numpy as np
import time
import re
import threading
import queue # Added queue import
from faster_whisper import WhisperModel
import sys

# --- Default Config (can be overridden in __init__) ---
DEFAULT_ACTIVATION_PHRASE = "wake up"
DEFAULT_ENDING_PHRASE = "thank you"
# --- Whisper prefers 16kHz ---
DEFAULT_SAMPLERATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_DURATION = 0.5
# --- Reduced duration to help prevent input overflow ---
DEFAULT_PROCESS_DURATION = 3 # Reduced from 5
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_MODEL_NAME = "tiny.en"

class VoiceListener:
    """
    A class to continuously listen for an activation phrase, then record commands
    until an ending phrase is detected. Puts the final command onto a queue.
    Uses print statements for status feedback instead of callbacks.
    """
    def __init__(self,
                 command_queue: queue.Queue, # REQUIRED: Queue to send commands to
                 activation_phrase: str = DEFAULT_ACTIVATION_PHRASE,
                 ending_phrase: str = DEFAULT_ENDING_PHRASE,
                 samplerate: int = DEFAULT_SAMPLERATE,
                 channels: int = DEFAULT_CHANNELS,
                 chunk_duration: float = DEFAULT_CHUNK_DURATION,
                 process_duration: float = DEFAULT_PROCESS_DURATION,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 model_name: str = DEFAULT_MODEL_NAME,
                 verbose: bool = True): # Control print statements
        """
        Initializes the VoiceListener.

        Args:
            command_queue: A thread-safe queue where finished commands (str) will be put.
            activation_phrase: The phrase to start command listening.
            ending_phrase: The phrase to stop command listening.
            # ... other parameters ...
            verbose: If True, print status messages.
        """
        self.command_queue = command_queue
        self.activation_phrase = self._normalize(activation_phrase)
        self.ending_phrase = self._normalize(ending_phrase)
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.process_duration = process_duration
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name
        self.verbose = verbose

        # Calculate sample sizes
        self.chunk_samples = int(self.chunk_duration * self.samplerate)
        self.process_samples = int(self.process_duration * self.samplerate)
        self.overlap_samples = self.process_samples // 2

        # --- Instance-specific Shared Resources ---
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        # Internal state simplified: just track if activated
        self._is_active = threading.Event()
        self._rec_thread = None
        self._trans_thread = None
        self._model = None

        self._log(f"VoiceListener initialized. Waiting for start(). Activation='{activation_phrase}', End='{ending_phrase}'")

    def _log(self, message: str):
        """Internal logging helper."""
        if self.verbose:
            print(f"[VoiceListener] {message}")

    def _normalize(self, text: str) -> str:
        """Removes punctuation, converts to lowercase, strips whitespace."""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def _load_model(self):
        """Loads the Whisper model."""
        if self._model is None:
            self._log(f"Loading Whisper model '{self.model_name}' on device '{self.device}' ({self.compute_type})...")
            try:
                # Make sure faster_whisper is installed
                self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
                self._log("Whisper model loaded successfully.")
            except ImportError:
                 self._log("ERROR: faster_whisper not found. Please install it: pip install faster-whisper")
                 raise
            except Exception as e:
                self._log(f"ERROR: Failed to load Whisper model: {e}")
                raise

    # --- Thread Target Methods ---
    def _record_audio(self):
        """Continuously records audio and puts chunks into the queue."""
        self._log("🎙️ Recording thread started.")
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    # Input overflows are common if processing is slow, log as warning
                     self._log(f"Audio callback status warning: {status}")
                # Only put data if the stop event is not set to prevent queue filling during shutdown
                if not self._stop_event.is_set():
                     self._audio_queue.put(indata.copy().astype(np.float32))

            # Check device capabilities before opening stream
            sd.check_input_settings(samplerate=self.samplerate, channels=self.channels, dtype='float32')
            self._log(f"Audio input check passed for {self.samplerate}Hz, {self.channels}ch.")

            with sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                dtype='float32', blocksize=self.chunk_samples,
                                callback=audio_callback):
                self._log("🎧 Recording stream active. Waiting for stop event.")
                self._stop_event.wait()
        except sd.PortAudioError as pae:
             self._log(f"ERROR: PortAudioError in recording thread: {pae}")
             self._log("Check if the sample rate is supported or if another application is using the microphone.")
        except Exception as e:
            self._log(f"ERROR in recording thread: {e}")
        finally:
            self._log("🛑 Recording thread finished.")

    def _transcribe_audio(self):
        """Continuously processes audio chunks and transcribes."""
        self._log("🧠 Transcription thread started.")
        if self._model is None:
             self._log("ERROR: Transcription thread started but model is not loaded.")
             return

        audio_buffer = np.array([], dtype=np.float32)
        full_command_transcript = ""

        try:
            while not self._stop_event.is_set():
                try:
                    # Use a shorter timeout to be more responsive to stop_event
                    chunk = self._audio_queue.get(timeout=0.5)
                    # Mark task done immediately after getting to prevent queue size issues if processing is slow
                    self._audio_queue.task_done()
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))

                    # Process chunks when buffer is sufficient
                    while len(audio_buffer) >= self.process_samples and not self._stop_event.is_set():
                        process_chunk = audio_buffer[:self.process_samples]
                        # Keep the remainder for overlap
                        audio_buffer = audio_buffer[self.process_samples - self.overlap_samples:]

                        # --- Transcribe ---
                        segments, info = self._model.transcribe(
                            process_chunk,
                            language="en",
                            beam_size=2, # Reduced beam_size for speed
                            condition_on_previous_text=False
                        )

                        # --- Process Segments ---
                        detected_in_chunk = False
                        for segment in segments:
                            if self._stop_event.is_set(): break
                            raw_text = segment.text.strip()
                            if not raw_text: continue
                            normalized = self._normalize(raw_text)

                            # State 1: Waiting for Activation
                            if not self._is_active.is_set():
                                if self.activation_phrase in normalized:
                                    self._log(f"✅ Activation Detected: '{raw_text}'")
                                    self._is_active.set()
                                    full_command_transcript = ""
                                    self._log("🎧 Listening for command...")
                                    detected_in_chunk = True
                                    break # Move to next audio chunk processing

                            # State 2: Listening for Command / Ending Phrase
                            else:
                                if self.verbose: print(f"   🗣️ ... '{raw_text}'")
                                full_command_transcript += raw_text + " "

                                if self.ending_phrase in normalized:
                                    self._log(f"✅ Command Finished (Ending Phrase): '{raw_text}'")
                                    final_transcript = full_command_transcript.strip()
                                    # --- Put command on the queue ---
                                    try:
                                         self.command_queue.put_nowait(final_transcript) # Use nowait if possible
                                         self._log(f"📨 Command sent for processing: '{final_transcript}'")
                                    except queue.Full:
                                         self._log("WARNING: Command queue is full. Discarding command.")

                                    self._is_active.clear()
                                    full_command_transcript = ""
                                    detected_in_chunk = True
                                    self._log(f"🎙️ Waiting for activation phrase: '{self.activation_phrase}'...")
                                    break # Move to next audio chunk processing

                        # If something significant happened, potentially reduce buffer to avoid reprocessing
                        if detected_in_chunk:
                            audio_buffer = audio_buffer[self.overlap_samples:]

                except queue.Empty:
                    # No audio data available within timeout, loop again
                    continue
                except Exception as e:
                    self._log(f"ERROR in transcription loop: {e}")
                    time.sleep(0.1) # Avoid tight loop on errors
        finally:
            self._log("🛑 Transcription thread finished.")

    # --- Public Methods ---
    def start(self):
        """Starts the voice listener threads."""
        if self._rec_thread is not None or self._trans_thread is not None:
            self._log("Listener already started.")
            return False
        self._log("Starting voice listener...")
        try:
            self._load_model() # Load model if not already loaded
            if self._model is None: raise RuntimeError("Model not loaded.")

            # Clear queues and reset events before starting threads
            self._stop_event.clear()
            self._is_active.clear()
            while not self._audio_queue.empty():
                try: self._audio_queue.get_nowait()
                except queue.Empty: break
                finally: self._audio_queue.task_done() # Ensure task_done is called

            # Start threads
            self._rec_thread = threading.Thread(target=self._record_audio, daemon=True)
            self._trans_thread = threading.Thread(target=self._transcribe_audio, daemon=True)
            self._rec_thread.start()
            self._trans_thread.start()
            self._log(f"Threads started. Waiting for activation phrase: '{self.activation_phrase}'...")
            return True
        except Exception as e:
            self._log(f"ERROR during listener start: {e}")
            self.stop() # Attempt cleanup if start fails
            return False

    def stop(self):
        """Signals the listener threads to stop and waits for them to finish."""
        if not self._stop_event.is_set(): # Prevent multiple stop calls if already stopping
             self._log("Stopping voice listener threads...")
             self._stop_event.set()

             # Wait briefly for threads to finish
             if self._trans_thread and self._trans_thread.is_alive():
                 self._trans_thread.join(timeout=2.0)
             if self._rec_thread and self._rec_thread.is_alive():
                 self._rec_thread.join(timeout=1.0)

             # Check if they actually stopped
             if self._trans_thread and self._trans_thread.is_alive():
                  self._log("Warning: Transcription thread did not join cleanly.")
             if self._rec_thread and self._rec_thread.is_alive():
                  self._log("Warning: Recording thread did not join cleanly.")

             self._rec_thread = None
             self._trans_thread = None
             self._log("Voice listener stopped.")


# --- Main Function for Independent Testing ---
if __name__ == "__main__":
    print("--- VoiceListener Independent Test ---")

    # 1. Create the queue
    test_command_queue = queue.Queue()

    # 2. Instantiate VoiceListener
    # You can customize parameters here for testing
    listener_instance = VoiceListener(
        command_queue=test_command_queue,
        activation_phrase="hello there", # Example: different phrase
        ending_phrase="goodbye",         # Example: different phrase
        process_duration=3,              # Use reduced duration
        verbose=True
    )

    # 3. Start the listener
    if not listener_instance.start():
        print("\n--- Test Failed: Listener did not start. Check errors above. ---")
        sys.exit(1) # Exit if listener failed to start

    print("\n--- Listener Started ---")
    print("Say 'hello there' to activate, then speak a command ending with 'goodbye'.")
    print("Press Ctrl+C to stop the test.")
    print("Waiting for commands on the queue...")

    # 4. Simulate a consumer reading from the queue
    try:
        while True:
            try:
                # Wait for a command to appear on the queue
                command = test_command_queue.get(timeout=1.0) # Timeout allows checking loop condition
                print(f"\n>>> COMMAND RECEIVED (from queue): '{command}'")
                test_command_queue.task_done() # Mark task as done
            except queue.Empty:
                # No command received yet, just continue waiting
                pass
            # Add a small sleep to prevent this loop from consuming too much CPU
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n--- Ctrl+C detected. Stopping listener... ---")
    except Exception as e:
        print(f"\n--- An error occurred in the test loop: {e} ---")
    finally:
        # 5. Handle Shutdown
        if listener_instance:
            listener_instance.stop()
        print("--- Test Finished ---")