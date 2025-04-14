import sounddevice as sd
import numpy as np
import wave
import time
import re
import threading
import queue
from faster_whisper import WhisperModel
# Remove the direct import of voice_leds here, handle LEDs via callbacks
# from voice_leds import turn_on_led, turn_off_led
from typing import Callable, Optional
import sys # For stderr printing in callback example

# --- Default Config (can be overridden in __init__) ---
DEFAULT_ACTIVATION_PHRASE = "wake up"
DEFAULT_ENDING_PHRASE = "thank you"
DEFAULT_SAMPLERATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_DURATION = 0.5
DEFAULT_PROCESS_DURATION = 5
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_MODEL_NAME = "tiny.en"

class VoiceListener:
    """
    A class to continuously listen for an activation phrase, then record commands
    until an ending phrase is detected, using separate threads for recording
    and transcription. Uses callbacks to communicate results.
    """
    def __init__(self,
                 activation_phrase: str = DEFAULT_ACTIVATION_PHRASE,
                 ending_phrase: str = DEFAULT_ENDING_PHRASE,
                 samplerate: int = DEFAULT_SAMPLERATE,
                 channels: int = DEFAULT_CHANNELS,
                 chunk_duration: float = DEFAULT_CHUNK_DURATION,
                 process_duration: float = DEFAULT_PROCESS_DURATION,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 model_name: str = DEFAULT_MODEL_NAME,
                 # --- Callbacks ---
                 on_activation: Optional[Callable[[str], None]] = None,
                 on_command_update: Optional[Callable[[str], None]] = None,
                 on_command_finish: Optional[Callable[[str], None]] = None,
                 on_listening_start: Optional[Callable[[], None]] = None,
                 on_listening_stop: Optional[Callable[[], None]] = None,
                 on_debug_log: Optional[Callable[[str], None]] = None):
        """
        Initializes the VoiceListener.

        Args:
            activation_phrase: The phrase to start command listening.
            ending_phrase: The phrase to stop command listening.
            samplerate: Audio sample rate (Whisper prefers 16000).
            channels: Number of audio channels.
            chunk_duration: How often the recording callback provides data (seconds).
            process_duration: Duration of audio chunks to feed to Whisper (seconds).
            device: Device for Whisper model ('cpu' or 'cuda').
            compute_type: Compute type for Whisper model ('int8', 'float16', etc.).
            model_name: Name of the Whisper model to load.
            on_activation: Callback function when activation phrase is detected. Takes detected phrase (str) as argument.
            on_command_update: Callback function with partial command transcript updates. Takes partial text (str) as argument.
            on_command_finish: Callback function when ending phrase is detected. Takes full command transcript (str) as argument.
            on_listening_start: Callback function when listener switches to command mode after activation.
            on_listening_stop: Callback function when listener stops command mode (after ending phrase or stop()).
            on_debug_log: Callback function for receiving debug/status messages. Takes message (str) as argument.
        """
        self.activation_phrase = self._normalize(activation_phrase)
        self.ending_phrase = self._normalize(ending_phrase)
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.process_duration = process_duration
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name

        # Calculate sample sizes
        self.chunk_samples = int(self.chunk_duration * self.samplerate)
        self.process_samples = int(self.process_duration * self.samplerate)
        # Overlap helps catch phrases at boundaries. Keep roughly half the process window.
        self.overlap_samples = self.process_samples // 2

        # Callbacks
        self.on_activation = on_activation
        self.on_command_update = on_command_update
        self.on_command_finish = on_command_finish
        self.on_listening_start = on_listening_start
        self.on_listening_stop = on_listening_stop
        self.on_debug_log = on_debug_log or (lambda msg: None) # Default to no-op

        # --- Instance-specific Shared Resources ---
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_listening_internal = threading.Event() # Internal state flag
        self._rec_thread = None
        self._trans_thread = None
        self._model = None

        self._log(f"VoiceListener initialized. Waiting for start(). Activation='{activation_phrase}', End='{ending_phrase}'")

    def _log(self, message: str):
        """Internal logging helper that uses the debug callback."""
        self.on_debug_log(f"[VoiceListener] {message}")

    def _normalize(self, text: str) -> str:
        """Removes punctuation, converts to lowercase, strips whitespace."""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def _load_model(self):
        """Loads the Whisper model."""
        if self._model is None:
            self._log(f"Loading Whisper model '{self.model_name}' on device '{self.device}' ({self.compute_type})...")
            try:
                self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
                self._log("Whisper model loaded successfully.")
            except Exception as e:
                self._log(f"ERROR: Failed to load Whisper model: {e}")
                raise # Reraise the exception to prevent starting without a model

    # --- Thread Target Methods ---
    def _record_audio(self):
        """Continuously records audio and puts chunks into the queue. (Runs in dedicated thread)"""
        self._log("🎙️ Recording thread started.")
        try:
            def audio_callback(indata, frames, time_info, status):
                """Called by sounddevice with new audio data."""
                if status:
                    self._log(f"Audio callback status warning: {status}")
                # Ensure data is float32, required by Whisper
                self._audio_queue.put(indata.copy().astype(np.float32))

            with sd.InputStream(samplerate=self.samplerate,
                                channels=self.channels,
                                dtype='float32',
                                blocksize=self.chunk_samples,
                                callback=audio_callback):
                self._log("🎧 Recording stream active. Waiting for stop event.")
                self._stop_event.wait() # Keep recording until stop_event is set

        except Exception as e:
            self._log(f"ERROR in recording thread: {e}")
            # Optionally signal main thread or attempt recovery
        finally:
            self._log("🛑 Recording thread finished.")

    def _transcribe_audio(self):
        """Continuously processes audio chunks and transcribes. (Runs in dedicated thread)"""
        self._log("🧠 Transcription thread started.")
        if self._model is None:
             self._log("ERROR: Transcription thread started but model is not loaded.")
             return # Cannot proceed without a model

        audio_buffer = np.array([], dtype=np.float32)
        full_command_transcript = "" # Accumulates transcript when listening

        try:
            while not self._stop_event.is_set():
                try:
                    # Get audio data, wait max 1 second if queue is empty
                    chunk = self._audio_queue.get(timeout=1.0)
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))

                    # Process when buffer has enough data
                    while len(audio_buffer) >= self.process_samples and not self._stop_event.is_set():
                        # Take the chunk to process
                        process_chunk = audio_buffer[:self.process_samples]
                        # Keep the remainder for overlap
                        audio_buffer = audio_buffer[self.process_samples - self.overlap_samples:]

                        # --- Transcribe ---
                        segments, info = self._model.transcribe(process_chunk,
                                                               language="en", # Assuming English, make configurable if needed
                                                               beam_size=5,
                                                               condition_on_previous_text=False) # Usually False for streaming chunks

                        # --- Process Segments ---
                        detected_in_chunk = False # Track if activation/ending found in this chunk
                        for segment in segments:
                            if self._stop_event.is_set(): break # Exit early if stopping

                            raw_text = segment.text.strip()
                            if not raw_text: continue # Skip empty segments

                            normalized = self._normalize(raw_text)
                            self._log(f"🔍 Segment: '{raw_text}' (Normalized: '{normalized}')")

                            # State 1: Waiting for Activation
                            if not self._is_listening_internal.is_set():
                                if self.activation_phrase in normalized:
                                    self._log(f"✅ Activation detected: '{raw_text}'")
                                    detected_in_chunk = True
                                    self._is_listening_internal.set() # Change state
                                    if self.on_listening_start:
                                        try: self.on_listening_start()
                                        except Exception as cb_e: self._log(f"Error in on_listening_start callback: {cb_e}")
                                    if self.on_activation:
                                        try: self.on_activation(raw_text)
                                        except Exception as cb_e: self._log(f"Error in on_activation callback: {cb_e}")
                                    full_command_transcript = "" # Reset command buffer
                                    # Consume the rest of this chunk's segments without processing further for activation
                                    # This prevents rapid re-activation if phrase is repeated in same chunk
                                    break # Exit segment loop for this chunk

                            # State 2: Listening for Command / Ending Phrase
                            else:
                                full_command_transcript += raw_text + " "
                                if self.on_command_update:
                                     try: self.on_command_update(raw_text) # Send partial update
                                     except Exception as cb_e: self._log(f"Error in on_command_update callback: {cb_e}")

                                if self.ending_phrase in normalized:
                                    self._log(f"🙏 Ending phrase detected: '{raw_text}'")
                                    detected_in_chunk = True
                                    final_transcript = full_command_transcript.strip()
                                    self._is_listening_internal.clear() # Go back to waiting state
                                    if self.on_listening_stop:
                                         try: self.on_listening_stop()
                                         except Exception as cb_e: self._log(f"Error in on_listening_stop callback: {cb_e}")
                                    if self.on_command_finish:
                                         try: self.on_command_finish(final_transcript)
                                         except Exception as cb_e: self._log(f"Error in on_command_finish callback: {cb_e}")
                                    full_command_transcript = "" # Reset for next time
                                    # Consume rest of segments in this chunk without processing further for ending phrase
                                    break # Exit segment loop for this chunk

                        # After processing segments for a chunk:
                        if detected_in_chunk:
                             # If activation/ending was found, clear buffer more aggressively
                             # to avoid reprocessing audio that led to the detection.
                             audio_buffer = audio_buffer[self.overlap_samples:] # Keep only overlap


                except queue.Empty:
                    # Queue was empty for the timeout duration, just loop again
                    continue
                except Exception as e:
                    self._log(f"ERROR in transcription loop: {e}")
                    time.sleep(0.1) # Avoid tight loop on continuous errors

        finally:
            self._log("🛑 Transcription thread finished.")

    # --- Public Methods ---
    def start(self):
        """Starts the voice listener threads (recording and transcription)."""
        if self._rec_thread is not None or self._trans_thread is not None:
            self._log("Listener already started.")
            return False

        self._log("Starting voice listener...")
        try:
            # Load model only when starting, if not already loaded
            self._load_model()
            if self._model is None:
                 raise RuntimeError("Failed to load Whisper model, cannot start.")

            # Check audio device before starting threads (optional but good practice)
            try:
                sd.check_input_settings(samplerate=self.samplerate, channels=self.channels)
                self._log(f"Audio input settings check passed for {self.samplerate} Hz, {self.channels} channels.")
            except Exception as audio_e:
                self._log(f"ERROR: Audio input device does not support settings: {audio_e}")
                return False


            self._stop_event.clear()
            self._is_listening_internal.clear() # Start in waiting state

            # Clear queues before starting
            while not self._audio_queue.empty():
                 try: self._audio_queue.get_nowait()
                 except queue.Empty: break

            # daemon=True allows program to exit even if these threads are running,
            # but we handle clean shutdown with stop_event and join() anyway.
            self._rec_thread = threading.Thread(target=self._record_audio, daemon=True)
            self._trans_thread = threading.Thread(target=self._transcribe_audio, daemon=True)

            self._rec_thread.start()
            self._trans_thread.start()
            self._log(f"Threads started. Waiting for activation phrase: '{self.activation_phrase}'...")
            return True

        except Exception as e:
            self._log(f"ERROR during listener start: {e}")
            # Ensure cleanup if threads partially started
            if self._rec_thread or self._trans_thread:
                self.stop()
            return False


    def stop(self):
        """Signals the listener threads to stop and waits for them to finish."""
        if self._rec_thread is None and self._trans_thread is None:
            self._log("Listener already stopped.")
            return

        self._log("Stopping voice listener threads...")
        self._stop_event.set() # Signal threads to stop

        # Wait for threads to finish
        if self._trans_thread and self._trans_thread.is_alive():
             self._log("Waiting for transcription thread to join...")
             self._trans_thread.join(timeout=5.0) # Add timeout
             if self._trans_thread.is_alive(): self._log("Warning: Transcription thread did not join.")
             else: self._log("Transcription thread joined.")

        if self._rec_thread and self._rec_thread.is_alive():
             self._log("Waiting for recording thread to join...")
             # Recording thread usually stops quickly once stop_event is set
             self._rec_thread.join(timeout=2.0) # Add timeout
             if self._rec_thread.is_alive(): self._log("Warning: Recording thread did not join.")
             else: self._log("Recording thread joined.")


        # Reset state
        self._rec_thread = None
        self._trans_thread = None
        # Ensure the listening stop callback is called if it was active
        if self._is_listening_internal.is_set():
            self._is_listening_internal.clear()
            if self.on_listening_stop:
                 try: self.on_listening_stop()
                 except Exception as cb_e: self._log(f"Error in final on_listening_stop callback: {cb_e}")

        self._log("Voice listener stopped.")

    def is_listening(self) -> bool:
        """Returns True if the listener is currently in command mode (post-activation)."""
        return self._is_listening_internal.is_set()

# --- Example Usage (Put this in your main program/class) ---
if __name__ == "__main__":

    # --- Define your callback functions in your main scope ---
    def my_activation_handler(phrase):
        print(f"\n ✨ ACTIVATION DETECTED! Phrase: '{phrase}'")
        # Add your logic here (e.g., turn on LED, change UI state)
        # Example: turn_on_led() # If you import your LED functions here

    def my_command_update_handler(partial_text):
        print(f"   🗣️ ... '{partial_text}'")
        # Update UI with partial transcript, etc.

    def my_command_finish_handler(full_command):
        print(f"\n ✅ COMMAND FINISHED! Full Command: '{full_command}'")
        # Process the command, turn off LED, etc.
        # Example: turn_off_led()
        print(f"\n 다시 듣기 '{listener.activation_phrase}'...\n") # "Listening again for..."

    def my_listening_start_handler():
         print("\n 🎧 LISTENING FOR COMMAND...")
         # Maybe turn on a specific LED color

    def my_listening_stop_handler():
         print("\n 🛑 LISTENING STOPPED.")
         # Maybe turn off LEDs or change color

    def my_debug_logger(message):
        print(f"[DEBUG] {message}")

    print("--- Main Program: Creating VoiceListener instance ---")
    listener = VoiceListener(
        # Customize phrases, model, device etc. here if needed
        # activation_phrase="computer",
        # model_name="base.en",
        # device="cuda", compute_type="float16", # If GPU available
        on_activation=my_activation_handler,
        on_command_update=my_command_update_handler,
        on_command_finish=my_command_finish_handler,
        on_listening_start=my_listening_start_handler,
        on_listening_stop=my_listening_stop_handler,
        on_debug_log=my_debug_logger
    )

    print("\n--- Main Program: Starting Listener ---")
    if listener.start():
        print("--- Main Program: Listener started successfully. Press Ctrl+C to stop. ---")
        try:
            # Keep the main thread alive while the listener runs in background threads
            while True:
                time.sleep(0.5) # Main thread doesn't need to do much work here
                # You could add other main program logic here if needed,
                # checking listener.is_listening() status, etc.

        except KeyboardInterrupt:
            print("\n--- Main Program: KeyboardInterrupt received ---")
        finally:
            print("--- Main Program: Stopping Listener ---")
            listener.stop()
            print("--- Main Program: Exiting ---")
    else:
        print("--- Main Program: Failed to start listener. Exiting. ---")