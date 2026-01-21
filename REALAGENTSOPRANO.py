import time
import threading
import queue
import sys
import random
import numpy as np
import sounddevice as sd

# --- Configuration ---
SAMPLE_RATE = 16000  # Input sample rate
TTS_SAMPLE_RATE = 32000 # Soprano output rate
VAD_THRESHOLD = 0.01 # RMS threshold for voice detection (adjust if too sensitive)
SILENCE_DURATION = 1.2 # Seconds of silence to consider "done speaking"

# Import Soprano
try:
    from soprano import SopranoTTS
except ImportError:
    print("Soprano not found. Please install it first: pip install .")
    sys.exit(1)

class InterruptiblePlayer:
    """
    Plays audio chunks but can be interrupted immediately.
    """
    def __init__(self):
        self.stop_event = threading.Event()
        self.stream = None
        self._lock = threading.Lock()

    def play(self, audio_generator):
        """
        Consumes an audio generator (yielding numpy arrays) and plays them.
        Stops loop if self.stop_event is set.
        """
        self.stop_event.clear()
        
        # Initialize output stream
        with sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype='float32') as self.stream:
            for chunk in audio_generator:
                if self.stop_event.is_set():
                    break
                
                # Convert torch tensor to numpy if needed
                if hasattr(chunk, 'cpu'):
                    chunk = chunk.cpu().numpy()
                
                # Write to stream
                self.stream.write(chunk)

    def stop(self):
        """Signals the player to stop immediately."""
        self.stop_event.set()

class MicrophoneListener:
    """
    Listens to the microphone in a separate thread.
    Detects voice activity (RMS > threshold).
    """
    def __init__(self, on_speech_start=None, on_speech_end=None):
        self.queue = queue.Queue()
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.speaking = False
        self.running = False
        self.silence_start_time = None

    def _callback(self, indata, frames, time_info, status):
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(indata**2))
        self.queue.put(rms)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.start()

    def _listen_loop(self):
        with sd.InputStream(callback=self._callback, channels=1, samplerate=SAMPLE_RATE):
            while self.running:
                try:
                    rms = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if rms > VAD_THRESHOLD:
                    if not self.speaking:
                        self.speaking = True
                        if self.on_speech_start:
                            self.on_speech_start()
                    self.silence_start_time = None
                
                elif self.speaking:
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                    
                    if time.time() - self.silence_start_time > SILENCE_DURATION:
                        self.speaking = False
                        if self.on_speech_end:
                            self.on_speech_end()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

class MockBrain:
    """Mock LLM & STT"""
    def transcribe(self):
        print("\n(Simulating STT for demo... assume you said 'Hello')")
        return "Hello there."

    def generate(self, text):
        responses = [
            "I heard you! I stopped speaking immediately because you interrupted me.",
            "Yes, I am listening. Go on.",
            "That's a great point. Soprano's low latency allows this snappy interaction.",
            "Interruption detected. I'm all ears.",
             "I can handle full duplex conversations easily.",
        ]
        return random.choice(responses)

class FullDuplexAgent:
    def __init__(self):
        print("Loading Soprano...")
        # Use 'auto' to automatically select CUDA/GPU if available
        self.tts = SopranoTTS(device='auto', backend='auto')
        self.player = InterruptiblePlayer()
        self.brain = MockBrain()
        self.listener = MicrophoneListener(
            on_speech_start=self.handle_interruption, 
            on_speech_end=self.handle_user_turn
        )
        self.is_agent_speaking = False

    def handle_interruption(self):
        """Called when user starts speaking"""
        if self.is_agent_speaking:
            print("\n[!] Barge-in detected! Stopping TTS...")
            self.player.stop()
            self.is_agent_speaking = False
        print("\n[User started speaking...]")

    def handle_user_turn(self):
        """Called when user finishes speaking"""
        print("[User finished speaking]")
        # In a real app, you'd get the audio buffer here and send to Whisper
        # For this demo, we mock the transcription
        text = self.brain.transcribe()
        self.respond(text)

    def respond(self, user_text):
        print(f"Thinking response to: '{user_text}'...")
        response_text = self.brain.generate(user_text)
        
        print(f"Agent: {response_text}")
        self.is_agent_speaking = True
        
        # Generate stream
        audio_stream = self.tts.infer_stream(response_text)
        
        # Play (interruptible)
        self.player.play(audio_stream)
        self.is_agent_speaking = False

    def run(self):
        print("\n--- Full Duplex Soprano Agent ---")
        print("1. Speak into your microphone.")
        print("2. The agent will respond.")
        print("3. Interrupt the agent while it speaks - it should stop immediately!")
        print("-----------------------------------")
        
        self.listener.start()
        try:
            # Keep main thread alive
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.listener.stop()
            print("\nGoodbye!")

if __name__ == "__main__":
    agent = FullDuplexAgent()
    agent.run()
