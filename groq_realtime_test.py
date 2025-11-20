"""
Real-time incremental audio translation with sentence-level TTS playback.

This script:
1. Captures audio from the microphone (German speech)
2. Uses Silero VAD to detect voice activity
3. Uses amplitude filtering to prevent speaker feedback loops
4. Translates incrementally (every 2s during continuous speech OR on pause)
5. Detects complete sentences using NLTK sentence tokenization
6. Only plays back complete sentences via PlayAI TTS
7. Supports long continuous speech with progressive updates

Example flow:
  "Hi my" → (incomplete, hold)
  "Hi my name is Julian." → (complete sentence, queue for TTS)
  "I am testing" → (incomplete, hold)
  "I am testing this system." → (complete sentence, queue for TTS)
"""

import os
import io
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from groq import Groq
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import nltk
from nltk.tokenize import sent_tokenize


class RealtimeTranslator:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,  # seconds
        translation_chunk_duration: float = 3.0,  # seconds of audio before sending
        speech_threshold: float = 0.5,
        amplitude_threshold: float = 0.02,  # RMS amplitude threshold to filter speaker playback
        source_language: str = "de",  # German input
        target_language: str = "en",  # English output
        model: str = "whisper-large-v3",
        max_workers: int = 3,
        enable_tts: bool = True,
        tts_voice: str = "Celeste-PlayAI",
        translation_prompt: str = None
    ):
        """
        Initialize real-time translator with streaming support and TTS playback.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of each audio chunk in seconds for VAD
            translation_chunk_duration: Duration of audio to accumulate before translating
            speech_threshold: VAD threshold (0.0 to 1.0)
            amplitude_threshold: RMS amplitude threshold to filter low-volume audio (speaker playback)
            source_language: Source language code (e.g., "de" for German)
            target_language: Target language (only "en" supported by Groq translations)
            model: Model to use for translation
            max_workers: Maximum parallel translation workers
            enable_tts: Whether to play back translations as speech
            tts_voice: Voice to use for TTS (e.g., "Celeste-PlayAI")
            translation_prompt: Optional prompt to guide translation style
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.translation_chunk_size = int(sample_rate * translation_chunk_duration)
        self.speech_threshold = speech_threshold
        self.amplitude_threshold = amplitude_threshold
        self.enable_tts = enable_tts
        self.tts_voice = tts_voice
        self.source_language = source_language
        self.target_language = target_language
        self.translation_prompt = translation_prompt
        
        # Initialize Silero VAD
        print("Loading Silero VAD model...")
        self.vad_model = load_silero_vad()
        
        # Initialize NLTK for sentence detection
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
        
        # Also try legacy punkt for compatibility
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Initialize Groq client for translation and TTS
        print(f"Initializing Groq client with model {model}...")
        print(f"Translation: {source_language.upper()} → {target_language.upper()}")
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        
        # Audio buffer and state
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speech_buffer = []
        self.is_speaking = False
        self.last_transcription_time = 0
        self.chunk_counter = 0
        self.playback_queue = queue.Queue()
        self.sentence_queue = queue.Queue()  # Queue for complete sentences
        
        # Incremental transcription state
        self.current_transcript = ""  # Full transcript so far
        self.spoken_sentences = set()  # Track what's already been spoken
        self.last_sentence_end = 0  # Character position of last complete sentence
        self.speech_start_time = 0
        self.last_incremental_check = 0
        # Incremental interval: transcribe every N seconds during long continuous speech
        # This ensures real-time updates even if user keeps talking
        self.incremental_interval = 2.0  # seconds
        
        # Threading
        self.running = False
        self.stream = None
        self.processor_thread = None
        self.transcription_workers = ThreadPoolExecutor(max_workers=max_workers)
        self.transcription_thread = None
        self.playback_thread = None
        self.sentence_tts_thread = None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice input stream."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_data = indata[:, 0].copy()
        else:
            audio_data = indata.copy()
        
        # Add to queue for processing
        self.audio_queue.put(audio_data)
    
    def detect_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech in audio chunk using Silero VAD.
        
        Args:
            audio_chunk: Audio data as numpy array
        
        Returns:
            Speech probability (0.0 to 1.0)
        """
        # Silero VAD expects exactly 512 samples for 16kHz (or 256 for 8kHz)
        window_size = 512 if self.sample_rate == 16000 else 256
        
        # If chunk is smaller than window, pad it
        if len(audio_chunk) < window_size:
            audio_chunk = np.pad(audio_chunk, (0, window_size - len(audio_chunk)))
        
        # Process audio in windows and aggregate speech probabilities
        speech_probs = []
        for i in range(0, len(audio_chunk), window_size):
            window = audio_chunk[i:i + window_size]
            
            # If last window is incomplete, pad it
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(window).float()
            
            # Get speech probability for this window
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            speech_probs.append(speech_prob)
        
        # Return maximum speech probability across all windows
        return max(speech_probs) if speech_probs else 0.0
    
    def process_audio(self):
        """Process audio from queue and detect speech, sending chunks for translation."""
        print("Audio processor started")
        
        while self.running:
            try:
                # Get audio chunk from queue (with timeout to check running flag)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Calculate RMS amplitude (energy level) to filter low-volume audio
                rms_amplitude = np.sqrt(np.mean(audio_chunk**2))
                
                # Filter out low-amplitude audio (speaker playback, background noise)
                if rms_amplitude < self.amplitude_threshold:
                    # Show muted status for low amplitude
                    bar = "░" * 50
                    print(f"\r🔇 AMP: [{bar}] {rms_amplitude:.4f} (below threshold {self.amplitude_threshold:.4f})", end="", flush=True)
                    continue
                
                # Detect speech (only if amplitude is high enough)
                speech_prob = self.detect_speech(audio_chunk)
                
                # Visualize speech detection with amplitude info
                bar_length = int(speech_prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                status = "🎤" if self.is_speaking else "⏹️"
                print(f"\r{status} VAD: [{bar}] {speech_prob:.2f} | AMP: {rms_amplitude:.4f}", end="", flush=True)
                
                # Speech detection logic
                if speech_prob >= self.speech_threshold:
                    if not self.is_speaking:
                        print("\n[🎤 Speech started - incremental translation...]")
                        self.is_speaking = True
                        self.speech_buffer = []
                        self.speech_start_time = time.time()
                        self.last_incremental_check = time.time()
                    
                    # Add to speech buffer
                    self.speech_buffer.append(audio_chunk)
                    
                    # HYBRID TRIGGER: Check both time-based AND size-based
                    current_time = time.time()
                    time_since_last_check = current_time - self.last_incremental_check
                    total_samples = sum(len(chunk) for chunk in self.speech_buffer)
                    
                    # Trigger transcription if:
                    # 1. Enough time has passed (incremental updates during long speech)
                    # 2. OR we have enough audio accumulated
                    should_transcribe = (
                        time_since_last_check >= self.incremental_interval or
                        total_samples >= self.translation_chunk_size
                    )
                    
                    if should_transcribe and len(self.speech_buffer) > 0:
                        # Send for incremental translation
                        self.transcribe_incrementally(self.speech_buffer.copy())
                        self.last_incremental_check = current_time
                        # Keep overlap for context
                        overlap_chunks = 3  # Keep last 1.5 seconds
                        self.speech_buffer = self.speech_buffer[-overlap_chunks:]
                
                elif self.is_speaking:
                    # Speech ended - do final transcription
                    print("\n[⏸️  Speech ended - finalizing...]")
                    
                    # Send any remaining audio
                    if self.speech_buffer:
                        self.transcribe_incrementally(self.speech_buffer.copy(), is_final=True)
                    
                    self.is_speaking = False
                    self.speech_buffer = []
                    # Reset transcript for next utterance
                    self.current_transcript = ""
                    self.last_sentence_end = 0
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in audio processing: {e}", file=sys.stderr)
    
    def transcribe_incrementally(self, audio_chunks, is_final=False):
        """Transcribe audio incrementally and extract complete sentences."""
        if not audio_chunks:
            return
        
        self.chunk_counter += 1
        chunk_id = self.chunk_counter
        
        # Submit to thread pool
        future = self.transcription_workers.submit(
            self._process_incremental_translation, 
            audio_chunks, 
            chunk_id,
            is_final
        )
    
    def _process_incremental_translation(self, audio_chunks, chunk_id, is_final=False):
        """
        Translate audio incrementally and extract complete sentences.
        Only queues NEW complete sentences for TTS.
        """
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(audio_chunks)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            import wave
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_buffer.seek(0)
            wav_buffer.name = f"speech_{chunk_id}.wav"
            
            # Translate using Groq
            duration = len(audio_data) / self.sample_rate
            start_time = time.time()
            
            # Build translation request parameters
            translation_params = {
                "file": (wav_buffer.name, wav_buffer.read()),
                "model": self.model,
                "response_format": "text",
                "temperature": 0.0
            }
            
            # Add optional prompt if provided
            if self.translation_prompt:
                translation_params["prompt"] = self.translation_prompt
            
            response = self.groq_client.audio.translations.create(**translation_params)
            
            translation = response.strip() if isinstance(response, str) else response.text.strip()
            
            elapsed = time.time() - start_time
            
            if not translation:
                return
            
            # Extract complete sentences using NLTK
            sentences = sent_tokenize(translation)
            
            # Determine which sentences are complete
            complete_sentences = []
            
            if is_final:
                # All sentences are complete on final pass
                complete_sentences = sentences
            else:
                # Last sentence might be incomplete (no ending punctuation)
                if translation.endswith(('.', '!', '?', ';', ':')):
                    complete_sentences = sentences
                else:
                    # Hold last sentence as it's incomplete
                    complete_sentences = sentences[:-1] if len(sentences) > 1 else []
            
            # Queue only NEW complete sentences
            new_sentences = []
            for sentence in complete_sentences:
                # Simple deduplication based on sentence text
                sentence_clean = sentence.strip()
                if sentence_clean and sentence_clean not in self.spoken_sentences:
                    self.spoken_sentences.add(sentence_clean)
                    new_sentences.append(sentence_clean)
                    self.sentence_queue.put(sentence_clean)
            
            # Display result
            if new_sentences:
                print(f"\n{'='*80}")
                print(f"🌍 Chunk #{chunk_id} ({duration:.1f}s audio, {elapsed:.1f}s latency):")
                print(f"   {self.source_language.upper()} → {self.target_language.upper()}:")
                for i, sent in enumerate(new_sentences, 1):
                    print(f"   [{i}] {sent}")
                print(f"{'='*80}\n")
            else:
                # Just show progress without complete sentences
                print(f"\r[📝 Transcribing... partial: \"{translation[:50]}...\"]", end="", flush=True)
            
        except Exception as e:
            print(f"\n❌ Error translating chunk #{chunk_id}: {e}", file=sys.stderr)
    
    def sentence_tts_worker(self):
        """Worker thread that processes sentences from queue and generates TTS."""
        print("Sentence TTS worker started")
        sentence_counter = 0
        
        while self.running:
            try:
                # Get sentence from queue (with timeout)
                sentence = self.sentence_queue.get(timeout=0.1)
                sentence_counter += 1
                
                # Generate TTS for this sentence
                start_time = time.time()
                
                response = self.groq_client.audio.speech.create(
                    model="playai-tts",
                    voice=self.tts_voice,
                    input=sentence,
                    response_format="wav"
                )
                
                audio_data = response.read()
                elapsed = time.time() - start_time
                
                print(f"🔊 TTS generated for sentence #{sentence_counter} ({elapsed:.2f}s)")
                
                # Queue for playback
                self.playback_queue.put((sentence_counter, audio_data))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ Error generating TTS: {e}", file=sys.stderr)
        
        print("Sentence TTS worker stopped")
    
    def playback_worker(self):
        """Worker thread that plays back TTS audio."""
        print("Playback worker started")
        
        while self.running:
            try:
                # Get audio from playback queue (with timeout)
                chunk_id, audio_data = self.playback_queue.get(timeout=0.1)
                
                # Load audio from bytes
                audio_buffer = io.BytesIO(audio_data)
                data, samplerate = sf.read(audio_buffer)
                
                print(f"🔊 Playing chunk #{chunk_id}...")
                
                # Play audio (blocks until complete)
                sd.play(data, samplerate)
                sd.wait()  # Wait until audio finishes playing
                
                print(f"✅ Finished playing chunk #{chunk_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ Error playing audio: {e}", file=sys.stderr)
        
        print("Playback worker stopped")
    
    def start(self):
        """Start the real-time translation."""
        if self.running:
            print("Already running!")
            return
        
        self.running = True
        
        # Start audio processor thread
        self.processor_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processor_thread.start()
        
        # Start TTS worker threads if TTS is enabled
        if self.enable_tts:
            self.sentence_tts_thread = threading.Thread(target=self.sentence_tts_worker, daemon=True)
            self.sentence_tts_thread.start()
            
            self.playback_thread = threading.Thread(target=self.playback_worker, daemon=True)
            self.playback_thread.start()
        
        # Start audio input stream
        print(f"\n{'='*80}")
        print(f"🎙️  Starting real-time streaming translation")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Translation: {self.source_language.upper()} → {self.target_language.upper()}")
        print(f"Model: {self.model}")
        print(f"Chunk size: {self.translation_chunk_size / self.sample_rate:.1f}s")
        print(f"Amplitude threshold: {self.amplitude_threshold:.4f} (filters speaker playback)")
        if self.enable_tts:
            print(f"TTS Voice: {self.tts_voice}")
            print(f"🔊 TTS Playback: ENABLED")
        print(f"{'='*80}\n")
        print(f"🔴 Recording... Speak {self.source_language.upper()} close to your microphone.")
        print(f"💡 Incremental translation: Updates every {self.incremental_interval}s during long speech")
        print(f"💡 Sentence detection: Only complete sentences are played back")
        if self.enable_tts:
            print(f"🔊 English TTS will play complete sentences as they're detected")
        print(f"💡 Tip: Lower speaker volume to ~30% to prevent feedback loops.")
        print(f"💡 Watch AMP values: Direct speech should be >0.05, playback <0.02")
        print("Press Ctrl+C to stop.\n")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        
        self.stream.start()
    
    def stop(self):
        """Stop the real-time translation."""
        if not self.running:
            return
        
        print("\n\n[🛑 Stopping...]")
        
        # Stop stream first
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Send any remaining speech for translation
        if self.is_speaking and self.speech_buffer:
            print("[⏸️  Sending final segment for translation...]")
            self.transcribe_incrementally(self.speech_buffer.copy(), is_final=True)
        
        # Wait for all translations to complete
        print("[⏳ Waiting for pending translations...]")
        self.transcription_workers.shutdown(wait=True)
        
        # Wait for sentence queue to empty
        if self.enable_tts and not self.sentence_queue.empty():
            print("[⏳ Waiting for sentence TTS generation...]")
            while not self.sentence_queue.empty():
                time.sleep(0.1)
        
        # Wait for playback queue to empty
        if self.enable_tts and not self.playback_queue.empty():
            print("[⏳ Waiting for audio playback to finish...]")
            while not self.playback_queue.empty():
                time.sleep(0.1)
            time.sleep(0.5)  # Small delay to ensure last audio finishes
        
        # Stop running flag (stops worker threads)
        self.running = False
        
        # Wait for processor thread
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        # Wait for sentence TTS thread
        if self.sentence_tts_thread and self.sentence_tts_thread.is_alive():
            self.sentence_tts_thread.join(timeout=2.0)
        
        # Wait for playback thread
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        print("✅ Stopped")
    
    def run(self):
        """Run the translator until interrupted."""
        try:
            self.start()
            # Keep running until interrupted
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n\n[⚠️  Interrupted by user]")
        finally:
            self.stop()


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Real-time Incremental Translation + TTS Test")
    print("Groq API (Whisper Translation + PlayAI TTS) + Silero VAD")
    print("German → English Translation with Voice Playback")
    print("Features:")
    print("  - Incremental translation during continuous speech")
    print("  - Sentence-level detection (NLTK)")
    print("  - Amplitude filtering to prevent feedback loops")
    print("="*80 + "\n")
    
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("Please set it with: export GROQ_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create and run translator with incremental sentence-based TTS
    translator = RealtimeTranslator(
        sample_rate=16000,
        chunk_duration=0.5,  # VAD check every 0.5 seconds
        translation_chunk_duration=2.5,  # Max audio before forcing transcription
        speech_threshold=0.5,
        amplitude_threshold=0.012,  # RMS amplitude threshold - tune based on your setup!
        source_language="de",  # German input
        target_language="en",  # English output (only option for Groq translations)
        model="whisper-large-v3",
        max_workers=3,  # Allow up to 3 parallel translations
        enable_tts=True,  # Enable TTS playback of complete sentences
        tts_voice="Celeste-PlayAI",  # PlayAI TTS voice
        translation_prompt=None  # Optional: "Specify context or spelling"
    )
    
    translator.run()


if __name__ == "__main__":
    main()

