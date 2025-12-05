import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI
import websocket
import json
import time
import threading
from open_language_eval.inference_clients.transcription_client import AudioTranscriptionInterface
from dotenv import load_dotenv

load_dotenv()

class AudioTranscriptionOpenAI(AudioTranscriptionInterface):
    """
    Service class for audio transcription using OpenAI Realtime API.

    Supports multiple models for transcription with flexible client configuration.
    """

    # Available models for each provider
    OPENAI_MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]

    def __init__(
        self,
        model: Optional[str] = "whisper-1",
        source_language: Optional[str] = None,
        api_key: Optional[str] = None,
        noise_reduction: Optional[Literal["near_field", "far_field"]] = None,
        prompt: Optional[str] = None,
        turn_detection: Optional[Literal["server_vad", "semantic_vad"]] = "server_vad",
        silence_duration_ms: Optional[int] = 500,
        eagerness: Optional[Literal["low", "medium", "high"]] = "medium",
        interrupt_response: Optional[bool] = True,
        log_events: Optional[bool] = False,
        completion_wait_sec: Optional[int] = 3,
    ):
        """
        Initialize the audio transcription service.

        Args:
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for transcription (e.g., "en", "es")
            api_key: API key for the provider. If None, reads from environment
        """
        super().__init__(model, self.OPENAI_MODELS, source_language)

        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found")
        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = [
            "Authorization: Bearer " + openai_api_key,
        ]

        # Configure turn detection
        if turn_detection == "server_vad":
            turn_detection_config = {
                "type": turn_detection,
                "silence_duration_ms": silence_duration_ms,
                "create_response": True,
                "interrupt_response": interrupt_response,
            }
        elif turn_detection == "semantic_vad":
            turn_detection_config = {
                "type": turn_detection,
                "eagerness": eagerness,
                "create_response": True,
                "interrupt_response": interrupt_response,
            }
        else:
            turn_detection_config = None

        # Configure session
        self.session_config = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "noise_reduction": {"type": noise_reduction} if noise_reduction else None,
                    "transcription": {
                        "model": model,
                        "language": source_language,
                    },
                    "turn_detection": turn_detection_config,
                },
            }
        }
        if prompt:
            self.session_config["audio"]["input"]["transcription"]["prompt"] = prompt

        self.transcriptions = []
        self.completion_wait_sec = completion_wait_sec
        self.last_transcript_received_time = None
        self.started = False
        self._error = None
        self._error_event = threading.Event()

        def on_open(ws):
            # update session
            ws.send(json.dumps({
                "type": "session.update",
                "session": self.session_config,
            }))

        def on_message(ws, msg):
            data = json.loads(msg)
            t = data.get("type", "")
            if log_events:
                print("-> event:", t)

            if t == "session.created":
                print("Session opened")

            if t == "session.updated":
                print("Session updated", data.get("session", ""))
                self.started = True

            if t == "input_audio_buffer.committed":
                # get item_id and previous_item_id from committed for reordering later
                item_id = data["item_id"]
                previous = data.get("previous_item_id")

                if item_id not in self.transcriptions:
                    self.transcriptions[item_id] = {"partial": "", "transcript": "", "previous": None}

                self.transcriptions[item_id]["previous"] = previous
                print("Input audio buffer committed", data)

            # Streamed audio from the assistant (PCM16 base64 deltas)
            if t == "conversation.item.input_audio_transcription.completed":
                print("Transcript:", data)

                item_id = data["item_id"]
                transcript = data["transcript"]

                # Ensure entry exists
                if item_id not in self.transcriptions:
                    self.transcriptions[item_id] = {"partial": "", "transcript": "", "previous": None}

                self.transcriptions[item_id]["transcript"] = transcript
                self.last_transcript_received_time = time.time()

                print("Completed transcript:", transcript)

                self.last_transcript_received_time = time.time()

            if t == "error":
                print("Server error:", data)
                self._error = Exception("Server error: " + data.get("message", ""))
                self._error_event.set()
                self.close()

        def on_error(ws, err):
            print("WebSocket error:", err)
            self._error = Exception("WebSocket error: " + str(err))
            self._error_event.set()
            self.close()

        def on_close(ws, code, reason):
            print("WebSocket Closed:", code, reason)

        # Initialize the websocket
        self.ws = websocket.WebSocketApp(
            url, header=headers,
            on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close,
        )

        # Start websocket loop in a background thread
        self._ws_thread = threading.Thread(
            target=self.ws.run_forever,
            daemon=True,
        )
        self._ws_thread.start()
        while self.started is False:
            if self._error_event.is_set():
                raise self._error
            time.sleep(0.1)

    def transcribe(
        self,
        audio_input: io.BytesIO,
    ) -> str:
        """
        Transcribe audio to text in the source language.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Transcribed text
        """
        self.transcriptions = {} # a linked list dict
        self.last_transcript_received_time = None
        self._error = None
        self._error_event.clear()
        self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_input.read()).decode("ascii")
        }))
        # commit because last VAD event might not be triggered
        self.ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

        # wait for the transcript to be completed
        while not self.last_transcript_received_time or (time.time() - self.last_transcript_received_time) <= self.completion_wait_sec:
            if self._error_event.is_set():
                raise self._error
            time.sleep(0.2)
            if not self.last_transcript_received_time:
                print("Waiting for transcript to start...")
            else:
                print("Waiting for transcript to complete. Time since last delta:", time.time() - self.last_transcript_received_time)

        return self.get_transcript()

    def get_session_config(self) -> dict:
        return self.session_config

    def get_transcript(self) -> str:
        # reconstruct chunks via previous_item_id linked list
        items = self.transcriptions   # dict: item_id -> {previous, partial, transcript}

        if not items:
            return ""

        # --- 1. Find the head item (previous_item_id is None) ---
        head = None
        for item_id, info in items.items():
            if info["previous"] is None:
                head = item_id
                break

        if head is None:
            # Fallback: no explicit head found → pick arbitrary stable item
            head = next(iter(items.keys()))

        # --- 2. Walk forward through the chain ---
        ordered_transcripts = []
        current = head

        while current:
            info = items[current]
            text = info["transcript"] or info["partial"] or ""
            ordered_transcripts.append(text)

            # find next item whose previous == current
            next_item = None
            for iid, d in items.items():
                if d["previous"] == current:
                    next_item = iid
                    break

            current = next_item

        # --- 3. Join together into final transcript ---
        return " ".join(ordered_transcripts)

    def close(self):
        # Graceful shutdown
        self.ws.close()
        if self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2)

    def __repr__(self) -> str:
        return f"AudioTranscriptionOpenAI(model='{self.model}', source_language='{self.source_language}')"
