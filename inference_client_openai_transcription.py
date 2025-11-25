import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI
import websocket
import json
import time

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
        silence_duration_ms: Optional[int] = 200,
        eagerness: Optional[Literal["low", "medium", "high"]] = "medium",
        interrupt_response: Optional[bool] = False,
        log_events: Optional[bool] = False,
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
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
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
                        "prompt": prompt,
                        "language": source_language,
                    },
                    "turn_detection": turn_detection_config,
                },
            }
        }

        self.transcript = None

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

            # Streamed audio from the assistant (PCM16 base64 deltas)
            if t == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                if self.transcript is None:
                    print("Transcript completed:", transcript)
                    self.transcript = transcript
                else:
                    print("Transcript updated:", transcript)
                    self.transcript += "\n" + transcript

            if t == "error":
                print("Server error:", data)

        def on_error(ws, err):
            print("WebSocket error:", err)

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
        self.transcript = None
        self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_input.read()).decode("ascii")
        }))

        # wait for the transcript to be completed
        while self.transcript is None:
            time.sleep(0.1)

        return self.transcript

    def get_session_config(self) -> dict:
        return self.session_config

    def get_transcript(self) -> str:
        return self.transcript

    def close(self):
        # Graceful shutdown
        self.ws.close()
        if self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2)

    def __repr__(self) -> str:
        return f"AudioTranscriptionOpenAI(model='{self.model}', source_language='{self.source_language}')"
