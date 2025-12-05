import os
import io
import base64
from typing import Optional, Literal, Union, Any
from openai import OpenAI


class AudioTranscriptionInterface:
    """
    Interface class for audio transcription.

    Supports multiple models for transcription with flexible client configuration.
    """

    def __init__(
        self,
        model: str,
        available_models: Optional[list[str]] = None,
        source_language: Optional[str] = None,
    ):
        """
        Initialize the audio transcription service.

        Args:
            model: Model ID to use
            available_models: List of available models for the provider
            source_language: Source language code for transcription (e.g., "en", "es")
        """
        self.source_language = source_language

        # Initialize the appropriate client
        self.model = model
        self.available_models = available_models

        # Validate model
        if self.model not in self.available_models:
            raise ValueError(
                f"Model '{self.model}' not available for AudioTranscriptionInterface. "
                f"Available models: {', '.join(self.available_models)}"
            )

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
    ) -> str:
        """
        Transcribe audio to text in the source language.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object
            language: Override source language for this transcription
            prompt: Optional prompt to guide the transcription
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Transcribed text
        """
        raise NotImplementedError

    def transcribe_batch(
        self,
        audio_inputs: list[tuple[Any, Union[str, bytes, io.BytesIO]]],
        max_workers: int = 10,
    ) -> dict[Any, str]:
        """
        Transcribe multiple audio files concurrently.

        Args:
            audio_inputs: List of tuples (identifier, audio_input)
            max_workers: Maximum number of concurrent workers

        Returns:
            List of tuples (identifier, transcription_text)
        """
        raise NotImplementedError

    def _prepare_audio_input(
        self, audio_input: Union[str, bytes, io.BytesIO]
    ) -> Union[io.BytesIO, object]:
        """
        Prepare audio input for API consumption.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Prepared audio file object
        """
        # If it's a file path
        if isinstance(audio_input, str):
            return open(audio_input, "rb")

        # If it's bytes, convert to BytesIO
        if isinstance(audio_input, bytes):
            audio_bytes = io.BytesIO(audio_input)
            audio_bytes.name = "audio.wav"
            audio_bytes.seek(0)
            return audio_bytes

        # If it's already BytesIO
        if isinstance(audio_input, io.BytesIO):
            if not hasattr(audio_input, "name"):
                audio_input.name = "audio.wav"
            audio_input.seek(0)
            return audio_input

        raise ValueError("audio_input must be a file path, bytes, or BytesIO object")

    def to_base64(self, audio_input: Union[str, bytes, io.BytesIO]) -> str:
        """
        Convert audio input to base64 encoded string.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Base64 encoded audio string
        """
        if isinstance(audio_input, str):
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            audio_bytes = audio_input.read()
        else:
            raise ValueError(
                "audio_input must be a file path, bytes, or BytesIO object"
            )

        return base64.b64encode(audio_bytes).decode("utf-8")

    def get_available_models(self) -> list[str]:
        """
        Get list of available models for current provider.

        Returns:
            List of model IDs
        """
        return self.available_models.copy()