import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranslationInterface:
    """
    Service class for audio translation using OpenAI and Groq APIs.

    Translates audio from any supported language to any target language.
    Supports multiple models for translation with flexible client configuration.
    """

    def __init__(
        self,
        model: str,
        available_models: Optional[list[str]] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ):
        """
        Initialize the audio translation service.

        Args:
            model: Model ID to use
            available_models: List of available models for the provider
            source_language: Source language code for translation (e.g., "es", "fr", "de")
            target_language: Target language code for translation (e.g., "en")

        Note:
            Translation always outputs English text (Whisper API limitation)
        """
        self.model = model
        self.available_models = available_models
        self.source_language = source_language
        self.target_language = target_language

        # Validate model
        if self.model not in self.available_models:
            raise ValueError(
                f"Model '{self.model}' not available for AudioTranslationInterface. "
                f"Available models: {', '.join(self.available_models)}"
            )

    def translate(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
    ) -> str:
        """
        Translate audio to target language

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Translated audio transcripts in target language
        """
        raise NotImplementedError

    def _prepare_audio_input(self, audio_input: Union[str, bytes, io.BytesIO]) -> Union[io.BytesIO, object]:
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
            if not hasattr(audio_input, 'name'):
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
            raise ValueError("audio_input must be a file path, bytes, or BytesIO object")

        return base64.b64encode(audio_bytes).decode('utf-8')

    def get_available_models(self) -> list[str]:
        """
        Get list of available models for current provider.

        Returns:
            List of model IDs
        """
        return self.available_models.copy()
