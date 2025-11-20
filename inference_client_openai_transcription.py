import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranscriptionService:
    """
    Service class for audio transcription using OpenAI and Groq APIs.

    Supports multiple models for transcription with flexible client configuration.
    """

    # Available models for each provider
    OPENAI_MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
    GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]

    def __init__(
        self,
        provider: Literal["openai", "groq"] = "groq",
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the audio transcription service.

        Args:
            provider: API provider to use ("openai" or "groq")
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for transcription (e.g., "en", "es")
            api_key: API key for the provider. If None, reads from environment
        """
        self.provider = provider
        self.source_language = source_language

        # Initialize the appropriate client
        if provider == "openai":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )
            self.model = model or "whisper-1"
            self.available_models = self.OPENAI_MODELS
        elif provider == "groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.environ.get("GROQ_API_KEY")
            )
            self.model = model or "whisper-large-v3"
            self.available_models = self.GROQ_MODELS
        else:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai' or 'groq'")

        # Validate model
        if self.model not in self.available_models:
            raise ValueError(
                f"Model '{self.model}' not available for provider '{provider}'. "
                f"Available models: {', '.join(self.available_models)}"
            )

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0
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
        audio_file = self._prepare_audio_input(audio_input)

        transcription_params = {
            "model": self.model,
            "file": audio_file,
            "temperature": temperature
        }

        # Add language if specified
        lang = language or self.source_language
        if lang:
            transcription_params["language"] = lang

        # Add prompt if specified
        if prompt:
            transcription_params["prompt"] = prompt

        transcription = self.client.audio.transcriptions.create(**transcription_params)
        return transcription.text

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

    def set_model(self, model: str):
        """
        Change the model being used.

        Args:
            model: New model ID

        Raises:
            ValueError: If model is not available for current provider
        """
        if model not in self.available_models:
            raise ValueError(
                f"Model '{model}' not available for provider '{self.provider}'. "
                f"Available models: {', '.join(self.available_models)}"
            )
        self.model = model

    def get_available_models(self) -> list[str]:
        """
        Get list of available models for current provider.

        Returns:
            List of model IDs
        """
        return self.available_models.copy()

    def __repr__(self) -> str:
        return (
            f"AudioTranscriptionService(provider='{self.provider}', "
            f"model='{self.model}', "
            f"source_language='{self.source_language}')"
        )
