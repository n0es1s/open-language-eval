import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranscriptionService(AudioTranscriptionInterface):
    """
    Service class for audio transcription using Groq API.

    Supports multiple models for transcription with flexible client configuration.
    """

    # Available models for each provider
    GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]

    def __init__(
        self,
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the audio transcription service.

        Args:
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for transcription (e.g., "en", "es")
            api_key: API key for the provider. If None, reads from environment
        """
        super().__init__(model or "whisper-large-v3", GROQ_MODELS, source_language)
        self.prompt = prompt
        self.temperature = temperature

        # Initialize the appropriate client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get("GROQ_API_KEY")
        )

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
    ) -> str:
        """
        Transcribe audio to text in the source language.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Transcribed text
        """
        audio_file = self._prepare_audio_input(audio_input)

        transcription_params = {
            "model": self.model,
            "file": audio_file,
            "temperature": self.temperature
        }

        # Add language if specified
        if self.source_language:
            transcription_params["language"] = self.source_language

        # Add prompt if specified
        if self.prompt:
            transcription_params["prompt"] = self.prompt

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
