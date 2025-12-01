import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranscriptionOpenAI(AudioTranscriptionInterface):
    """
    Service class for audio transcription using OpenAI.

    Supports multiple models for transcription with flexible client configuration.
    """

    # Available models for each provider
    OPENAI_MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]

    def __init__(
        self,
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the audio transcription service.

        Args:
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for transcription (e.g., "en", "es")
            api_key: API key for the provider. If None, reads from environment
        """
        super().__init__(model or "whisper-1", self.OPENAI_MODELS, source_language)

        # Initialize the appropriate client
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
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
        # TODO: implement transcription
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"AudioTranscriptionOpenAI(model='{self.model}', source_language='{self.source_language}')"
