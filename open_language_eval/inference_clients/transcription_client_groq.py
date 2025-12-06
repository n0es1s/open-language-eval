import io
import os
from typing import Any, Optional, Union
from open_language_eval.inference_clients.transcription_client import AudioTranscriptionInterface
from openai import OpenAI
import numpy as np


class AudioTranscriptionGroq(AudioTranscriptionInterface):
    """
    Service class for audio transcription using Groq API.

    Supports multiple models for transcription with flexible client configuration.
    """

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
        super().__init__(model or "whisper-large-v3", self.GROQ_MODELS, source_language)
        self.prompt = prompt
        self.temperature = temperature

        # Initialize the appropriate client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get("GROQ_API_KEY")
        )

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> str:
        """
        Transcribe audio to text in the source language.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object

        Returns:
            Transcribed text
        """
        audio_file = self._prepare_audio_input(audio_input, sample_rate)

        transcription_params = {
            "model": self.model,
            "file": audio_file,
            "temperature": self.temperature,
        }

        # Add language if specified
        if not self.translate:
            if self.source_language:
                transcription_params["language"] = self.source_language

        # Add prompt if specified
        if self.prompt:
            transcription_params["prompt"] = self.prompt

        if self.translate:
            transcription = self.client.audio.translations.create(
                **transcription_params
            )
        else:
            transcription = self.client.audio.transcriptions.create(
                **transcription_params
            )

        return transcription.text

    def __repr__(self) -> str:
        return (
            f"AudioTranscriptionGroq(model='{self.model}', "
            f"source_language='{self.source_language}', "
            f"prompt='{self.prompt}', "
            f"temperature='{self.temperature}')"
        )
