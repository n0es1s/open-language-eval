import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Optional, Union
from open_language_eval.inference_clients.transcription_client import AudioTranscriptionInterface
from openai import OpenAI


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
        results = {}

        def transcribe_single(
            item: tuple[Any, Union[str, bytes, io.BytesIO]],
        ) -> tuple[Any, str]:
            """Helper function to transcribe a single audio input."""
            identifier, audio_input = item
            try:
                transcription = self.transcribe(audio_input)
                return (identifier, transcription)
            except Exception as e:
                # Return error message as transcription
                print(f"Error transcribing {identifier}: {e}")
                return (identifier, f"ERROR: {str(e)}")

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(transcribe_single, item): item for item in audio_inputs
            }

            # Collect results as they complete
            for future in as_completed(future_to_item):
                id, result = future.result()
                results[id] = result

        return results

    def __repr__(self) -> str:
        return (
            f"AudioTranscriptionGroq(model='{self.model}', "
            f"source_language='{self.source_language}', "
            f"prompt='{self.prompt}', "
            f"temperature='{self.temperature}')"
        )
