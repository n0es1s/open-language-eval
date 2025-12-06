import os
import io
import base64
from typing import Optional, Literal, Union, Any
from openai import OpenAI
import numpy as np
import soundfile as sf
from librosa import resample
from concurrent.futures import ThreadPoolExecutor, as_completed


class AudioTranscriptionInterface:
    """
    Interface class for audio transcription.

    Supports multiple models for transcription with flexible client configuration.
    """

    def __init__(
        self,
        model: str,
        available_models: list[str],
        source_language: str,
        target_sample_rate: Optional[int] = 16000,
        format: Optional[str] = "WAV",
        subtype: Optional[str] = None,
    ):
        """
        Initialize the audio transcription service.

        Args:
            model: Model ID to use
            available_models: List of available models for the provider
            source_language: Source language code for transcription (e.g., "en", "es")
            target_sample_rate: Target sample rate for audio input
            format: Audio format for transcription
            subtype: Audio subtype for transcription
        """

        # Initialize the appropriate client
        self.model = model
        self.available_models = available_models
        self.source_language = source_language
        self.target_sample_rate = target_sample_rate
        self.format = format
        self.subtype = subtype if subtype else sf.default_subtype()

        # Validate model
        if self.model not in self.available_models:
            raise ValueError(
                f"Model '{self.model}' not available for AudioTranscriptionInterface. "
                f"Available models: {', '.join(self.available_models)}"
            )

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
    ) -> str:
        """
        Transcribe audio to text in the source language.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object
            language: Override source language for this transcription

        Returns:
            Transcribed text
        """
        raise NotImplementedError

    def transcribe_batch(
        self,
        audio_inputs: list[tuple[Any, Union[str, bytes, io.BytesIO, np.ndarray]]],
        sample_rate: Optional[int] = None,
        max_workers: int = 10,
    ) -> dict[Any, str]:
        """
        Transcribe multiple audio files concurrently.

        Args:
            audio_inputs: List of tuples (identifier, audio_input)
            sample_rate: Sample rate for audio input
            max_workers: Maximum number of concurrent workers

        Returns:
            List of tuples (identifier, transcription_text)
        """
        results = {}

        def transcribe_single(
            item: tuple[Any, Union[str, bytes, io.BytesIO, np.ndarray]],
        ) -> tuple[Any, str]:
            """Helper function to transcribe a single audio input."""
            identifier, audio_input = item
            try:
                transcription = self.transcribe(audio_input, sample_rate)
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

    def _prepare_audio_input(
        self, audio_input: Union[str, bytes, io.BytesIO, np.ndarray], sample_rate: Optional[int] = None
    ) -> Union[io.BytesIO, object]:
        """
        Prepare audio input for API consumption.

        Args:
            audio_input: Audio file path, bytes, BytesIO object, or numpy array
            sample_rate: Sample rate of audio input

        Returns:
            Prepared audio file object
        """

        if isinstance(audio_input, np.ndarray):
            audio_array = audio_input
            if not sample_rate or sample_rate != self.target_sample_rate:
                print(f"Resampling audio from {sample_rate or 'unknown'} to {self.target_sample_rate}")
                audio_array = resample(audio_array, orig_sr=sample_rate, target_sr=self.target_sample_rate)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_array, self.target_sample_rate, format=self.format, subtype=self.subtype)
            audio_bytes.seek(0)
            audio_bytes.name = "audio.wav"
            return audio_bytes

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