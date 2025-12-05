import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Optional, Union

from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.protocol.transcription.request import TranscriptionRequest
from openai import OpenAI


class AudioTranscriptionInterface:
    """
    Interface class for audio transcription.

    Supports multiple models for transcription with flexible client configuration.
    """

    GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]
    GROQ_TRANSLATION_MODELS = ["whisper-large-v3"]
    OPENAI_MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
    OPENAI_TRANSLATION_MODELS = []
    MISTRAL_MODELS = ["voxtrall-small-24b", "voxtrall-mini-3b"]
    MISTRAL_TRANSLATION_MODELS = []

    def __init__(
        self,
        provider: Literal["groq", "openai", "mistral"] = "groq",
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        translate: bool = False,
    ):
        """
        Initialize the AudioTranscriptionInterface.

        Args:
            provider: The provider to use for transcription (either "groq" or "openai").
            model: The model to use for transcription.
            source_language: The language of the source text.
            api_key: The API key to use for transcription.
            prompt: The prompt to use for transcription.
            temperature: The temperature to use for transcription.
            translate: Whether to translate the transcription.
        """

        self.provider = provider
        self.prompt = prompt
        self.temperature = temperature
        self.source_language = source_language
        self.api_key = api_key
        self.translate = translate

        # Initialize the appropriate client and model
        if provider == "groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.environ.get("GROQ_API_KEY"),
            )
            self.available_models = (
                self.GROQ_TRANSLATION_MODELS if translate else self.GROQ_MODELS
            )
        elif provider == "openai":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            )
            self.available_models = (
                self.OPENAI_TRANSLATION_MODELS if translate else self.OPENAI_MODELS
            )
        elif provider == "mistral":
            self.client = OpenAI(
                base_url="http://<your-server-host>:8000/v1",
                api_key="EMPTY",
            )
            self.available_models = (
                self.MISTRAL_TRANSLATION_MODELS if translate else self.MISTRAL_MODELS
            )
        if model is None:
            self.model = self.available_models[0]

        if self.model not in self.available_models:
            raise ValueError(f"Model '{self.model}' not available for {self.provider}")

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
            if self.provider == "mistral":
                transcription_params["audio"] = transcription_params.pop("file")
                req = TranscriptionRequest(**transcription_params).to_openai(
                    exclude=("top_p", "seed")
                )
                transcription = self.client.audio.transcriptions.create(**req)
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
        if self.provider == "mistral":
            if not isinstance(audio_input, str):
                raise ValueError("audio_input must be a file path for Mistral")
            audio = Audio.from_file(audio_input, strict=False)

            return RawAudio.from_audio(audio)

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

    def __repr__(self) -> str:
        return (
            f"AudioTranslation(model='{self.model}', "
            f"provider='{self.provider}', "
            f"source_language='{self.source_language}', "
            f"prompt='{self.prompt}', "
            f"temperature='{self.temperature}')"
            f"translate='{self.translate}')"
        )
