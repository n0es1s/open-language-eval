import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranslationGroq(AudioTranslationInterface):
    """
    Service class for audio translation using Groq API.

    Translates audio from any supported language to English.
    Supports multiple models for translation with flexible client configuration.
    """

    # Available models for each provider
    GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]

    def __init__(
        self,
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the audio translation service.

        Args:
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for translation (e.g., "es", "fr", "de")
            target_language: Target language code for translation (e.g., "en")
            api_key: API key for the provider. If None, reads from environment

        Note:
            Translation always outputs English text (Whisper API limitation)
        """
        super().__init__(
            source_language=source_language,
            target_language=target_language,
            available_models=self.GROQ_MODELS
        )
        self.model = model or "whisper-large-v3"
        # Validate model
        if self.model not in self.available_models:
            raise ValueError(
                f"Model '{self.model}' not available for AudioTranslationGroq. "
                f"Available models: {', '.join(self.available_models)}"
            )

        # Initialize the appropriate client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get("GROQ_API_KEY")
        )



    def translate(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        """
        Translate audio to English text.

        Args:
            audio_input: Audio file path, bytes, or BytesIO object
            prompt: Optional prompt to guide the translation
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Translated text in English
        """
        audio_file = self._prepare_audio_input(audio_input)

        translation_params = {
            "model": self.model,
            "file": audio_file,
            "temperature": temperature
        }

        # Add prompt if specified
        if prompt:
            translation_params["prompt"] = prompt

        translation = self.client.audio.translations.create(**translation_params)
        return translation.text

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
            f"AudioTranslationService(provider='{self.provider}', "
            f"model='{self.model}', "
            f"source_language='{self.source_language}')"
        )
