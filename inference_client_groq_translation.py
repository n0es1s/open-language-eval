import os
import io
import base64
from typing import Optional, Literal, Union
from openai import OpenAI


class AudioTranslationGroq(AudioTranslationInterface):
    """
    Service class for audio translation using Groq API.

    Translates audio from any supported language to any target language.
    Supports multiple models for translation with flexible client configuration.
    """

    # Available models for each provider
    GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]

    def __init__(
        self,
        model: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize the audio translation service.

        Args:
            model: Model ID to use. If None, uses default for provider
            source_language: Source language code for translation (e.g., "es", "fr", "de")
            target_language: Target language code for translation (e.g., "en")
            api_key: API key for the provider. If None, reads from environment
            prompt: Optional prompt to guide the translation
            temperature: Sampling temperature (0.0 to 1.0)

        Note:
            Translation always outputs English text (Whisper API limitation)
        """
        super().__init__(
            model=model or "whisper-large-v3",
            available_models=self.GROQ_MODELS,
            source_language=source_language,
            target_language=target_language,
        )
        self.prompt = prompt
        self.temperature = temperature

        # Initialize the appropriate client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get("GROQ_API_KEY")
        )

    def translate(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
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
            "temperature": self.temperature
        }

        # Add prompt if specified
        if self.prompt:
            translation_params["prompt"] = self.prompt

        translation = self.client.audio.translations.create(**translation_params)
        return translation.text

    def __repr__(self) -> str:
        return (
            f"AudioTranslationGroq(model='{self.model}', "
            f"source_language='{self.source_language}', "
            f"target_language='{self.target_language}', "
            f"prompt='{self.prompt}', "
            f"temperature='{self.temperature}')"
        )
