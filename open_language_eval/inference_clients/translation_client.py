import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

from openai import OpenAI

from open_language_eval.assets.prompts import TRANSLATION_PROMPT


class TranslationInterface:
    def __init__(
        self,
        model: str,
        provider: Literal["openai", "groq"],
        source_language: str,
        target_language: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the TranslationInterface.

        Args:
            model: The model to use for translation.
            provider: The provider to use for translation (either "groq" or "openai").
            source_language: The language of the source text.
            target_language: The language of the translation.
            api_key: The API key to use for translation.
            temperature: The temperature to use for translation.
        """
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        self.provider = provider

        if provider == "openai":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            )
        elif provider == "groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.environ.get("GROQ_API_KEY"),
            )

    def translate(self, text: str) -> str:
        """Translate a single text using the translation model.

        Args:
            text: The text to translate.

        Returns:
            The translated text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator with expertise in multiple languages.",
                    },
                    {
                        "role": "user",
                        "content": TRANSLATION_PROMPT.format(
                            source_language=self.source_language,
                            target_language=self.target_language,
                            text=text,
                        ),
                    },
                ],
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.split("</think>")[-1]
            return result

        except Exception as e:
            print(f"Error translating text: {e}")
            return None

    def batch_translate(self, texts: list[tuple], max_workers: int = 10) -> dict:
        """
        Translate multiple texts concurrently using ThreadPoolExecutor.

        Args:
            texts: List of tuples containing (id, text) to translate
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary with ids as keys and translated texts as values
        """
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_id = {
                executor.submit(self.translate, text): id_ for id_, text in texts
            }

            # Collect results in a dictionary
            results = {}
            for future in as_completed(future_to_id):
                id_ = future_to_id[future]
                results[id_] = future.result()

            return results
