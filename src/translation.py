import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import TRANSLATION_PROMPT


class Translation:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        source_language: str,
        target_language: str,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature

    def translate(self, text: str) -> str:
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
            result = response.choices[0].message.content
            return result

        except Exception as e:
            print(f"Error translating text: {e}")
            return None

    def batch_translate(self, texts: list[str], max_workers: int = 10) -> list[str]:
        """
        Translate multiple texts concurrently using ThreadPoolExecutor.

        Args:
            texts: List of texts to translate

        Returns:
            List of translated texts in the same order as input
        """
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_index = {
                executor.submit(self.translate, text): i for i, text in enumerate(texts)
            }

            # Collect results in order
            results = [None] * len(texts)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()

            return results

    def batch_translate_csv(
        self, file_path: str, text_column: str, prefix: str, max_workers: int = 10
    ) -> pd.DataFrame:
        file_path = Path(file_path)
        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        translated_texts = self.batch_translate(texts, max_workers)
        df[
            prefix + self.source_language + "_" + self.target_language + "_translation"
        ] = translated_texts
        df.to_csv(file_path, index=False)
        return df


if __name__ == "__main__":
    load_dotenv(".env")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    client = OpenAI(base_url="https://api.groq.com/openai/v1/", api_key=GROQ_API_KEY)

    language = "hu"

    # model = "openai/gpt-oss-120b"
    model = "qwen/qwen3-32b"

    source_language = "Hungarian"
    target_language = "English"
    temperature = 0.0
    translation = Translation(
        client=client,
        model=model,
        source_language=source_language,
        target_language=target_language,
        temperature=temperature,
    )
    results_df = translation.batch_translate_csv(
        file_path=f"../data/{language}_transcription_results.csv",
        text_column="w_l_v3_tbpredicted_transcription",
        # prefix="gpt_oss_120b_",
        prefix="qwen3_32b_",
        max_workers=10,
    )
