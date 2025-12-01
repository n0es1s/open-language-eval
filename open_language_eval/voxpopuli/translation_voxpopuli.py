import json
from typing import Literal, Optional

from dotenv import load_dotenv
from tqdm.auto import tqdm

from open_language_eval.inference_clients.translation_client import TranslationInterface


class TranslationVoxPopuli(TranslationInterface):
    """Translation interface specifically designed for VoxPopuli dataset processing.

    This class extends TranslationInterface to handle batch translation of VoxPopuli
    dataset samples. It processes translations in configurable batches and saves
    results incrementally to prevent data loss during long-running operations.
    """

    def __init__(
        self,
        model: str,
        provider: Literal["openai", "groq"],
        source_language: str,
        target_language: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_workers: int = 10,
        batch_size: int = 100,
    ):
        """Initialize the TranslationVoxPopuli instance.

        Args:
            model: The model identifier to use for translation (e.g., "qwen/qwen3-32b").
            provider: The API provider, either "openai" or "groq".
            source_language: The language of the source text (e.g., "Hungarian", "English").
            target_language: The language to translate to (e.g., "English", "German").
            api_key: Optional API key for the provider. If None, will use environment
                variable. Defaults to None.
            temperature: Sampling temperature for translation generation. Lower values
                (e.g., 0.0) produce more deterministic outputs. Defaults to 0.0.
            max_workers: Maximum number of concurrent translation API calls within
                each batch. Defaults to 10.
            batch_size: Number of samples to process before saving results to disk.
                Defaults to 100.
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        super().__init__(
            model=model,
            provider=provider,
            source_language=source_language,
            target_language=target_language,
            api_key=api_key,
            temperature=temperature,
        )

    def translate_data(self, input_file_path: str, output_file_path: str):
        """Translate VoxPopuli dataset samples and save results incrementally.

        This method reads a VoxPopuli dataset file, translates the transcriptions in
        batches, and saves the results after each batch to prevent data loss. The
        output file includes translation metadata and is updated progressively.

        Args:
            input_file_path: Path to the input JSON file containing VoxPopuli samples
                with transcriptions to translate.
            output_file_path: Path where the translated results will be saved. The file
                is overwritten after each batch with all results collected so far.

        Note:
            The input file is expected to have a specific structure:
            - "metadata" key containing dataset information
            - "samples" key containing a list of samples, each with:
                - "audio_info" containing "audio_id"
                - "transcriptions" containing "model_original" (text to translate)

            The output file will include the same structure with an additional
            "translations" key for each sample containing the translated text.
        """
        with open(input_file_path, "r") as f:
            results = json.load(f)

        results["metadata"]["translation_model"] = self.model
        results["metadata"]["translation_provider"] = self.provider
        results["metadata"]["translation_source_language"] = self.source_language
        results["metadata"]["translation_target_language"] = self.target_language

        samples = results["samples"]
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size

        # Process samples in batches with progress bar
        for batch_idx in tqdm(
            range(0, len(samples), self.batch_size),
            total=total_batches,
            desc="Translating batches",
        ):
            batch_samples = samples[batch_idx : batch_idx + self.batch_size]

            # Create batch of (id, text) tuples
            batch_data = [
                (
                    sample["audio_info"]["audio_id"],
                    sample["transcriptions"]["model_original"],
                )
                for sample in batch_samples
            ]

            # Translate the batch
            translations = self.batch_translate(
                batch_data, max_workers=self.max_workers
            )

            # Update results with translations
            for sample in batch_samples:
                audio_id = sample["audio_info"]["audio_id"]
                if "translations" not in sample:
                    sample["translations"] = {}
                sample["translations"][self.target_language] = translations[audio_id]

            # Write results to file after each batch
            with open(output_file_path, "w") as f:
                json.dump(results, f, indent=4)

            print(
                f"Translated {batch_idx + self.batch_size} samples and saved to {output_file_path}"
            )


if __name__ == "__main__":
    load_dotenv()
    translation_voxpopuli = TranslationVoxPopuli(
        model="qwen/qwen3-32b",
        provider="groq",
        source_language="Hungarian",
        target_language="English",
    )
    translation_voxpopuli.translate_data(
        input_file_path="outputs/hu/hu_transcription_whisper_large_v3_turbo.json",
        output_file_path="outputs/hu/hu_whisper_large_v3_turbo_qwen3-32b_translated.json",
    )
