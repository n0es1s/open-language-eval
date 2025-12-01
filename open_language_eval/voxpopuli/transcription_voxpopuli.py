import io
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Optional

import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm.auto import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

from open_language_eval.evals.transcription_evaluation import TranscriptionWER
from open_language_eval.inference_clients.transcription_client import (
    AudioTranscriptionInterface,
)


class TranscriptionVoxPopuli(AudioTranscriptionInterface, TranscriptionWER):
    def __init__(
        self,
        language: str,
        output_path: str,
        provider: str,
        model: Optional[str] = None,
        max_workers: int = 10,
        translate: bool = False,
    ):
        self.language = language
        self.output_path = output_path
        self.provider = provider
        self.model = model
        self.max_workers = max_workers
        self.translate = translate
        self.transforms = jiwer.Compose(
            [
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        if language == "en":
            self.normalizer = EnglishTextNormalizer()
        else:
            self.normalizer = None
        AudioTranscriptionInterface.__init__(
            self,
            provider=provider,
            model=model,
            source_language=language,
            api_key=None,
            prompt=None,
            temperature=0.0,
            translate=translate,
        )
        TranscriptionWER.__init__(self, self.transforms, self.normalizer)

    def transcribe_dataset(
        self,
        files_path: Optional[str] = None,
        tsv_file_path: Optional[str] = None,
        split: str = "test",
        batch_size: int = 100,
        max_workers: int = None,
        max_samples: int = None,
    ):
        """
        Transcribe audio samples from the VoxPopuli dataset with concurrent processing.

        Args:
            split: Dataset split to use (default: "test")
            batch_size: Number of samples to process in each batch
            max_workers: Maximum number of concurrent workers for transcription (default: 10)
            max_samples: Maximum number of samples to process (None for all)
        """
        if max_workers is None:
            max_workers = self.max_workers

        if files_path is not None:
            files_path = Path(files_path)
            if tsv_file_path is not None:
                tsv_file_path = Path(tsv_file_path)
            else:
                tsv_file_path = files_path / ".." / f"asr_{split}.tsv"

            df = pd.read_csv(tsv_file_path, sep="\t")
            dataset = self._row_generator(df)

            total_samples = len(df) if not max_samples else min(max_samples, len(df))
        else:
            dataset = load_dataset(
                "facebook/voxpopuli",
                self.language,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            total_samples = max_samples

        results = {
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "source_language": self.language,
                "dataset": f"facebook/voxpopuli ({self.language})",
                "num_samples": 0,
                "timestamp": datetime.now().isoformat(),
            },
            "samples": [],
        }
        batch_samples = []
        batch_audio_inputs = []
        sample_count = 0

        # Count total samples for progress bar (if not streaming, we could get exact count)
        print(f"Starting transcription for {self.language} dataset...")
        print(f"Provider: {self.provider}, Model: {self.model}")
        print(f"Concurrent workers: {max_workers}, Batch size: {batch_size}")

        # Process dataset in batches
        for i, sample in enumerate(tqdm(dataset, total=total_samples)):
            if max_samples and i >= max_samples:
                break

            # Get audio data
            if files_path is None:
                audio_array = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]
            else:
                sample["audio_id"] = sample["id"]
                audio_file_path = files_path / f"{sample['id']}.wav"
                audio_array, sampling_rate = librosa.load(audio_file_path, sr=None)

            # Convert numpy array to audio bytes
            audio_bytes = self._audio_array_to_bytes(audio_array, sampling_rate)

            # Store sample info and audio
            batch_samples.append(
                (sample["audio_id"], sample, audio_array, sampling_rate)
            )
            batch_audio_inputs.append((sample["audio_id"], audio_bytes))

            sample_count += 1
            results["metadata"]["num_samples"] = sample_count

            # Process batch when full
            if len(batch_samples) >= batch_size:
                self._process_batch(
                    batch_samples, batch_audio_inputs, results, max_workers
                )
                # Save results incrementally after each batch
                self._save_results(results)
                batch_samples = []
                batch_audio_inputs = []

        # Process remaining samples
        if batch_samples:
            self._process_batch(batch_samples, batch_audio_inputs, results, max_workers)
            # Save final batch results
            self._save_results(results)

        # Print summary statistics
        self._print_summary(results["samples"])

        return results

    def _row_generator(self, data):
        """
        Yield rows from either:
        - a custom dataset object (already iterable)
        - a pandas DataFrame (yields each row as a dict)
        """
        # Case 1: pandas DataFrame
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                # convert to dict, or keep as Series if you prefer
                yield row.to_dict()

        # Case 2: any other iterable dataset
        elif isinstance(data, Iterable):
            for item in data:
                yield item

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _audio_array_to_bytes(
        self, audio_array: np.ndarray, sampling_rate: int
    ) -> io.BytesIO:
        """Convert audio array to BytesIO object in WAV format."""
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_array, sampling_rate, format="WAV")
        audio_buffer.seek(0)
        audio_buffer.name = "audio.wav"
        return audio_buffer

    def _process_batch(
        self,
        batch_samples: list,
        batch_audio_inputs: list,
        results: dict,
        max_workers: int,
    ):
        """Process a batch of samples concurrently."""
        print(f"\nProcessing batch of {len(batch_samples)} samples...")

        batch_results = []

        # Transcribe batch concurrently
        transcriptions = self.transcribe_batch(
            batch_audio_inputs, max_workers=max_workers
        )

        # Process each sample
        for audio_id, sample, audio_array, sampling_rate in batch_samples:
            transcription_text = transcriptions.get(audio_id, "")

            # Get reference text
            reference_original = sample["normalized_text"]

            # Normalize texts
            if self.normalizer:
                reference_normalized = self.normalizer(reference_original)
                transcription_normalized = self.normalizer(transcription_text)
            else:
                reference_normalized = reference_original
                transcription_normalized = transcription_text

            # Calculate WER using inherited evaluate method
            wer_score = self.calculate_wer(reference_original, transcription_text)

            # Create result entry
            sample_result = {
                "audio_info": {
                    "audio_id": sample["audio_id"],
                    "speaker_id": sample["speaker_id"],
                    "gender": sample["gender"],
                    "duration_seconds": round(len(audio_array) / sampling_rate, 2),
                    "sampling_rate": sampling_rate,
                },
                "transcriptions": {
                    "reference_original": reference_original,
                    "reference_normalized": reference_normalized,
                    "model_original": transcription_text,
                    "model_normalized": transcription_normalized,
                },
                "metrics": {"wer": round(wer_score, 4)},
            }
            results["samples"].append(sample_result)
            batch_results.append(sample_result)

        self.print_batch_summary(batch_results)

    def _save_results(self, results: dict):
        """Save results to JSON file incrementally."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(
            f"✓ Saved a total of {len(results['samples'])} results to: {self.output_path}"
        )

    def _print_summary(self, results: list):
        """Print summary statistics."""
        if not results:
            print("No results to summarize.")
            return

        wer_scores = [r["metrics"]["wer"] for r in results]
        avg_wer = sum(wer_scores) / len(wer_scores)

        print("\n" + "=" * 60)
        print("TRANSCRIPTION SUMMARY")
        print("=" * 60)
        print(f"Total samples processed: {len(results)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Min WER: {min(wer_scores):.4f}")
        print(f"Max WER: {max(wer_scores):.4f}")
        print(f"Provider: {self.provider}")
        print(f"Model: {self.model}")
        print(f"Language: {self.language}")
        print("=" * 60)

    def print_batch_summary(self, batch_results: list):
        """Print summary statistics for a batch of results."""
        wer_scores = [r["metrics"]["wer"] for r in batch_results]
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"Batch completed: {len(batch_results)} samples processed")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Min WER: {min(wer_scores):.4f}")
        print(f"Max WER: {max(wer_scores):.4f}")
