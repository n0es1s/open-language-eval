import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jiwer
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


class Transcription:
    def __init__(
        self,
        client: OpenAI,
        model: str = "whisper-large-v3",
        temperature: float = 0.0,
        source_language: str = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.source_language = source_language

    def transcribe_audio(self, audio_file_path: str, translate: bool = False) -> str:
        # Open the audio file
        with open(audio_file_path, "rb") as file:
            # Create a transcription of the audio file
            if translate:
                transcription = self.client.audio.translations.create(
                    file=file,  # Required audio file
                    model=self.model,  # Required model to use for transcription
                    response_format="json",  # Optional
                    temperature=self.temperature,  # Optional
                )
            else:
                transcription = self.client.audio.transcriptions.create(
                    file=file,  # Required audio file
                    model=self.model,  # Required model to use for transcription
                    response_format="json",  # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
                    language=self.source_language,  # Optional
                    temperature=self.temperature,  # Optional
                )
            # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
            return transcription.text

    def get_wer(self, original_text: str, predicted_text: str) -> float:
        transforms = jiwer.Compose(
            [
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        wer = jiwer.wer(
            original_text,
            predicted_text,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )
        return wer

    def batch_transcribe_audio(
        self,
        prefix: str,
        output_path: str,
        language: str = None,
        translate: bool = False,
        audio_file_path: str = "../data/",
        max_workers: int = 10,
    ) -> pd.DataFrame:
        if self.source_language is not None:
            language = self.source_language
        elif language is not None:
            language = language
        else:
            raise ValueError("Language folder must be specified")

        input_folder = Path(audio_file_path) / language

        # Read the TSV file into a pandas dataframe

        tsv_path = input_folder / "asr_test.tsv"
        df = pd.read_csv(tsv_path, sep="\t")

        print(f"Loaded {len(df)} rows from {tsv_path}")
        print(f"Columns: {df.columns.tolist()}")

        # Path to audio files
        audio_folder = input_folder / "test_part_0"

        # Helper function to process a single audio file
        def process_audio_file(idx, row):
            file_id = row["id"]
            original_text = row[
                "normalized_text"
            ]  # or 'raw_text' depending on preference

            # Construct audio file path (assuming .wav extension)
            audio_file_path = audio_folder / f"{file_id}.wav"

            if not audio_file_path.exists():
                print(f"Warning: Audio file not found: {audio_file_path}")
                predicted_text = None
                duration = None
                ttft = None
            else:
                try:
                    predicted_text = self.transcribe_audio(
                        str(audio_file_path), translate=translate
                    )
                    print(f"[{idx + 1}/{len(df)}] Transcribed: {file_id}")
                except Exception as e:
                    print(f"Error transcribing {file_id}: {e}")
                    predicted_text = None

            result = {
                "idx": idx,
                "id": file_id,
                "original_transcription": original_text,
                prefix + "predicted_transcription": predicted_text,
            }
            if not translate:
                result[prefix + "wer"] = self.get_wer(original_text, predicted_text)

            return result

        # Use ThreadPoolExecutor to transcribe files concurrently
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_audio_file, idx, row): idx
                for idx, row in df.iterrows()
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"Unexpected error processing row {idx}: {e}")

        # Sort results by original index to maintain order
        results.sort(key=lambda x: x["idx"])

        # Extract data for dataframe
        results_df = pd.DataFrame(results).drop(columns=["idx"])

        # Save results to CSV
        output_path = Path(output_path)

        # Check if file already exists and merge if it does
        if os.path.exists(output_path):
            print(f"\nFile {output_path} already exists. Merging with existing data...")
            existing_df = pd.read_csv(output_path)

            # Merge on 'id' column, with new results taking precedence (_x suffix from results_df)
            merged_df = existing_df.merge(
                results_df.drop(columns=["original_transcription"]),
                on="id",
                how="outer",
                suffixes=("_old", ""),
            )

            # For columns that exist in both, keep the new values (without suffix)
            # Drop the old columns (with _old suffix)
            cols_to_drop = [col for col in merged_df.columns if col.endswith("_old")]
            merged_df = merged_df.drop(columns=cols_to_drop)

            results_df = merged_df
            print(
                f"Merged {len(existing_df)} existing rows with {len(results_df)} total rows"
            )

        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print("\nFirst few results:")
        print(results_df.head())
        return results_df


if __name__ == "__main__":
    load_dotenv(".env")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    client = OpenAI(base_url="https://api.groq.com/openai/v1/", api_key=GROQ_API_KEY)
    language = "en"

    for model in ["whisper-large-v3", "whisper-large-v3-turbo"]:
        for translate in [False]:  # , False]:
            transcription = Transcription(
                client=client,
                model=model,
                source_language=language,
                temperature=0.0,
            )

            if not translate:
                if model == "whisper-large-v3":
                    prefix = "w_l_v3_tb_"
                else:
                    prefix = "w_l_v3_turbo_tl_"
            else:
                if model == "whisper-large-v3-turbo":
                    continue
                else:
                    prefix = "w_l_v3_tl_"

            results_df = transcription.batch_transcribe_audio(
                prefix=prefix,
                output_path=f"../data/{language}_transcription_results.csv",
                language=language,
                audio_file_path="../data/",
                max_workers=10,
                translate=translate,
            )

    # ===============================
    # Whisper Large V3
    # ===============================

    # model = "whisper-large-v3"

    # transcription = Transcription(
    #     client=client, model=model, source_language=language, temperature=0.0
    # )
    # results_df = transcription.batch_transcribe_audio(
    #     prefix="w_l_v3_tb",
    #     output_path="../data/{language}_transcription_results.csv",
    #     language=language,
    #     audio_file_path="../data/",
    #     max_workers=10,
    # )

    # ===============================
    # Whisper Large V3 with Translation
    # ===============================

    # model = "whisper-large-v3"

    # transcription = Transcription(
    #     client=client, model=model, source_language=language, temperature=0.0
    # )
    # results_df = transcription.batch_transcribe_audio(
    #     prefix="w_l_v3_tl_",
    #     output_path=f"../data/{language}_transcription_results.csv",
    #     language=language,
    #     audio_file_path="../data/",
    #     max_workers=10,
    #     translate=True,
    # )

    # ===============================
    # Whisper Large V3 Turbo
    # ===============================

    # model = "whisper-large-v3-turbo"

    # transcription = Transcription(
    #     client=client, model=model, source_language=language, temperature=0.0
    # )
    # results_df = transcription.batch_transcribe_audio(
    #     prefix="w_l_v3_turbo_tl_",
    #     output_path="../data/hu_transcription_results.csv",
    #     language=language,
    #     audio_file_path="../data/",
    #     max_workers=10,
    # )
