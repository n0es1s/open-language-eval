from pathlib import Path

from dotenv import load_dotenv

from open_language_eval.assets.language_code_map import language_code_map
from open_language_eval.voxpopuli.transcription_voxpopuli import TranscriptionVoxPopuli
from open_language_eval.voxpopuli.translation_voxpopuli import TranslationVoxPopuli

load_dotenv()

# ============ PARAMETERS ==========

# 1. Languages in the VoxPopuli dataset to use
languages = ["cs", "pl", "hu"]

# 2. Models to use for the transcription, as well as the provider (groq or openai)
transcription_models = [
    ("whisper-large-v3", "groq"),
    ("whisper-large-v3-turbo", "groq"),
]

# 3. Models to use for text translations, as well as the provider (groq or openai)
translation_available_transcription_models = ["whisper-large-v3"]

# 4. Models to use for text translations, as well as the provider (groq or openai)
translation_models = [("qwen/qwen3-32b", "groq"), ("gpt-4o-mini", "openai")]

# 5. Input Directory
input_directory = "data"

# 6. Output Directory
output_directory = "outputs"

# ============ END OF PARAMETERS ==========


for language in languages:
    for transcription_model, transcription_provider in transcription_models:
        transcription_voxpopuli = TranscriptionVoxPopuli(
            language=language,
            output_path=Path(output_directory)
            / f"{language}/transcription/{transcription_model.replace('/', '_')}.json",
            provider=transcription_provider,
            model=transcription_model,
            translate=False,
        )

        transcription_voxpopuli.transcribe_dataset(
            files_path=Path(input_directory) / f"{language}/test_part_0",
        )

        if language != "en":
            if transcription_model in translation_available_transcription_models:
                transcription_voxpopuli = TranscriptionVoxPopuli(
                    language=language,
                    output_path=Path(output_directory)
                    / f"{language}/transcription/{transcription_model.replace('/', '_')}_translated.json",
                    provider=transcription_provider,
                    model=transcription_model,
                    translate=True,
                )

                transcription_voxpopuli.transcribe_dataset(
                    files_path=Path(input_directory) / f"{language}/test_part_0",
                )

        for translation_model, translation_provider in translation_models:
            if language != "en":
                translation_voxpopuli = TranslationVoxPopuli(
                    model=translation_model,
                    provider=translation_provider,
                    source_language=language_code_map[language],
                    target_language="English",
                )
                translation_voxpopuli.translate_data(
                    input_file_path=Path(output_directory)
                    / f"{language}/transcription/{transcription_model.replace('/', '_')}.json",
                    output_file_path=Path(output_directory)
                    / f"{language}/translation/en_{transcription_model.replace('/', '_')}_{translation_model.replace('/', '_')}.json",
                )
            else:
                for target_language in languages:
                    if target_language == language:
                        continue
                    translation_voxpopuli = TranslationVoxPopuli(
                        model=translation_model,
                        provider=translation_provider,
                        source_language=language_code_map[language],
                        target_language=language_code_map[target_language],
                    )
                    translation_voxpopuli.translate_data(
                        input_file_path=Path(output_directory)
                        / f"{language}/transcription/{transcription_model.replace('/', '_')}.json",
                        output_file_path=Path(output_directory)
                        / f"{language}/translation/{target_language}_{transcription_model.replace('/', '_')}_{translation_model.replace('/', '_')}.json",
                    )
