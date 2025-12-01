import os

from dotenv import load_dotenv

from open_language_eval.voxpopuli.transcription_voxpopuli import TranscriptionVoxPopuli
from open_language_eval.voxpopuli.translation_voxpopuli import TranslationVoxPopuli

load_dotenv()

languages = ["hu"]

language_code_map = {
    "hu": "Hungarian",
}

transcription_models = [
    ("whisper-large-v3", "groq"),
    ("whisper-large-v3-turbo", "groq"),
]
translation_available_transcription_models = ["whisper-large-v3"]

translation_models = [("qwen/qwen3-32b", "groq"), ("gpt-4o-mini", "openai")]


for language in languages:
    if not os.path.exists(f"outputs/{language}/transcription"):
        os.makedirs(f"outputs/{language}/transcription")
    if not os.path.exists(f"outputs/{language}/translation"):
        os.makedirs(f"outputs/{language}/translation")

    for transcription_model, transcription_provider in transcription_models:
        for translation_model, translation_provider in translation_models:
            transcription_voxpopuli = TranscriptionVoxPopuli(
                language="hu",
                output_path=f"outputs/{language}/transcription/{transcription_model.replace('/', '_')}.json",
                provider=transcription_provider,
                model=transcription_model,
                translate=False,
            )

            # transcription_voxpopuli.transcribe_dataset(max_samples=100)
            transcription_voxpopuli.transcribe_dataset(
                files_path=f"data/{language}/test_part_0",
            )

            if transcription_model in translation_available_transcription_models:
                transcription_voxpopuli = TranscriptionVoxPopuli(
                    language="hu",
                    output_path=f"outputs/{language}/transcription/{transcription_model.replace('/', '_')}_translated.json",
                    provider=transcription_provider,
                    model=transcription_model,
                    translate=True,
                )

                # transcription_voxpopuli.transcribe_dataset(max_samples=100)
                transcription_voxpopuli.transcribe_dataset(
                    files_path=f"data/{language}/test_part_0",
                )

            translation_voxpopuli = TranslationVoxPopuli(
                model=translation_model,
                provider=translation_provider,
                source_language=language_code_map[language],
                target_language="English",
            )
            translation_voxpopuli.translate_data(
                input_file_path=f"outputs/{language}/transcription/{transcription_model.replace('/', '_')}.json",
                output_file_path=f"outputs/{language}/translation/{transcription_model.replace('/', '_')}_{translation_model.replace('/', '_')}.json",
            )
