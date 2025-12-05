import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from open_language_eval.evals.translation_llm_judge import TranslationJudge

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

language = "de"
input_path = Path("outputs") / language / "translation"
output_path = Path("outputs") / language / "evals"

if not output_path.exists():
    output_path.mkdir(parents=True)

# Find all files in the directory
files = list(input_path.glob("*"))

for file in files:
    judge = TranslationJudge(
        client=client,
        model="qwen/qwen3-32b",
        source_language="German",
        target_language="English",
    )

    with open(file, "r") as f:
        data = json.load(f)

    translation_pairs = []

    for sample in data["samples"]:
        source_text = sample["transcriptions"]["model_original"]
        for target_language in sample["translations"].keys():
            target_text = sample["translations"][target_language]
            translation_pairs.append(
                (sample["audio_info"]["audio_id"], source_text, target_text)
            )
    results = judge.batch_assess_translations(
        translation_pairs, output_path=output_path / f"{file.stem}_llm_judge.json"
    )
