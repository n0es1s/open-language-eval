import json

from anthropic import Anthropic
from openai import OpenAI
from typiing import Union

ClientType = Union[OpenAI, Anthropic]


class JudgeVoxpopuli:
    def __init__(
        self, client: ClientType, model: str, source_language: str, target_language: str
    ):
        self.client = client
        self.model = model
        self.source_language = source_language
        self.target_language = target_language

    def judge_results(self, file_path: str):
        translation_pairs = []
        with open(file_path, "r") as f:
            results = json.load(f)

        for sample in results["samples"]:
            source_text = sample["transcriptions"]["reference_original"]
            for target_language in sample["translations"].keys():
                target_text = sample["translations"][target_language]
                translation_pairs.append(
                    (sample["audio_info"]["audio_id"], source_text, target_text)
                )
