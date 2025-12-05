import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Tuple, Union

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from open_language_eval.assets.prompts import (
    SEVERITY_PROMPT_DQ4,
    SEVERITY_PROMPT_DR4,
    SEVERITY_PROMPT_GEMMA,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

ClientType = Union[OpenAI, Anthropic]

severity_prompts = {
    "gemma": SEVERITY_PROMPT_GEMMA,
    "dr4": SEVERITY_PROMPT_DR4,
    "dq4": SEVERITY_PROMPT_DQ4,
}

severity_scale = "dr4"  # "gemma", "dr4", "dq4"

severity_prompt = severity_prompts[severity_scale]
if severity_scale == "gemma":
    severity_type = Literal["critical", "major", "minor"]
    severity_description = "Error severity: critical, major, or minor"
elif severity_scale == "dr4":
    severity_type = int
    severity_description = "Error severity: 1, 2, 3, or 4"
elif severity_scale == "dq4":
    severity_type = float
    severity_description = "Error severity between 1 and 4"


class TranslationError(BaseModel):
    """Represents a single translation error."""

    error: str = Field(description="Description of the error")
    classification: Literal[
        "accuracy",
        "fluency",
        "style",
        "terminology",
        "non-translation",
        "other",
        "no-error",
    ] = Field(description="Error category (e.g., accuracy, fluency, style, etc.)")
    severity: severity_type = Field(description=severity_description)
    snippet: str = Field(
        description="The snippet from the translation containing the error"
    )


class TranslationAssessment(BaseModel):
    """Contains a list of translation errors found during assessment."""

    errors: List[TranslationError] = Field(
        description="List of translation errors identified"
    )


class TranslationJudge:
    """A judge for evaluating translation quality using Large Language Models.

    This class uses LLM-based evaluation to assess translation quality by identifying
    and classifying errors in translated text. It supports both OpenAI and Anthropic
    clients and returns structured error assessments including error type, severity,
    and specific text snippets.
    """

    def __init__(
        self,
        client: ClientType,
        model: str,
        source_language: str,
        target_language: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
    ):
        """Initialize the TranslationJudge.

        Args:
            client: Either an OpenAI or Anthropic client instance for API calls.
            model: The model identifier to use (e.g., "gpt-4", "claude-sonnet-4-5").
            source_language: The language of the source text (e.g., "English", "Spanish").
            target_language: The language of the translation (e.g., "German", "French").
            temperature: Sampling temperature for the LLM. Lower values (e.g., 0.0) produce
                more deterministic outputs. Defaults to 0.0.
            max_tokens: Maximum number of tokens for the LLM response. Only applies to
                Anthropic clients. Defaults to 16384.
        """
        self.client = client
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = SYSTEM_PROMPT + severity_prompt
        self.user_prompt = USER_PROMPT

    def _assess_with_openai(self, user_message: str) -> TranslationAssessment:
        """Handle OpenAI API calls with structured outputs."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,  # Model that supports structured outputs
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=TranslationAssessment,
            temperature=self.temperature,
        )

        return response.choices[0].message.parsed

    def _assess_with_anthropic(self, user_message: str) -> TranslationAssessment:
        """Handle Anthropic API calls with structured outputs."""
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self.max_tokens,
            # Use tools/tool_choice for structured output with Anthropic
            tools=[
                {
                    "name": "record_translation_errors",
                    "description": "Record the translation errors found in the assessment",
                    "input_schema": TranslationAssessment.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "record_translation_errors"},
        )

        # Extract the tool use from the response
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"), None
        )

        if tool_use:
            return TranslationAssessment(**tool_use.input)
        else:
            # Fallback: return empty assessment if no tool use found
            return TranslationAssessment(errors=[])

    def assess_translation(
        self,
        source_text: str,
        translated_text: str,
    ) -> TranslationAssessment:
        """
        Assess translation quality using an LLM judge.

        Args:
            source_text: The original text in the source language
            translated_text: The translated text

        Returns:
            TranslationAssessment: A Pydantic object containing a list of translation errors
        """
        # Format the user prompt with the provided inputs
        user_message = self.user_prompt.format(
            source_language=self.source_language,
            source_segment=source_text,
            target_language=self.target_language,
            target_segment=translated_text,
        )

        # Detect client type and make appropriate API call
        client_type = type(self.client).__module__

        if "openai" in client_type.lower():
            return self._assess_with_openai(user_message)
        elif "anthropic" in client_type.lower():
            return self._assess_with_anthropic(user_message)
        else:
            raise ValueError(
                f"Unsupported client type: {client_type}. Must be OpenAI or Anthropic client."
            )

    def batch_assess_translations(
        self,
        translation_pairs: List[Tuple[Any, str, str]],
        output_path: str,
        max_workers: int = 10,
        batch_size: int = 100,
    ) -> Dict[Any, Dict[str, Any]]:
        """Concurrently assess multiple translations with progress tracking.

        Args:
            translation_pairs: List of (id, source_text, translated_text) tuples
            output_path: Path to save the results
            max_workers: Maximum number of concurrent API calls (default: 10)
            batch_size: Number of translation pairs to process in each batch (default: 100)

        Returns:
            Dictionary with IDs as keys and assessment results as values
        """

        results_dict = {}
        total_count = len(translation_pairs)
        processed_count = 0
        batch_count = 0

        print(f"Starting concurrent evaluation of {total_count} translation pairs...")
        print(f"Using {max_workers} concurrent workers")
        print(f"Results will be saved to {output_path} every {batch_size} translations")

        # Create a function to process a single translation pair
        def process_pair(
            pair_id: Any, source_text: str, translated_text: str
        ) -> Tuple[Any, Dict[str, Any]]:
            retries = 0
            while retries < 3:
                try:
                    assessment = self.assess_translation(source_text, translated_text)

                    # Build result dictionary with serialized errors
                    result = {
                        "pair_id": pair_id,
                        "source_text": source_text,
                        "translated_text": translated_text,
                        "num_errors": len(assessment.errors),
                        "errors": [error.model_dump() for error in assessment.errors],
                    }

                    break

                except Exception as e:
                    print(f"\nError processing pair {pair_id}: {str(e)}")

                    result = {
                        "pair_id": pair_id,
                        "source_text": source_text,
                        "translated_text": translated_text,
                        "num_errors": -1,
                        "errors": [],
                        "error": str(e),
                    }
                    retries += 1

            if result["num_errors"] == -1:
                print(f"\nError processing pair {pair_id}: {result['error']}")

            return pair_id, result

        def save_results(results: Dict[Any, Dict[str, Any]], batch_num: int):
            """Save results to JSON file."""
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(
                    f"\n✓ Batch {batch_num}: Saved {len(results)} results to {output_path}"
                )
            except Exception as e:
                print(f"\n✗ Error saving results to {output_path}: {str(e)}")

        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(process_pair, pair_id, src, tgt): pair_id
                for pair_id, src, tgt in translation_pairs
            }

            # Process completed futures with progress bar
            with tqdm(total=total_count, desc="Assessing translations") as pbar:
                for future in as_completed(future_to_id):
                    pair_id, result = future.result()
                    results_dict[pair_id] = result
                    processed_count += 1

                    # Update progress bar with error count info
                    pbar.set_postfix({"errors": result["num_errors"]})
                    pbar.update(1)

                    # Save results after every batch_size translations
                    if processed_count % batch_size == 0:
                        batch_count += 1
                        save_results(results_dict, batch_count)

        # Save any remaining results
        if processed_count % batch_size != 0:
            batch_count += 1
            save_results(results_dict, batch_count)

        print(f"\nCompleted! All {total_count} translations assessed.")

        return results_dict


if __name__ == "__main__":
    # Load environment variables
    config = load_dotenv(".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Initialize clients
    openai_client = OpenAI(api_key=openai_api_key)
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=groq_api_key
    )
    claude_client = Anthropic(api_key=anthropic_api_key)

    # # Example 1: Assess an English to German translation with OpenAI
    # print("=" * 80)
    # print("Example 1: English to German (with intentional errors) - GPT-5.1")
    # print("=" * 80)

    # source_text = "The quick brown fox jumps over the lazy dog."
    # translated_text= "flinke braune Fuchs springt unter den faulen Hund."

    # judge_1 = TranslationJudge(
    #     client=openai_client,
    #     model="gpt-5.1-2025-11-13",
    #     source_language="English",
    #     target_language="German",
    # )

    # result_1 = judge_1.assess_translation(
    #     source_text=source_text,
    #     translated_text=translated_text,
    # )

    # print("\nResult 1")
    # print(f"\nErrors found: {len(result_1.errors)}")
    # for i, error in enumerate(result_1.errors, 1):
    #     print(f"\n{i}. {error.error}")
    #     print(f"   Classification: {error.classification}")
    #     print(f"   Severity: {error.severity}")
    #     print(f"   Snippet: '{error.snippet}'")

    # # Example 2: Assess an English to German translation with Groq
    # print("=" * 80)
    # print("Example 2: English to German (with intentional errors) - Qwen3-32b on Groq")
    # print("=" * 80)

    # judge_2 = TranslationJudge(
    #     client=groq_client,
    #     model="qwen/qwen3-32b",
    #     source_language="English",
    #     target_language="German",
    # )

    # result_2 = judge_2.assess_translation(
    #     source_text=source_text,
    #     translated_text=translated_text,
    # )

    # print("\nResult 2")
    # print(f"\nErrors found: {len(result_2.errors)}")
    # for i, error in enumerate(result_2.errors, 1):
    #     print(f"\n{i}. {error.error}")
    #     print(f"   Classification: {error.classification}")
    #     print(f"   Severity: {error.severity}")
    #     print(f"   Snippet: '{error.snippet}'")

    # # Example 3: Assess an English to German translation with Claude
    # print("=" * 80)
    # print("Example 3: English to German (with intentional errors) - Claude 4.5 Sonnet")
    # print("=" * 80)

    # judge_3 = TranslationJudge(
    #     client=claude_client,
    #     model="claude-sonnet-4-5-20250929",
    #     source_language="English",
    #     target_language="German",
    # )

    # result_3 = judge_3.assess_translation(
    #     source_text=source_text,
    #     translated_text=translated_text,
    # )

    # print("\nResult 3")
    # print(f"\nErrors found: {len(result_3.errors)}")
    # for i, error in enumerate(result_3.errors, 1):
    #     print(f"\n{i}. {error.error}")
    #     print(f"   Classification: {error.classification}")
    #     print(f"   Severity: {error.severity}")
    #     print(f"   Snippet: '{error.snippet}'")
