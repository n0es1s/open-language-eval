import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Literal, Tuple, Dict, Any
from pydantic import BaseModel, Field
from prompts import SYSTEM_PROMPT, USER_PROMPT, SEVERITY_PROMPT_GEMMA, SEVERITY_PROMPT_DR4, SEVERITY_PROMPT_DQ4
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path


ClientType = Union[OpenAI, Anthropic]

severity_prompts = {
    "gemma": SEVERITY_PROMPT_GEMMA,
    "dr4": SEVERITY_PROMPT_DR4,
    "dq4": SEVERITY_PROMPT_DQ4
}

severity_scale = "dr4" # "gemma", "dr4", "dq4"

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
    classification: Literal["accuracy", "fluency", "style", "terminology", "non-translation", "other", "no-error"] = Field(description="Error category (e.g., accuracy, fluency, style, etc.)")
    severity: severity_type = Field(description=severity_description)
    snippet: str = Field(description="The snippet from the translation containing the error")

class TranslationAssessment(BaseModel):
            """Contains a list of translation errors found during assessment."""
            errors: List[TranslationError] = Field(description="List of translation errors identified")    


class TranslationJudge():
    def __init__(
        self, 
        client: ClientType, 
        model: str,
        source_language: str,
        target_language: str,
        temperature: float = 0.0,
        max_tokens: int = 16384):
        

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
            model= self.model,  # Model that supports structured outputs
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
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
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=self.max_tokens,
            # Use tools/tool_choice for structured output with Anthropic
            tools=[
                {
                    "name": "record_translation_errors",
                    "description": "Record the translation errors found in the assessment",
                    "input_schema": TranslationAssessment.model_json_schema()
                }
            ],
            tool_choice={"type": "tool", "name": "record_translation_errors"}
        )
        
        # Extract the tool use from the response
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None
        )
        
        if tool_use:
            return TranslationAssessment(**tool_use.input)
        else:
            # Fallback: return empty assessment if no tool use found
            return TranslationAssessment(errors=[])



    def assess_translation(self,
        source_text: str,
        translated_text: str,
    ) -> TranslationAssessment:
        """
        Assess translation quality using an LLM judge.
        
        Args:
            source_text: The original text in the source language
            source_language: The language of the source text (e.g., "English", "Spanish")
            translated_text: The translated text
            translated_language: The language of the translation
            client: Either an OpenAI client or Anthropic client object
            
        Returns:
            TranslationAssessment: A Pydantic object containing a list of translation errors
        """
        # Format the user prompt with the provided inputs
        user_message = self.user_prompt.format(
            source_language=self.source_language,
            source_segment=source_text,
            target_language=self.target_language,
            target_segment=translated_text
        )
        
        # Detect client type and make appropriate API call
        client_type = type(self.client).__module__
        
        if 'openai' in client_type.lower():
            return self._assess_with_openai(user_message)
        elif 'anthropic' in client_type.lower():
            return self._assess_with_anthropic(user_message)
        else:
            raise ValueError(f"Unsupported client type: {client_type}. Must be OpenAI or Anthropic client.")


    def batch_assess_translations(
        self,
        translation_pairs: List[Tuple[str, str]],
        output_csv_path: str,
        batch_size: int = 100,
        max_workers: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Concurrently assess multiple translations and save results to CSV intermittently.
        Each error creates a new row in the CSV for the same translation pair.
        
        Args:
            translation_pairs: List of (source_text, translated_text) tuples
            output_csv_path: Path where the CSV file will be saved
            batch_size: Number of results to accumulate before saving to CSV (default: 10)
            max_workers: Maximum number of concurrent API calls (default: 5)
            
        Returns:
            List of dictionaries containing all assessment results
        """
        results = []
        pending_rows = []
        completed_count = 0
        total_count = len(translation_pairs)
        
        # Initialize CSV file with headers
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        csv_headers = [
            'index',
            'pair_id',
            'source_text',
            'translated_text',
            'num_errors',
            'error_description',
            'error_classification',
            'error_severity',
            'error_snippet'
        ]
        
        # Write CSV headers
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
        
        print(f"Starting concurrent evaluation of {total_count} translation pairs...")
        print(f"Using {max_workers} concurrent workers")
        print(f"Results will be saved to: {output_csv_path}")
        
        # Create a function to process a single translation pair
        def process_pair(pair_id: int, source_text: str, translated_text: str) -> Dict[str, Any]:
            try:
                assessment = self.assess_translation(source_text, translated_text)
                
                # Build result dictionary
                result = {
                    'pair_id': pair_id,
                    'source_text': source_text,
                    'translated_text': translated_text,
                    'num_errors': len(assessment.errors),
                    'errors': assessment.errors
                }
                
                return result
                
            except Exception as e:
                print(f"Error processing pair {pair_id}: {str(e)}")
                return {
                    'pair_id': pair_id,
                    'source_text': source_text,
                    'translated_text': translated_text,
                    'num_errors': -1,
                    'errors': [],
                    'error': str(e)
                }
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_pair, i, src, tgt): i
                for i, (src, tgt) in enumerate(translation_pairs)
            }
            
            # Process completed futures
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)
                
                # Convert result to CSV rows (one row per error)
                csv_rows = self._convert_result_to_csv_rows(result)
                pending_rows.extend(csv_rows)
                completed_count += 1
                
                print(f"Completed {completed_count}/{total_count} - Pair {result['pair_id']}: {result['num_errors']} errors found")
                
                # Save to CSV intermittently
                if len(pending_rows) >= batch_size:
                    self._append_results_to_csv(output_csv_path, pending_rows, csv_headers)
                    print(f"Saved {len(pending_rows)} rows to CSV")
                    pending_rows = []
        
        # Save any remaining results
        if pending_rows:
            self._append_results_to_csv(output_csv_path, pending_rows, csv_headers)
            print(f"Saved final {len(pending_rows)} rows to CSV")
        
        # Sort results by pair_id for consistency
        results.sort(key=lambda x: x['pair_id'])
        
        print(f"\nCompleted! All {total_count} translations assessed.")
        print(f"Results saved to: {output_csv_path}")
        
        return results
    
    
    def _convert_result_to_csv_rows(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a single result with multiple errors into multiple CSV rows.
        Each error gets its own row with an index.
        """
        csv_rows = []
        pair_id = result['pair_id']
        source_text = result['source_text']
        translated_text = result['translated_text']
        num_errors = result['num_errors']
        
        # Handle error case
        if 'error' in result:
            return [{
                'index': 0,
                'pair_id': pair_id,
                'source_text': source_text,
                'translated_text': translated_text,
                'num_errors': num_errors,
                'error_description': f"ERROR: {result['error']}",
                'error_classification': '',
                'error_severity': '',
                'error_snippet': ''
            }]
        
        # If no errors, create one row with empty error fields
        if num_errors == 0:
            return [{
                'index': 0,
                'pair_id': pair_id,
                'source_text': source_text,
                'translated_text': translated_text,
                'num_errors': num_errors,
                'error_description': '',
                'error_classification': '',
                'error_severity': '',
                'error_snippet': ''
            }]
        
        # Create one row per error
        for i, error in enumerate(result['errors']):
            csv_rows.append({
                'index': i,
                'pair_id': pair_id,
                'source_text': source_text,
                'translated_text': translated_text,
                'num_errors': num_errors,
                'error_description': error.error,
                'error_classification': error.classification,
                'error_severity': str(error.severity),
                'error_snippet': error.snippet
            })
        
        return csv_rows
    
    
    def _append_results_to_csv(
        self,
        csv_path: str,
        results: List[Dict[str, Any]],
        headers: List[str]
    ) -> None:
        """Helper method to append results to CSV file."""
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writerows(results)

    

if __name__ == "__main__":

    # Load environment variables
    config = load_dotenv(".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


    # Initialize clients
    openai_client = OpenAI(api_key=openai_api_key)
    groq_client = OpenAI(base_url = "https://api.groq.com/openai/v1", api_key=groq_api_key)
    claude_client = Anthropic(api_key=anthropic_api_key)

    # Example 1: Assess an English to German translation with OpenAI
    print("=" * 80)
    print("Example 1: English to German (with intentional errors) - GPT-5.1")
    print("=" * 80)
    
    source_text = "The quick brown fox jumps over the lazy dog."
    translated_text= "flinke braune Fuchs springt unter den faulen Hund."

    judge_1 = TranslationJudge(
        client=openai_client,
        model="gpt-5.1-2025-11-13",
        source_language="English",
        target_language="German",
    )
    
    result_1 = judge_1.assess_translation(
        source_text=source_text,
        translated_text=translated_text,
    )
    
    print("\nResult 1")
    print(f"\nErrors found: {len(result_1.errors)}")
    for i, error in enumerate(result_1.errors, 1):
        print(f"\n{i}. {error.error}")
        print(f"   Classification: {error.classification}")
        print(f"   Severity: {error.severity}")
        print(f"   Snippet: '{error.snippet}'")

    # Example 2: Assess an English to German translation with Groq
    print("=" * 80)
    print("Example 2: English to German (with intentional errors) - Qwen3-32b on Groq")
    print("=" * 80)

    judge_2 = TranslationJudge(
        client=groq_client,
        model="qwen/qwen3-32b",
        source_language="English",
        target_language="German",
    )
    
    result_2 = judge_2.assess_translation(
        source_text=source_text,
        translated_text=translated_text,
    )
    
    print("\nResult 2")
    print(f"\nErrors found: {len(result_2.errors)}")
    for i, error in enumerate(result_2.errors, 1):
        print(f"\n{i}. {error.error}")
        print(f"   Classification: {error.classification}")
        print(f"   Severity: {error.severity}")
        print(f"   Snippet: '{error.snippet}'")

    # Example 3: Assess an English to German translation with Claude
    print("=" * 80)
    print("Example 3: English to German (with intentional errors) - Claude 4.5 Sonnet")
    print("=" * 80)

    judge_3 = TranslationJudge(
        client=claude_client,
        model="claude-sonnet-4-5-20250929",
        source_language="English",
        target_language="German",
    )
    
    result_3 = judge_3.assess_translation(
        source_text=source_text,
        translated_text=translated_text,
    )

    print("\nResult 3")
    print(f"\nErrors found: {len(result_3.errors)}")
    for i, error in enumerate(result_3.errors, 1):
        print(f"\n{i}. {error.error}")
        print(f"   Classification: {error.classification}")
        print(f"   Severity: {error.severity}")
        print(f"   Snippet: '{error.snippet}'")

    # Example 4: Batch assess multiple translations concurrently
    print("\n" + "=" * 80)
    print("Example 4: Batch assess multiple translations - GPT-5.1")
    print("=" * 80)
    
    translation_pairs = [
        ("The quick brown fox jumps over the lazy dog.", "flinke braune Fuchs springt unter den faulen Hund."),
        ("Hello, how are you?", "Hallo, wie geht es dir?"),
        ("I love programming.", "Ich liebe Programmierung."),
        ("The weather is nice today.", "Das Wetter ist heute schön."),
        ("Can you help me?", "Kannst du mir helfen?"),
    ]
    
    results = judge_1.batch_assess_translations(
        translation_pairs=translation_pairs,
        output_csv_path="translation_results.csv",
        batch_size=2,
        max_workers=3
    )
    
    print(f"\nBatch assessment complete. Processed {len(results)} translation pairs.")