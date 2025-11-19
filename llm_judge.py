import os
from typing import List, Union
from pydantic import BaseModel, Field
from prompts import SYSTEM_PROMPT, USER_PROMPT, SEVERITY_PROMPT_GEMMA
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

ClientType = Union[OpenAI, Anthropic]

class TranslationError(BaseModel):
    """Represents a single translation error."""
    error: str = Field(description="Description of the error")
    classification: str = Field(description="Error category (e.g., accuracy, fluency, style, etc.)")
    severity: str = Field(description="Error severity: critical, major, or minor")
    snippet: str = Field(description="The snippet from the translation containing the error")


class TranslationAssessment(BaseModel):
    """Contains a list of translation errors found during assessment."""
    errors: List[TranslationError] = Field(description="List of translation errors identified")


def assess_translation(
    source_text: str,
    source_language: str,
    translated_text: str,
    translated_language: str,
    client: ClientType,
    model: str, # OpenAI or Anthropic client
    temperature: float = 0.0,
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
    user_message = USER_PROMPT.format(
        source_language=source_language,
        source_segment=source_text,
        target_language=translated_language,
        target_segment=translated_text
    ) + SEVERITY_PROMPT_GEMMA
    
    # Detect client type and make appropriate API call
    client_type = type(client).__module__
    
    if 'openai' in client_type.lower():
        return _assess_with_openai(client, user_message, model, temperature)
    elif 'anthropic' in client_type.lower():
        return _assess_with_anthropic(client, user_message, model, temperature)
    else:
        raise ValueError(f"Unsupported client type: {client_type}. Must be OpenAI or Anthropic client.")


def _assess_with_openai(client, user_message: str, model: str, temperature: float = 0.0) -> TranslationAssessment:
    """Handle OpenAI API calls with structured outputs."""
    response = client.beta.chat.completions.parse(
        model= model,  # Model that supports structured outputs
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_format=TranslationAssessment,
        temperature=temperature,
    )
    
    return response.choices[0].message.parsed


def _assess_with_anthropic(client, user_message: str, model: str, temperature: float = 0.0) -> TranslationAssessment:
    """Handle Anthropic API calls with structured outputs."""
    response = client.messages.create(
        model=model,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
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


if __name__ == "__main__":

    # Load environment variables
    config = load_dotenv(".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")


    # Example 1: Assess an English to Spanish translation with OpenAI
    print("=" * 80)
    print("Example 1: English to German (with intentional errors)")
    print("=" * 80)
    
    # Initialize clients
    openai_client = OpenAI(api_key=openai_api_key)
    groq_client = OpenAI(base_url = "https://api.groq.com/openai/v1", api_key=groq_api_key)
    
    source_text = "The quick brown fox jumps over the lazy dog."
    translated_text= "flinke braune Fuchs springt unter den faulen Hund."
    
    result_1 = assess_translation(
        source_text=source_text,
        source_language="English",
        translated_text=translated_text,
        translated_language="German",
        client=openai_client,
        model="gpt-4o",
        temperature=0.0
    )
    
    print("\nResult 1")
    print(f"\nErrors found: {len(result_1.errors)}")
    for i, error in enumerate(result_1.errors, 1):
        print(f"\n{i}. {error.error}")
        print(f"   Classification: {error.classification}")
        print(f"   Severity: {error.severity}")
        print(f"   Snippet: '{error.snippet}'")
    
    result_2 = assess_translation(
        source_text=source_text,
        source_language="English",
        translated_text=translated_text,
        translated_language="German",
        client=groq_client,
        model="qwen/qwen3-32b",
        temperature=0.0
    )
    
    print("\nResult 2")
    print(f"\nErrors found: {len(result_2.errors)}")
    for i, error in enumerate(result_2.errors, 1):
        print(f"\n{i}. {error.error}")
        print(f"   Classification: {error.classification}")
        print(f"   Severity: {error.severity}")
        print(f"   Snippet: '{error.snippet}'")
    