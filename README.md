# Open Language Eval

A Python-based evaluation framework for audio transcription and translation models. Provides standardized benchmarks and metrics for assessing model performance using the VoxPopuli dataset.

## Features

- **Audio Transcription Testing**: Evaluate transcription models with WER (Word Error Rate) metrics
- **Multi-Provider Support**: Works with both OpenAI and Groq APIs
- **Text Normalization**: Uses Whisper normalizer for accurate WER calculation
- **Flexible Configuration**: Command-line arguments for customization
- **JSON Output**: Comprehensive results tracking with progressive updates

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up API keys:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   export OPENAI_API_KEY="your-openai-api-key"  # Optional
   ```

## Usage

There are several scripts that can be run depending on the requirements and are listed below.

### Transcription Testing
This script can be used to try out some of the inference models, evaluate the word error rate ("WER") and listen to the source audio playbacks.

Basic usage (uses defaults: Groq, whisper-large-v3, English, 5 samples, no audio playback):
```bash
python transcription_test.py
```

#### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--language` | `-l` | Language code (en, es, de, fr, etc.) | `en` |
| `--num-samples` | `-n` | Number of samples to test | `5` |
| `--play-audio` | `-p` | Play audio during testing | `False` |
| `--provider` | | API provider (openai, groq) | `groq` |
| `--model` | `-m` | Model to use | Provider default |
| `--output` | `-o` | Output JSON file path | `transcription_results.json` |

#### Examples

Test with 10 English samples and play audio:
```bash
python transcription_test.py -n 10 -p
```

Test Spanish transcription:
```bash
python transcription_test.py -l es -n 5
```

Use OpenAI's Whisper model:
```bash
python transcription_test.py --provider openai --model whisper-1
```

Test with custom output file:
```bash
python transcription_test.py -n 20 -o results_20_samples.json
```

Combine multiple options:
```bash
python transcription_test.py -l de -n 15 -p --provider groq --model whisper-large-v3-turbo
```

#### Output

The script generates a JSON file with:
- **Metadata**: Provider, model, language, timestamp
- **Sample Results**: Audio info, original/normalized transcriptions, WER scores
- **Summary Statistics**: Average, best, and worst WER scores

Example output structure:
```json
{
  "metadata": {
    "provider": "groq",
    "model": "whisper-large-v3",
    "source_language": "en",
    "dataset": "facebook/voxpopuli (en)",
    "num_samples": 5,
    "play_audio": false,
    "timestamp": "2025-11-19T..."
  },
  "samples": [
    {
      "sample_index": 1,
      "audio_info": { ... },
      "transcriptions": {
        "reference_original": "...",
        "reference_normalized": "...",
        "model_original": "...",
        "model_normalized": "..."
      },
      "metrics": { "wer": 0.0652 }
    }
  ],
  "summary": {
    "total_samples": 5,
    "average_wer": 0.0434,
    "best_wer": 0.0123,
    "worst_wer": 0.0652
  }
}
```
### VoxPopuli Inference

This script can be used to run the speech to text models chained with translation models on the VoxPopuli dataset.

The top of the script contains the following parameters that can be set by the user:
1. Languages. The languages within the VoxPopuli dataset to use. 
E.g. `["de", "en"]`
2. Transcription models. The models to use for transcription as well as the provider. E.g. `[("whisper_large_v3", "groq")]`.
3. Models with built-in translation capabilities. Some models, such as Whisper V3 Large, have built in capabilities to translate text to English after transcribing the audio. If this functionality is to be used, the model name should be added to the list.
4. Translation models.The models to use for text translations as well as the provider. E.g. `[("qwen/qwen3-32b", "groq"), ("gpt-4o-mini", "openai")]`.
5. Input directory. The folder where the VoxPopuli data is stored. The folder should contain subfolders for each language e.g. "de", "en" etc. Within each folder, there should be a folder called "test_part_0" containing the audio and a file called asr_test.tsv containing the associated metadata. Note that these are the default names the data would be downloaded as. The directory can also be set as `None` in which case the HuggingFace dataloader will be utilised to load the data - This is not recommended as it is much slower.
6. Output directory. The folder where the outputs are to be stored.

## Components

### AudioTranscriptionInterface

Service class for audio transcription supporting multiple providers and models.

```python
from open_language_eval.inference_clients.transcription_client import AudioTranscriptionInterface 

# Initialize service
service = AudioTranscriptionService(
    provider="groq",
    model="whisper-large-v3",
    source_language="en"
)

# Transcribe audio
transcription = service.transcribe(audio_file)

# Transcribe multiple audio files concurrently
transcription = service.transcribe_batch([
  ("id_1", audio_file_1),
  ("id_2", audio_file_2),
  ...
])
```

**Supported Models:**
- **OpenAI**: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`
- **Groq**: `whisper-large-v3`, `whisper-large-v3-turbo`

### Translation Interface

Service class for text translations supporting multiple providers and models.

```python
from open_language_eval.inference_clients.translation_client import TranslationInterface

service = TranslationInterface(
  model = "qwen/qwen3-32b"
  provider = "groq",
  source_language = "English"
  target_language = "German"
)

# Translate text
service.translate("text")

# Translate multiple texts concurrently
service.batch_translate([
  ("id_1", "text_1),
  ("id_2", "text_2)
  ...
])

```

### VoxPopuli Transcription

An extension of the transcription interface that is specifically tailored to run on the VoxPopuli dataset.

```python
from open_language_eval.voxpopuli.transcription_voxpopuli import TranscriptionVoxPopuli

transcription_voxpopuli = TranscriptionVoxPopuli(
            language="de",
            output_path="file_path.json",
            provider="groq",
            model="whisper_large_v3",
            translate=False, # can be set to True for models supporting built in translations
        )

# transcription using the HuggingFace dataloader
transcription_voxpopuli.transcribe_dataset()

# transcription using pre-downloaded files (recommended)
transcription_voxpopuli.transcribe_dataset(
    file_path = "file_directory"
)

```
Note that this class inherits from the WER class (see below) and calculates the WER as it transcribes.

### VoxPopuli Translation

An extension of the translation interface that is specifically tailored to run on the transcriptions of the VoxPopuli dataset.

```python
from open_language_eval.voxpopuli.translation_voxpopuli import TranslationVoxPopuli

translation_voxpopuli = TranslationVoxPopuli(
                    model="qwen/qwen3-32b",
                    provider="groq",
                    source_language="German"
                    target_language="English",
                )

translation_voxpopuli.translate_data(
    input_file_path="file_path" # Path to transcription output file
    output_file_path="file_path.json",
)

```

### LLM as a Judge Translation Evaluation

A class to facilitate reference-free evaluations (referred to as quality evaluations or "QE" in literature) of translations.

The class is compatible with the Anthropic model client and the OpenAI client (which can be used for groq served models too).

```python
from open_language_eval.evals.translation_llm_judge import TranslationJudge

openai_client = OpenAI(api_key= <API_KEY>)

# Initialise Judge
judge = TranslationJudge(
    client=openai_client,
    model="gpt-5.1-2025-11-13",
    source_language="English",
    target_language="German",
)

# Assess single translation
result = judge_1.assess_translation(
    source_text="source_text",
    translated_text="translated_text",
)

result = batch_assess_translation(
  translation_pairs = [
    ("id_1", "source_text_1", "translated_text_1"),
    ("id_2", "source_text_2", "translated_text_2"),
    ...,

  ],
  output_path = "file_path.json"

)

```

### WER

```python
from open_language_eval.evals.transcription_evaluation import TranscriptionWER
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

# Set up basic WER evaluator
evaluator = TranscriptionWER()

evaluator.calculate_wer("reference text", "source text")

# Set up WER evaluator with text transforms
transforms = jiwer.Compose(
            [
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]

evaluator = TranscriptionWER(transforms=transforms)

evaluator.calculate_wer("reference text", "source text")

# Set up WER evaluator with normalizer

evaluator = TranscriptionWER(normalizer=EnglishTextNormalizer())

evaluator.calculate_wer("reference text", "source text")

```

## License

MIT

