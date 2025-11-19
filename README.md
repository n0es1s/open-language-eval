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

### Transcription Testing

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

### Output

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

## Components

### AudioTranscriptionService

Service class for audio transcription supporting multiple providers and models.

```python
from inference_client_transcription import AudioTranscriptionService

# Initialize service
service = AudioTranscriptionService(
    provider="groq",
    model="whisper-large-v3",
    source_language="en"
)

# Transcribe audio
transcription = service.transcribe(audio_file)
```

**Supported Models:**
- **OpenAI**: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`
- **Groq**: `whisper-large-v3`, `whisper-large-v3-turbo`

## License

MIT

