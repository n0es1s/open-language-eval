import argparse
import io
import json
from datetime import datetime
import itertools
import soundfile as sf
from datasets import load_dataset
import sounddevice as sd
from librosa import resample
from evaluate import load
from whisper_normalizer.english import EnglishTextNormalizer
from open_language_eval.inference_clients.transcription_client import (
    AudioTranscriptionInterface,
)
from whisper_normalizer.basic import BasicTextNormalizer
from inference_client_openai_transcription import AudioTranscriptionOpenAI
from inference_client_groq_transcription import AudioTranscriptionGroq

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Test audio transcription models with VoxPopuli dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--language",
    "-l",
    type=str,
    default="en",
    choices=["en", "es", "de", "fr", "pl", "it", "ro", "hu", "cs", "nl", "fi", "hr", "sk"],
    help="Language code for the dataset"
)
parser.add_argument(
    "--num-samples", "-n", type=int, default=5, help="Number of audio samples to test"
)
parser.add_argument(
    "--play-audio", "-p", action="store_true", help="Play audio samples during testing"
)
parser.add_argument(
    "--provider",
    type=str,
    choices=["openai", "groq"],
    default="groq",
    help="API provider to use",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    help="Model to use (if not specified, uses provider default)",
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="transcription_results.json",
    help="Output JSON file path",
)
parser.add_argument(
    "--noise-reduction",
    type=str,
    choices=["near_field", "far_field"],
    default=None,
    help="which noise reduction method to use"
)

args = parser.parse_args()

TARGET_SAMPLING_RATE = 24000 # target sampling rate given by openai, they only support 24 kHz for now

wer_metric = load("wer")
if args.language == "en":
    normalizer = EnglishTextNormalizer()
else:
    normalizer = BasicTextNormalizer()

# Load VoxPopuli dataset with specified language
print(f"Loading VoxPopuli {args.language.upper()} dataset...")
dataset = load_dataset(
    "facebook/voxpopuli",
    args.language,
    split="test",
    streaming=True,
    trust_remote_code=True,
)

# Initialize transcription service
print(f"Initializing {args.provider} transcription service...")
if args.provider == "openai":
    transcription_service = AudioTranscriptionOpenAI(
        model=args.model,
        source_language=args.language,
        noise_reduction=args.noise_reduction
    )
elif args.provider == "groq":
    transcription_service = AudioTranscriptionGroq(
        model=args.model,
        source_language=args.language,
        noise_reduction=args.noise_reduction
    )
else:
    raise ValueError(f"Unknown provider: {args.provider}")

# Process multiple samples
num_samples = args.num_samples
wer_scores = []
results = {
    "metadata": {
        "provider": args.provider,
        "model": transcription_service.model,
        "source_language": args.language,
        "dataset": f"facebook/voxpopuli ({args.language})",
        "num_samples": num_samples,
        "play_audio": args.play_audio,
        "timestamp": datetime.now().isoformat(),
    },
    "samples": [],
}
if args.provider == "openai":
    results["metadata"]["session_config"] = transcription_service.get_session_config()

output_file = args.output
print(f"Results will be saved to: {output_file}")
print(f"Audio playback: {'enabled' if args.play_audio else 'disabled'}")
print()

for i, sample in enumerate(dataset):
    if i >= num_samples:
        break

    print(f"\n{'=' * 60}")
    print(f"Sample {i + 1}/{num_samples}")
    print(f"{'=' * 60}")
    print(f"Audio ID: {sample['audio_id']}")
    print(f"Expected transcription: {sample['normalized_text']}")
    print(f"Speaker: {sample['speaker_id']} ({sample['gender']})")

    # Get audio data
    audio_array = sample['audio']['array']
    sampling_rate = sample['audio']['sampling_rate']

    if sampling_rate!=TARGET_SAMPLING_RATE:
        print(f"resampling audio from {sampling_rate} to 24000")
        audio_array = resample(audio_array, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)

    audio_duration_seconds = round(len(audio_array) / TARGET_SAMPLING_RATE, 2)
    print(f"\nAudio duration: {audio_duration_seconds} seconds")

    # Play audio clip if enabled
    if args.play_audio:
        print("Playing audio clip...")
        sd.play(audio_array, TARGET_SAMPLING_RATE)
        sd.wait()  # Wait until audio finishes playing

    # Convert audio array to file-like object for transcription
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio_array, sampling_rate, format="WAV")
    audio_bytes.seek(0)
    audio_bytes.name = "audio.wav"

    # Transcribe using service
    print("\nTranscribing...")
    transcription_text = transcription_service.transcribe(audio_bytes)

    print(f"\nModel transcription: {transcription_text}")

    # Calculate Word Error Rate with normalization
    norm_reference = normalizer(sample["normalized_text"])
    norm_hypothesis = normalizer(transcription_text)
    wer_score = wer_metric.compute(
        references=[norm_reference], predictions=[norm_hypothesis]
    )
    wer_scores.append(wer_score)
    print(f"Word Error Rate (WER): {wer_score:.2%}")

    # Collect sample results
    sample_result = {
        "sample_index": i + 1,
        "audio_info": {
            "audio_id": sample['audio_id'],
            "speaker_id": sample['speaker_id'],
            "gender": sample['gender'],
            "duration_seconds": audio_duration_seconds,
            "sampling_rate": sampling_rate
        },
        "transcriptions": {
            "reference_original": sample["normalized_text"],
            "reference_normalized": norm_reference,
            "model_original": transcription_text,
            "model_normalized": norm_hypothesis,
        },
        "metrics": {"wer": round(wer_score, 4)},
    }
    results["samples"].append(sample_result)

    # Write updated results to JSON file after each sample
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

# Add summary statistics to results
results["summary"] = {
    "total_samples": len(wer_scores),
    "average_wer": round(sum(wer_scores) / len(wer_scores), 4),
    "best_wer": round(min(wer_scores), 4),
    "worst_wer": round(max(wer_scores), 4),
}

# Write final results with summary
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(
    f"Average WER across {num_samples} samples: {sum(wer_scores) / len(wer_scores):.2%}"
)
print(f"Best WER: {min(wer_scores):.2%}")
print(f"Worst WER: {max(wer_scores):.2%}")
print(f"\nFinal results saved to {output_file}")
