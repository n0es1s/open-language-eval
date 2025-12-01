from typing import Callable, Optional

import jiwer


class TranscriptionWER:
    """A class for calculating Word Error Rate (WER) for transcription evaluation.

    This class provides methods to compute WER between reference and predicted
    transcriptions, with optional text normalization and transformation capabilities.
    """

    def __init__(
        self,
        transforms: Optional[Callable] = None,
        normalizer: Optional[Callable] = None,
    ):
        """Initialize the TranscriptionWER evaluator.

        Args:
            transforms: Optional callable to transform both reference and hypothesis
                text before WER calculation. Common transforms include removing
                punctuation, lowercasing, etc.
            normalizer: Optional callable to normalize English text before WER calculation. This helps deal with differences in British and American English.
        """
        self.transforms = transforms
        self.normalizer = normalizer

    def calculate_wer(self, reference: str, prediction: str) -> float:
        """Calculate Word Error Rate (WER) between reference and prediction texts.

        WER is computed as (S + D + I) / N, where S is the number of substitutions,
        D is the number of deletions, I is the number of insertions, and N is the
        number of words in the reference.

        Args:
            reference: The ground truth transcription text.
            prediction: The predicted/hypothesis transcription text.

        Returns:
            The Word Error Rate as a float. A value of 0.0 indicates perfect
            transcription, while values > 1.0 indicate the hypothesis has more
            errors than the number of words in the reference.
        """
        if self.normalizer is not None:
            reference = self.normalizer(reference)
            prediction = self.normalizer(prediction)

        wer = jiwer.wer(
            reference,
            prediction,
            reference_transform=self.transforms,
            hypothesis_transform=self.transforms,
        )

        return wer
