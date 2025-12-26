import torch
from transformers import LogitsProcessor


class MinPLogitsProcessor(LogitsProcessor):
    """
    Implements min_p sampling from llama-cpp.

    Filters out tokens with probability less than min_p * max_probability.
    This helps reduce low-quality samples by removing unlikely tokens.
    """

    def __init__(self, min_p: float = 0.05, filter_value: float = -float("Inf")):
        """
        Args:
            min_p: Minimum probability threshold relative to max probability.
                   Tokens with prob < min_p * max_prob are filtered out.
            filter_value: Value to assign to filtered tokens (default -inf).
        """
        if not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be between 0 and 1, got {min_p}")
        self.min_p = min_p
        self.filter_value = filter_value

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.min_p <= 0.0:
            return scores

        # Apply softmax to get probabilities
        probs = torch.softmax(scores, dim=-1)

        # Get max probability for each batch
        max_probs = probs.max(dim=-1, keepdim=True).values

        # Calculate threshold: tokens below min_p * max_prob are filtered
        threshold = max_probs * self.min_p

        # Filter tokens below threshold
        indices_to_remove = probs < threshold
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores
