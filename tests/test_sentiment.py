import pytest

from src.sentiment import predict_sentiment


def test_empty_review_raises_error():
    """Empty reviews should not be accepted by VibeSig."""

    with pytest.raises(ValueError, match="Review text cannot be empty."):
        predict_sentiment(None, "   ")