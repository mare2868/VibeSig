from src.calibration import (
    POSITIVE_TO_NEUTRAL_THRESHOLD,
    calibrate_sentiment_prediction,
)


def test_threshold_is_expected_value():
    assert POSITIVE_TO_NEUTRAL_THRESHOLD == 0.97


def test_high_confidence_positive_remains_positive():
    result = calibrate_sentiment_prediction(
        "Positive",
        0.99,
    )

    assert result == "Positive"


def test_low_confidence_positive_becomes_neutral():
    result = calibrate_sentiment_prediction(
        "Positive",
        0.96,
    )

    assert result == "Neutral"


def test_positive_at_threshold_remains_positive():
    result = calibrate_sentiment_prediction(
        "Positive",
        0.97,
    )

    assert result == "Positive"


def test_neutral_is_not_changed():
    result = calibrate_sentiment_prediction(
        "Neutral",
        0.80,
    )

    assert result == "Neutral"


def test_negative_is_not_changed():
    result = calibrate_sentiment_prediction(
        "Negative",
        0.70,
    )

    assert result == "Negative"


def test_custom_threshold_can_be_used():
    result = calibrate_sentiment_prediction(
        "Positive",
        0.94,
        positive_threshold=0.95,
    )

    assert result == "Neutral"