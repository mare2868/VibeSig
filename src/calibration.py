"""Post-inference sentiment calibration for VibeSig."""


POSITIVE_TO_NEUTRAL_THRESHOLD = 0.97


def calibrate_sentiment_prediction(
    sentiment,
    confidence,
    positive_threshold=POSITIVE_TO_NEUTRAL_THRESHOLD,
):
    """
    Calibrate a sentiment prediction using the validated
    Positive-to-Neutral confidence threshold.

    Positive predictions below the threshold are reclassified
    as Neutral. Negative and Neutral predictions are unchanged.
    """

    if (
        sentiment == "Positive"
        and confidence < positive_threshold
    ):
        return "Neutral"

    return sentiment