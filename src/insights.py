def generate_vibe_insight(summary):
    """Generate a deterministic business insight from sentiment metrics."""

    if not summary:
        raise ValueError("Summary cannot be empty.")

    required_fields = [
        "vibe_score",
        "positive_pct",
        "neutral_pct",
        "negative_pct",
    ]

    for field in required_fields:
        if field not in summary:
            raise ValueError(
                f"Missing required summary field: {field}"
            )

    vibe_score = summary["vibe_score"]
    positive_pct = summary["positive_pct"]
    neutral_pct = summary["neutral_pct"]
    negative_pct = summary["negative_pct"]

    # Overall interpretation
    if vibe_score >= 75:
        overall_sentiment = "Strongly positive"

        interpretation = (
            "Customer sentiment is strongly positive, "
            "indicating a highly favorable overall experience."
        )

    elif vibe_score >= 60:
        overall_sentiment = "Positive"

        interpretation = (
            "Customer sentiment is generally positive, "
            "although some feedback may still require attention."
        )

    elif vibe_score >= 40:
        overall_sentiment = "Mixed"

        interpretation = (
            "Customer sentiment is mixed, with no clear "
            "dominance of positive or negative feedback."
        )

    else:
        overall_sentiment = "Negative"

        interpretation = (
            "Customer sentiment is predominantly negative, "
            "indicating potential customer experience issues."
        )

    # Positive vs. negative gap
    sentiment_gap = positive_pct - negative_pct

    if sentiment_gap >= 20:
        key_signal = (
            f"Positive feedback exceeds negative feedback "
            f"by {sentiment_gap:.1f} percentage points."
        )

    elif sentiment_gap <= -20:
        key_signal = (
            f"Negative feedback exceeds positive feedback "
            f"by {abs(sentiment_gap):.1f} percentage points."
        )

    else:
        key_signal = (
            "Positive and negative feedback are relatively "
            "close, suggesting a divided customer experience."
        )

    # Attention signal
    if negative_pct >= 30:
        attention = (
            "Negative feedback represents a significant share "
            "of the analyzed reviews and should be investigated."
        )

    elif negative_pct >= 15:
        attention = (
            "A meaningful share of negative feedback is present "
            "and may warrant further review."
        )

    else:
        attention = (
            "Negative feedback remains relatively limited "
            "within the analyzed sample."
        )

    return {
        "overall_sentiment": overall_sentiment,
        "interpretation": interpretation,
        "key_signal": key_signal,
        "attention": attention,
        "sentiment_gap": sentiment_gap,
        "positive_pct": positive_pct,
        "neutral_pct": neutral_pct,
        "negative_pct": negative_pct,
        "vibe_score": vibe_score,
    }