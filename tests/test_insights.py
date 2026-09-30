import pytest

from src.insights import generate_vibe_insight


def test_positive_vibe_insight():
    summary = {
        "vibe_score": 71.0,
        "positive_pct": 62.0,
        "neutral_pct": 18.0,
        "negative_pct": 20.0,
    }

    insight = generate_vibe_insight(summary)

    assert insight["overall_sentiment"] == "Positive"
    assert insight["sentiment_gap"] == 42.0

    assert (
        "Positive feedback exceeds negative feedback"
        in insight["key_signal"]
    )

    assert (
        "meaningful share of negative feedback"
        in insight["attention"]
    )


def test_strongly_positive_vibe_insight():
    summary = {
        "vibe_score": 85.0,
        "positive_pct": 80.0,
        "neutral_pct": 10.0,
        "negative_pct": 10.0,
    }

    insight = generate_vibe_insight(summary)

    assert insight["overall_sentiment"] == "Strongly positive"
    assert insight["sentiment_gap"] == 70.0

    assert (
        "relatively limited"
        in insight["attention"]
    )


def test_mixed_vibe_insight():
    summary = {
        "vibe_score": 50.0,
        "positive_pct": 40.0,
        "neutral_pct": 20.0,
        "negative_pct": 40.0,
    }

    insight = generate_vibe_insight(summary)

    assert insight["overall_sentiment"] == "Mixed"
    assert insight["sentiment_gap"] == 0.0

    assert (
        "relatively close"
        in insight["key_signal"]
    )


def test_negative_vibe_insight():
    summary = {
        "vibe_score": 30.0,
        "positive_pct": 20.0,
        "neutral_pct": 20.0,
        "negative_pct": 60.0,
    }

    insight = generate_vibe_insight(summary)

    assert insight["overall_sentiment"] == "Negative"
    assert insight["sentiment_gap"] == -40.0

    assert (
        "Negative feedback exceeds positive feedback"
        in insight["key_signal"]
    )

    assert (
        "significant share"
        in insight["attention"]
    )


def test_empty_summary_raises_error():
    with pytest.raises(
        ValueError,
        match="Summary cannot be empty",
    ):
        generate_vibe_insight({})


def test_missing_required_field_raises_error():
    summary = {
        "vibe_score": 71.0,
        "positive_pct": 62.0,
        "negative_pct": 20.0,
    }

    with pytest.raises(
        ValueError,
        match="Missing required summary field",
    ):
        generate_vibe_insight(summary)