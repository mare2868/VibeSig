import pandas as pd

from src.pipeline import run_feedback_pipeline


class FakeClassifier:
    """Fake sentiment classifier for pipeline testing."""

    def __call__(self, texts, **kwargs):
        predictions = []

        for text in texts:
            text_lower = text.lower()

            if "amazing" in text_lower:
                predictions.append(
                    {"label": "Positive", "score": 0.90}
                )
            elif "terrible" in text_lower:
                predictions.append(
                    {"label": "Negative", "score": 0.80}
                )
            else:
                predictions.append(
                    {"label": "Neutral", "score": 0.70}
                )

        return predictions


def test_run_feedback_pipeline():
    classifier = FakeClassifier()

    df = pd.DataFrame(
        {
            "review": [
                "The food was amazing.",
                "The service was terrible.",
                "The experience was acceptable.",
                "",
                None,
            ]
        }
    )

    result = run_feedback_pipeline(
        classifier,
        df,
        "review",
    )

    results_df = result["results"]
    summary = result["summary"]

    assert len(results_df) == 3

    # Raw predictions must preserve the original
    # classifier output.
    assert list(results_df["raw_sentiment"]) == [
        "Positive",
        "Negative",
        "Neutral",
    ]

    # VibeSig sentiment uses the calibration layer.
    # Positive at 0.90 is recalibrated to Neutral
    # because the validated threshold is 0.97.
    assert list(results_df["sentiment"]) == [
        "Neutral",
        "Negative",
        "Neutral",
    ]

    assert list(results_df["confidence"]) == [
        0.90,
        0.80,
        0.70,
    ]

    assert summary["total_reviews"] == 3

    assert summary["positive"] == 0
    assert summary["negative"] == 1
    assert summary["neutral"] == 2

    assert round(summary["positive_pct"], 2) == 0.00
    assert round(summary["neutral_pct"], 2) == 66.67
    assert round(summary["negative_pct"], 2) == 33.33

    assert round(summary["vibe_score"], 2) == 33.33
    assert summary["average_confidence"] == 80.0