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

    assert list(results_df["sentiment"]) == [
        "Positive",
        "Negative",
        "Neutral",
    ]

    assert summary["total_reviews"] == 3
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert summary["neutral"] == 1

    assert round(summary["positive_pct"], 2) == 33.33
    assert round(summary["neutral_pct"], 2) == 33.33
    assert round(summary["negative_pct"], 2) == 33.33

    assert summary["vibe_score"] == 50.0
    assert summary["average_confidence"] == 80.0