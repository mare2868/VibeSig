import pytest

from src.batch import analyze_reviews


class FakeClassifier:
    """Simple classifier used to test batch processing without loading the AI model."""

    def __call__(self, texts, **kwargs):
        predictions = []

        for text in texts:
            if "amazing" in text.lower():
                predictions.append(
                    {"label": "Positive", "score": 0.99}
                )
            else:
                predictions.append(
                    {"label": "Negative", "score": 0.95}
                )

        return predictions


def test_analyze_reviews_returns_predictions():
    classifier = FakeClassifier()

    reviews = [
        "The food was amazing.",
        "The service was terrible.",
    ]

    results = analyze_reviews(
        classifier,
        reviews,
    )

    assert len(results) == 2

    assert results[0]["text"] == "The food was amazing."
    assert results[0]["sentiment"] == "Positive"
    assert results[0]["confidence"] == 0.99

    assert results[1]["sentiment"] == "Negative"
    assert results[1]["confidence"] == 0.95


def test_analyze_reviews_removes_empty_reviews():
    classifier = FakeClassifier()

    reviews = [
        "The food was amazing.",
        "",
        "   ",
        None,
        "The service was terrible.",
    ]

    results = analyze_reviews(
        classifier,
        reviews,
    )

    assert len(results) == 2


def test_empty_reviews_list_raises_error():
    classifier = FakeClassifier()

    with pytest.raises(
        ValueError,
        match="Reviews list cannot be empty.",
    ):
        analyze_reviews(classifier, [])


def test_reviews_without_valid_text_raise_error():
    classifier = FakeClassifier()

    with pytest.raises(
        ValueError,
        match="Reviews must contain valid text.",
    ):
        analyze_reviews(
            classifier,
            ["", "   ", None],
        )