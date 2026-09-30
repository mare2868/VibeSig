from src.evaluation import evaluate_sentiment_predictions


def test_perfect_predictions():
    actual = [
        "Negative",
        "Neutral",
        "Positive",
    ]

    predicted = [
        "Negative",
        "Neutral",
        "Positive",
    ]

    result = evaluate_sentiment_predictions(
        actual,
        predicted,
    )

    assert result["accuracy"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["total_samples"] == 3


def test_confusion_matrix_shape_and_labels():
    actual = [
        "Negative",
        "Neutral",
        "Positive",
    ]

    predicted = [
        "Negative",
        "Positive",
        "Positive",
    ]

    result = evaluate_sentiment_predictions(
        actual,
        predicted,
    )

    assert result["labels"] == [
        "Negative",
        "Neutral",
        "Positive",
    ]

    assert len(result["confusion_matrix"]) == 3

    assert all(
        len(row) == 3
        for row in result["confusion_matrix"]
    )


def test_per_class_metrics_are_returned():
    actual = [
        "Negative",
        "Neutral",
        "Positive",
        "Positive",
    ]

    predicted = [
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
    ]

    result = evaluate_sentiment_predictions(
        actual,
        predicted,
    )

    assert "Negative" in result["per_class"]
    assert "Neutral" in result["per_class"]
    assert "Positive" in result["per_class"]

    assert (
        result["per_class"]["Positive"]["support"]
        == 2
    )


def test_handles_missing_predicted_class():
    actual = [
        "Negative",
        "Neutral",
        "Positive",
    ]

    predicted = [
        "Negative",
        "Negative",
        "Negative",
    ]

    result = evaluate_sentiment_predictions(
        actual,
        predicted,
    )

    assert result["accuracy"] == 1 / 3

    assert (
        result["per_class"]["Positive"]["precision"]
        == 0.0
    )


def test_rejects_empty_inputs():
    try:
        evaluate_sentiment_predictions([], [])
        assert False

    except ValueError as error:
        assert "cannot be empty" in str(error)


def test_rejects_different_lengths():
    actual = [
        "Negative",
        "Positive",
    ]

    predicted = [
        "Negative",
    ]

    try:
        evaluate_sentiment_predictions(
            actual,
            predicted,
        )

        assert False

    except ValueError as error:
        assert "same length" in str(error)