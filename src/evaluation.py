"""Model evaluation utilities for VibeSig."""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


SENTIMENT_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
]


def evaluate_sentiment_predictions(
    actual_sentiments,
    predicted_sentiments,
):
    """
    Evaluate sentiment predictions against ground-truth labels.

    Returns overall accuracy, macro-averaged precision, recall,
    F1 score, per-class metrics, and the confusion matrix.
    """

    actual = list(actual_sentiments)
    predicted = list(predicted_sentiments)

    if not actual or not predicted:
        raise ValueError(
            "Actual and predicted sentiment values cannot be empty."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted sentiment values must have "
            "the same length."
        )

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=SENTIMENT_LABELS,
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=SENTIMENT_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=SENTIMENT_LABELS,
    )

    per_class = {}

    for index, label in enumerate(SENTIMENT_LABELS):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": SENTIMENT_LABELS.copy(),
        "total_samples": len(actual),
    }