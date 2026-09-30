import pandas as pd

from src.sentiment import load_sentiment_model
from src.pipeline import run_feedback_pipeline
from src.evaluation import evaluate_sentiment_predictions


THRESHOLDS = [
    None,
    0.97,
]


df = pd.read_csv(
    "data/yelp_vibesig_sample.csv"
)

sample = pd.concat(
    [
        df[df["actual_sentiment"] == label].sample(
            n=500,
            random_state=2026,
        )
        for label in [
            "Negative",
            "Neutral",
            "Positive",
        ]
    ],
    ignore_index=True,
)

model = load_sentiment_model()

result = run_feedback_pipeline(
    model,
    sample,
    "text",
)

predictions = result["results"].copy()

rows = []

for threshold in THRESHOLDS:

    adjusted = predictions["sentiment"].copy()

    changed = 0

    if threshold is not None:
        mask = (
            (predictions["sentiment"] == "Positive")
            & (predictions["confidence"] < threshold)
        )

        adjusted.loc[mask] = "Neutral"

        changed = int(mask.sum())

    evaluation = evaluate_sentiment_predictions(
        sample["actual_sentiment"],
        adjusted,
    )

    rows.append(
        {
            "threshold": (
                "baseline"
                if threshold is None
                else threshold
            ),
            "changed_predictions": changed,
            "accuracy": evaluation["accuracy"],
            "macro_precision": evaluation[
                "macro_precision"
            ],
            "macro_recall": evaluation[
                "macro_recall"
            ],
            "macro_f1": evaluation["macro_f1"],
            "neutral_recall": evaluation[
                "per_class"
            ]["Neutral"]["recall"],
            "positive_recall": evaluation[
                "per_class"
            ]["Positive"]["recall"],
        }
    )


comparison = pd.DataFrame(rows)

print()
print("=" * 90)
print("POSITIVE -> NEUTRAL THRESHOLD EXPERIMENT")
print("=" * 90)

print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)