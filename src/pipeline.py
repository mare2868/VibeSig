import pandas as pd

from src.batch import analyze_reviews, calculate_summary
from src.csv_processor import prepare_reviews_dataframe


def run_feedback_pipeline(
    classifier,
    df,
    text_column,
    batch_size=16,
):
    """Run the complete VibeSig customer feedback analysis pipeline."""

    clean_df = prepare_reviews_dataframe(
        df,
        text_column,
    )

    reviews = clean_df[text_column].tolist()

    predictions = analyze_reviews(
        classifier,
        reviews,
        batch_size=batch_size,
    )

    results_df = pd.DataFrame(predictions)

    summary = calculate_summary(predictions)

    return {
        "results": results_df,
        "summary": summary,
    }