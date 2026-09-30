import pandas as pd


def prepare_reviews_dataframe(df, text_column):
    """Validate and prepare a dataframe containing customer reviews."""

    if df.empty:
        raise ValueError("CSV file cannot be empty.")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found in the CSV file."
        )

    clean_df = df.copy()

    clean_df = clean_df[
        clean_df[text_column].notna()
    ]

    clean_df[text_column] = (
        clean_df[text_column]
        .astype(str)
        .str.strip()
    )

    clean_df = clean_df[
        clean_df[text_column] != ""
    ]

    if clean_df.empty:
        raise ValueError(
            "The selected column does not contain valid reviews."
        )

    clean_df = clean_df.reset_index(drop=True)

    return clean_df