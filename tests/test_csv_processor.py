import pandas as pd
import pytest

from src.csv_processor import prepare_reviews_dataframe


def test_prepare_reviews_dataframe_removes_invalid_reviews():
    df = pd.DataFrame(
        {
            "review": [
                "Great food!",
                "   ",
                None,
                "Terrible service.",
            ]
        }
    )

    result = prepare_reviews_dataframe(
        df,
        "review",
    )

    assert len(result) == 2
    assert result.iloc[0]["review"] == "Great food!"
    assert result.iloc[1]["review"] == "Terrible service."


def test_missing_text_column_raises_error():
    df = pd.DataFrame(
        {
            "comment": [
                "Great food!",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Column 'review' was not found in the CSV file.",
    ):
        prepare_reviews_dataframe(
            df,
            "review",
        )


def test_column_without_valid_reviews_raises_error():
    df = pd.DataFrame(
        {
            "review": [
                "",
                "   ",
                None,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="The selected column does not contain valid reviews.",
    ):
        prepare_reviews_dataframe(
            df,
            "review",
        )