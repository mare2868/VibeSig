import pandas as pd

from src.drivers import (
    detect_feedback_drivers,
    summarize_negative_drivers,
)


def test_detect_multiple_feedback_drivers():
    text = (
        "The food was overpriced and "
        "the service was extremely slow."
    )

    drivers = detect_feedback_drivers(text)

    assert drivers == [
        "Service",
        "Food / Product",
        "Price / Value",
        "Wait Time",
    ]


def test_detect_staff_and_cleanliness():
    text = (
        "The waiter was rude and "
        "the bathroom was dirty."
    )

    drivers = detect_feedback_drivers(text)

    assert "Service" in drivers
    assert "Staff" in drivers
    assert "Cleanliness" in drivers


def test_unknown_feedback_returns_other():
    drivers = detect_feedback_drivers(
        "I simply did not enjoy the experience."
    )

    assert drivers == ["Other"]


def test_empty_feedback_returns_other():
    assert detect_feedback_drivers("") == ["Other"]
    assert detect_feedback_drivers(None) == ["Other"]


def test_summarize_negative_drivers():
    results = [
        {
            "text": "The service was slow.",
            "sentiment": "Negative",
            "confidence": 0.95,
        },
        {
            "text": "The food was overpriced.",
            "sentiment": "Negative",
            "confidence": 0.92,
        },
        {
            "text": "Great food and excellent service.",
            "sentiment": "Positive",
            "confidence": 0.98,
        },
    ]

    summary = summarize_negative_drivers(results)

    assert summary["negative_reviews"] == 2

    drivers = {
        item["driver"]: item["mentions"]
        for item in summary["drivers"]
    }

    assert drivers["Service"] == 1
    assert drivers["Wait Time"] == 1
    assert drivers["Food / Product"] == 1
    assert drivers["Price / Value"] == 1

    
    driver_percentages = {
        item["driver"]: item["negative_review_pct"]
        for item in summary["drivers"]
    }

    assert driver_percentages["Service"] == 50.0
    assert driver_percentages["Wait Time"] == 50.0
    assert driver_percentages["Food / Product"] == 50.0
    assert driver_percentages["Price / Value"] == 50.0


def test_summary_ignores_positive_reviews():
    results = [
        {
            "text": "The service was excellent.",
            "sentiment": "Positive",
            "confidence": 0.99,
        }
    ]

    summary = summarize_negative_drivers(results)

    assert summary["negative_reviews"] == 0
    assert summary["total_driver_mentions"] == 0
    assert summary["drivers"] == []


def test_empty_results_raises_error():
    try:
        summarize_negative_drivers([])

    except ValueError as error:
        assert str(error) == "Results cannot be empty."

    else:
        raise AssertionError(
            "Expected ValueError for empty results."
        )

def test_summary_accepts_dataframe():
    results = pd.DataFrame(
        [
            {
                "text": "The service was slow.",
                "sentiment": "Negative",
                "confidence": 0.95,
            },
            {
                "text": "The food was excellent.",
                "sentiment": "Positive",
                "confidence": 0.98,
            },
        ]
    )

    summary = summarize_negative_drivers(results)

    assert summary["negative_reviews"] == 1
    assert summary["total_driver_mentions"] == 2

    drivers = {
        item["driver"]: item["mentions"]
        for item in summary["drivers"]
    }

    assert drivers["Service"] == 1
    assert drivers["Wait Time"] == 1
