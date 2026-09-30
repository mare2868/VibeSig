from src.recommendations import generate_action_recommendations


def test_returns_empty_list_when_no_drivers():
    recommendations = generate_action_recommendations([])

    assert recommendations == []


def test_generates_recommendations_from_prioritized_drivers():
    prioritized_drivers = [
        {
            "driver": "Food / Product",
            "negative_review_pct": 45.0,
            "priority": "High",
        },
        {
            "driver": "Quality",
            "negative_review_pct": 40.0,
            "priority": "High",
        },
    ]

    recommendations = generate_action_recommendations(
        prioritized_drivers
    )

    assert len(recommendations) == 2

    assert recommendations[0]["driver"] == "Food / Product"
    assert recommendations[0]["priority"] == "High"
    assert recommendations[0]["negative_review_pct"] == 45.0

    assert recommendations[1]["driver"] == "Quality"


def test_limits_number_of_recommendations():
    prioritized_drivers = [
        {
            "driver": "Food / Product",
            "negative_review_pct": 45.0,
            "priority": "High",
        },
        {
            "driver": "Quality",
            "negative_review_pct": 45.0,
            "priority": "High",
        },
        {
            "driver": "Staff",
            "negative_review_pct": 40.0,
            "priority": "High",
        },
        {
            "driver": "Wait Time",
            "negative_review_pct": 35.0,
            "priority": "Medium",
        },
    ]

    recommendations = generate_action_recommendations(
        prioritized_drivers,
        max_recommendations=3,
    )

    assert len(recommendations) == 3

    assert [
        item["driver"]
        for item in recommendations
    ] == [
        "Food / Product",
        "Quality",
        "Staff",
    ]


def test_skips_other_driver():
    prioritized_drivers = [
        {
            "driver": "Other",
            "negative_review_pct": 50.0,
            "priority": "High",
        },
        {
            "driver": "Service",
            "negative_review_pct": 30.0,
            "priority": "Medium",
        },
    ]

    recommendations = generate_action_recommendations(
        prioritized_drivers
    )

    assert len(recommendations) == 1
    assert recommendations[0]["driver"] == "Service"


def test_preserves_priority_and_percentage():
    prioritized_drivers = [
        {
            "driver": "Wait Time",
            "negative_review_pct": 35.0,
            "priority": "Medium",
        }
    ]

    recommendations = generate_action_recommendations(
        prioritized_drivers
    )

    result = recommendations[0]

    assert result["priority"] == "Medium"
    assert result["negative_review_pct"] == 35.0


def test_recommendation_contains_action_text():
    prioritized_drivers = [
        {
            "driver": "Staff",
            "negative_review_pct": 40.0,
            "priority": "High",
        }
    ]

    recommendations = generate_action_recommendations(
        prioritized_drivers
    )

    recommendation = recommendations[0]["recommendation"]

    assert isinstance(recommendation, str)
    assert len(recommendation) > 20