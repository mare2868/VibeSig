import re
from collections import Counter


DRIVER_KEYWORDS = {
    "Service": [
        "service",
        "customer service",
        "rude",
        "ignored",
        "unhelpful",
    ],
    "Staff": [
        "staff",
        "waiter",
        "waitress",
        "server",
        "employee",
        "manager",
    ],
    "Food / Product": [
        "food",
        "meal",
        "dish",
        "product",
        "order",
    ],
    "Price / Value": [
        "price",
        "prices",
        "expensive",
        "overpriced",
        "value",
        "cost",
    ],
    "Wait Time": [
        "wait",
        "waited",
        "waiting",
        "slow",
        "delay",
        "delayed",
        "minutes",
    ],
    "Quality": [
        "quality",
        "poor",
        "awful",
        "terrible",
        "horrible",
        "disappointing",
    ],
    "Cleanliness": [
        "dirty",
        "clean",
        "cleanliness",
        "unclean",
        "bathroom",
        "restroom",
    ],
}


def _contains_keyword(text, keyword):
    """Check whether a keyword appears as a complete term."""

    pattern = rf"\b{re.escape(keyword)}\b"

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def detect_feedback_drivers(text):
    """Detect one or more feedback drivers in a review."""

    if not isinstance(text, str) or not text.strip():
        return ["Other"]

    detected_drivers = []

    for driver, keywords in DRIVER_KEYWORDS.items():

        if any(
            _contains_keyword(text, keyword)
            for keyword in keywords
        ):
            detected_drivers.append(driver)

    if not detected_drivers:
        return ["Other"]

    return detected_drivers


def summarize_negative_drivers(results):
    """Summarize driver mentions across negative reviews."""

    if results is None or len(results) == 0:
        raise ValueError("Results cannot be empty.")

    if hasattr(results, "to_dict"):
        results = results.to_dict(
            orient="records"
        )

    negative_reviews = [
        result
        for result in results
        if result.get("sentiment") == "Negative"
    ]

    if not negative_reviews:
        return {
            "negative_reviews": 0,
            "total_driver_mentions": 0,
            "drivers": [],
        }

    driver_counts = Counter()

    for result in negative_reviews:

        drivers = detect_feedback_drivers(
            result.get("text", "")
        )

        driver_counts.update(drivers)

    total_mentions = sum(driver_counts.values())

    driver_summary = []

    for driver, count in driver_counts.most_common():

        percentage = (
            count / total_mentions * 100
            if total_mentions > 0
            else 0
        )

        negative_review_pct = (
            count / len(negative_reviews) * 100
            if negative_reviews
            else 0
        )

        driver_summary.append(
            {
                "driver": driver,
                "mentions": count,
                "percentage": percentage,
                "negative_review_pct": negative_review_pct,
            }
        )

    return {
        "negative_reviews": len(negative_reviews),
        "total_driver_mentions": total_mentions,
        "drivers": driver_summary,
    }

def prioritize_negative_drivers(driver_summary):
    """Assign business attention levels to negative feedback drivers."""

    if not driver_summary:
        raise ValueError("Driver summary cannot be empty.")

    drivers = driver_summary.get("drivers", [])

    priorities = []

    for driver in drivers:
        percentage = driver["negative_review_pct"]

        if percentage >= 40:
            priority = "High"
        elif percentage >= 20:
            priority = "Medium"
        else:
            priority = "Low"

        priorities.append(
            {
                **driver,
                "priority": priority,
            }
        )

    return priorities