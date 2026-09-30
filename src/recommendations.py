"""Rule-based action recommendations for VibeSig."""

DRIVER_RECOMMENDATIONS = {
    "Service": (
        "Review recurring service issues and identify opportunities "
        "to improve the customer experience."
    ),
    "Food / Product": (
        "Review product consistency and recurring product-related "
        "complaints."
    ),
    "Price / Value": (
        "Examine whether customer expectations align with pricing "
        "and perceived value."
    ),
    "Wait Time": (
        "Examine customer wait-time patterns and potential "
        "operational bottlenecks."
    ),
    "Staff": (
        "Review staff-related complaints and identify recurring "
        "customer service behaviors."
    ),
    "Quality": (
        "Investigate recurring quality issues across negative "
        "customer feedback."
    ),
    "Other": (
        "Review uncategorized negative feedback to identify emerging "
        "themes not covered by existing drivers."
    ),
}


def generate_action_recommendations(
    prioritized_drivers,
    max_recommendations=3,
):
    """
    Generate actionable recommendations from prioritized
    negative-feedback drivers.

    Recommendations are ordered according to the existing
    driver prioritization and limited to the most relevant signals.
    """

    if not prioritized_drivers:
        return []

    recommendations = []

    for item in prioritized_drivers:
        driver = item["driver"]

        if driver == "Other":
            continue

        recommendation = DRIVER_RECOMMENDATIONS.get(driver)

        if recommendation is None:
            continue

        recommendations.append(
            {
                "driver": driver,
                "priority": item["priority"],
                "negative_review_pct": item[
                    "negative_review_pct"
                ],
                "recommendation": recommendation,
            }
        )

        if len(recommendations) >= max_recommendations:
            break

    return recommendations