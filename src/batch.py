def analyze_reviews(classifier, reviews, batch_size=16):
    """Analyze sentiment for multiple customer reviews in batches."""

    if not reviews:
        raise ValueError("Reviews list cannot be empty.")

    clean_reviews = []

    for review in reviews:
        if review and review.strip():
            clean_reviews.append(review.strip())

    if not clean_reviews:
        raise ValueError("Reviews must contain valid text.")

    predictions = classifier(
        clean_reviews,
        truncation=True,
        max_length=512,
        batch_size=batch_size,
    )

    results = []

    for review, prediction in zip(clean_reviews, predictions):
        results.append(
            {
                "text": review,
                "sentiment": prediction["label"],
                "confidence": prediction["score"],
            }
        )

    return results

def calculate_summary(results):
    """Calculate aggregate sentiment metrics for analyzed reviews."""

    if not results:
        raise ValueError("Results cannot be empty.")

    total = len(results)

    positive = sum(
        1 for result in results
        if result["sentiment"] == "Positive"
    )

    neutral = sum(
        1 for result in results
        if result["sentiment"] == "Neutral"
    )

    negative = sum(
        1 for result in results
        if result["sentiment"] == "Negative"
    )

    positive_pct = (positive / total) * 100
    neutral_pct = (neutral / total) * 100
    negative_pct = (negative / total) * 100

    vibe_score = (
        (positive + 0.5 * neutral) / total
    ) * 100

    average_confidence = (
        sum(result["confidence"] for result in results)
        / total
    ) * 100

    return {
        "total_reviews": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_pct": positive_pct,
        "neutral_pct": neutral_pct,
        "negative_pct": negative_pct,
        "vibe_score": vibe_score,
        "average_confidence": average_confidence,
    }