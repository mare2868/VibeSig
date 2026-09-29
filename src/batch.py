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