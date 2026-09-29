from transformers import pipeline


MODEL_NAME = "abhishek1005/smartreview-distilroberta-sentiment"


def load_sentiment_model():
    """Load the sentiment analysis model used by VibeSig."""
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
    )

    return classifier

def predict_sentiment(classifier, text):
    """Predict sentiment and confidence for a single customer review."""

    if not text or not text.strip():
        raise ValueError("Review text cannot be empty.")

    result = classifier(
        text,
        truncation=True,
        max_length=512,
    )[0]

    return {
        "sentiment": result["label"],
        "confidence": result["score"],
    }

