import streamlit as st

from src.sentiment import load_sentiment_model, predict_sentiment


st.set_page_config(
    page_title="VibeSig",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def get_sentiment_model():
    """Load and cache the sentiment model."""
    return load_sentiment_model()


st.title("VibeSig")
st.subheader("Turn customer feedback into actionable signals.")

st.write(
    "Analyze customer feedback with AI-powered sentiment intelligence."
)

review_text = st.text_area(
    "Customer review",
    placeholder="Paste a customer review here...",
    height=180,
)

analyze_button = st.button(
    "Analyze feedback",
    type="primary",
)

if analyze_button:
    if not review_text.strip():
        st.warning("Please enter a customer review.")
    else:
        with st.spinner("Analyzing feedback..."):
            classifier = get_sentiment_model()
            result = predict_sentiment(
                classifier,
                review_text,
            )

        st.success("Analysis complete.")

        st.metric(
            "Sentiment",
            result["sentiment"],
        )

        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )