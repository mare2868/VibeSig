import pandas as pd
import streamlit as st
import altair as alt

from src.sentiment import load_sentiment_model, predict_sentiment
from src.pipeline import run_feedback_pipeline


st.set_page_config(
    page_title="VibeSig",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def get_sentiment_model():
    """Load and cache the sentiment model."""
    return load_sentiment_model()

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_summary" not in st.session_state:
    st.session_state.batch_summary = None


st.title("VibeSig")
st.subheader("Turn customer feedback into actionable signals.")

st.write(
    "Analyze customer feedback with AI-powered sentiment intelligence."
)


single_tab, batch_tab = st.tabs(
    [
        "Single Review",
        "Batch Analysis",
    ]
)


# ---------------------------------------------------------
# SINGLE REVIEW
# ---------------------------------------------------------

with single_tab:

    st.subheader("Analyze a single customer review")

    review_text = st.text_area(
        "Customer review",
        placeholder="Paste a customer review here...",
        height=180,
    )

    analyze_button = st.button(
        "Analyze feedback",
        type="primary",
        key="single_analyze",
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

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Sentiment",
                    result["sentiment"],
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence']:.1%}",
                )


# ---------------------------------------------------------
# BATCH ANALYSIS
# ---------------------------------------------------------

with batch_tab:

    st.subheader("Analyze customer feedback from CSV")

    st.write(
        "Upload a CSV file containing customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.success(
                f"CSV loaded successfully — {len(df):,} rows detected."
            )

            st.write("Preview")

            st.dataframe(
                df.head(),
                use_container_width=True,
            )

            st.write(
                "Select the column containing customer reviews"
            )

            text_column = st.selectbox(
                "Review column",
                options=list(df.columns),
                index=(
                    list(df.columns).index("text")
                    if "text" in df.columns
                    else 0
                ),
            )

            st.info(
                f"Selected column: {text_column}"
            )

            max_reviews = min(len(df), 1000)

            review_limit = st.number_input(
                "Number of reviews to analyze",
                min_value=1,
                max_value=max_reviews,
                value=min(100, max_reviews),
                step=50,
                help="Start with a smaller sample for faster analysis.",
            )

            st.caption(
                f"VibeSig will analyze {review_limit:,} "
                f"of the {len(df):,} available reviews."
            )

            analyze_csv_button = st.button(
                "Analyze CSV",
                type="primary",
                key="batch_analyze",
            )

            if analyze_csv_button:

                analysis_df = df.head(
                    int(review_limit)
                ).copy()

                with st.spinner(
                    f"Analyzing {review_limit:,} customer reviews..."
                ):
                    classifier = get_sentiment_model()

                    analysis = run_feedback_pipeline(
                        classifier,
                        analysis_df,
                        text_column,
                    )

                st.session_state.batch_results = analysis["results"]
                st.session_state.batch_summary = analysis["summary"]

            # -------------------------------------------------
            # DISPLAY STORED ANALYSIS
            # -------------------------------------------------

            if (
                st.session_state.batch_results is not None
                and st.session_state.batch_summary is not None
            ):

                results_df = st.session_state.batch_results
                summary = st.session_state.batch_summary

                st.success("Batch analysis complete.")

                st.subheader("Customer Vibe")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Vibe Score",
                        f"{summary['vibe_score']:.1f}/100",
                    )

                with col2:
                    st.metric(
                        "Reviews analyzed",
                        f"{summary['total_reviews']:,}",
                    )

                with col3:
                    st.metric(
                        "Avg. confidence",
                        f"{summary['average_confidence']:.1f}%",
                    )

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric(
                        "Positive",
                        f"{summary['positive_pct']:.1f}%",
                    )

                with col5:
                    st.metric(
                        "Neutral",
                        f"{summary['neutral_pct']:.1f}%",
                    )

                with col6:
                    st.metric(
                        "Negative",
                        f"{summary['negative_pct']:.1f}%",
                    )

                    st.caption(
                    f"Processed in "
                    f"{summary['processing_time_seconds']:.2f} seconds "
                    f"• {summary['reviews_per_second']:.2f} reviews/second"
                )

                # ---------------------------------------------
                # SENTIMENT DISTRIBUTION
                # ---------------------------------------------

                st.subheader("Sentiment Distribution")

                sentiment_chart = pd.DataFrame(
                    {
                        "Sentiment": [
                            "Positive",
                            "Neutral",
                            "Negative",
                        ],
                        "Percentage": [
                            summary["positive_pct"],
                            summary["neutral_pct"],
                            summary["negative_pct"],
                        ],
                    }
                )

                sentiment_chart["Label"] = (
                    sentiment_chart["Percentage"]
                    .map(lambda value: f"{value:.1f}%")
                )

                sentiment_order = [
                    "Positive",
                    "Neutral",
                    "Negative",
                ]

                bars = (
                    alt.Chart(sentiment_chart)
                    .mark_bar(
                        cornerRadiusEnd=6,
                        height=34,
                    )
                    .encode(
                        y=alt.Y(
                            "Sentiment:N",
                            sort=sentiment_order,
                            title=None,
                            axis=alt.Axis(
                                labelFontSize=14,
                                labelPadding=12,
                            ),
                        ),
                        x=alt.X(
                            "Percentage:Q",
                            scale=alt.Scale(
                                domain=[0, 100]
                            ),
                            title=None,
                            axis=alt.Axis(
                                labels=False,
                                ticks=False,
                                domain=False,
                                grid=False,
                            ),
                        ),
                        color=alt.Color(
                            "Sentiment:N",
                            scale=alt.Scale(
                                domain=sentiment_order,
                                range=[
                                    "#22C55E",
                                    "#F59E0B",
                                    "#EF4444",
                                ],
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Sentiment:N",
                                title="Sentiment",
                            ),
                            alt.Tooltip(
                                "Percentage:Q",
                                title="Share",
                                format=".1f",
                            ),
                        ],
                    )
                )

                labels = (
                    alt.Chart(sentiment_chart)
                    .mark_text(
                        align="left",
                        baseline="middle",
                        dx=8,
                        fontSize=14,
                        fontWeight="bold",
                    )
                    .encode(
                        y=alt.Y(
                            "Sentiment:N",
                            sort=sentiment_order,
                        ),
                        x=alt.X(
                            "Percentage:Q"
                        ),
                        text="Label:N",
                    )
                )

                chart = (
                    (bars + labels)
                    .properties(
                        height=190,
                    )
                    .configure_view(
                        strokeWidth=0
                    )
                )

                st.altair_chart(
                    chart,
                    use_container_width=True,
                )

                # ---------------------------------------------
                # RESULTS TABLE
                # ---------------------------------------------

                st.subheader("Analyzed Reviews")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                )

                # ---------------------------------------------
                # DOWNLOAD RESULTS
                # ---------------------------------------------

                csv_results = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download analyzed CSV",
                    data=csv_results,
                    file_name="vibesig_analyzed_reviews.csv",
                    mime="text/csv",
                    key="download_results",
                )

        except Exception as error:

            st.error(
                f"Unable to process CSV file: {error}"
            )