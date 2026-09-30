import pandas as pd
import streamlit as st
import altair as alt

from src.sentiment import load_sentiment_model, predict_sentiment
from src.pipeline import run_feedback_pipeline
from src.insights import generate_vibe_insight


st.set_page_config(
    page_title="VibeSig | Customer Feedback Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_sentiment_model():
    """Load and cache the sentiment model."""
    return load_sentiment_model()

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_summary" not in st.session_state:
    st.session_state.batch_summary = None

st.write("")

st.caption("CUSTOMER FEEDBACK INTELLIGENCE")

st.title("VibeSig")

st.markdown(
    "### Turn customer feedback into actionable signals."
)

st.write(
    "AI-powered sentiment intelligence for individual reviews "
    "and customer feedback datasets."
)

st.caption(
    "Sentiment classification • Vibe Score • Batch analytics • CSV export"
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

    st.write(
        "Enter a customer review to identify its sentiment "
        "and model confidence."
    )

    input_col, empty_col = st.columns([2.2, 1])

    with input_col:

        review_text = st.text_area(
            "Customer review",
            placeholder="Paste a customer review here...",
            height=160,
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

            sentiment = result["sentiment"]
            confidence = result["confidence"]

            st.caption("ANALYSIS RESULT")

            results_area, results_space = st.columns([2.2, 1])

            with results_area:

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "Customer sentiment",
                        sentiment,
                    )

                    if sentiment == "Positive":
                        st.success(
                            "Strong positive customer signal."
                        )

                    elif sentiment == "Negative":
                        st.error(
                            "Negative customer signal detected."
                        )

                    else:
                        st.warning(
                            "Mixed or neutral customer signal."
                        )

                with result_col2:

                    st.metric(
                        "Model confidence",
                        f"{confidence:.1%}",
                    )

                    st.progress(
                        min(float(confidence), 1.0)
                    )

            
# ---------------------------------------------------------
# BATCH ANALYSIS
# ---------------------------------------------------------

with batch_tab:

    st.subheader("Analyze customer feedback at scale")

    st.write(
        "Upload a CSV dataset to measure customer sentiment, "
        "calculate your Vibe Score, and explore feedback patterns."
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

            st.markdown("### Configure analysis")

            config_col1, config_col2, config_space = st.columns(
                [1.2, 1, 0.8]
            )

            with config_col1:

                text_column = st.selectbox(
                    "Review column",
                    options=list(df.columns),
                    index=(
                        list(df.columns).index("text")
                        if "text" in df.columns
                        else 0
                    ),
                )

            max_reviews = min(len(df), 1000)

            with config_col2:

                review_limit = st.number_input(
                    "Reviews to analyze",
                    min_value=1,
                    max_value=max_reviews,
                    value=min(100, max_reviews),
                    step=50,
                    help=(
                        "Start with a smaller sample "
                        "for faster analysis."
                    ),
                )

            st.caption(
                f"Ready to analyze {review_limit:,} reviews "
                f"from `{text_column}` • "
                f"{len(df):,} total rows available"
            )

            analyze_csv_button = st.button(
                "Analyze feedback",
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
                insight = generate_vibe_insight(summary)

                st.caption("CUSTOMER FEEDBACK OVERVIEW")
                st.subheader("Customer Vibe")

                st.write(
                    "A consolidated view of customer sentiment "
                    "across the analyzed feedback."
                )

                # ---------------------------------------------
                # PRIMARY KPIs
                # ---------------------------------------------

                kpi1, kpi2, kpi3 = st.columns(3)

                with kpi1:
                    st.metric(
                        "Vibe Score",
                        f"{summary['vibe_score']:.1f}",
                        help=(
                            "Overall customer sentiment score "
                            "on a 0–100 scale."
                        ),
                    )
                    st.caption("Overall sentiment • 0–100")

                with kpi2:
                    st.metric(
                        "Reviews analyzed",
                        f"{summary['total_reviews']:,}",
                    )
                    st.caption("Customer feedback processed")

                with kpi3:
                    st.metric(
                        "Model confidence",
                        f"{summary['average_confidence']:.1f}%",
                    )
                    st.caption("Average prediction confidence")

                st.divider()

                # ---------------------------------------------
                # SENTIMENT MIX
                # ---------------------------------------------

                st.markdown("### Sentiment mix")

                sentiment_col1, sentiment_col2, sentiment_col3 = (
                    st.columns(3)
                )

                with sentiment_col1:
                    st.metric(
                        "Positive",
                        f"{summary['positive_pct']:.1f}%",
                        help=(
                            f"{summary['positive']:,} "
                            "positive reviews"
                        ),
                    )

                with sentiment_col2:
                    st.metric(
                        "Neutral",
                        f"{summary['neutral_pct']:.1f}%",
                        help=(
                            f"{summary['neutral']:,} "
                            "neutral reviews"
                        ),
                    )

                with sentiment_col3:
                    st.metric(
                        "Negative",
                        f"{summary['negative_pct']:.1f}%",
                        help=(
                            f"{summary['negative']:,} "
                            "negative reviews"
                        ),
                    )

                st.caption(
                    f"Processed in "
                    f"{summary['processing_time_seconds']:.2f} seconds"
                    f" • "
                    f"{summary['reviews_per_second']:.2f} reviews/second"
                )
                # ---------------------------------------------
                # SENTIMENT DISTRIBUTION
                # ---------------------------------------------

                st.markdown("### Sentiment Distribution")

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
                # VIBE INSIGHT
                # ---------------------------------------------

                st.markdown("### Vibe Insight")

                st.caption(
                    "RULE-BASED CUSTOMER SIGNAL INTERPRETATION"
                )

                st.markdown(
                    f"**{insight['overall_sentiment']} "
                    f"customer sentiment**"
                )

                st.write(
                    insight["interpretation"]
                )

                insight_col1, insight_col2 = st.columns(2)

                with insight_col1:
                    st.info(
                        f"**Key signal**\n\n"
                        f"{insight['key_signal']}"
                    )

                with insight_col2:
                    if summary["negative_pct"] >= 30:
                        st.error(
                            f"**Attention signal**\n\n"
                            f"{insight['attention']}"
                        )

                    elif summary["negative_pct"] >= 15:
                        st.warning(
                            f"**Attention signal**\n\n"
                            f"{insight['attention']}"
                        )

                    else:
                        st.success(
                            f"**Attention signal**\n\n"
                            f"{insight['attention']}"
                        )

                st.divider()

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