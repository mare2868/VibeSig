import pandas as pd
import streamlit as st
import altair as alt

from src.sentiment import load_sentiment_model, predict_sentiment
from src.pipeline import run_feedback_pipeline
from src.insights import generate_vibe_insight
from src.drivers import (
    summarize_negative_drivers,
    prioritize_negative_drivers,
)
from src.recommendations import generate_action_recommendations
from src.evaluation import evaluate_sentiment_predictions


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
                driver_summary = summarize_negative_drivers(
                    results_df
            )
                driver_priorities = prioritize_negative_drivers(
                     driver_summary
            )
                action_recommendations = generate_action_recommendations(
                    driver_priorities
            )
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
                # NEGATIVE FEEDBACK DRIVERS
                # ---------------------------------------------

                st.markdown("### Negative Feedback Drivers")

                st.write(
                    "Most common themes detected across "
                    "negative customer reviews."
                )

                if driver_summary["negative_reviews"] > 0:

                    st.caption(
                        f"Based on "
                        f"{driver_summary['negative_reviews']:,} "
                        f"negative reviews"
                    )

                    drivers_df = pd.DataFrame(
                        driver_summary["drivers"]
                    )

                    drivers_df["Label"] = (
                        drivers_df["negative_review_pct"]
                         .map(lambda value: f"{value:.1f}%")
                    )

                    driver_chart = (
                        alt.Chart(drivers_df)
                        .mark_bar(
                            cornerRadiusEnd=6,
                            height=30,
                        )
                        .encode(
                            y=alt.Y(
                                "driver:N",
                                sort="-x",
                                title=None,
                                axis=alt.Axis(
                                    labelFontSize=13,
                                    labelPadding=10,
                                ),
                            ),
                            x=alt.X(
                                "negative_review_pct:Q",
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
                            tooltip=[
                                alt.Tooltip(
                                    "driver:N",
                                    title="Driver",
                                ),
                                alt.Tooltip(
                                    "mentions:Q",
                                    title="Negative reviews",
                                ),
                                alt.Tooltip(
                                    "negative_review_pct:Q",
                                    title="% of negative reviews",
                                    format=".1f",
                                ),
                            ],
                        )
                    )

                    driver_labels = (
                        alt.Chart(drivers_df)
                        .mark_text(
                            align="left",
                            baseline="middle",
                            dx=8,
                            fontSize=13,
                            fontWeight="bold",
                        )
                        .encode(
                            y=alt.Y(
                                "driver:N",
                                sort="-x",
                            ),
                            x=alt.X(
                                "negative_review_pct:Q"
                            ),
                           text="Label:N",
                        )
                    )

                    drivers_chart = (
                        (driver_chart + driver_labels)
                        .properties(
                            height=260,
                        )
                        .configure_view(
                            strokeWidth=0
                        )
                    )

                    st.altair_chart(
                        drivers_chart,
                        use_container_width=True,
                    )

                    st.caption(
                        "% of negative reviews mentioning each driver. "
                        "A review may mention more than one driver, "
                        "so percentages do not need to total 100%."
                    )
                                        # -----------------------------------------
                    # PRIORITY SIGNALS
                    # -----------------------------------------

                    st.markdown("#### Priority Signals")

                    high_priority = [
                        item
                        for item in driver_priorities
                        if item["priority"] == "High"
                    ]

                    medium_priority = [
                        item
                        for item in driver_priorities
                        if item["priority"] == "Medium"
                    ]

                    low_priority = [
                        item
                        for item in driver_priorities
                        if item["priority"] == "Low"
                    ]

                    priority_col1, priority_col2, priority_col3 = (
                        st.columns(3)
                    )

                    with priority_col1:
                        st.markdown("**High**")

                        if high_priority:
                            for item in high_priority:
                                st.write(
                                    f"{item['driver']} "
                                    f"· {item['negative_review_pct']:.1f}%"
                                )
                        else:
                            st.caption("No high-priority signals")

                    with priority_col2:
                        st.markdown("**Medium**")

                        if medium_priority:
                            for item in medium_priority:
                                st.write(
                                    f"{item['driver']} "
                                    f"· {item['negative_review_pct']:.1f}%"
                                )
                        else:
                            st.caption("No medium-priority signals")

                    with priority_col3:
                        st.markdown("**Low**")

                        if low_priority:
                            for item in low_priority:
                                st.write(
                                    f"{item['driver']} "
                                    f"· {item['negative_review_pct']:.1f}%"
                                )
                        else:
                            st.caption("No low-priority signals")

                    st.caption(
                        "Priority levels are rule-based and reflect "
                        "how frequently each driver appears in negative reviews."
                    )

                                        # -----------------------------------------
                    # RECOMMENDED ACTIONS
                    # -----------------------------------------

                    st.markdown("#### Recommended Actions")

                    st.write(
                        "Suggested areas for further business review "
                        "based on the highest-priority feedback signals."
                    )

                    if action_recommendations:

                        for index, item in enumerate(
                            action_recommendations,
                            start=1,
                        ):
                            st.markdown(
                                f"**{index}. {item['driver']}** "
                                f"· {item['negative_review_pct']:.1f}% "
                                f"of negative reviews"
                            )

                            st.write(
                                item["recommendation"]
                            )

                    else:

                        st.info(
                            "No action recommendations are available "
                            "for the current analysis."
                        )

                    st.caption(
                        "Recommendations are rule-based prompts for "
                        "further investigation, not automated decisions."
                    )
                else:

                    st.info(
                        "No negative reviews were detected, "
                        "so feedback drivers are not available."
                    )

                st.divider()

                                # ---------------------------------------------
                # MODEL PERFORMANCE
                # ---------------------------------------------

                if "actual_sentiment" in analysis_df.columns:

                    evaluation_df = analysis_df[
                        analysis_df[text_column]
                        .notna()
                        & analysis_df[text_column]
                        .astype(str)
                        .str.strip()
                        .ne("")
                    ].copy()

                    actual_sentiments = (
                        evaluation_df["actual_sentiment"]
                        .astype(str)
                        .str.strip()
                    )

                    valid_labels = {
                        "Negative",
                        "Neutral",
                        "Positive",
                    }

                    evaluation_mask = actual_sentiments.isin(
                        valid_labels
                    )

                    actual_sentiments = (
                        actual_sentiments[
                            evaluation_mask
                        ].reset_index(drop=True)
                    )

                    predicted_sentiments = (
                        results_df.loc[
                            evaluation_mask.to_numpy(),
                            "sentiment",
                        ].reset_index(drop=True)
                    )

                    if len(actual_sentiments) > 0:

                        evaluation = evaluate_sentiment_predictions(
                            actual_sentiments,
                            predicted_sentiments,
                        )

                        st.divider()

                        st.subheader("Model Performance")

                        st.caption(
                            "PERFORMANCE AGAINST AVAILABLE "
                            "GROUND-TRUTH LABELS"
                        )

                        st.write(
                            f"Evaluation based on "
                            f"{evaluation['total_samples']:,} "
                            f"labeled reviews."
                        )

                        metric1, metric2, metric3, metric4 = (
                            st.columns(4)
                        )

                        with metric1:
                            st.metric(
                                "Accuracy",
                                f"{evaluation['accuracy'] * 100:.1f}%",
                            )

                        with metric2:
                            st.metric(
                                "Macro Precision",
                                f"{evaluation['macro_precision'] * 100:.1f}%",
                            )

                        with metric3:
                            st.metric(
                                "Macro Recall",
                                f"{evaluation['macro_recall'] * 100:.1f}%",
                            )

                        with metric4:
                            st.metric(
                                "Macro F1",
                                f"{evaluation['macro_f1'] * 100:.1f}%",
                            )

                        # -------------------------------------
                        # PERFORMANCE BY CLASS
                        # -------------------------------------

                        st.markdown("#### Performance by Class")

                        class_rows = []

                        for label in evaluation["labels"]:
                            metrics = evaluation[
                                "per_class"
                            ][label]

                            class_rows.append(
                                {
                                    "Sentiment": label,
                                    "Precision": f"{metrics['precision'] * 100:.1f}%",
                                    "Recall": f"{metrics['recall'] * 100:.1f}%",
                                    "F1": f"{metrics['f1'] * 100:.1f}%",
                                    "Support": metrics["support"],
                                }
                            )

                        class_df = pd.DataFrame(class_rows)

                        st.dataframe(
                            class_df,
                            use_container_width=True,
                            hide_index=True,
                                    )

                        # -------------------------------------
                        # CONFUSION MATRIX
                        # -------------------------------------

                        st.markdown("#### Confusion Matrix")

                        st.caption(
                            "Rows represent actual sentiment; "
                            "columns represent VibeSig predictions."
                        )

                        matrix_df = pd.DataFrame(
                            evaluation["confusion_matrix"],
                            index=evaluation["labels"],
                            columns=evaluation["labels"],
                        )

                        matrix_long = (
                            matrix_df
                            .rename_axis("Actual")
                            .reset_index()
                            .melt(
                                id_vars="Actual",
                                var_name="Predicted",
                                value_name="Count",
                            )
                        )

                        matrix_chart = (
                            alt.Chart(matrix_long)
                            .mark_rect(
                                cornerRadius=4,
                            )
                            .encode(
                                x=alt.X(
                                    "Predicted:N",
                                    title="Predicted",
                                    sort=evaluation["labels"],
                                ),
                                y=alt.Y(
                                    "Actual:N",
                                    title="Actual",
                                    sort=evaluation["labels"],
                                ),
                                color=alt.Color(
                                    "Count:Q",
                                    title="Reviews",
                                ),
                                tooltip=[
                                    alt.Tooltip(
                                        "Actual:N",
                                        title="Actual",
                                    ),
                                    alt.Tooltip(
                                        "Predicted:N",
                                        title="Predicted",
                                    ),
                                    alt.Tooltip(
                                        "Count:Q",
                                        title="Reviews",
                                    ),
                                ],
                            )
                        )

                        matrix_labels = (
                            alt.Chart(matrix_long)
                            .mark_text(
                                fontSize=16,
                                fontWeight="bold",
                            )
                            .encode(
                                x=alt.X(
                                    "Predicted:N",
                                    sort=evaluation["labels"],
                                ),
                                y=alt.Y(
                                    "Actual:N",
                                    sort=evaluation["labels"],
                                ),
                                text=alt.Text(
                                    "Count:Q",
                                    format="d",
                                ),
                            )
                        )

                        confusion_chart = (
                            (matrix_chart + matrix_labels)
                            .properties(
                                height=300,
                            )
                            .configure_view(
                                strokeWidth=0
                            )
                        )

                        st.altair_chart(
                            confusion_chart,
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