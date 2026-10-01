# VibeSig

### Customer Feedback Intelligence

**Turn customer feedback into actionable signals.**

VibeSig is an AI-powered customer feedback intelligence application that transforms individual reviews and customer feedback datasets into structured sentiment signals, business insights, and actionable recommendations.

The project combines transformer-based sentiment analysis, confidence calibration, model evaluation, feedback-driver detection, rule-based business interpretation, and an interactive Streamlit application.

---

## Project Overview

Organizations collect large volumes of customer feedback, but turning unstructured reviews into useful information requires more than assigning a positive or negative label.

VibeSig was developed to explore a broader question:

> **How can customer feedback be transformed into measurable signals that help identify what customers are experiencing and where attention may be required?**

The application analyzes customer reviews and provides:

- Sentiment classification
- Model confidence
- Calibrated sentiment predictions
- Customer Vibe Score
- Sentiment distribution
- Negative feedback drivers
- Priority signals
- Rule-based recommended actions
- Model performance metrics
- Confusion matrix analysis
- CSV export of analyzed feedback

---

## Business Problem

Customer reviews contain valuable information about product quality, service, staff behavior, waiting times, pricing, and the overall customer experience.

However, manually reviewing hundreds or thousands of comments is slow and difficult to scale.

Traditional sentiment classification also has an important limitation: knowing whether a review is positive or negative does not necessarily explain **why customers are dissatisfied or what should be investigated first**.

VibeSig addresses this problem through a layered feedback intelligence pipeline that moves from raw customer comments toward interpretable business signals.

---

## Solution

VibeSig processes customer feedback through several analytical stages:

```text
Customer Reviews
       |
       v
Sentiment Model
       |
       v
Confidence Calibration
       |
       v
Sentiment Classification
       |
       +--------------------+
       |                    |
       v                    v
 Customer Vibe       Negative Reviews
       |                    |
       v                    v
Sentiment Mix        Feedback Drivers
                            |
                            v
                     Priority Signals
                            |
                            v
                   Recommended Actions
```

The goal is not to automate business decisions.

Instead, VibeSig provides structured signals that can help analysts and decision-makers identify patterns that deserve further investigation.

---

## Key Features

### Single Review Analysis

Users can enter an individual customer review and obtain its predicted sentiment and model confidence.

### Batch Feedback Analysis

Users can upload CSV datasets and analyze customer feedback at scale.

The application provides:

- Total reviews analyzed
- Positive, neutral, and negative distribution
- Average model confidence
- Processing performance
- Customer Vibe Score

### Customer Vibe Score

VibeSig summarizes the overall sentiment of the analyzed feedback into a score ranging from **0 to 100**.

This provides a compact indicator of the overall customer feedback signal while preserving the underlying sentiment distribution for interpretation.

### Negative Feedback Drivers

Negative reviews are analyzed for recurring customer experience themes, including:

- Food / Product
- Quality
- Staff
- Wait Time
- Service
- Price / Value
- Other

A single review may contain multiple drivers.

### Priority Signals

Detected negative-feedback drivers are classified into rule-based priority levels:

- High
- Medium
- Low

Priority reflects how frequently each driver appears across negative customer feedback.

### Recommended Actions

For the highest-priority signals, VibeSig generates rule-based prompts for further business investigation.

These recommendations are intentionally designed as **decision-support signals rather than automated decisions**.

---

## Dataset

The project uses the **Yelp Review Full** dataset as the primary source of customer-review text for development and model evaluation.

A balanced development sample was created containing:

- 15,000 reviews
- 6,000 Positive
- 6,000 Negative
- 3,000 Neutral

Sentiment labels were derived from the original review ratings to support the three-class sentiment analysis used by VibeSig.

The development dataset is maintained locally and is intentionally excluded from the public repository through `.gitignore`. It is used for model development, benchmarking, and evaluation but is **not required to run the VibeSig application**.

The deployed application operates independently of the development dataset: users can analyze individual reviews or upload their own CSV files for batch analysis.

---

## Sentiment Model

VibeSig uses a transformer-based sentiment classifier from the Hugging Face ecosystem.

The model produces:

```text
Review
   |
   +--> Sentiment label
   |
   +--> Confidence score
```

During development, model behavior was evaluated against available ground-truth labels rather than relying only on prediction confidence.

This revealed an important classification issue.

---

## Model Evaluation

Initial evaluation showed strong performance for clearly positive feedback but weaker discrimination of neutral sentiment.

A balanced evaluation sample containing:

- 100 Negative reviews
- 100 Neutral reviews
- 100 Positive reviews

was used to investigate this behavior.

The baseline experiment showed:

| Metric | Baseline |
|---|---:|
| Accuracy | 61.0% |
| Macro Precision | 62.5% |
| Macro Recall | 61.0% |
| Macro F1 | 58.0% |
| Neutral Recall | 23.0% |
| Positive Recall | 94.0% |

The main issue was therefore not general model failure.

The analysis showed that many **Neutral reviews were being classified as Positive**.

---

## Confidence Calibration Experiment

Further analysis compared confidence distributions between correctly classified positive reviews and neutral reviews incorrectly classified as positive.

Correct positive predictions showed very high confidence:

```text
Mean confidence: 0.977
Median confidence: 0.997
```

Neutral reviews incorrectly classified as positive also frequently received high model confidence:

```text
Mean confidence: 0.891
Median confidence: 0.968
```

This demonstrated that confidence alone could not be interpreted as classification correctness.

A threshold experiment was therefore performed to test whether lower-confidence positive predictions could be conservatively reassigned to Neutral.

Several thresholds were evaluated.

A threshold of:

```text
0.97
```

provided a useful balance between recovering neutral reviews and preserving positive classification performance.

The calibration rule implemented in VibeSig is conceptually:

```text
IF predicted sentiment = Positive
AND confidence < 0.97
THEN calibrated sentiment = Neutral
```

The original prediction is preserved separately from the calibrated sentiment, allowing model behavior to remain auditable.

---

## Current Model Performance

On the current 100-review labeled evaluation sample, the calibrated pipeline produced:

| Metric | Result |
|---|---:|
| Accuracy | 68.0% |
| Macro Precision | 71.4% |
| Macro Recall | 66.1% |
| Macro F1 | 63.9% |

### Performance by Class

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 100.0% | 51.3% | 67.8% | 39 |
| Neutral | 27.8% | 62.5% | 38.5% | 16 |
| Positive | 86.4% | 84.4% | 85.4% | 45 |

These results are displayed directly in the application together with the confusion matrix.

The evaluation also highlights an important limitation: **neutral sentiment remains the most difficult class**, particularly in terms of precision.

Rather than hiding this limitation, VibeSig exposes model performance as part of the analytical workflow.

---

## Model Transparency

VibeSig preserves both:

```text
raw_sentiment
```

and:

```text
sentiment
```

in analyzed results.

`raw_sentiment` represents the original model prediction.

`sentiment` represents the prediction after the confidence-calibration rule has been applied.

This makes it possible to inspect where calibration changed the original model output.

---

## Business Intelligence Layer

VibeSig goes beyond sentiment classification by converting predictions into higher-level signals.

The analytical flow is:

```text
Sentiment
    ↓
Negative Feedback
    ↓
Feedback Drivers
    ↓
Frequency Analysis
    ↓
Priority Signals
    ↓
Recommended Actions
```

This layer is deliberately rule-based and interpretable.

The objective is to provide analysts with understandable signals rather than opaque automated recommendations.

---

## Interactive Application

VibeSig includes a Streamlit interface with two primary workflows.

### Single Review

Analyze one customer review interactively.

### Batch Analysis

Upload a CSV file and:

1. Select the review column
2. Choose the number of reviews to analyze
3. Run sentiment analysis
4. Review the Customer Vibe
5. Explore sentiment distribution
6. Inspect negative feedback drivers
7. Review priority signals
8. Examine recommended actions
9. Review model performance when ground-truth labels are available
10. Download the analyzed dataset

---

## Testing & Reliability

VibeSig includes automated tests covering the major analytical components:

```text
tests/test_batch.py
tests/test_calibration.py
tests/test_csv_processor.py
tests/test_drivers.py
tests/test_evaluation.py
tests/test_insights.py
tests/test_pipeline.py
tests/test_recommendations.py
tests/test_sentiment.py
```

Current test status:

```text
47 passed
```

The test suite validates sentiment processing, CSV handling, calibration behavior, feedback-driver detection, model evaluation, business insights, recommendations, and the end-to-end analysis pipeline.

---

## Project Structure

```text
VibeSig/
│
├── app/
│   └── app.py
│
├── benchmarks/
│   └── evaluate_positive_threshold.py
│
├── src/
│   ├── batch.py
│   ├── calibration.py
│   ├── csv_processor.py
│   ├── drivers.py
│   ├── evaluation.py
│   ├── insights.py
│   ├── pipeline.py
│   ├── recommendations.py
│   └── sentiment.py
│
├── tests/
│   ├── test_batch.py
│   ├── test_calibration.py
│   ├── test_csv_processor.py
│   ├── test_drivers.py
│   ├── test_evaluation.py
│   ├── test_insights.py
│   ├── test_pipeline.py
│   ├── test_recommendations.py
│   └── test_sentiment.py
│
└── README.md
```

---

## Technology Stack

VibeSig is built with:

- Python
- pandas
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Streamlit
- pytest

---

## Design Principles

Several principles guided the development of VibeSig:

**Measure before modifying.**  
Model behavior was evaluated before calibration rules were introduced.

**Preserve transparency.**  
Raw and calibrated predictions remain available for comparison.

**Separate prediction from decision-making.**  
Business recommendations are presented as signals for investigation rather than automated decisions.

**Expose limitations.**  
Performance metrics and confusion matrices are included directly in the application.

**Build for reproducibility.**  
Calibration experiments and automated tests are maintained as part of the repository.

---

## Limitations

VibeSig is currently a portfolio and analytical prototype rather than a production customer-experience platform.

Current limitations include:

- Sentiment calibration is based on empirical threshold analysis and should be validated on additional datasets.
- Neutral sentiment remains challenging to classify accurately.
- Feedback-driver detection uses interpretable rule-based logic rather than a dedicated topic-modeling system.
- Business recommendations are rule-based and are not causal conclusions.
- Yelp reviews represent a specific type of customer-feedback domain and may not generalize directly to every industry.
- Larger-scale production deployment would require additional monitoring, validation, security, and model-governance controls.

---

## Future Improvements

Potential extensions include:

- Evaluation on larger independent validation datasets
- Domain-specific sentiment calibration
- Comparison with additional transformer models
- Topic modeling and semantic clustering
- Explainability analysis
- Time-based customer sentiment trends
- Interactive filtering by feedback driver
- Advanced dashboard capabilities
- API-based inference
- Production deployment and monitoring

---

## Project Purpose

VibeSig was developed as an applied Data Science and AI portfolio project demonstrating an end-to-end workflow:

```text
Business Problem
      ↓
Data Preparation
      ↓
NLP Model
      ↓
Model Evaluation
      ↓
Error Analysis
      ↓
Calibration
      ↓
Business Signals
      ↓
Interactive Application
      ↓
Testing & Validation
```

The project demonstrates not only how to generate machine-learning predictions, but also how to **evaluate, interpret, challenge, improve, and translate those predictions into useful analytical signals**.