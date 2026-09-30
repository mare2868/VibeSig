import statistics
import time

import pandas as pd

from src.pipeline import run_feedback_pipeline
from src.sentiment import load_sentiment_model


CSV_PATH = "data/yelp_vibesig_sample.csv"
TEXT_COLUMN = "text"
N_REVIEWS = 50

# Alternating order to reduce order-related bias.
TEST_SEQUENCE = [8, 16, 16, 8, 8, 16]


print("Loading dataset...")

df = pd.read_csv(CSV_PATH).head(N_REVIEWS)

print(f"Reviews loaded: {len(df)}")
print("Loading sentiment model...")

model = load_sentiment_model()

print("\nStarting repeated VibeSig batch-size benchmark...\n")

benchmark_results = []

for run_number, batch_size in enumerate(TEST_SEQUENCE, start=1):

    print(
        f"Run {run_number}/{len(TEST_SEQUENCE)} "
        f"— batch_size={batch_size}"
    )

    start = time.perf_counter()

    result = run_feedback_pipeline(
        model,
        df,
        TEXT_COLUMN,
        batch_size=batch_size,
    )

    elapsed = time.perf_counter() - start
    reviews_per_second = len(df) / elapsed

    benchmark_results.append(
        {
            "run": run_number,
            "batch_size": batch_size,
            "seconds": elapsed,
            "reviews_per_second": reviews_per_second,
            "vibe_score": result["summary"]["vibe_score"],
        }
    )

    print(
        f"Completed: {elapsed:.2f}s "
        f"({reviews_per_second:.2f} reviews/s)"
    )
    print(
        f"Vibe Score: "
        f"{result['summary']['vibe_score']:.1f}\n"
    )


results_df = pd.DataFrame(benchmark_results)


print("=" * 70)
print("INDIVIDUAL RUNS")
print("=" * 70)

display_results = results_df.copy()

display_results["seconds"] = (
    display_results["seconds"].round(2)
)

display_results["reviews_per_second"] = (
    display_results["reviews_per_second"].round(2)
)

print(display_results.to_string(index=False))


print("\n" + "=" * 70)
print("SUMMARY BY BATCH SIZE")
print("=" * 70)

summary_rows = []

for batch_size in sorted(results_df["batch_size"].unique()):

    subset = results_df[
        results_df["batch_size"] == batch_size
    ]

    times = subset["seconds"].tolist()
    throughputs = subset["reviews_per_second"].tolist()

    summary_rows.append(
        {
            "batch_size": batch_size,
            "runs": len(subset),
            "median_seconds": statistics.median(times),
            "mean_seconds": statistics.mean(times),
            "median_reviews_per_second": statistics.median(
                throughputs
            ),
            "mean_reviews_per_second": statistics.mean(
                throughputs
            ),
        }
    )


summary_df = pd.DataFrame(summary_rows)

for column in [
    "median_seconds",
    "mean_seconds",
    "median_reviews_per_second",
    "mean_reviews_per_second",
]:
    summary_df[column] = summary_df[column].round(2)

print(summary_df.to_string(index=False))


best = summary_df.loc[
    summary_df["median_reviews_per_second"].idxmax()
]

print("\n" + "=" * 70)
print("BEST CONFIGURATION BY MEDIAN THROUGHPUT")
print("=" * 70)

print(
    f"batch_size={int(best['batch_size'])} | "
    f"median time={best['median_seconds']:.2f}s | "
    f"median throughput="
    f"{best['median_reviews_per_second']:.2f} reviews/s"
)