"""
detect.py

Loads the trained model + encoder, scores new data, and reports
anomalies -- no labels required, since in real use we won't have them.
"""

import sys
import joblib
import pandas as pd

from features import load_raw, build_feature_matrix, COLUMN_NAMES

MODEL_PATH = "model.joblib"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["encoder"]


def score(df: pd.DataFrame, model, encoder) -> pd.DataFrame:
    X, _ = build_feature_matrix(df, encoder=encoder)
    predictions = model.predict(X)          # 1 = normal, -1 = anomaly
    scores = model.decision_function(X)      # higher = more normal, lower/negative = more anomalous

    result = df.copy()
    result["anomaly_flag"] = (predictions == -1)
    result["anomaly_score"] = scores
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect.py <path-to-csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Loading model from {MODEL_PATH}...")
    model, encoder = load_model()

    print(f"Loading data from {input_path}...")
    df = load_raw(input_path)

    print("Scoring...")
    result = score(df, model, encoder)

    flagged = result[result["anomaly_flag"]].sort_values("anomaly_score")
    print(f"\n{len(flagged)} anomalies flagged out of {len(result)} rows checked "
          f"({len(flagged) / len(result) * 100:.1f}%)")

    if len(flagged) > 0:
        print("\nTop 10 most anomalous rows (lowest score = most anomalous):")
        cols_to_show = ["duration", "protocol_type", "service", "flag",
                         "src_bytes", "dst_bytes", "num_failed_logins",
                         "anomaly_score"]
        print(flagged[cols_to_show].head(10).to_string(index=False))

    output_path = "flagged_anomalies.csv"
    flagged.to_csv(output_path, index=False)
    print(f"\nFull flagged results saved to {output_path}")


if __name__ == "__main__":
    main()
