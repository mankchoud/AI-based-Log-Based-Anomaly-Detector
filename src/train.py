"""
train.py

Trains an Isolation Forest on NORMAL-ONLY traffic from the NSL-KDD
training set, then validates it against the labeled test set.

The model never sees an attack example during training -- it only
learns what "normal" looks like. Anomaly detection at inference time
is then just: "does this look like what I learned as normal, or not?"
"""

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_fscore_support

from features import load_raw, build_feature_matrix

TRAIN_PATH = "data/NSL_KDD_Train.csv"
TEST_PATH = "data/NSL_KDD_Test.csv"
MODEL_PATH = "model.joblib"


def main():
    print("Loading training data...")
    train_df = load_raw(TRAIN_PATH)

    # Core design choice: train ONLY on normal traffic.
    normal_only = train_df[train_df["is_attack"] == 0]
    print(f"Training on {len(normal_only)} normal rows "
          f"(out of {len(train_df)} total training rows)")

    X_train, encoder = build_feature_matrix(normal_only)

    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    print(f"Saving model + encoder to {MODEL_PATH}")
    joblib.dump({"model": model, "encoder": encoder}, MODEL_PATH)

    # --- Validation against the held-out test set ---
    print("\nLoading test data for validation...")
    test_df = load_raw(TEST_PATH)
    X_test, _ = build_feature_matrix(test_df, encoder=encoder)

    predictions = model.predict(X_test)  # 1 = normal, -1 = anomaly
    predicted_attack = (predictions == -1).astype(int)
    true_attack = test_df["is_attack"].to_numpy()

    print("\n--- Validation results (test set, never seen during training) ---")
    print(classification_report(
        true_attack, predicted_attack,
        target_names=["normal", "attack"],
    ))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_attack, predicted_attack, average="binary"
    )
    print(f"Summary -> precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}")


if __name__ == "__main__":
    main()
