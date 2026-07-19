"""
features.py

Loads the NSL-KDD dataset and prepares a focused set of features for an
unsupervised anomaly detector.

We deliberately don't use all 41 raw features — only the ones tied to
"suspicious access pattern" behavior (failed logins, connection counts,
service diversity, host-level access patterns), which mirrors what a
real auth/access log would offer. Fewer, well-understood features beat
more features you can't explain in an interview.
"""

import pandas as pd

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
]

NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes",
    "num_failed_logins", "logged_in",
    "count", "srv_count", "same_srv_rate", "diff_srv_rate", "serror_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_diff_srv_rate",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, encoder=None):
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=float)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    else:
        cat_encoded = encoder.transform(df[CATEGORICAL_FEATURES])

    X = np.hstack([numeric, cat_encoded])
    return X, encoder
