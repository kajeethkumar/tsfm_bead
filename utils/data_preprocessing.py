import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, b_id):
    building = df[df["building_id"] == b_id].copy(deep=True)

    building["meter_reading"] = building["meter_reading"].fillna(
        building["meter_reading"].median()
    )

    X = building["meter_reading"].values.reshape(-1, 1)
    y = building["anomaly"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled.ravel(), y

def train_test_split(X, y, train_days=183, val_days=61, test_days=122, freq_per_day=24):
    train_size = train_days * freq_per_day
    val_size = val_days * freq_per_day
    test_size = test_days * freq_per_day

    total_required_size = train_size + val_size + test_size

    if len(X) < total_required_size:
        raise ValueError(
            f"Need {total_required_size} samples, got {len(X)}"
        )

    # 1. Training set (First 183 days)
    X_train = X[:train_size]
    y_train = y[:train_size]

    # 2. Validation set (Next 61 days)
    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    # 3. Testing set (Final 122 days)
    X_test = X[train_size + val_size : train_size + val_size + test_size]
    y_test = y[train_size + val_size : train_size + val_size + test_size]

    return X_train, X_val, X_test, y_train, y_val, y_test