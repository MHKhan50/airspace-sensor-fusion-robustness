import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

SENSORS = {
    "rgb": ["rgb_shape", "rgb_motion", "rgb_brightness"],
    "audio": ["audio_low", "audio_high", "audio_periodicity"],
    "telemetry": ["telemetry_speed", "telemetry_altitude"],
}


def train_fusion_models(X, y):
    models = {}

    for sensor, columns in SENSORS.items():
        available = X[f"{sensor}_available"] == 1

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )

        model.fit(X.loc[available, columns], y.loc[available])
        models[sensor] = model

    return models


def predict_fused(models, X):
    classes = models["rgb"].classes_
    total = np.zeros((len(X), len(classes)))
    weights = np.zeros(len(X))

    for sensor, columns in SENSORS.items():
        available = X[f"{sensor}_available"].to_numpy()

        probabilities = models[sensor].predict_proba(X[columns])

        total += probabilities * available[:, None]
        weights += available

    weights = np.maximum(weights, 1)
    return classes[np.argmax(total / weights[:, None], axis=1)]


def evaluate():
    df = pd.read_csv("data/synthetic_sensor_data.csv")
    feature_columns = [
        col for columns in SENSORS.values() for col in columns
    ] + [f"{sensor}_available" for sensor in SENSORS]

    X = df[feature_columns].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_fusion_models(X_train, y_train)

    conditions = ["normal", "rgb_missing", "audio_missing",
                  "telemetry_missing"]

    results = []

    for condition in conditions:
        test = X_test.copy()

        if condition.endswith("_missing"):
            sensor = condition.replace("_missing", "")
            test[SENSORS[sensor]] = 0
            test[f"{sensor}_available"] = 0

        predictions = predict_fused(models, test)

        results.append({
            "condition": condition,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, average="macro"
            )
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = evaluate()
    print("\nAvailability-Aware Sensor Fusion")
    print(results.to_string(index=False))

    results.to_csv(
        "artifacts/fusion_results.csv", index=False
    )
    print("\nSaved artifacts/fusion_results.csv")
