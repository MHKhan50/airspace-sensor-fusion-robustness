import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from skysentinel.data import FEATURES
from skysentinel.model import build


SENSOR_GROUPS = {
    "rgb": ["rgb_shape", "rgb_motion", "rgb_brightness"],
    "audio": ["audio_low", "audio_high", "audio_periodicity"],
    "telemetry": ["telemetry_speed", "telemetry_altitude"],
}


def evaluate_conditions(df):
    X = df[FEATURES].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    conditions = ["normal", "rgb_missing", "audio_missing",
                  "telemetry_missing", "all_sensors_noisy"]

    for condition in conditions:
        test = X_test.copy()

        if condition.endswith("_missing"):
            sensor = condition.replace("_missing", "")
            test[SENSOR_GROUPS[sensor]] = 0
            test[f"{sensor}_available"] = 0

        elif condition == "all_sensors_noisy":
            rng = np.random.default_rng(42)
            feature_cols = FEATURES[:8]
            noise = rng.normal(0, 0.15, test[feature_cols].shape)
            test[feature_cols] = np.clip(
                test[feature_cols].to_numpy() + noise, 0, 1
            )

        model = build()
        model.fit(X_train, y_train)
        predictions = model.predict(test)

        results.append({
            "condition": condition,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, average="macro"
            )
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_sensor_data.csv")
    results = evaluate_conditions(df)

    print("\nRAMF Robustness Experiment")
    print(results.to_string(index=False))

    results.to_csv("artifacts/robustness_results.csv", index=False)
    print("\nSaved artifacts/robustness_results.csv")
