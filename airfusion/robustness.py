import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from airfusion.data import FEATURES
from airfusion.model import build

SENSOR_GROUPS = {
    "rgb": ["rgb_shape", "rgb_motion", "rgb_brightness"],
    "audio": ["audio_low", "audio_high", "audio_periodicity"],
    "telemetry": ["telemetry_speed", "telemetry_altitude"],
}

AVAILABILITY_COLUMNS = [
    "rgb_available",
    "audio_available",
    "telemetry_available",
]

MISSING_CONDITIONS = {
    "rgb_missing": ("rgb",),
    "audio_missing": ("audio",),
    "telemetry_missing": ("telemetry",),
    "rgb_audio_missing": ("rgb", "audio"),
    "rgb_telemetry_missing": ("rgb", "telemetry"),
    "audio_telemetry_missing": ("audio", "telemetry"),
    "all_sensors_missing": ("rgb", "audio", "telemetry"),
}


def corrupt_data(X, condition, seed=42):
    X = X.copy()

    if condition in MISSING_CONDITIONS:
        for sensor in MISSING_CONDITIONS[condition]:
            X[SENSOR_GROUPS[sensor]] = 0
            X[f"{sensor}_available"] = 0

    elif condition == "all_sensors_noisy":
        rng = np.random.default_rng(seed)
        columns = FEATURES[:8]
        noise = rng.normal(0, 0.15, X[columns].shape)
        X[columns] = np.clip(
            X[columns].to_numpy() + noise, 0, 1
        )

    return X


def evaluate_conditions(df):
    X = df[FEATURES].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build()
    model.fit(X_train, y_train)

    conditions = [
        "normal",
        "rgb_missing",
        "audio_missing",
        "telemetry_missing",
        "all_sensors_noisy",
        "rgb_audio_missing",
        "rgb_telemetry_missing",
        "audio_telemetry_missing",
        "all_sensors_missing",
    ]

    results = []

    for condition in conditions:
        test = corrupt_data(X_test, condition)

        available = test[AVAILABILITY_COLUMNS].sum(axis=1) > 0
        coverage = available.mean()

        if available.sum() == 0:
            accuracy = np.nan
            macro_f1 = np.nan
        else:
            predictions = model.predict(test.loc[available])
            actual = y_test.loc[available]

            accuracy = accuracy_score(actual, predictions)
            macro_f1 = f1_score(
                actual, predictions,
                average="macro",
                zero_division=0
            )

        results.append({
            "condition": condition,
            "coverage": coverage,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_sensor_data.csv")
    results = evaluate_conditions(df)

    print("\nExtended Sensor Robustness Evaluation")
    print(results.to_string(index=False))

    results.to_csv(
        "artifacts/robustness_results.csv",
        index=False
    )

    print("\nSaved artifacts/robustness_results.csv")
