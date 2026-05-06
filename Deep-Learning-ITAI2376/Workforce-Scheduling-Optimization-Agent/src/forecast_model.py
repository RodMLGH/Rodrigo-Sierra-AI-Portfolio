"""
forecast_model.py

Agent 1: Demand Forecasting Pipeline

This script trains a lightweight LSTM model on synthetic manufacturing
demand history and generates a staffing forecast for the next 7 days.

Input:
- data/demand_history.csv

Output:
- outputs/forecast.json
- outputs/forecast.csv

Project role:
This script represents the technical deep learning component of
Agent 1 (Demand Forecasting Agent).
"""


# Imports

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Configuration

RANDOM_SEED = 42
WINDOW_SIZE = 14
FORECAST_DAYS = 7
EPOCHS = 60
BATCH_SIZE = 16


# File paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEMAND_FILE = DATA_DIR / "demand_history.csv"
FORECAST_JSON_FILE = OUTPUTS_DIR / "forecast.json"
FORECAST_CSV_FILE = OUTPUTS_DIR / "forecast.csv"


# Shift and skills

SHIFTS = ["Morning", "Afternoon", "Night"]

SKILLS = [
    "Machine Operator",
    "Quality Inspector",
    "Maintenance Technician",
    "Packaging Operator",
]


# Helper functions

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_output_folder() -> None:
    """Create outputs folder if it does not exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_demand_data() -> pd.DataFrame:
    """Load the synthetic demand history dataset."""
    if not DEMAND_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DEMAND_FILE}. "
            "Run python src/generate_data.py first."
        )

    df = pd.read_csv(DEMAND_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Data preparation

def aggregate_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate demand to Version 1 forecasting level:
    date + shift + skill_required.

    Important:
    We use average demand across production lines instead of total demand.
    This keeps the first scheduling version feasible for a 20-employee
    proof of concept.
    """
    agg_df = (
        df.groupby(["date", "day_of_week", "shift", "skill_required"], as_index=False)
        .agg(
            required_staff=("required_staff", "mean"),
            production_units=("production_units", "mean"),
            is_weekend=("is_weekend", "max"),
            is_peak_day=("is_peak_day", "max"),
        )
    )

    # Round required staff to whole people.
    agg_df["required_staff"] = agg_df["required_staff"].round().astype(int)
    agg_df["required_staff"] = agg_df["required_staff"].clip(lower=1)

    return agg_df.sort_values(["skill_required", "shift", "date"]).reset_index(drop=True)


def create_lstm_sequences(
    agg_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    Create LSTM sequences.

    The model learns from historical required_staff values for each
    shift + skill combination.

    X shape:
    - samples, window_size, 1

    y shape:
    - samples, 1
    """
    scaler = MinMaxScaler()

    # Fit scaler on all required_staff values.
    agg_df = agg_df.copy()
    agg_df["required_staff_scaled"] = scaler.fit_transform(
        agg_df[["required_staff"]]
    )

    X, y = [], []

    for skill in SKILLS:
        for shift in SHIFTS:
            series_df = agg_df[
                (agg_df["skill_required"] == skill) &
                (agg_df["shift"] == shift)
            ].sort_values("date")

            values = series_df["required_staff_scaled"].values

            if len(values) <= window_size:
                continue

            for i in range(window_size, len(values)):
                X.append(values[i - window_size:i])
                y.append(values[i])

    X = np.array(X)
    y = np.array(y)

    # LSTM expects 3D input: samples, timesteps, features
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((-1, 1))

    return X, y, scaler, agg_df


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into training and testing sets.

    We use a simple chronological-style split based on sequence order.
    """
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# LSTM model

def build_lstm_model(window_size: int = WINDOW_SIZE) -> Sequential:
    """
    Build a lightweight LSTM model.

    This model is intentionally simple because this is a proof of concept,
    not a production-grade forecasting system.
    """
    model = Sequential([
        tf.keras.Input(shape=(window_size, 1)),
        LSTM(32),
        Dropout(0.10),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray
):
    """Train the LSTM model."""
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    return history


def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler
) -> float:
    """
    Evaluate the LSTM model using Mean Absolute Error in original units.

    This tells us approximately how many workers off the forecast is.
    """
    predictions_scaled = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions_scaled)
    actuals = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actuals, predictions)
    return mae


# Forecast generation

def get_next_dates(last_date: pd.Timestamp, days: int = FORECAST_DAYS) -> list[pd.Timestamp]:
    """Generate the next forecast dates."""
    return [last_date + timedelta(days=i) for i in range(1, days + 1)]


def forecast_next_week(
    model: Sequential,
    agg_df: pd.DataFrame,
    scaler: MinMaxScaler,
    window_size: int = WINDOW_SIZE,
    forecast_days: int = FORECAST_DAYS
) -> pd.DataFrame:
    """
    Generate forecast for the next 7 days for each shift + skill combination.

    Forecast level:
    day + shift + skill_required
    """
    forecast_rows = []
    last_date = agg_df["date"].max()
    next_dates = get_next_dates(last_date, forecast_days)

    for skill in SKILLS:
        for shift in SHIFTS:
            series_df = agg_df[
                (agg_df["skill_required"] == skill) &
                (agg_df["shift"] == shift)
            ].sort_values("date")

            recent_values = series_df["required_staff_scaled"].values[-window_size:]

            if len(recent_values) < window_size:
                raise ValueError(
                    f"Not enough historical data for {skill} - {shift}."
                )

            current_window = recent_values.copy()

            for forecast_date in next_dates:
                X_input = current_window.reshape((1, window_size, 1))
                prediction_scaled = model.predict(X_input, verbose=0)[0][0]

                # Keep scaled prediction in reasonable range.
                prediction_scaled = np.clip(prediction_scaled, 0, 1)

                prediction_original = scaler.inverse_transform(
                    np.array([[prediction_scaled]])
                )[0][0]

                forecast_required_staff = int(round(prediction_original))
                forecast_required_staff = max(1, forecast_required_staff)

                forecast_rows.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "day_of_week": forecast_date.strftime("%A"),
                    "shift": shift,
                    "skill_required": skill,
                    "forecast_required_staff": forecast_required_staff
                })

                # Recursive forecasting: feed prediction back into the window.
                current_window = np.append(current_window[1:], prediction_scaled)

    forecast_df = pd.DataFrame(forecast_rows)

    # Sort for readability.
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df.sort_values(
        ["date", "shift", "skill_required"]
    ).reset_index(drop=True)
    forecast_df["date"] = forecast_df["date"].dt.strftime("%Y-%m-%d")

    return forecast_df


def save_forecast(forecast_df: pd.DataFrame) -> None:
    """Save forecast as JSON and CSV."""
    records = forecast_df.to_dict(orient="records")

    with open(FORECAST_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    forecast_df.to_csv(FORECAST_CSV_FILE, index=False)


# Main function

def main() -> None:
    """Run the full demand forecasting pipeline."""
    set_random_seed()
    ensure_output_folder()

    print("Loading demand history...")
    raw_df = load_demand_data()
    print(f"Loaded demand rows: {len(raw_df)}")

    print("\nAggregating demand for Version 1 forecast level...")
    agg_df = aggregate_demand(raw_df)
    print(f"Aggregated rows: {len(agg_df)}")
    print("Forecasting level: date + shift + skill_required")

    print("\nCreating LSTM sequences...")
    X, y, scaler, agg_df = create_lstm_sequences(agg_df)
    print(f"Sequence samples: {len(X)}")
    print(f"Input shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

    print("\nBuilding LSTM model...")
    model = build_lstm_model()

    print("\nTraining LSTM model...")
    train_model(model, X_train, y_train)

    print("\nEvaluating model...")
    mae = evaluate_model(model, X_test, y_test, scaler)
    print(f"Test MAE: {mae:.2f} workers")

    print("\nGenerating next-week forecast...")
    forecast_df = forecast_next_week(model, agg_df, scaler)

    save_forecast(forecast_df)

    print("\nForecast generated successfully.")
    print(f"Forecast JSON: {FORECAST_JSON_FILE}")
    print(f"Forecast CSV: {FORECAST_CSV_FILE}")

    print("\nForecast sample:")
    print(forecast_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
