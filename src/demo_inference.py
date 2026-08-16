"""
Demo: Real-time Sleep Stage Inference

This script simulates how the trained model would work in the actual Android app:
1. Receive sensor data from watch (simulated)
2. Extract features from 1-minute epoch
3. Predict sleep stage (binary: Deep vs Light)
4. Decide if it's a good time to wake up

This is a SIMULATION of what the Kotlin app will do!
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple

from features import extract_epoch_features, simplify_labels
from mock_data_generator import generate_sleep_night


class SleepStagePredictor:
    """
    Real-time sleep stage predictor.

    This class would be ported to Kotlin for the Android app.
    """

    def __init__(self, model_path: str = 'models/sleep_stage_model.pkl'):
        """Load the trained model."""
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data.get('scaler')
        self.feature_columns = model_data['feature_columns']

        print(f"Model loaded! Classes: {list(self.label_encoder.classes_)}")

    def predict(self, features: dict) -> Tuple[str, dict]:
        """
        Predict sleep stage from features.

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (predicted_stage, probabilities)
        """
        # Create feature vector in correct order
        feature_vector = []
        for col in self.feature_columns:
            value = features.get(col, np.nan)
            feature_vector.append(value)

        # Handle missing values
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        # Scale features if scaler is available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        # Predict
        prediction_encoded = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

        # Create probability dictionary
        prob_dict = {
            self.label_encoder.classes_[i]: prob
            for i, prob in enumerate(probabilities)
        }

        return prediction, prob_dict


class SmartAlarm:
    """
    Smart alarm logic for binary classification.

    Decides when to trigger the alarm based on sleep stage and time window.
    """

    def __init__(
        self,
        target_time_hours: float,
        window_minutes: int = 30
    ):
        """
        Initialize smart alarm.

        Args:
            target_time_hours: Target wake time (hours since sleep start)
            window_minutes: Minutes before target to start looking for good time
        """
        self.target_time = target_time_hours
        self.window_start = target_time_hours - (window_minutes / 60)

        print(f"\nSmart Alarm configured:")
        print(f"  Target time: {target_time_hours:.1f} hours after sleep")
        print(f"  Window: starts at {self.window_start:.2f} hours")
        print(f"  Wake when: 'Light' (Wake, N1, N2)")

    def should_trigger(
        self,
        current_time_hours: float,
        sleep_stage: str,
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Decide if alarm should trigger.

        Args:
            current_time_hours: Current time (hours since sleep start)
            sleep_stage: Predicted sleep stage ('Deep' or 'Light')
            confidence: Model confidence in prediction

        Returns:
            Tuple of (should_trigger, reason)
        """
        # Before window - never trigger
        if current_time_hours < self.window_start:
            return False, "Before wake window"

        # Past target time - always trigger
        if current_time_hours >= self.target_time:
            return True, f"Target time reached (stage: {sleep_stage})"

        # In window - trigger if Light (OK to wake)
        if sleep_stage == "Light":
            if confidence > 0.6:  # Only if confident
                return True, f"Light sleep detected ({confidence:.0%} confident) - OK to wake!"
            else:
                return False, f"Light but low confidence ({confidence:.0%})"

        return False, f"Deep sleep - waiting for lighter stage"


def simulate_night(
    predictor: SleepStagePredictor,
    alarm: SmartAlarm,
    duration_hours: float = 7.0,
    realtime: bool = False
):
    """
    Simulate a night of sleep with smart alarm.

    Args:
        predictor: Sleep stage predictor
        alarm: Smart alarm configuration
        duration_hours: Total sleep duration
        realtime: If True, add delays for visualization
    """
    print("\n" + "=" * 60)
    print("SIMULATING NIGHT OF SLEEP (Binary: Deep vs Light)")
    print("=" * 60)

    # Generate synthetic sleep data
    df = generate_sleep_night(duration_hours=duration_hours, seed=42)

    epoch_duration = 60  # seconds (1 minute epoch)
    sampling_freq = 64
    samples_per_epoch = epoch_duration * sampling_freq
    total_epochs = len(df) // samples_per_epoch

    print(f"\nTotal epochs: {total_epochs} ({total_epochs:.1f} minutes)")
    print("\nStarting simulation...\n")

    alarm_triggered = False

    for epoch_idx in range(total_epochs):
        start_idx = epoch_idx * samples_per_epoch
        end_idx = start_idx + samples_per_epoch

        df_epoch = df.iloc[start_idx:end_idx]
        current_time = (epoch_idx * 60) / 3600  # hours (1 min epochs)

        # Extract features (this is what the app would do)
        features = extract_epoch_features(df_epoch, epoch_idx, total_epochs)

        # Predict sleep stage
        predicted_stage, probabilities = predictor.predict(features)
        confidence = probabilities.get(predicted_stage, 0)

        # Get actual stage (for comparison) - simplified to binary
        actual_stage = df_epoch['Sleep_Stage'].mode().iloc[0]
        actual_binary = simplify_labels(np.array([actual_stage]), 'binary')[0]

        # Check alarm
        should_wake, reason = alarm.should_trigger(current_time, predicted_stage, confidence)

        # Display status (every 5 epochs to reduce output)
        if epoch_idx % 5 == 0 or should_wake:
            status = "ALARM!" if should_wake else "zzz"
            match = "OK" if predicted_stage == actual_binary else "X"

            print(f"[{current_time:5.2f}h] {status:6s} "
                  f"Pred: {predicted_stage:8s} ({confidence:4.0%}) "
                  f"Actual: {actual_binary:8s} {match} "
                  f"| {reason}")

        if should_wake and not alarm_triggered:
            alarm_triggered = True
            print("\n" + "=" * 40)
            print("WAKE UP! Smart alarm triggered!")
            print(f"Time: {current_time:.2f} hours into sleep")
            print(f"Stage: {predicted_stage} (confidence: {confidence:.0%})")
            print("=" * 40 + "\n")

            if not realtime:
                break

        if realtime:
            time.sleep(0.1)  # Slow down for visualization

    if not alarm_triggered:
        print("\nSimulation ended without alarm trigger")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def main():
    """Run the demo."""
    # Check if model exists
    model_path = 'models/sleep_stage_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found! Run train.py first.")
        print("Running: python src/train.py")
        os.system("cd /home/afik.s/PycharmProjects/SleepWise && python src/train.py")

    # Load predictor
    predictor = SleepStagePredictor(model_path)

    # Configure smart alarm
    # User wants to wake up after 7 hours, with 30 minute window
    alarm = SmartAlarm(
        target_time_hours=7.0,
        window_minutes=30
    )

    # Run simulation
    simulate_night(predictor, alarm, duration_hours=7.5, realtime=False)


if __name__ == "__main__":
    main()
