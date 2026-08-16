"""
Mock Data Generator for SleepWise POC
Generates synthetic sleep data that mimics the DREAMT dataset format.

This allows us to test the full pipeline before downloading the real dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Sleep stage characteristics (realistic ranges)
SLEEP_STAGE_PARAMS = {
    'W': {  # Wake
        'hr_mean': 75, 'hr_std': 8,
        'movement_mean': 0.5, 'movement_std': 0.3,
        'hrv_mean': 50, 'hrv_std': 15,
        'temp_mean': 36.0, 'temp_std': 0.3,
    },
    'N1': {  # Light sleep stage 1
        'hr_mean': 65, 'hr_std': 5,
        'movement_mean': 0.2, 'movement_std': 0.15,
        'hrv_mean': 55, 'hrv_std': 12,
        'temp_mean': 35.5, 'temp_std': 0.2,
    },
    'N2': {  # Light sleep stage 2
        'hr_mean': 60, 'hr_std': 4,
        'movement_mean': 0.1, 'movement_std': 0.1,
        'hrv_mean': 60, 'hrv_std': 10,
        'temp_mean': 35.3, 'temp_std': 0.2,
    },
    'N3': {  # Deep sleep
        'hr_mean': 55, 'hr_std': 3,
        'movement_mean': 0.05, 'movement_std': 0.05,
        'hrv_mean': 70, 'hrv_std': 8,
        'temp_mean': 35.0, 'temp_std': 0.15,
    },
    'R': {  # REM
        'hr_mean': 68, 'hr_std': 10,
        'movement_mean': 0.15, 'movement_std': 0.2,
        'hrv_mean': 45, 'hrv_std': 18,
        'temp_mean': 35.4, 'temp_std': 0.25,
    },
}

# Typical sleep cycle pattern (90 minutes per cycle)
SLEEP_CYCLE_PATTERN = ['W', 'N1', 'N2', 'N3', 'N2', 'R']
EPOCHS_PER_STAGE = [2, 3, 8, 6, 4, 7]  # ~30 epochs per cycle (15 min)


def generate_sleep_night(
    duration_hours: float = 7.0,
    sampling_freq: int = 64,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate one night of synthetic sleep data.

    Args:
        duration_hours: Length of sleep in hours
        sampling_freq: Samples per second (64Hz like DREAMT)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns matching DREAMT format
    """
    if seed is not None:
        np.random.seed(seed)

    total_seconds = int(duration_hours * 3600)
    total_samples = total_seconds * sampling_freq
    epoch_duration = 30  # seconds per epoch (sleep stage label)
    samples_per_epoch = epoch_duration * sampling_freq
    total_epochs = total_seconds // epoch_duration

    # Generate sleep stage sequence
    stages = []
    while len(stages) < total_epochs:
        # Add some randomness to cycle pattern
        cycle = SLEEP_CYCLE_PATTERN.copy()
        for stage, base_epochs in zip(cycle, EPOCHS_PER_STAGE):
            n_epochs = max(1, base_epochs + np.random.randint(-2, 3))
            stages.extend([stage] * n_epochs)

    stages = stages[:total_epochs]

    # Generate raw signals
    timestamps = np.arange(total_samples) / sampling_freq
    hr_values = np.zeros(total_samples)
    acc_x = np.zeros(total_samples)
    acc_y = np.zeros(total_samples)
    acc_z = np.zeros(total_samples)
    temp_values = np.zeros(total_samples)
    bvp_values = np.zeros(total_samples)
    eda_values = np.zeros(total_samples)
    ibi_values = np.zeros(total_samples)
    sleep_stages = np.empty(total_samples, dtype=object)

    for epoch_idx, stage in enumerate(stages):
        start_sample = epoch_idx * samples_per_epoch
        end_sample = min(start_sample + samples_per_epoch, total_samples)

        params = SLEEP_STAGE_PARAMS[stage]
        n_samples = end_sample - start_sample

        # Heart rate (1 Hz in real data, but we'll generate at 64 Hz for simplicity)
        hr_values[start_sample:end_sample] = np.random.normal(
            params['hr_mean'], params['hr_std'], n_samples
        )

        # Accelerometer (movement)
        movement = np.random.exponential(params['movement_mean'], n_samples)
        acc_x[start_sample:end_sample] = movement * np.random.randn(n_samples)
        acc_y[start_sample:end_sample] = movement * np.random.randn(n_samples)
        acc_z[start_sample:end_sample] = 1 + movement * np.random.randn(n_samples) * 0.1

        # Temperature
        temp_values[start_sample:end_sample] = np.random.normal(
            params['temp_mean'], params['temp_std'], n_samples
        )

        # BVP (simplified sine wave with noise)
        t = np.arange(n_samples) / sampling_freq
        hr_for_bvp = params['hr_mean'] / 60  # Convert BPM to Hz
        bvp_values[start_sample:end_sample] = (
            np.sin(2 * np.pi * hr_for_bvp * t) +
            np.random.randn(n_samples) * 0.2
        )

        # EDA (slowly varying)
        eda_values[start_sample:end_sample] = np.random.exponential(0.1, n_samples)

        # IBI (inter-beat interval in ms)
        ibi_values[start_sample:end_sample] = 60000 / hr_values[start_sample:end_sample]

        # Sleep stage label
        sleep_stages[start_sample:end_sample] = stage

    # Clip to valid ranges
    hr_values = np.clip(hr_values, 40, 120)
    temp_values = np.clip(temp_values, 33, 38)

    # Create DataFrame
    df = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'BVP': bvp_values,
        'ACC_X': acc_x,
        'ACC_Y': acc_y,
        'ACC_Z': acc_z,
        'TEMP': temp_values,
        'EDA': eda_values,
        'HR': hr_values,
        'IBI': ibi_values,
        'Sleep_Stage': sleep_stages,
    })

    return df


def generate_dataset(
    n_participants: int = 10,
    output_dir: str = 'data/mock',
    seed: int = 42
) -> None:
    """
    Generate mock dataset with multiple participants.

    Args:
        n_participants: Number of synthetic participants
        output_dir: Directory to save CSV files
        seed: Base random seed
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_participants):
        print(f"Generating participant {i+1}/{n_participants}...")

        # Vary sleep duration between participants
        duration = np.random.uniform(5.5, 8.0)

        df = generate_sleep_night(
            duration_hours=duration,
            seed=seed + i
        )

        filename = f"{output_dir}/participant_{i+1:03d}.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename} ({len(df)} samples, {duration:.1f} hours)")

    print(f"\nDone! Generated {n_participants} participants in {output_dir}/")


if __name__ == "__main__":
    generate_dataset(n_participants=10)
