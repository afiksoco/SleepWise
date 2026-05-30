"""
Feature Extraction for SleepWise

Extracts meaningful features from raw sensor data for sleep stage classification.
Each 1-minute epoch of raw data becomes one row of features.

Features focused on what Galaxy Watch 5 can provide:
- Heart Rate (continuous)
- Skin Temperature
- (Movement/Accelerometer - not in DREAMT dataset)

Note: HRV features removed as DREAMT dataset has no IBI data.
When using real Galaxy Watch data with IBI, HRV can be added back.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def extract_hr_features(hr_values: np.ndarray) -> Dict[str, float]:
    """
    Extract heart rate features.

    Heart rate patterns during sleep:
    - Wake: Higher, more variable
    - Light sleep: Decreasing
    - Deep sleep: Lowest, most stable
    - REM: Variable, can be higher
    """
    hr_clean = hr_values[~np.isnan(hr_values)]
    hr_clean = hr_clean[(hr_clean > 30) & (hr_clean < 200)]

    if len(hr_clean) < 5:
        return {
            'hr_mean': np.nan,
            'hr_std': np.nan,
            'hr_min': np.nan,
            'hr_max': np.nan,
            'hr_range': np.nan,
            'hr_cv': np.nan,
            'hr_median': np.nan,
            'hr_iqr': np.nan,
            'hr_skew': np.nan,
        }

    mean_hr = np.mean(hr_clean)
    std_hr = np.std(hr_clean)

    # Skewness (asymmetry of distribution)
    skew = np.mean(((hr_clean - mean_hr) / std_hr) ** 3) if std_hr > 0 else 0

    return {
        'hr_mean': mean_hr,
        'hr_std': std_hr,
        'hr_min': np.min(hr_clean),
        'hr_max': np.max(hr_clean),
        'hr_range': np.max(hr_clean) - np.min(hr_clean),
        'hr_cv': (std_hr / mean_hr) * 100 if mean_hr > 0 else np.nan,
        'hr_median': np.median(hr_clean),
        'hr_iqr': np.percentile(hr_clean, 75) - np.percentile(hr_clean, 25),
        'hr_skew': skew,
    }


def extract_temp_features(temp_values: np.ndarray) -> Dict[str, float]:
    """
    Extract temperature features.

    Body temperature during sleep:
    - Generally decreases during sleep onset
    - Lowest during deep sleep
    - Rises before waking
    """
    temp_clean = temp_values[~np.isnan(temp_values)]
    temp_clean = temp_clean[(temp_clean > 30) & (temp_clean < 40)]

    if len(temp_clean) < 5:
        return {
            'temp_mean': np.nan,
            'temp_std': np.nan,
            'temp_trend': np.nan,
        }

    # Calculate trend (slope)
    x = np.arange(len(temp_clean))
    trend = np.polyfit(x, temp_clean, 1)[0] if len(temp_clean) > 1 else 0

    return {
        'temp_mean': np.mean(temp_clean),
        'temp_std': np.std(temp_clean),
        'temp_trend': trend,
    }


def extract_acc_features(
    acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
) -> Dict[str, float]:
    """
    Extract accelerometer movement features, normalized to be comparable
    across the Empatica E4 (training) and Galaxy Watch (inference).

    Both devices are reduced to GRAVITY-REMOVED magnitude in units of g:
    - E4 raw ACC is 1/64 g per LSB, gravity included → divide by 64, then
      subtract the per-epoch median (robust gravity + posture baseline).
    - The Galaxy Watch already sends |sqrt(x^2+y^2+z^2) - 9.81| in m/s^2,
      which the Android side divides by 9.81 → gravity-removed g.

    Features are scale- and rate-robust on purpose (E4 is 32-64 Hz, the watch
    delivers far fewer samples/epoch):
    - acc_std:        std of gravity-removed magnitude (g)
    - acc_move_ratio: fraction of samples whose motion exceeds MOVE_THRESH_G
    """
    MOVE_THRESH_G = 0.02
    mag = np.sqrt(acc_x.astype(float) ** 2 + acc_y.astype(float) ** 2 + acc_z.astype(float) ** 2)
    mag = mag[~np.isnan(mag)]
    if len(mag) < 5:
        return {'acc_std': 0.0, 'acc_move_ratio': 0.0}
    mag_g = mag / 64.0                      # E4 LSB → g
    motion = np.abs(mag_g - np.median(mag_g))  # gravity + posture removed
    return {
        'acc_std': float(np.std(motion)),
        'acc_move_ratio': float(np.mean(motion > MOVE_THRESH_G)),
    }


def extract_epoch_features(
    df_epoch: pd.DataFrame,
    epoch_index: int = 0,
    total_epochs: int = 1
) -> Dict[str, float]:
    """
    Extract all features from a single epoch.

    Returns features for: HR (9), Temperature (3), Accelerometer (2) = 14 base
    features. (HRV removed - watch can't supply it live.)
    """
    features = {}

    # Heart rate features (9)
    features.update(extract_hr_features(df_epoch['HR'].values))

    # Temperature features (3)
    features.update(extract_temp_features(df_epoch['TEMP'].values))

    # Accelerometer movement features (2) — present in DREAMT (ACC_X/Y/Z) and
    # streamed live from the Galaxy Watch.
    if all(c in df_epoch.columns for c in ('ACC_X', 'ACC_Y', 'ACC_Z')):
        features.update(extract_acc_features(
            df_epoch['ACC_X'].values, df_epoch['ACC_Y'].values, df_epoch['ACC_Z'].values
        ))
    else:
        features.update({'acc_std': 0.0, 'acc_move_ratio': 0.0})

    return features


def add_temporal_features(features_df: pd.DataFrame, lookback: int = 4) -> pd.DataFrame:
    """
    Add temporal features based on previous epochs.

    This helps the model understand patterns over time:
    - Is HR going up or down?
    - What was the pattern in the last few minutes?
    """
    df = features_df.copy()

    # Key features to track over time (HR and Temp only - available from Galaxy Watch)
    key_features = ['hr_mean', 'temp_mean']

    for feature in key_features:
        if feature not in df.columns:
            continue

        # Previous epoch values (lag features)
        for lag in range(1, lookback + 1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

        # Rolling statistics (trends)
        df[f'{feature}_rolling_mean'] = df[feature].rolling(window=lookback, min_periods=1).mean()
        df[f'{feature}_rolling_std'] = df[feature].rolling(window=lookback, min_periods=1).std()

        # Trend direction (positive = increasing, negative = decreasing)
        df[f'{feature}_trend'] = df[feature].diff(periods=lookback)

        # Rate of change
        df[f'{feature}_roc'] = df[feature].pct_change(periods=lookback)

    # HR stability (low std = stable = likely deep sleep)
    df['hr_stability'] = df['hr_mean'].rolling(window=lookback, min_periods=1).std()

    # Sleep cycle position estimate (90 min = 90 epochs of 1 min)
    df['sleep_cycle_position'] = (df.index % 90) / 90  # 0-1 within 90-min cycle

    # Fill NaN values from lag features with forward/backward fill
    df = df.bfill().ffill().fillna(0)

    return df


def extract_features_from_participant(
    df: pd.DataFrame,
    epoch_duration_sec: int = 60,
    sampling_freq: int = 64,
    add_temporal: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features from a full night of sleep data.

    Args:
        df: DataFrame with raw sensor data (DREAMT format)
        epoch_duration_sec: Duration of each epoch in seconds (60s = 1 minute)
        sampling_freq: Sampling frequency in Hz
        add_temporal: Whether to add temporal features (recommended)

    Returns:
        Tuple of (features_df, labels_array)
    """
    samples_per_epoch = epoch_duration_sec * sampling_freq
    total_samples = len(df)
    total_epochs = total_samples // samples_per_epoch

    all_features = []
    all_labels = []

    for epoch_idx in range(total_epochs):
        start_idx = epoch_idx * samples_per_epoch
        end_idx = start_idx + samples_per_epoch

        df_epoch = df.iloc[start_idx:end_idx]

        # Extract features
        features = extract_epoch_features(df_epoch, epoch_idx, total_epochs)
        all_features.append(features)

        # Get majority label for this epoch
        label = df_epoch['Sleep_Stage'].mode().iloc[0]
        all_labels.append(label)

    features_df = pd.DataFrame(all_features)
    labels_array = np.array(all_labels)

    # Add temporal features (previous epochs, trends)
    if add_temporal:
        features_df = add_temporal_features(features_df, lookback=4)

    return features_df, labels_array


def simplify_labels(labels: np.ndarray, scheme: str = 'binary') -> np.ndarray:
    """
    Simplify sleep stage labels.

    Args:
        labels: Original labels (W, N1, N2, N3, R)
        scheme: Simplification scheme
            - 'binary': Light (W+N1+N2), Deep (N3+R)

    Returns:
        Simplified labels
    """
    # Convert to object dtype to allow longer strings
    simplified = labels.astype(object)

    # Handle special labels from DREAMT dataset
    simplified[simplified == 'P'] = 'W'
    simplified[simplified == 'Missing'] = 'W'

    if scheme == 'binary':
        # Good to wake vs bad to wake (original: REM grouped with deep)
        simplified[simplified == 'W'] = 'Light'
        simplified[simplified == 'N1'] = 'Light'
        simplified[simplified == 'N2'] = 'Light'
        simplified[simplified == 'N3'] = 'Deep'
        simplified[simplified == 'R'] = 'Deep'
    elif scheme == 'binary_n3':
        # Deep = N3 ONLY. Everything else (Wake, N1, N2, REM) is wakeable.
        # REM is easy to rouse from and moves like light sleep, so it belongs
        # with Light for a smart-wake alarm — this also lets the accelerometer
        # cleanly separate the (very still) N3 from the rest.
        simplified[simplified == 'W'] = 'Light'
        simplified[simplified == 'N1'] = 'Light'
        simplified[simplified == 'N2'] = 'Light'
        simplified[simplified == 'R'] = 'Light'
        simplified[simplified == 'N3'] = 'Deep'

    return simplified


if __name__ == "__main__":
    from mock_data_generator import generate_sleep_night

    print("Generating test data...")
    df = generate_sleep_night(duration_hours=1, seed=42)
    print(f"Raw data shape: {df.shape}")

    print("\nExtracting features...")
    features_df, labels = extract_features_from_participant(df)
    print(f"Features shape: {features_df.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\nFeature columns:")
    print(features_df.columns.tolist())

    print("\nLabel distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    print("\nSimplified (binary):")
    simplified = simplify_labels(labels, 'binary')
    unique, counts = np.unique(simplified, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
