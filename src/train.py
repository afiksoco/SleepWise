"""
Model Training for SleepWise

Trains a Random Forest classifier to predict sleep stages from sensor features.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from typing import Tuple, List

from features import extract_features_from_participant, simplify_labels


def load_all_participants(data_dir: str) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Load and process data from all participants.

    Args:
        data_dir: Directory containing participant CSV files

    Returns:
        Tuple of (features_df, labels_array, participant_ids)
    """
    # Support both mock data (participant_*.csv) and real DREAMT data (S*_whole_df.csv)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "participant_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "S*_whole_df.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No participant files found in {data_dir}")

    all_features = []
    all_labels = []
    all_participant_ids = []

    for i, csv_file in enumerate(csv_files):
        print(f"Processing {os.path.basename(csv_file)}...")

        df = pd.read_csv(csv_file)
        features_df, labels = extract_features_from_participant(df)

        # Add participant ID for later splitting
        features_df['participant_id'] = i
        all_features.append(features_df)
        all_labels.append(labels)
        all_participant_ids.extend([i] * len(labels))

    # Combine all participants
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels, all_participant_ids


def train_test_split_by_participant(
    features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data by participant (not by sample) for proper evaluation.

    This is CRITICAL! If we split by sample, the same person's data
    could be in both train and test, leading to overly optimistic results.

    Args:
        features: Feature DataFrame with 'participant_id' column
        labels: Label array
        test_size: Fraction of participants to use for testing
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)

    participant_ids = features['participant_id'].unique()
    n_test = max(1, int(len(participant_ids) * test_size))

    # Randomly select test participants
    test_participants = np.random.choice(participant_ids, n_test, replace=False)
    train_participants = [p for p in participant_ids if p not in test_participants]

    print(f"Train participants: {len(train_participants)}, Test participants: {len(test_participants)}")

    # Split data
    train_mask = features['participant_id'].isin(train_participants)
    test_mask = features['participant_id'].isin(test_participants)

    X_train = features[train_mask].drop('participant_id', axis=1)
    X_test = features[test_mask].drop('participant_id', axis=1)
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 1000,
    max_depth: int = 6,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, LabelEncoder, StandardScaler]:
    """
    Train a Random Forest classifier.

    Why Random Forest?
    - Works well with tabular data (our features)
    - Handles missing values reasonably
    - Provides feature importance (interpretability)
    - Relatively fast training and inference
    - Good out-of-the-box performance

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained model, label encoder, and scaler
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    print(f"\nTraining Random Forest with {n_estimators} trees...")
    print(f"Classes: {label_encoder.classes_}")

    # Handle missing values
    X_train_clean = X_train.fillna(X_train.median())

    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Manual class weights for sleep stages (binary scheme)
    # Higher weight for Deep to avoid waking during deep sleep/REM
    # Balance: want ~50% Deep recall while maintaining good Light recall
    class_weights = {
        'Light': 1.0,   # Wake + N1 + N2 - OK to wake
        'Deep': 8.0,    # N3 + REM - important not to wake during
    }

    # Convert class names to encoded indices
    encoded_weights = {
        label_encoder.transform([cls])[0]: weight
        for cls, weight in class_weights.items()
        if cls in label_encoder.classes_
    }

    print(f"Class weights: {class_weights}")

    # Train model with manual class weights
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=encoded_weights,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train_scaled, y_encoded)

    print("Training complete!")

    return model, label_encoder, scaler


def evaluate_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        label_encoder: Label encoder
        scaler: Feature scaler
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Handle missing values and scale
    X_test_clean = X_test.fillna(X_test.median())
    X_test_scaled = scaler.transform(X_test_clean)

    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test_encoded, y_pred,
        target_names=label_encoder.classes_
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    print(cm_df)

    # Feature importance
    print("\nTop 10 Important Features:")
    importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'feature_importance': importance
    }


def save_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    feature_columns: List[str],
    output_dir: str = 'models'
) -> str:
    """
    Save trained model and metadata.

    Args:
        model: Trained model
        label_encoder: Label encoder
        scaler: Feature scaler
        feature_columns: List of feature names (order matters!)
        output_dir: Directory to save model

    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)

    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_columns': feature_columns,
    }

    output_path = os.path.join(output_dir, 'sleep_stage_model.pkl')
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")

    # Also save as readable text
    info_path = os.path.join(output_dir, 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write("SleepWise Sleep Stage Classification Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n")
        f.write(f"Number of features: {len(feature_columns)}\n")
        f.write(f"Feature columns:\n")
        for col in feature_columns:
            f.write(f"  - {col}\n")

    return output_path


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("SleepWise Model Training")
    print("=" * 50)

    # Configuration
    DATA_DIR = '/home/afik.s/PycharmProjects/SleepWise/data/dreamt'  # Real DREAMT data (use 'data/mock' for synthetic)
    LABEL_SCHEME = 'binary'  # Options: '4class', '3class', 'binary'

    # Load data
    print("\n[1/5] Loading data...")
    features, labels, _ = load_all_participants(DATA_DIR)
    print(f"Total samples: {len(labels)}")

    # Simplify labels
    print(f"\n[2/5] Simplifying labels ({LABEL_SCHEME})...")
    labels = simplify_labels(labels, LABEL_SCHEME)
    print("Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    # Split by participant
    print("\n[3/5] Splitting data by participant...")
    X_train, X_test, y_train, y_test = train_test_split_by_participant(
        features, labels, test_size=0.2
    )
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Train model
    print("\n[4/5] Training model...")
    model, label_encoder, scaler = train_model(X_train, y_train)

    # Evaluate
    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(model, label_encoder, scaler, X_test, y_test)

    # Save model
    feature_columns = [c for c in X_train.columns if c != 'participant_id']
    save_model(model, label_encoder, scaler, feature_columns)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
