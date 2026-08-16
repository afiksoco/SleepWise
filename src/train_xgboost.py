"""
XGBoost Training for SleepWise

XGBoost typically handles class imbalance better than Random Forest.
"""

import os
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter

from features import extract_features_from_participant, simplify_labels


def load_all_participants(data_dir: str):
    """Load and process data from all participants."""
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
        features_df['participant_id'] = i
        all_features.append(features_df)
        all_labels.append(labels)
        all_participant_ids.extend([i] * len(labels))

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels, all_participant_ids


def train_test_split_by_participant(features, labels, participant_ids, test_size=0.2, random_state=42):
    """Split data by participant to avoid data leakage."""
    unique_participants = list(set(participant_ids))
    np.random.seed(random_state)
    np.random.shuffle(unique_participants)

    n_test = max(1, int(len(unique_participants) * test_size))
    test_participants = set(unique_participants[:n_test])
    train_participants = set(unique_participants[n_test:])

    participant_array = np.array(participant_ids)
    train_mask = np.isin(participant_array, list(train_participants))
    test_mask = np.isin(participant_array, list(test_participants))

    feature_cols = [c for c in features.columns if c != 'participant_id']

    X_train = features.loc[train_mask, feature_cols]
    X_test = features.loc[test_mask, feature_cols]
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    print(f"Train participants: {len(train_participants)}, Test participants: {len(test_participants)}")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    return X_train, X_test, y_train, y_test


def main():
    print("=" * 50)
    print("SleepWise XGBoost Training")
    print("=" * 50)

    DATA_DIR = 'data/dreamt'
    LABEL_SCHEME = '4class'

    # Load data
    print("\n[1/5] Loading data...")
    features, labels, participant_ids = load_all_participants(DATA_DIR)
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
        features, labels, participant_ids, test_size=0.2, random_state=42
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Handle missing values
    X_train_clean = X_train.fillna(X_train.median())
    X_test_clean = X_test.fillna(X_train.median())

    # Calculate class weights for XGBoost
    class_counts = Counter(y_train_encoded)
    total = len(y_train_encoded)
    n_classes = len(class_counts)

    # Scale weights: most frequent class gets weight 1
    max_count = max(class_counts.values())
    scale_pos_weights = {cls: max_count / count for cls, count in class_counts.items()}

    print(f"\n[4/5] Training XGBoost...")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Sample weights scale: {scale_pos_weights}")

    # Create sample weights
    sample_weights = np.array([scale_pos_weights[y] for y in y_train_encoded])

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    model.fit(X_train_clean, y_train_encoded, sample_weight=sample_weights)
    print("Training complete!")

    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred_encoded = model.predict(X_test_clean)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
    print(f"       {' '.join([f'{c:>5}' for c in label_encoder.classes_])}")
    for i, row in enumerate(cm):
        print(f"{label_encoder.classes_[i]:>5} {' '.join([f'{v:>5}' for v in row])}")

    # Feature importance
    print("\nTop 10 Important Features:")
    feature_names = X_train_clean.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10).to_string(index=False))

    # Save model
    os.makedirs('models', exist_ok=True)
    output_path = 'models/sleep_stage_xgboost.pkl'
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }, output_path)
    print(f"\nModel saved to: {output_path}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    return accuracy


if __name__ == "__main__":
    main()
