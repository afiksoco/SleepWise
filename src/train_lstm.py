"""
LSTM Training for SleepWise

LSTM (Long Short-Term Memory) uses sequences of epochs to learn temporal patterns.
This is ideal for sleep stage prediction because:
- Sleep stages transition gradually (N1→N2→N3→N2→REM)
- Physiological patterns over time are predictive
- 90-minute sleep cycles have recognizable patterns
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        # Don't add temporal features - LSTM will learn these
        features_df, labels = extract_features_from_participant(df, add_temporal=False)
        features_df['participant_id'] = i
        all_features.append(features_df)
        all_labels.append(labels)
        all_participant_ids.extend([i] * len(labels))

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels, all_participant_ids


def create_sequences(features, labels, participant_ids, sequence_length=10):
    """
    Create sequences for LSTM input.

    Each sequence contains `sequence_length` consecutive epochs.
    We only create sequences within the same participant to avoid mixing data.

    Args:
        features: DataFrame of features
        labels: Array of labels
        participant_ids: List of participant IDs
        sequence_length: Number of epochs per sequence (10 = 5 minutes)

    Returns:
        X_sequences, y_sequences (labels for last epoch in each sequence)
    """
    feature_cols = [c for c in features.columns if c != 'participant_id']
    X = features[feature_cols].values
    y = labels
    participants = np.array(participant_ids)

    sequences = []
    sequence_labels = []
    sequence_participants = []

    unique_participants = np.unique(participants)

    for pid in unique_participants:
        # Get indices for this participant
        mask = participants == pid
        indices = np.where(mask)[0]

        # Create sequences within this participant's data
        for i in range(len(indices) - sequence_length + 1):
            seq_indices = indices[i:i + sequence_length]

            # Only create sequence if indices are consecutive
            if np.all(np.diff(seq_indices) == 1):
                sequences.append(X[seq_indices])
                sequence_labels.append(y[seq_indices[-1]])  # Label of last epoch
                sequence_participants.append(pid)

    return np.array(sequences), np.array(sequence_labels), np.array(sequence_participants)


def train_test_split_by_participant(X_seq, y_seq, participants, test_size=0.2, random_state=42):
    """Split sequences by participant."""
    unique_participants = list(set(participants))
    np.random.seed(random_state)
    np.random.shuffle(unique_participants)

    n_test = max(1, int(len(unique_participants) * test_size))
    test_participants = set(unique_participants[:n_test])
    train_participants = set(unique_participants[n_test:])

    train_mask = np.isin(participants, list(train_participants))
    test_mask = np.isin(participants, list(test_participants))

    X_train = X_seq[train_mask]
    X_test = X_seq[test_mask]
    y_train = y_seq[train_mask]
    y_test = y_seq[test_mask]

    print(f"Train participants: {len(train_participants)}, Test participants: {len(test_participants)}")
    print(f"Train sequences: {len(y_train)}, Test sequences: {len(y_test)}")

    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape, n_classes):
    """
    Build LSTM model for sleep stage classification.

    Architecture:
    - Bidirectional LSTM: Looks at sequence both forward and backward
    - Dropout: Prevents overfitting
    - Dense layers: Classification head
    """
    model = Sequential([
        # First LSTM layer - learns sequence patterns
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),

        # Second LSTM layer - learns higher-level patterns
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),

        # Classification head
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),

        # Output layer
        Dense(n_classes, activation='softmax')
    ])

    return model


def main():
    print("=" * 50)
    print("SleepWise LSTM Training")
    print("=" * 50)

    DATA_DIR = 'data/dreamt'
    LABEL_SCHEME = '4class'
    SEQUENCE_LENGTH = 10  # 10 epochs = 5 minutes of context

    # Load data
    print("\n[1/6] Loading data...")
    features, labels, participant_ids = load_all_participants(DATA_DIR)
    print(f"Total epochs: {len(labels)}")
    print(f"Features per epoch: {len([c for c in features.columns if c != 'participant_id'])}")

    # Simplify labels
    print(f"\n[2/6] Simplifying labels ({LABEL_SCHEME})...")
    labels = simplify_labels(labels, LABEL_SCHEME)
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    # Create sequences
    print(f"\n[3/6] Creating sequences (length={SEQUENCE_LENGTH})...")
    X_seq, y_seq, seq_participants = create_sequences(
        features, labels, participant_ids, sequence_length=SEQUENCE_LENGTH
    )
    print(f"Total sequences: {len(y_seq)}")
    print(f"Sequence shape: {X_seq.shape}")  # (n_sequences, sequence_length, n_features)

    # Split by participant
    print("\n[4/6] Splitting by participant...")
    X_train, X_test, y_train, y_test = train_test_split_by_participant(
        X_seq, y_seq, seq_participants, test_size=0.2, random_state=42
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)

    print(f"Classes: {label_encoder.classes_}")

    # Convert to categorical (one-hot)
    y_train_cat = to_categorical(y_train_encoded, n_classes)
    y_test_cat = to_categorical(y_test_encoded, n_classes)

    # Normalize features
    print("\nNormalizing features...")
    n_samples_train, seq_len, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    # Reshape to 2D for scaling
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples_train, seq_len, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_samples_test, seq_len, n_features)

    # Handle NaN values
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    # Calculate class weights
    class_counts = Counter(y_train_encoded)
    total = len(y_train_encoded)
    class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
    print(f"Class weights: {class_weights}")

    # Build model
    print("\n[5/6] Building and training LSTM...")
    input_shape = (seq_len, n_features)
    model = build_lstm_model(input_shape, n_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[6/6] Evaluating model...")
    y_pred_proba = model.predict(X_test_scaled)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
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

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/sleep_stage_lstm.keras')
    joblib.dump({
        'label_encoder': label_encoder,
        'scaler': scaler,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_names': [c for c in features.columns if c != 'participant_id']
    }, 'models/lstm_metadata.pkl')

    print(f"\nModel saved to: models/sleep_stage_lstm.keras")
    print(f"Metadata saved to: models/lstm_metadata.pkl")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    return accuracy


if __name__ == "__main__":
    main()
