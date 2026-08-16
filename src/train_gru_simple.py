"""
Train a simpler GRU model for TFLite compatibility

GRU (Gated Recurrent Unit) is more TFLite-friendly than Bidirectional LSTM.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import glob

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from features import extract_features_from_participant, simplify_labels


def load_all_participants():
    """Load and process data from all participants."""
    data_dirs = [
        '/home/afik.s/Downloads/afik_gal_fp_data',
        '/home/afik.s/Downloads/moredata_sleep'
    ]

    all_features = []
    all_labels = []
    all_participant_ids = []

    pid = 0
    for data_dir in data_dirs:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "participant_*.csv")))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(data_dir, "S*_whole_df.csv")))

        for csv_file in csv_files:
            try:
                print(f"Processing {os.path.basename(csv_file)}...")
                df = pd.read_csv(csv_file)
                features_df, labels = extract_features_from_participant(df, add_temporal=False)
                if len(labels) > 10:
                    features_df['participant_id'] = pid
                    all_features.append(features_df)
                    all_labels.append(labels)
                    all_participant_ids.extend([pid] * len(labels))
                    pid += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"Loaded {pid} participants")

    if not all_features:
        return None, None, None

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels, all_participant_ids


def create_sequences_by_participant(features, labels, participant_ids, sequence_length=10):
    """Create sequences within each participant's data."""
    feature_cols = [c for c in features.columns if c != 'participant_id']
    X = features[feature_cols].values
    y = labels
    participants = np.array(participant_ids)

    sequences = []
    sequence_labels = []

    unique_participants = np.unique(participants)

    for pid in unique_participants:
        mask = participants == pid
        indices = np.where(mask)[0]

        for i in range(len(indices) - sequence_length + 1):
            seq_indices = indices[i:i + sequence_length]
            if np.all(np.diff(seq_indices) == 1):
                sequences.append(X[seq_indices])
                sequence_labels.append(y[seq_indices[-1]])

    return np.array(sequences), np.array(sequence_labels)


def train_gru_model():
    print("=" * 50)
    print("Training Simple GRU Model (TFLite Compatible)")
    print("=" * 50)

    # Load data
    print("\n[1/6] Loading data...")
    features_df, labels, participant_ids = load_all_participants()
    if features_df is None:
        print("No data found!")
        return

    print(f"Total epochs: {len(labels)}")

    # Simplify labels to 4 classes
    labels = simplify_labels(labels, '4class')

    # Filter out any None/unknown labels
    valid_mask = labels != 'Unknown'
    features_df = features_df[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask]
    participant_ids = [pid for pid, valid in zip(participant_ids, valid_mask) if valid]
    print(f"After filtering: {len(labels)} epochs")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    print(f"Classes: {list(le.classes_)}")

    # Get feature columns
    feature_cols = [c for c in features_df.columns if c != 'participant_id']
    print(f"Using {len(feature_cols)} features")

    # Scale features
    scaler = StandardScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

    # Handle NaN
    features_df = features_df.fillna(0)

    # Create sequences
    print("\n[2/6] Creating sequences...")
    seq_length = 10
    X_seq, y_seq = create_sequences_by_participant(features_df, labels_encoded, participant_ids, seq_length)
    print(f"Sequences: {X_seq.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Boost Deep sleep weight
    deep_idx = list(le.classes_).index('Deep')
    class_weights[deep_idx] *= 2.0
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Build simple GRU model (more TFLite compatible)
    print("\n[3/6] Building GRU model...")
    n_features = X_seq.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, n_features)),

        # Single GRU layer (unidirectional for better compatibility)
        tf.keras.layers.GRU(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),

        # Dense layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train
    print("\n[4/6] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[5/6] Evaluating...")
    y_pred = model.predict(X_test).argmax(axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Per-class recall
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class recall:")
    for i, cls in enumerate(le.classes_):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {cls}: {recall*100:.1f}%")

    # Save model
    print("\n[6/6] Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/sleep_stage_gru.keras')

    # Save metadata
    metadata = {
        'sequence_length': seq_length,
        'feature_names': feature_cols,
        'label_encoder': le,
        'scaler': scaler,
        'class_weights': class_weight_dict
    }
    joblib.dump(metadata, 'models/gru_metadata.pkl')

    print("\n" + "=" * 50)
    print("GRU Model Training Complete!")
    print("=" * 50)

    return model, metadata


if __name__ == "__main__":
    train_gru_model()
