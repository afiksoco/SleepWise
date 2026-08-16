"""
Train Neural Network for TFLite Export

Creates a Keras model that can be converted to TensorFlow Lite
for on-device inference on Android.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

from features import extract_features_from_participant, simplify_labels

# Configuration
DATA_DIR = Path('data/dreamt')
MODEL_DIR = Path('models')
LABEL_SCHEME = 'binary'
RANDOM_STATE = 42


def load_all_data(data_dir: Path):
    """Load and process all participant data."""
    all_features = []
    all_labels = []
    participant_ids = []

    csv_files = sorted(data_dir.glob('*_whole_df.csv'))

    for csv_file in csv_files:
        participant_id = csv_file.stem.replace('_whole_df', '')
        print(f"Processing {csv_file.name}...")

        df = pd.read_csv(csv_file)

        # Extract features
        features_df, labels = extract_features_from_participant(df)

        # Add participant ID for later splitting
        features_df['participant_id'] = participant_id

        all_features.append(features_df)
        all_labels.extend(labels)
        participant_ids.extend([participant_id] * len(labels))

    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)

    print(f"Total samples: {len(y)}")
    return X, y


def create_model(input_dim: int, num_classes: int = 2):
    """Create a neural network model (TFLite compatible)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def main():
    print("=" * 50)
    print("TFLite Model Training")
    print("=" * 50)

    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)

    # Load data
    print("\n[1/6] Loading data...")
    X, y = load_all_data(DATA_DIR)

    # Simplify labels
    print(f"\n[2/6] Simplifying labels ({LABEL_SCHEME})...")
    y = simplify_labels(y, LABEL_SCHEME)

    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

    # Split by participant
    print("\n[3/6] Splitting data...")
    participants = X['participant_id'].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(participants)

    split_idx = int(len(participants) * 0.8)
    train_participants = participants[:split_idx]
    test_participants = participants[split_idx:]

    train_mask = X['participant_id'].isin(train_participants)
    test_mask = X['participant_id'].isin(test_participants)

    X_train = X[train_mask].drop('participant_id', axis=1)
    X_test = X[test_mask].drop('participant_id', axis=1)
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Get feature columns (important for Android app)
    feature_columns = list(X_train.columns)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"Classes: {list(label_encoder.classes_)}")

    # Scale features
    print("\n[4/6] Preparing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))

    # Compute class weights - boost Deep significantly for smart alarm use case
    # We want HIGH Deep recall (don't wake during deep sleep)
    class_counts = np.bincount(y_train_encoded)
    total = len(y_train_encoded)
    class_weights = {
        0: total / (2 * class_counts[0]) * 2.0,  # Deep - 2x boost for high recall
        1: total / (2 * class_counts[1]) * 0.5,  # Light - reduce weight
    }

    print(f"Class weights: {class_weights}")

    # Create and compile model
    print("\n[5/6] Training neural network...")
    model = create_model(len(feature_columns), num_classes=2)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train with early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    print("\n[6/6] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(
        y_test_encoded, y_pred_classes,
        target_names=label_encoder.classes_
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    print(f"       {label_encoder.classes_[0]:>5s}  {label_encoder.classes_[1]:>5s}")
    for i, row in enumerate(cm):
        print(f"{label_encoder.classes_[i]:5s}  {row[0]:5d}  {row[1]:5d}")

    # Convert to TFLite
    print("\n" + "=" * 50)
    print("EXPORTING TO TFLITE")
    print("=" * 50)

    # Convert model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = MODEL_DIR / 'sleep_stage_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # Save metadata for Android app
    metadata = {
        'feature_names': feature_columns,
        'class_names': list(label_encoder.classes_),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'input_shape': [1, len(feature_columns)],
        'output_shape': [1, 2],
    }

    metadata_path = MODEL_DIR / 'tflite_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    print("\n" + "=" * 50)
    print("DONE! Copy these files to Android app:")
    print(f"  {tflite_path}")
    print(f"  {metadata_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
