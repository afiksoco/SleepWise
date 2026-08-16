"""
Export LSTM model to TensorFlow Lite for Android

TFLite is a lightweight format that runs on mobile devices.
"""

import os
import numpy as np
import tensorflow as tf
import joblib

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def export_to_tflite():
    print("=" * 50)
    print("Exporting LSTM to TensorFlow Lite")
    print("=" * 50)

    # Load the trained Keras model
    print("\n[1/4] Loading Keras model...")
    model = tf.keras.models.load_model('models/sleep_stage_lstm.keras')
    model.summary()

    # Load metadata
    metadata = joblib.load('models/lstm_metadata.pkl')
    print(f"\nModel info:")
    print(f"  Sequence length: {metadata['sequence_length']} epochs")
    print(f"  Features: {len(metadata['feature_names'])}")
    print(f"  Classes: {list(metadata['label_encoder'].classes_)}")

    # Convert to TFLite
    print("\n[2/4] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable Select TF ops for LSTM compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    # Force older op versions for compatibility
    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_lower_tensor_list_ops = False

    # Don't use default optimizations which can bump op versions
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    tflite_model = converter.convert()

    # Save TFLite model
    print("\n[3/4] Saving TFLite model...")
    tflite_path = 'models/sleep_stage_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Get file sizes
    keras_size = os.path.getsize('models/sleep_stage_lstm.keras') / 1024 / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024 / 1024

    print(f"\n  Keras model:  {keras_size:.2f} MB")
    print(f"  TFLite model: {tflite_size:.2f} MB")
    print(f"  Compression:  {(1 - tflite_size/keras_size) * 100:.1f}% smaller")

    # Note: Verification requires Flex delegate which isn't available locally
    # The model will work on Android with tensorflow-lite-select-tf-ops
    print("\n[4/4] Model exported successfully!")
    print("  (Verification skipped - requires Flex delegate on Android)")

    # Save metadata for Android
    android_metadata = {
        'sequence_length': metadata['sequence_length'],
        'feature_names': metadata['feature_names'],
        'class_names': list(metadata['label_encoder'].classes_),
        'scaler_mean': metadata['scaler'].mean_.tolist(),
        'scaler_scale': metadata['scaler'].scale_.tolist(),
    }

    import json
    metadata_path = 'models/tflite_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(android_metadata, f, indent=2)

    print(f"\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nFiles created:")
    print(f"  1. {tflite_path} - TFLite model for Android")
    print(f"  2. {metadata_path} - Metadata (feature names, classes, scaler)")

    print(f"\n" + "=" * 50)
    print("How to use in Android:")
    print("=" * 50)
    print("""
1. Copy files to Android project:
   - models/sleep_stage_model.tflite → app/src/main/assets/
   - models/tflite_metadata.json → app/src/main/assets/

2. Add TFLite dependency to build.gradle:
   implementation 'org.tensorflow:tensorflow-lite:2.14.0'

3. Load and run in Kotlin:
   ```kotlin
   val interpreter = Interpreter(loadModelFile("sleep_stage_model.tflite"))

   // Input: [1, 10, 23] = 1 batch, 10 epochs, 23 features
   val input = Array(1) { Array(10) { FloatArray(23) } }

   // Fill input with normalized sensor data...

   // Output: [1, 4] = probabilities for [Deep, Light, REM, Wake]
   val output = Array(1) { FloatArray(4) }

   interpreter.run(input, output)

   val classes = listOf("Deep", "Light", "REM", "Wake")
   val prediction = classes[output[0].indices.maxByOrNull { output[0][it] }!!]
   ```
""")

    return tflite_path


if __name__ == "__main__":
    export_to_tflite()
