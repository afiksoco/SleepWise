"""
Export GRU model to TensorFlow Lite for Android

GRU is more TFLite-compatible than Bidirectional LSTM.
"""

import os
import json
import numpy as np
import tensorflow as tf
import joblib

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def export_to_tflite():
    print("=" * 50)
    print("Exporting GRU to TensorFlow Lite")
    print("=" * 50)

    # Load the trained Keras model
    print("\n[1/4] Loading Keras model...")
    model = tf.keras.models.load_model('models/sleep_stage_gru.keras')
    model.summary()

    # Load metadata
    metadata = joblib.load('models/gru_metadata.pkl')
    print(f"\nModel info:")
    print(f"  Sequence length: {metadata['sequence_length']} epochs")
    print(f"  Features: {len(metadata['feature_names'])}")
    print(f"  Classes: {list(metadata['label_encoder'].classes_)}")

    # Convert to TFLite - GRU should convert without needing Select TF ops
    print("\n[2/4] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Try standard builtin ops first (more compatible)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    # Don't use optimizations that might bump op versions
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
        print("  Converted with TFLITE_BUILTINS only (best compatibility)")
    except Exception as e:
        print(f"  Builtin-only failed: {e}")
        print("  Retrying with Select TF ops...")

        # Fall back to including Select TF ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()
        print("  Converted with Select TF ops")

    # Save TFLite model
    print("\n[3/4] Saving TFLite model...")
    tflite_path = 'models/sleep_stage_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Get file sizes
    keras_size = os.path.getsize('models/sleep_stage_gru.keras') / 1024 / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024 / 1024

    print(f"\n  Keras model:  {keras_size:.2f} MB")
    print(f"  TFLite model: {tflite_size:.2f} MB")
    print(f"  Compression:  {(1 - tflite_size/keras_size) * 100:.1f}% smaller")

    # Verify the model works
    print("\n[4/4] Verifying TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input shape:  {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")

        # Test inference
        test_input = np.random.randn(1, 10, 23).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"  Test output:  {output[0]}")
        print("  Verification PASSED!")
    except Exception as e:
        print(f"  Verification failed: {e}")
        print("  Model may still work on Android with Flex delegate")

    # Save metadata for Android
    android_metadata = {
        'sequence_length': metadata['sequence_length'],
        'feature_names': metadata['feature_names'],
        'class_names': list(metadata['label_encoder'].classes_),
        'scaler_mean': metadata['scaler'].mean_.tolist(),
        'scaler_scale': metadata['scaler'].scale_.tolist(),
    }

    metadata_path = 'models/tflite_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(android_metadata, f, indent=2)

    print(f"\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nFiles created:")
    print(f"  1. {tflite_path} - TFLite model for Android")
    print(f"  2. {metadata_path} - Metadata (feature names, classes, scaler)")

    print(f"\nNext steps:")
    print(f"  1. Copy models/sleep_stage_model.tflite to Android assets/")
    print(f"  2. Copy models/tflite_metadata.json to Android assets/")
    print(f"  3. Rebuild and test on phone")

    return tflite_path


if __name__ == "__main__":
    export_to_tflite()
