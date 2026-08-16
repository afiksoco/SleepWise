#!/usr/bin/env python3
"""
SleepWise Proof of Concept - Full Pipeline

This script runs the complete ML pipeline:
1. Generate mock data (simulating DREAMT dataset)
2. Train the sleep stage classifier
3. Run a demo simulation

Run this to see the entire system working end-to-end!
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    print("=" * 70)
    print("   SleepWise - Proof of Concept")
    print("   Smart Sleep Stage Classification & Alarm System")
    print("=" * 70)

    # Step 1: Generate mock data
    print("\n" + "=" * 70)
    print("STEP 1: Generating Mock Dataset")
    print("=" * 70)
    print("(This simulates the DREAMT dataset you'll download from PhysioNet)")

    from src.mock_data_generator import generate_dataset
    generate_dataset(n_participants=10, output_dir='data/mock', seed=42)

    # Step 2: Train model
    print("\n" + "=" * 70)
    print("STEP 2: Training Sleep Stage Classifier")
    print("=" * 70)

    from src.train import main as train_main
    train_main()

    # Step 3: Demo inference
    print("\n" + "=" * 70)
    print("STEP 3: Running Smart Alarm Simulation")
    print("=" * 70)

    from src.demo_inference import main as demo_main
    demo_main()

    # Summary
    print("\n" + "=" * 70)
    print("POC COMPLETE!")
    print("=" * 70)
    print("""
What we demonstrated:
1. ✓ Generated synthetic sleep data (mock DREAMT format)
2. ✓ Extracted features from raw sensor data
3. ✓ Trained a Random Forest classifier
4. ✓ Simulated real-time sleep stage prediction
5. ✓ Smart alarm triggered during light sleep

Next steps:
1. Download real DREAMT dataset from PhysioNet
2. Replace 'data/mock' with 'data/dreamt' in train.py
3. Retrain on real data
4. Port to Android (Kotlin) app

Files created:
- data/mock/         → Synthetic participant data
- models/            → Trained model (.pkl)
- src/               → Python source code
""")


if __name__ == "__main__":
    main()
