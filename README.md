# SleepWise — Sleep-Stage Model (training & research)

On-device sleep-stage classification for the **SleepWise** smart alarm. A phone
runs a TensorFlow Lite model on live **heart-rate + accelerometer** data streamed
from a Galaxy Watch, and wakes the user during **light sleep** within their chosen
window. This repo holds the **model training pipeline and experiments** — the
Android app and backend live in separate repos.

The model is a small dense classifier (`[1,32] → [1,2]`, Deep vs Light). The
32-feature vector (9 HR stats + temporal derivatives + 2 accelerometer features,
with skin-temp pinned to a constant — see below) is computed on-device per
1-minute epoch; the same feature contract is used here for training.

## TL;DR results

The original DREAMT-trained model looked great on paper but failed on real watch
data (it labelled entire nights "Light"). Three root causes, all fixed:

| # | Problem | Fix | Effect |
|---|---------|-----|--------|
| 1 | **Data leakage** — random train/test split mixed a subject's epochs across both | participant-grouped CV | "89%" Deep recall → honest **~46%** |
| 2 | **Temperature bug** — trained on skin temp, but the watch has no live temp sensor (fed a constant 34 °C) | train with temp pinned to the same constant | Deep recall 37% → **86%** |
| 3 | **Weak teacher** — DREAMT is sleep-apnea patients, only **3.5%** deep sleep | retrain on a consumer-watch, healthy-sleeper dataset (Walch) | see below |

**Final model (Walch, honest participant-grouped 5-fold CV):**

| Model | Deep recall | Deep precision | Light recall |
|-------|-------------|----------------|--------------|
| Old deployed (watch conditions) | ~37% | ~6% | ~80% |
| **Walch-only (deployed)** | **83%** | **24%** | 64% |
| DREAMT+Walch combined | 86% | 17% | 66% |

Walch-only is deployed: same Deep recall as combined but **~1.5× the precision**
(pooling DREAMT's apnea data dilutes it). Deployed artifact:
[`models/walch_binary/`](models/walch_binary/) — drop-in `.tflite` + metadata.

## Datasets (download separately — NOT in this repo)

Both are PhysioNet datasets whose licenses prohibit redistribution, and they are
multi-GB. Download them yourself and place under `data/`:

- **Walch 2019 "sleep-accel"** (primary) — Apple Watch accel (in *g*) + PPG heart
  rate + PSG sleep stages, 31 subjects, apnea excluded.
  https://physionet.org/content/sleep-accel/1.0.0/
  → `data/sleep-accel-dl/physionet.org/files/sleep-accel/1.0.0/`
- **DREAMT** — Empatica E4 (BVP→HR, ACC, skin temp) + PSG, sleep-disorder cohort.
  https://physionet.org/content/dreamt/2.0.0/
  → `data/dreamt/S*_whole_df.csv`

Why Walch matches our hardware better: consumer wrist device (not a research
band), accelerometer already in *g* (no unit-conversion guesswork), sparse PPG HR
like a real watch delivers, and a healthy population with real deep sleep (~12%
vs DREAMT's 3.5%).

## Pipeline

```bash
conda activate sleepwise           # TF 2.16, scikit-learn, pandas

python src/extract_cache.py                 # DREAMT  -> dreamt_feats.pkl
python src/load_sleepaccel.py               # Walch   -> sleepaccel_feats.pkl
python src/experiment_temp_bug.py           # proves the temp bug + leakage (DREAMT)
python src/compare_datasets.py              # cross-dataset comparison (the results table)
DATASET=walch python src/build_final_model.py   # trains + exports the deployed tflite
```

## Files

| File | What |
|------|------|
| `src/features.py` | Feature extraction (HR/temp/accel + temporal derivatives) and label schemes |
| `src/extract_cache.py` | DREAMT → cached feature matrix |
| `src/load_sleepaccel.py` | Walch loader (accel-in-g, sparse HR, 30s→60s labels, NaN-impute missing accel) |
| `src/experiment_temp_bug.py` | Participant-grouped experiment proving the temperature-constant bug |
| `src/compare_datasets.py` | DREAMT vs Walch vs combined, incl. cross-device generalization |
| `src/build_final_model.py` | Trains the shipped model and exports `.tflite` + metadata |
| `models/walch_binary/` | The deployed model + metadata (scaler params, feature order, class names) |

## Design notes

- **Binary by design.** The alarm only needs "don't-wake-me deep" vs "ok-to-wake".
  A 4-class (Wake/Light/Deep/REM) model was tested for the display hypnogram but
  is weaker at the alarm-critical Deep decision, so the alarm stays binary.
- **Temp is a dead feature at inference** (the watch can't measure skin temp
  live), so it's pinned to the training-time constant — the model leans on heart
  rate (the strongest, most device-transferable signal) and motion.
- **Over-calling Deep is the safe direction** for an alarm: a false "Deep" just
  waits (a guaranteed fallback alarm covers it); a false "Light" would wake you
  *from* deep sleep. That's why we optimise Deep recall and tolerate lower
  precision.
