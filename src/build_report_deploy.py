"""
Deployable 4-STAGE REPORT model (Wake / Light / Deep / REM).

Twin of build_deploy_model.py but 4-class softmax instead of binary. This is the
model behind the morning report — NOT the wake decision (that stays the binary
alarm model). It reuses the EXACT same 45-feature vector, so the on-device
feature extractor needs no change: only a second .tflite + metadata.

Same design as the alarm deploy: pooled Walch + DREAMT + Wearanize+, presence
masks (has_temp/has_hrv), modality dropout, per-subject causal z-scores. No LF/HF
frequency HRV (needs on-device FFT and gave no REM gain in C3). Grouped 5-fold by
subject; test rows forced to masks=1 (the watch always has temp+HRV).

Exports models/report_v1/ (sleep_stage_model.tflite + tflite_metadata.json).
"""
import os, sys, json, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))
from build_combined_model import load
from build_deploy_model import FEATS, nan_standardize_fit, apply_std, modality_dropout, HAS_TEMP_I, HAS_HRV_I
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

CLASSES = ['Wake', 'Light', 'Deep', 'REM']          # output index order (ships in metadata)
CIDX = {c: i for i, c in enumerate(CLASSES)}


def prep4(df):
    lab = simplify_labels(df['label_raw'].values, '4class')
    ok = np.isin(lab, CLASSES)
    df = df[ok].reset_index(drop=True); lab = lab[ok]
    y = np.array([CIDX[c] for c in lab])
    g = df['participant_id'].values; coh = df['cohort'].values
    df = df.copy()
    df['has_temp'] = df['temp_mean'].notna().astype('float32')
    df['has_hrv'] = df['hrv_rmssd'].notna().astype('float32')
    X = df[FEATS].astype('float32').values
    return X, y, g, coh


def make(n, k=4):
    m = tf.keras.Sequential([tf.keras.layers.Input((n,)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(k, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy')
    return m


def class_weights(y, power=0.65):
    # sqrt-balanced: full 'balanced' weights over-boost the rare Deep/REM and crush
    # Light recall (majority class) -> tanks overall accuracy/kappa. power=0.5 keeps
    # Deep/REM visible in the report without destroying the majority stage.
    w = compute_class_weight('balanced', classes=np.unique(y), y=y) ** power
    d = {c: v for c, v in zip(np.unique(y), w)}
    return np.array([d[t] for t in y], dtype='float32')


def main():
    df = load()
    X, y, g, coh = prep4(df)
    print(f"cohorts {sorted(set(coh))}  n={len(y)}")
    u, c = np.unique(y, return_counts=True)
    print("stage dist:", {CLASSES[i]: int(n) for i, n in zip(u, c)})

    seed = np.random.RandomState(42)
    oof = np.full(len(y), -1)
    for tr, te in GroupKFold(5).split(X, y, g):
        Xtr = modality_dropout(X[tr], coh[tr], rng=seed)
        mu, sd = nan_standardize_fit(Xtr); Ztr = apply_std(Xtr, mu, sd)
        m = make(len(FEATS))
        m.fit(Ztr, y[tr], sample_weight=class_weights(y[tr]), validation_split=0.15,
              epochs=60, batch_size=64, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        Xte = X[te].copy(); Xte[:, HAS_TEMP_I] = 1.0; Xte[:, HAS_HRV_I] = 1.0   # watch = masks 1
        oof[te] = m.predict(apply_std(Xte, mu, sd), verbose=0).argmax(1)

    acc = accuracy_score(y, oof); kappa = cohen_kappa_score(y, oof)
    p, r, f, s = precision_recall_fscore_support(y, oof, labels=[0, 1, 2, 3], zero_division=0)
    print(f"\nOVERALL accuracy={acc*100:.1f}%   Cohen's kappa={kappa:.3f}   (grouped 5-fold, masks=1)")
    print(f"{'stage':<7}{'P':>6}{'R':>6}{'F1':>7}{'n':>8}")
    for i, cl in enumerate(CLASSES):
        print(f"{cl:<7}{p[i]*100:>5.0f}%{r[i]*100:>5.0f}%{f[i]:>7.2f}{s[i]:>8}")

    # ---- ship on ALL data ----
    seed2 = np.random.RandomState(7)
    Xd = modality_dropout(X, coh, rng=seed2)
    mu, sd = nan_standardize_fit(Xd); Z = apply_std(Xd, mu, sd)
    model = make(len(FEATS))
    model.fit(Z, y, sample_weight=class_weights(y), validation_split=0.1, epochs=80, batch_size=64,
              verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    OUT = 'models/report_v1'; os.makedirs(OUT, exist_ok=True)
    model.export(OUT + '/saved')
    tflite = tf.lite.TFLiteConverter.from_saved_model(OUT + '/saved').convert()
    open(OUT + '/sleep_stage_model.tflite', 'wb').write(tflite)
    json.dump({'feature_names': FEATS, 'class_names': CLASSES,
               'scaler_mean': mu.tolist(), 'scaler_scale': sd.tolist(),
               'input_shape': [1, len(FEATS)], 'output_shape': [1, 4],
               'label_scheme': '4class', 'missing_fill': 'standardize-then-zero; presence via has_temp/has_hrv',
               'overall_accuracy': round(float(acc), 4), 'cohen_kappa': round(float(kappa), 4),
               'trained_on': 'Walch + DREAMT + Wearanize+ (3 cohorts); 4-stage REPORT model (Wake/Light/Deep/REM); masks + modality dropout'},
              open(OUT + '/tflite_metadata.json', 'w'), indent=2)
    print(f"\nEXPORTED {OUT}/  ({len(FEATS)} features, 4 classes, acc={acc*100:.1f}% kappa={kappa:.3f})")


if __name__ == '__main__':
    main()
