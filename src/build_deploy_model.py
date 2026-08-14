"""
Deployable Stage-3 model: ONE TFLite that consumes the watch's REAL temp + HRV.

Combines the causal-feature precision recipe (the 24%->35% win) with real temp +
time-domain HRV from DREAMT, using presence masks (has_temp/has_hrv) so Walch's
missing temp/HRV are handled honestly.

Two safeguards for on-device correctness (where masks are ALWAYS 1):
  * modality dropout — randomly hide DREAMT temp/HRV in training so the masks
    can't become a "this row is low-Deep DREAMT" shortcut that suppresses Deep.
  * per-subject personalized temp/HRV z-scores (device-scale robustness).

Exports app-ready artifacts:
  models/deploy_v3/sleep_stage_model.tflite
  models/deploy_v3/tflite_metadata.json   (feature_names incl. has_temp/has_hrv, scaler, threshold)

Reports: Walch-cohort Deep (masks=0) vs baseline B0, plus a masks=1 STRESS CHECK
(force has_temp/has_hrv=1 on Walch rows) to prove Deep doesn't collapse.
"""
import os, sys, json, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))
from build_combined_model import load, HR_ACCEL, TEMP, HRV_TIME
from features import simplify_labels
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# personalized (per-subject causal z) columns added by add_causal inside load()
PERS_DEPLOY = ['temp_mean_z', 'hrv_rmssd_z', 'hrv_ibi_mean_z']
TEMP_F = TEMP                     # temp_mean, temp_std, temp_trend
HRV_F = HRV_TIME                  # rmssd, sdnn, pnn50, ibi_mean  (no LF/HF: not worth on-device FFT)
FEATS = HR_ACCEL + TEMP_F + HRV_F + PERS_DEPLOY + ['has_temp', 'has_hrv']
TEMP_IDX = [FEATS.index(c) for c in TEMP_F + ['temp_mean_z']]
HRV_IDX = [FEATS.index(c) for c in HRV_F + ['hrv_rmssd_z', 'hrv_ibi_mean_z']]
HAS_TEMP_I, HAS_HRV_I = FEATS.index('has_temp'), FEATS.index('has_hrv')


def prep(df):
    y = (simplify_labels(df['label_raw'].values, 'binary_n3') == 'Deep').astype(int)
    g = df['participant_id'].values
    coh = df['cohort'].values
    df = df.copy()
    df['has_temp'] = df['temp_mean'].notna().astype('float32')
    df['has_hrv'] = df['hrv_rmssd'].notna().astype('float32')
    X = df[FEATS].astype('float32').values
    return X, y, g, coh


def nan_standardize_fit(X):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd < 1e-6] = 1.0
    return mu, sd


def apply_std(X, mu, sd):
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0)          # missing -> 0 AFTER standardize (masks carry presence)


def cohort_balanced_weights(y, coh, deep_mult=0.6):
    """Equalize the 4 (cohort x class) groups so deep PREVALENCE is identical across
    cohorts. This breaks the 'has temp/HRV = 1  ->  low-Deep DREAMT' correlation that
    was dragging the on-watch alarm down. deep_mult(<1) trims Deep weight for precision."""
    w = np.zeros(len(y), float)
    for c in np.unique(coh):
        for cls in (0, 1):
            mm = (coh == c) & (y == cls)
            if mm.sum() > 0:
                w[mm] = (1.0 / mm.sum()) * (deep_mult if cls == 1 else 1.0)
    return (w * len(y) / w.sum()).astype('float32')


def modality_dropout(X, coh, p=0.5, rng=None):
    """Randomly hide temp and/or HRV on DREAMT rows (set feats NaN + mask 0)."""
    X = X.copy()
    d = np.where(coh == 'dreamt')[0]
    for idxs, has_i in [(TEMP_IDX, HAS_TEMP_I), (HRV_IDX, HAS_HRV_I)]:
        drop = d[rng.random(len(d)) < p]
        X[np.ix_(drop, idxs)] = np.nan
        X[drop, has_i] = 0.0
    return X


def make(n):
    m = tf.keras.Sequential([tf.keras.layers.Input((n,)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy')
    return m


def train_fold(Xtr, ytr, coh_tr, rng):
    Xtr = modality_dropout(Xtr, coh_tr, rng=rng)
    mu, sd = nan_standardize_fit(Xtr)
    Ztr = apply_std(Xtr, mu, sd)
    sw = cohort_balanced_weights(ytr, coh_tr)          # equal Deep prevalence across cohorts
    m = make(len(FEATS))
    m.fit(Ztr, ytr, sample_weight=sw, validation_split=0.15, epochs=60, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return m, mu, sd


def deep_pr(y, p, thr):
    pred = (p >= thr).astype(int); cm = confusion_matrix(y, pred, labels=[0, 1])
    rec = cm[1, 1] / max(cm[1].sum(), 1); prec = cm[1, 1] / max(cm[:, 1].sum(), 1)
    return prec, rec


def main():
    df = load()
    X, y, g, coh = prep(df)
    print(f"deploy features: {len(FEATS)}  (masks: has_temp, has_hrv)")
    print(f"n={len(y)} Deep={y.mean()*100:.1f}% (walch={y[coh=='walch'].mean()*100:.1f}% dreamt={y[coh=='dreamt'].mean()*100:.1f}%)")

    seed = np.random.RandomState(42)
    oof = np.zeros(len(y)); oof_m1 = np.zeros(len(y))    # oof_m1 = masks forced to 1 (deployment condition)
    for tr, te in GroupKFold(5).split(X, y, g):
        m, mu, sd = train_fold(X[tr], y[tr], coh[tr], seed)
        oof[te] = m.predict(apply_std(X[te], mu, sd), verbose=0)[:, 1]
        # stress test: same test rows but pretend temp+HRV are present (as on the watch)
        Xm1 = X[te].copy(); Xm1[:, HAS_TEMP_I] = 1.0; Xm1[:, HAS_HRV_I] = 1.0
        oof_m1[te] = m.predict(apply_std(Xm1, mu, sd), verbose=0)[:, 1]

    wm = coh == 'walch'
    def thr_at_recall(pv, target=0.83):
        p, r, th = precision_recall_curve(y[wm], pv[wm]); i = int(np.argmin(np.abs(r - target)))
        return float(th[min(i, len(th) - 1)])
    thr = thr_at_recall(oof)
    pr, rc = deep_pr(y[wm], oof[wm], thr)
    pr1, rc1 = deep_pr(y[wm], oof_m1[wm], thr)
    # DEPLOYMENT operating point: recalibrate threshold on the masks=1 predictions
    thr_dep = thr_at_recall(oof_m1)
    prd, rcd = deep_pr(y[wm], oof_m1[wm], thr_dep)
    print(f"\n[masks=0] threshold {thr:.3f}: WALCH Deep precision={pr*100:.1f}% recall={rc*100:.1f}%   [B0: 35.4% / 83.0%]")
    print(f"[masks=1, SAME thr]     : WALCH Deep precision={pr1*100:.1f}% recall={rc1*100:.1f}%  (raw collapse)")
    print(f"[masks=1, RECALIBRATED thr {thr_dep:.3f}] DEPLOYMENT point: precision={prd*100:.1f}% recall={rcd*100:.1f}%")
    print("  ^ this last line is the true on-watch operating point (watch is always masks=1).")
    thr = thr_dep  # ship the deployment-calibrated threshold

    # ---- ship on ALL data ----
    seed2 = np.random.RandomState(7)
    Xd = modality_dropout(X, coh, rng=seed2)
    mu, sd = nan_standardize_fit(Xd); Z = apply_std(Xd, mu, sd)
    sw = cohort_balanced_weights(y, coh)
    model = make(len(FEATS))
    model.fit(Z, y, sample_weight=sw, validation_split=0.1, epochs=80, batch_size=64, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    OUT = 'models/deploy_v3'; os.makedirs(OUT, exist_ok=True)
    model.export(OUT + '/saved')
    tflite = tf.lite.TFLiteConverter.from_saved_model(OUT + '/saved').convert()
    open(OUT + '/sleep_stage_model.tflite', 'wb').write(tflite)
    json.dump({'feature_names': FEATS, 'class_names': ['Light', 'Deep'],
               'scaler_mean': mu.tolist(), 'scaler_scale': sd.tolist(),
               'input_shape': [1, len(FEATS)], 'output_shape': [1, 2], 'deep_index': 1,
               'operating_threshold': thr, 'label_scheme': 'binary_n3',
               'missing_fill': 'standardize-then-zero; presence via has_temp/has_hrv',
               'trained_on': 'Walch + DREAMT; REAL temp + time-domain HRV; masks + modality dropout'},
              open(OUT + '/tflite_metadata.json', 'w'), indent=2)
    print(f"\nEXPORTED {OUT}/  ({len(FEATS)} features, thr={thr:.3f})")
    print("feature order:", FEATS)


if __name__ == '__main__':
    main()
