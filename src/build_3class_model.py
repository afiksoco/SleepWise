"""
4-class variant of build_final_model.py — Wake / Light / Deep / REM.

Same 32-feature ANDROID_ORDER contract, same participant-grouped 5-fold CV, same
Dense net but a Dense(4, softmax) head. Purpose: measure HONEST per-class quality
(full 4x4 confusion matrix) so we can decide whether the 4-stage report is
trustworthy — especially REM/Wake, which are hard from HR+accel without EEG.

The alarm still runs the binary model; this model only feeds the report display.
Walch-only (DREAMT is an apnea population and not representative).
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
sys.path.insert(0, os.path.dirname(__file__))
from features import simplify_labels

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, '..', 'models', 'walch_3class')
ANDROID_ORDER = [
    'hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
    'temp_mean','temp_std','temp_trend',
    'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
    'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
    'temp_mean_lag1','temp_mean_lag2','temp_mean_lag3','temp_mean_lag4',
    'temp_mean_rolling_mean','temp_mean_rolling_std','temp_mean_trend','temp_mean_roc',
    'hr_stability','sleep_cycle_position',
    'acc_std','acc_move_ratio',
]
CONST_TEMP = {'temp_mean':34.0,'temp_std':0.0,'temp_trend':0.0,'temp_mean_lag1':34.0,
    'temp_mean_lag2':34.0,'temp_mean_lag3':34.0,'temp_mean_lag4':34.0,
    'temp_mean_rolling_mean':34.0,'temp_mean_rolling_std':0.0,'temp_mean_trend':0.0,'temp_mean_roc':0.0}
# Fixed, readable class order (not LabelEncoder's alphabetical).
CLASSES = ['Wake', 'Light', 'Deep']


def load(pkl):
    df = pd.read_pickle(pkl)
    lab = simplify_labels(df['label_raw'].values, '3class')
    ok = np.isin(lab, CLASSES)
    df, lab = df[ok].reset_index(drop=True), lab[ok]
    for c, v in CONST_TEMP.items():
        df[c] = v
    X = df.reindex(columns=ANDROID_ORDER).fillna(0).values.astype('float32')
    return X, lab.astype(str), df['participant_id'].values


def make():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(ANDROID_ORDER),)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def cls_to_idx(lab):
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx[x] for x in lab])


def print_cm(cm, title):
    n = len(CLASSES)
    total = cm.sum()
    print(f"\n{title}")
    print("            " + "".join(f"{c:>8}" for c in CLASSES) + "   | recall")
    for i in range(n):
        row = cm[i]
        rec = row[i] / row.sum() if row.sum() else 0.0
        print(f"  true {CLASSES[i]:<5} " + "".join(f"{v:>8}" for v in row) + f"   | {rec*100:5.1f}%")
    print("  precision  " + "".join(
        f"{(cm[j, j] / cm[:, j].sum() * 100 if cm[:, j].sum() else 0):>7.1f}%" for j in range(n)))
    acc = np.trace(cm) / total if total else 0
    print(f"  overall accuracy = {acc*100:.1f}%   (n={total})")


def main():
    X, lab, g = load(os.path.join(HERE, '..', 'sleepaccel_feats.pkl'))
    y = cls_to_idx(lab)
    print(f"[walch 4-class] {len(y)} epochs, {len(np.unique(g))} subjects")
    for i, c in enumerate(CLASSES):
        print(f"    {c:<5}: {(y==i).sum():5d}  ({100*(y==i).mean():4.1f}%)")

    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    f1s, kappas = [], []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        cw = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=y[tr])
        cwd = {i: cw[i] for i in range(len(CLASSES))}
        m = make()
        m.fit(Xtr, y[tr], validation_split=0.15, epochs=60, batch_size=64, class_weight=cwd, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)])
        pred = m.predict(Xte, verbose=0).argmax(1)
        cm += confusion_matrix(y[te], pred, labels=np.arange(len(CLASSES)))
        f1s.append(f1_score(y[te], pred, average='macro'))
        kappas.append(cohen_kappa_score(y[te], pred))
        print(f"  fold {fold+1} done", flush=True)

    print_cm(cm, "HONEST participant-grouped 5-fold CV confusion matrix")
    print(f"  macro-F1 = {np.mean(f1s)*100:.1f}%   Cohen's kappa = {np.mean(kappas):.3f}")

    # Binary-collapse sanity: how does the 4-class model do at the ALARM job
    # (Deep vs not-Deep)? This is what actually drives the wake decision.
    di = CLASSES.index('Deep')
    tp = cm[di, di]; fn = cm[di].sum() - tp; fp = cm[:, di].sum() - tp
    print(f"\n  [alarm view] Deep-vs-rest: recall={tp/(tp+fn)*100:.1f}%  precision={tp/(tp+fp)*100:.1f}%")

    if os.environ.get('EXPORT') == '1':
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        cw = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=y)
        cwd = {i: cw[i] for i in range(len(CLASSES))}
        model = make()
        model.fit(Xs, y, validation_split=0.1, epochs=80, batch_size=64, class_weight=cwd, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])
        os.makedirs(OUT, exist_ok=True)
        saved = os.path.join(OUT, 'saved'); model.export(saved)
        tfl = tf.lite.TFLiteConverter.from_saved_model(saved).convert()
        with open(os.path.join(OUT, 'sleep_stage_model.tflite'), 'wb') as f:
            f.write(tfl)
        meta = {'feature_names': ANDROID_ORDER, 'class_names': CLASSES,
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                'input_shape': [1, len(ANDROID_ORDER)], 'output_shape': [1, len(CLASSES)],
                'label_scheme': '3class', 'trained_on': 'walch; temp pinned to 34C constant'}
        with open(os.path.join(OUT, 'tflite_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"\nEXPORTED -> {OUT}/  class order {dict(enumerate(CLASSES))}")


if __name__ == '__main__':
    main()
