"""
Build the deployable binary alarm model (Walch-only by default; DATASET=combined
to pool DREAMT+Walch).

- feature contract: identical 32-feature ANDROID_ORDER, temp pinned constant
  (drop-in .tflite swap for the Android app, no client code change)
- classes via LabelEncoder -> ['Deep','Light'] (matches the deployed metadata)
- reports honest participant-grouped 5-fold CV, THEN fits the shipped model on
  ALL data and exports tflite + metadata (StandardScaler params included).

The from_keras_model TFLite converter hits an MLIR crash on TF 2.16, so we export
via SavedModel then convert.
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from features import simplify_labels

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, '..', 'models', 'walch_binary')
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


def load(pkl, pid_offset):
    df = pd.read_pickle(pkl)
    lab = simplify_labels(df['label_raw'].values, 'binary_n3')
    ok = np.isin(lab, ['Deep', 'Light'])
    df, lab = df[ok].reset_index(drop=True), lab[ok]
    for c, v in CONST_TEMP.items():
        df[c] = v
    X = df.reindex(columns=ANDROID_ORDER).fillna(0).values.astype('float32')
    return X, lab.astype(str), df['participant_id'].values + pid_offset


def make():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(ANDROID_ORDER),)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def main():
    DATASET = os.environ.get('DATASET', 'walch')   # 'walch' or 'combined'
    Xd, ld, gd = load(os.path.join(HERE, '..', 'dreamt_feats.pkl'), 0)
    Xw, lw, gw = load(os.path.join(HERE, '..', 'sleepaccel_feats.pkl'), 100000)
    if DATASET == 'combined':
        X, lab, g = np.vstack([Xd, Xw]), np.concatenate([ld, lw]), np.concatenate([gd, gw])
    else:
        X, lab, g = Xw, lw, gw
    le = LabelEncoder(); y = le.fit_transform(lab)     # ['Deep','Light'] -> Deep=0, Light=1
    classes = list(le.classes_); deep = classes.index('Deep')
    print(f"[{DATASET}] {len(y)} epochs, {len(np.unique(g))} subjects, Deep={100*(lab=='Deep').mean():.1f}%\n", flush=True)

    cm = np.zeros((2, 2), int)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y[tr]); cw = {0: cw[0], 1: cw[1]}; cw[deep] *= 1.5
        m = make()
        m.fit(Xtr, y[tr], validation_split=0.15, epochs=60, batch_size=64, class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)])
        cm += confusion_matrix(y[te], m.predict(Xte, verbose=0).argmax(1), labels=[0, 1])
    dr = cm[deep, deep] / cm[deep].sum(); dp = cm[deep, deep] / cm[:, deep].sum()
    print(f"HONEST grouped-CV: Deep recall={dr*100:.1f}% precision={dp*100:.1f}%\n", flush=True)

    # shipped model on ALL data
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y); cw = {0: cw[0], 1: cw[1]}; cw[deep] *= 1.5
    model = make()
    model.fit(Xs, y, validation_split=0.1, epochs=80, batch_size=64, class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])
    os.makedirs(OUT, exist_ok=True)
    saved = os.path.join(OUT, 'saved'); model.export(saved)
    tfl = tf.lite.TFLiteConverter.from_saved_model(saved).convert()
    with open(os.path.join(OUT, 'sleep_stage_model.tflite'), 'wb') as f:
        f.write(tfl)
    meta = {'feature_names': ANDROID_ORDER, 'class_names': classes,
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
            'input_shape': [1, len(ANDROID_ORDER)], 'output_shape': [1, 2],
            'label_scheme': 'binary_n3', 'deep_class_weight_mult': 1.5,
            'trained_on': f'{DATASET}; temp pinned to 34C constant'}
    with open(os.path.join(OUT, 'tflite_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"EXPORTED -> {OUT}/  class order {dict(enumerate(classes))}", flush=True)


if __name__ == '__main__':
    main()
