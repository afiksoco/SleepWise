"""
Retrain the deployed dense [1,N]->[1,2] sleep-stage classifier with:
  - Accelerometer features (acc_std, acc_move_ratio) appended at the end
  - Deep = N3 ONLY label scheme (REM is now wakeable/Light)
  - Reduced Deep class weight (was 8x; movement signal lets us dial it down)

Matches the Android inference contract: a single flat feature vector per epoch
(Android computes the temporal features itself from its 5-epoch buffer), NOT a
sequence model. Feature column order is pinned to Android's computeTemporalFeatures
output, with the two accel features appended last.
"""
import os, sys, json, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, tensorflow as tf, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features_from_participant, simplify_labels

# Exact 30-feature order Android's computeTemporalFeatures emits, then accel appended.
ANDROID_ORDER = [
    'hr_mean','hr_std','hr_min','hr_max','hr_range','hr_cv','hr_median','hr_iqr','hr_skew',
    'temp_mean','temp_std','temp_trend',
    'hr_mean_lag1','hr_mean_lag2','hr_mean_lag3','hr_mean_lag4',
    'hr_mean_rolling_mean','hr_mean_rolling_std','hr_mean_trend','hr_mean_roc',
    'temp_mean_lag1','temp_mean_lag2','temp_mean_lag3','temp_mean_lag4',
    'temp_mean_rolling_mean','temp_mean_rolling_std','temp_mean_trend','temp_mean_roc',
    'hr_stability','sleep_cycle_position',
    'acc_std','acc_move_ratio',          # appended — positions 30,31
]
MAX_SUBJECTS = int(os.environ.get('MAX_SUBJECTS', '30'))
DREAMT = os.path.expanduser('~/PycharmProjects/SleepWise/data/dreamt')

def load():
    files = sorted(glob.glob(os.path.join(DREAMT, 'S*_whole_df.csv')))[:MAX_SUBJECTS]
    feats, labs, pids = [], [], []
    pid = 0
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['HR','TEMP','ACC_X','ACC_Y','ACC_Z','Sleep_Stage'])
            fdf, lab = extract_features_from_participant(df, add_temporal=True)
            if len(lab) > 10:
                fdf['participant_id'] = pid
                feats.append(fdf); labs.append(lab); pids += [pid]*len(lab); pid += 1
                print(f"  {os.path.basename(f)}: {len(lab)} epochs", flush=True)
        except Exception as e:
            print(f"  {os.path.basename(f)} ERROR {e}", flush=True)
    return pd.concat(feats, ignore_index=True), np.concatenate(labs), np.array(pids)

def main():
    print("[1/5] loading DREAMT (max %d subjects)..." % MAX_SUBJECTS, flush=True)
    fdf, labels, pids = load()
    print(f"total epochs: {len(labels)}", flush=True)

    labels = simplify_labels(labels, 'binary_n3')
    valid = labels != 'Unknown'
    fdf, labels, pids = fdf[valid].reset_index(drop=True), labels[valid], pids[valid]
    # distribution
    u,c = np.unique(labels, return_counts=True)
    print("label dist:", dict(zip(u, c.tolist())), flush=True)

    # pin column order
    X = fdf.reindex(columns=ANDROID_ORDER).fillna(0).values.astype('float32')
    le = LabelEncoder(); y = le.fit_transform(labels)
    print("classes:", list(le.classes_), flush=True)  # expect ['Deep','Light']

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte, ptr, pte = train_test_split(
        Xs, y, pids, test_size=0.2, random_state=42, stratify=y)

    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    cw = dict(enumerate(cw))
    deep_idx = list(le.classes_).index('Deep')
    cw[deep_idx] *= 1.5   # mild safety boost (was 8x in old pipeline)
    print("class weights:", cw, flush=True)

    print("[2/5] building dense model...", flush=True)
    n = X.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("[3/5] training...", flush=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=80, batch_size=64,
              class_weight=cw, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])

    print("[4/5] evaluating...", flush=True)
    yp = model.predict(Xte, verbose=0).argmax(1)
    print(classification_report(yte, yp, target_names=le.classes_), flush=True)
    cm = confusion_matrix(yte, yp)
    print("confusion (rows=true, cols=pred), classes:", list(le.classes_), flush=True)
    print(cm, flush=True)
    for i, cls in enumerate(le.classes_):
        rec = cm[i,i]/cm[i].sum() if cm[i].sum() else 0
        prec = cm[i,i]/cm[:,i].sum() if cm[:,i].sum() else 0
        print(f"  {cls}: recall={rec*100:.1f}% precision={prec*100:.1f}%", flush=True)

    print("[5/5] exporting tflite + metadata...", flush=True)
    os.makedirs('models', exist_ok=True)
    model.save('models/sleep_stage_accel.keras')
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = conv.convert()
    with open('models/sleep_stage_model.tflite','wb') as f: f.write(tfl)
    meta = {
        'feature_names': ANDROID_ORDER,
        'class_names': list(le.classes_),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'input_shape': [1, n],
        'output_shape': [1, len(le.classes_)],
        'label_scheme': 'binary_n3',
        'deep_class_weight_mult': 1.5,
    }
    with open('models/tflite_metadata.json','w') as f: json.dump(meta, f, indent=2)
    print("DONE. tflite size:", os.path.getsize('models/sleep_stage_model.tflite'), "bytes", flush=True)

if __name__ == '__main__':
    main()
