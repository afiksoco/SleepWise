"""
Cross-dataset comparison — the real test of whether a better dataset helps.

All feature vectors use the 32-feature ANDROID_ORDER with CONSTANT temp (34/0/0),
so DREAMT (Empatica E4, apnea patients) and Walch (Apple Watch, healthy) are
directly comparable and both match what the watch feeds at inference.

Runs (all participant-grouped, leak-free):
  1. DREAMT in-domain CV
  2. Walch  in-domain CV
  3. train DREAMT -> test Walch     (cross-device generalization -- our real problem)
  4. train Walch  -> test DREAMT
  5. DREAMT+Walch combined CV

Headline result: Walch-only gives Deep 83% recall / 24% precision — ~3.5x the
Deep precision of the DREAMT-trained model (7%). Combining with DREAMT dilutes
precision, so Walch-only is the deployed model.
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from features import simplify_labels

HERE = os.path.dirname(__file__)
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


def load(pkl):
    df = pd.read_pickle(pkl)
    lab = simplify_labels(df['label_raw'].values, 'binary_n3')
    ok = np.isin(lab, ['Deep', 'Light'])
    df, lab = df[ok].reset_index(drop=True), lab[ok]
    for c, v in CONST_TEMP.items():
        df[c] = v
    y = (lab == 'Deep').astype(int)   # 1=Deep, 0=Light
    return df, y, df['participant_id'].values


def build():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(ANDROID_ORDER),)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def fit(Xtr_df, ytr):
    Xtr = Xtr_df.reindex(columns=ANDROID_ORDER).fillna(0).values.astype('float32')
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
    cw = {0: cw[0], 1: cw[1] * 1.5}
    m = build()
    m.fit(Xtr, ytr, validation_split=0.15, epochs=60, batch_size=64, class_weight=cw, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)])
    return m, sc


def ev(m, sc, Xte_df, yte):
    Xte = sc.transform(Xte_df.reindex(columns=ANDROID_ORDER).fillna(0).values.astype('float32'))
    return confusion_matrix(yte, m.predict(Xte, verbose=0).argmax(1), labels=[0, 1])


def rp(cm):
    dr = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0
    dp = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() else 0
    lr = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0
    lp = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() else 0
    return dr, dp, lr, lp


def cv(df, y, g, label):
    acc = np.zeros((2, 2), int)
    for tr, te in GroupKFold(5).split(df, y, g):
        m, sc = fit(df.iloc[tr], y[tr]); acc += ev(m, sc, df.iloc[te], y[te])
    return label, rp(acc)


def cross(df_tr, y_tr, df_te, y_te, label):
    m, sc = fit(df_tr, y_tr)
    return label, rp(ev(m, sc, df_te, y_te))


def main():
    dD, yD, gD = load(os.path.join(HERE, '..', 'dreamt_feats.pkl'))
    dW, yW, gW = load(os.path.join(HERE, '..', 'sleepaccel_feats.pkl'))
    gW2 = gW + 1000
    print(f"DREAMT: {len(yD)} epochs Deep={100*yD.mean():.1f}%")
    print(f"Walch : {len(yW)} epochs Deep={100*yW.mean():.1f}%\n", flush=True)

    results = [
        cv(dD, yD, gD, '1. DREAMT in-domain CV'),
        cv(dW, yW, gW, '2. Walch  in-domain CV'),
        cross(dD, yD, dW, yW, '3. train DREAMT -> test WALCH'),
        cross(dW, yW, dD, yD, '4. train WALCH  -> test DREAMT'),
    ]
    dC = pd.concat([dD, dW], ignore_index=True); yC = np.concatenate([yD, yW]); gC = np.concatenate([gD, gW2])
    results.append(cv(dC, yC, gC, '5. DREAMT+WALCH combined CV'))

    print(f"{'scenario':<34}{'Deep rec':>9}{'Deep prec':>10}{'Light rec':>10}{'Light prec':>11}")
    print('-' * 78)
    for label, (dr, dp, lr, lp) in results:
        print(f"{label:<34}{dr*100:>8.1f}%{dp*100:>9.1f}%{lr*100:>9.1f}%{lp*100:>10.1f}%")


if __name__ == '__main__':
    main()
