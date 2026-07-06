"""
Prove the temperature-constant bug and compare fixes — on DREAMT.

Splits by PARTICIPANT (GroupKFold) — no epoch leakage — and evaluates every
model under WATCH-LIKE inference (skin temp pinned to the 34C constant the
Android app actually feeds, since the watch has no live skin-temp sensor).

Variants:
  A_real   baseline (temp from DREAMT), tested with real temp    [the old, leaky-free eval]
  A_watch  baseline (temp from DREAMT), tested with CONST temp    [reality on the watch]
  B_const  model TRAINED with const temp, tested with const temp  [fix: 32 feats, no app change]
  C_notemp model trained WITHOUT temp (21 feats)                  [fix: needs app change]

Headline result (30 subjects): the "89% Deep recall" was participant leakage
(honest baseline ~46%); the temp bug drops it further to ~37%; training with
temp neutralised (B_const) recovers Deep recall to ~86%.
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

CACHE = os.path.join(os.path.dirname(__file__), '..', 'dreamt_feats.pkl')
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
NOTEMP_ORDER = [c for c in ANDROID_ORDER if 'temp' not in c]
CONST_TEMP = {'temp_mean':34.0,'temp_std':0.0,'temp_trend':0.0,'temp_mean_lag1':34.0,
    'temp_mean_lag2':34.0,'temp_mean_lag3':34.0,'temp_mean_lag4':34.0,
    'temp_mean_rolling_mean':34.0,'temp_mean_rolling_std':0.0,'temp_mean_trend':0.0,'temp_mean_roc':0.0}


def watchify(df):
    d = df.copy()
    for c, v in CONST_TEMP.items():
        d[c] = v
    return d


def build(n):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def train_eval(Xtr_df, ytr, Xte_df, yte, cols, deep_idx):
    Xtr = Xtr_df.reindex(columns=cols).fillna(0).values.astype('float32')
    Xte = Xte_df.reindex(columns=cols).fillna(0).values.astype('float32')
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    cw = dict(enumerate(cw)); cw[deep_idx] *= 1.5
    m = build(len(cols))
    m.fit(Xtr, ytr, validation_split=0.15, epochs=60, batch_size=64, class_weight=cw, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)])
    return confusion_matrix(yte, m.predict(Xte, verbose=0).argmax(1), labels=[0, 1])


def metrics(cm, deep, light):
    def rp(i):
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0
        return rec, prec
    return (*rp(deep), *rp(light))


def main():
    df = pd.read_pickle(CACHE)
    labels = simplify_labels(df['label_raw'].values, 'binary_n3')
    ok = labels != 'Unknown'
    df, labels = df[ok].reset_index(drop=True), labels[ok]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); y = le.fit_transform(labels)
    classes = list(le.classes_); deep = classes.index('Deep'); light = classes.index('Light')
    pids = df['participant_id'].values
    u, c = np.unique(labels, return_counts=True)
    print(f"epochs={len(y)} subjects={df['participant_id'].nunique()} classes={classes}")
    print(f"label dist: {dict(zip(u, c.tolist()))}  (Deep={100*c[list(u).index('Deep')]/len(y):.1f}%)\n", flush=True)

    variants = ['A_real', 'A_watch', 'B_const', 'C_notemp']
    acc = {v: np.zeros((2, 2), int) for v in variants}
    for tr, te in GroupKFold(5).split(df, y, pids):
        tr_df, te_df, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]
        acc['A_real']   += train_eval(tr_df, ytr, te_df, yte, ANDROID_ORDER, deep)
        acc['A_watch']  += train_eval(tr_df, ytr, watchify(te_df), yte, ANDROID_ORDER, deep)
        acc['B_const']  += train_eval(watchify(tr_df), ytr, watchify(te_df), yte, ANDROID_ORDER, deep)
        acc['C_notemp'] += train_eval(tr_df, ytr, te_df, yte, NOTEMP_ORDER, deep)

    notes = {'A_real': 'baseline, REAL temp test (leak-free)',
             'A_watch': 'baseline, WATCH const-temp  <-- reality',
             'B_const': 'FIX: train+test const temp (no app change)',
             'C_notemp': 'FIX: drop temp entirely (needs app change)'}
    print(f"{'variant':<10}{'Deep rec':>9}{'Deep prec':>10}{'Light rec':>10}{'Light prec':>11}   note")
    print('-' * 78)
    for v in variants:
        dr, dp, lr, lp = metrics(acc[v], deep, light)
        print(f"{v:<10}{dr*100:>8.1f}%{dp*100:>9.1f}%{lr*100:>9.1f}%{lp*100:>10.1f}%   {notes[v]}")
    print("\n(participant-grouped 5-fold; metrics pooled across folds)")


if __name__ == '__main__':
    main()
