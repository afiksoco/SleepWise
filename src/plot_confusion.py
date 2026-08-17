"""
Confusion matrix for the DEPLOYED alarm model (binary Deep vs Light) at the
deployment operating point (recall ~0.83), Walch cohort, grouped 5-fold CV,
on-watch condition (masks=1). Saves a PNG for the defense deck.
"""
import os, sys, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_combined_model import load
from build_deploy_model import (FEATS, prep, nan_standardize_fit, apply_std,
                                 modality_dropout, make, cohort_balanced_weights,
                                 HAS_TEMP_I, HAS_HRV_I)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix
import tensorflow as tf

df = load()
X, y, g, coh = prep(df)

seed = np.random.RandomState(42)
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, g):
    Xtr = modality_dropout(X[tr], coh[tr], rng=seed)
    mu, sd = nan_standardize_fit(Xtr); Ztr = apply_std(Xtr, mu, sd)
    m = make(len(FEATS))
    m.fit(Ztr, y[tr], sample_weight=cohort_balanced_weights(y[tr], coh[tr]),
          validation_split=0.15, epochs=60, batch_size=64, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    Xm1 = X[te].copy(); Xm1[:, HAS_TEMP_I] = 1.0; Xm1[:, HAS_HRV_I] = 1.0
    oof[te] = m.predict(apply_std(Xm1, mu, sd), verbose=0)[:, 1]

wm = coh == 'walch'
yv, pv = y[wm], oof[wm]
prec, rec, thr = precision_recall_curve(yv, pv)
i = int(np.argmin(np.abs(rec - 0.83)))
op_thr = float(thr[min(i, len(thr) - 1)])
pred = (pv >= op_thr).astype(int)

# rows = actual (Light, Deep), cols = predicted (Light, Deep)
cm = confusion_matrix(yv, pred, labels=[0, 1])
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
recall = tp / (tp + fn); precision = tp / (tp + fp)
light_prec = tn / (tn + fn); light_rec = tn / (tn + fp)
acc = (tp + tn) / cm.sum()
print(f"thr={op_thr:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Deep P={precision:.3f} R={recall:.3f} | Light P={light_prec:.3f} R={light_rec:.3f} | acc={acc:.3f}")

# ── plot ──
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
labels = ['Light', 'Deep']
fig, ax = plt.subplots(figsize=(6.4, 6))
im = ax.imshow(cm, cmap='OrRd')
ax.set_xticks([0, 1], labels=[f'Pred\n{l}' for l in labels])
ax.set_yticks([0, 1], labels=[f'Actual\n{l}' for l in labels])
rowsum = cm.sum(axis=1, keepdims=True)
for r in range(2):
    for cix in range(2):
        n = cm[r, cix]; pct = 100 * n / rowsum[r, 0]
        color = 'white' if n > cm.max() * 0.55 else '#101922'
        tag = {(0,0):'TN',(0,1):'FP',(1,0):'FN',(1,1):'TP'}[(r,cix)]
        ax.text(cix, r, f'{tag}\n{n:,}\n{pct:.0f}%', ha='center', va='center',
                fontsize=13, color=color, fontweight='bold')
ax.set_title('Alarm model confusion matrix\n(Deep vs Light · Walch · grouped 5-fold · masks=1 · recall-calibrated)',
             fontsize=11.5)
sub = (f'Deep recall {recall*100:.0f}%   Deep precision {precision*100:.0f}%   '
       f'Light precision {light_prec*100:.0f}%   accuracy {acc*100:.0f}%')
fig.text(0.5, 0.02, sub, ha='center', fontsize=10.5, color='#55636F')
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.expanduser('~/Downloads/confusion_alarm.png')
fig.savefig(out, dpi=150)
print(f"SAVED {out}")
