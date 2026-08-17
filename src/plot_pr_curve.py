"""
Precision-Recall curve for the DEPLOYED alarm model (binary Deep vs Light),
measured at the on-watch operating condition (masks=1), Walch cohort, grouped
5-fold CV. Marks the 83%-recall operating point (the deployment threshold) and
the deep-prevalence no-skill baseline. Saves a PNG for the defense deck.

Reuses the exact deploy pipeline so the curve reflects the shipped model.
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
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf

df = load()
X, y, g, coh = prep(df)
print(f"cohorts {sorted(set(coh))} n={len(y)}")

seed = np.random.RandomState(42)
oof = np.zeros(len(y))          # masks=1 (on-watch deployment condition)
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
pr_auc = auc(rec, prec)
prevalence = yv.mean()

# operating point: threshold that gives recall ~0.83
i = int(np.argmin(np.abs(rec - 0.83)))
op_p, op_r = prec[i], rec[i]
op_thr = float(thr[min(i, len(thr) - 1)])
print(f"AUC={pr_auc:.3f} prevalence={prevalence:.3f} | op: P={op_p:.3f} R={op_r:.3f} thr={op_thr:.3f}")

# ── plot ──
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec, prec, color='#D9765A', lw=2.4, label=f'Deployed model (AUC = {pr_auc:.2f})')
ax.axhline(prevalence, ls='--', color='#9CACB9', lw=1.4,
           label=f'No-skill baseline (deep prevalence = {prevalence*100:.0f}%)')
# operating point
ax.scatter([op_r], [op_p], s=130, color='#2E5163', zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'Operating point\nrecall {op_r*100:.0f}%  ·  precision {op_p*100:.0f}%\nthreshold {op_thr:.3f}',
            xy=(op_r, op_p), xytext=(op_r - 0.5, op_p + 0.22),
            fontsize=11, color='#101922',
            arrowprops=dict(arrowstyle='->', color='#2E5163', lw=1.5))
ax.set_xlabel('Recall  (fraction of real deep sleep caught)')
ax.set_ylabel('Precision  (fraction of "Deep" calls correct)')
ax.set_title('Deep-sleep detection — Precision vs Recall\n(deployed 4-signal model, Walch cohort, grouped 5-fold, on-watch masks=1)',
             fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.25); ax.legend(loc='upper right', framealpha=0.95)
fig.tight_layout()
out = os.path.expanduser('~/Downloads/pr_curve_alarm.png')
fig.savefig(out, dpi=150)
print(f"SAVED {out}")
